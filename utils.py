import tensorflow as tf


IMG_SIZE = 256


def count_data_size(data):
    return len(data)


def random_crop(img):
    return tf.image.random_crop(img,
                                size=[IMG_SIZE, IMG_SIZE, 3])


def center_crop(img, crop_size=178):
    return tf.image.central_crop(img, crop_size / IMG_SIZE)


def normalize(img):
    # convert img vals from [0, 255] to [-1, 1]
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img


def denormalize(img):
    # convert img vals from [-1, 1] to [0, 1]
    return img / 2.0 + 0.5


def random_horizontal_flip(img):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
    return img


def resize(img, size=128):
    img = tf.image.resize(img, 
                          [size, size],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def read_and_decode_img(img_path):
    # from image path to string
    img_string = tf.io.read_file(img_path)
    # from jpg-encoded image to a uint8
    img = tf.io.decode_jpeg(img_string)
    return img


def preprocess_img(img, center_crop_size=178, size=128, use_aug=True):
    if use_aug:
        img = random_horizontal_flip(img)
    img = center_crop(img, center_crop_size)
    img = resize(img, size)
    img = normalize(img)
    return img


def preprocess_for_training(img, label_org):
    # For image preprocessing
    img = preprocess_img(img)
    # Generate target domain labels randomly
    label_trg = tf.random.shuffle(label_org)

    return img, label_org, label_trg


def get_gradient_penalty(x, x_gen, discriminator):
    """
    for the implementation of the gradient penalty, I referred to the links below.
    https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
    https://qiita.com/triwave33/items/72c7fceea2c6e48c8c07
    """
    # shape=[x.shape[0], 1, 1, 1] to generate a random number for every sample
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as tape:
        # to get a gradient w.r.t x_hat, we need to record the value on the tape
        tape.watch(x_hat)
        d_hat = discriminator(x_hat, training=True)
    
    gradients = tape.gradient(d_hat, x_hat)
    l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
    gp_loss = tf.reduce_mean((l2_norm - 1.0) ** 2)
    return gp_loss


def get_classification_loss(target, logits):
    logits = tf.squeeze(logits)
    # Compute binary or softmax cross entropy loss.
    loss_total = tf.keras.losses.BinaryCrossentropy(from_logits=True)(target, 
                                                                      logits)
    loss = tf.reduce_mean(loss_total)
    return loss


def get_mean_for_loss(out_src):
    return tf.reduce_mean(out_src)


def get_l1_loss(x_real, x_rec):
    return tf.reduce_mean(tf.abs(x_real - x_rec))


@tf.function
def get_lr_decay_factor(epoch, max_epoch, init_lr=0.0001):
    return tf.cast(2.0 * init_lr * (- tf.cast(epoch, tf.float32) / tf.cast(max_epoch, tf.float32) + 1.0), dtype=tf.float32)


@tf.function
def update_lr(gen_opt, disc_opt, max_epoch, epoch, g_lr=0.0001, d_lr=0.0001):
    if g_lr != d_lr:
        decayed_lr = get_lr_decay_factor(epoch, max_epoch, g_lr)
        gen_opt.lr.assign(decayed_lr)
        disc_opt.lr.assign(decayed_lr)
        # to debug
        print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))
    else:
        g_decayed_lr =  get_lr_decay_factor(epoch, 
                                            max_epoch,
                                            g_lr)
        d_decayed_lr =  get_lr_decay_factor(epoch, 
                                            max_epoch,
                                            d_lr)
        gen_opt.lr.assign(g_decayed_lr)
        disc_opt.lr.assign(d_decayed_lr)
        # to debug
        print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))


@tf.function
def train_step(step, 
               gen, 
               disc, 
               x_real, 
               label_org, 
               label_trg, 
               lambda_cls, 
               lambda_gp, 
               lambda_rec, 
               num_critic_updates, 
               disc_opt, 
               gen_opt):

    g_loss_fake = None
    g_loss_cls = None
    g_loss = None

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        # For the discriminator
        #Compute loss with real images
        real_out_src, real_out_cls = disc(x_real, training=True)
        d_loss_real = - get_mean_for_loss(real_out_src)
        d_loss_cls = get_classification_loss(label_org, real_out_cls)
        # Compute loss with fake images
        x_fake = gen(x_real, label_trg, training=False)
        fake_out_src, fake_out_cls = disc(x_fake, training=True)
        d_loss_fake = get_mean_for_loss(fake_out_src)
        # Compute loss for gradient penalty
        d_loss_gp = get_gradient_penalty(x_real, x_fake, disc)
        # Compute the total loss for the discriminator
        d_loss = d_loss_real + d_loss_fake + lambda_gp * d_loss_gp + lambda_cls * d_loss_cls

        # For the generator
        if step % num_critic_updates == 0:
            # Compute loss for original-to-target domain
            x_fake = gen(x_real, label_trg)
            gen_out_src, gen_out_cls = disc(x_fake)
            g_loss_fake = - get_mean_for_loss(gen_out_src)
            g_loss_cls = get_classification_loss(label_trg, gen_out_cls)
            # Compute loss for target-to-original domain
            x_rec = gen(x_fake, label_trg)
            g_loss_rec = get_l1_loss(x_real, x_rec)
            # Compute the total loss for the generator
            g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls

    # Calculate the gradients and update params for the discriminator and the generator
    disc_gradients = disc_tape.gradient(d_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_gradients, disc.trainable_variables))
    if g_loss is not None:
        gen_gradients = gen_tape.gradient(g_loss, gen.trainable_variables)
        gen_opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    return d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp, d_loss, g_loss_fake, g_loss_cls, g_loss


    @tf.function
    def print_log(epoch, start, end, losses):
        print("\nTime taken for epoch {} is {} sec\n".format(epoch, 
                                                             end-start))
        print("d_loss_real: {}, d_loss_fake: {}, d_loss_cls: {}, d_loss_gp: {}, " +\
            "d_loss: {}, g_loss_fake: {}, g_loss_cls: {}, g_loss: {}".format(losses[0],
                                                                            losses[1],
                                                                            losses[2],
                                                                            losses[3],
                                                                            losses[4],
                                                                            losses[5],
                                                                            losses[6],
                                                                            losses[7]))

    def preprocess_for_testing(img, c_trg):
        x = tf.expand_dims(img, axis=0)
        c = tf.expand_dims(c_trg, axis=0)
        x = tf.convert_to_tensor(x)
        c = tf.convert_to_tensor(c)

        return x, c


    def save_img(tensor, fpath):
        h, w, c = tensor.shape
        tensor = denormalize(tensor)
        tensor = tf.cast(tensor, dtype=tf.uint8) * 255
        tensor = tf.reshape(tensor, [h//2, w//2, c])
        bstr = tf.io.encode_jpeg(tensor)
        with open(fpath, "wb") as f:
            f.write(bstr)


    @tf.function
    def save_test_results(model, img_list, trg_list, fpath):
        results = []
        for img, c_trg in zip(img_list, trg_list):
            img = preprocess_img(img, use_aug=False)
            x, c = preprocess_for_testing(img, c_trg)
            result = model(x, c)
            result = tf.squeeze(result, axis=0)
            results.append(result)
        tensor = tf.concat(results, axis=1)
        save_img(tensor, fpath)