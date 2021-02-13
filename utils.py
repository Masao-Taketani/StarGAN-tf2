import matplotlib.pyplot as plt

import tensorflow as tf

from model import build_model


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


def preprocess_img(img,
                   center_crop_size=178,
                   size=128,
                   use_aug=True,
                   do_center_crop=True,
                   do_normalize=True):

    if use_aug:
        img = random_horizontal_flip(img)
    if do_center_crop:
        img = center_crop(img, center_crop_size)
    img = resize(img, size)
    if do_normalize:
        img = normalize(img)
    return img


def preprocess_for_training(img, label_org):
    # For image preprocessing
    img = preprocess_img(img)
    # Generate target domain labels randomly
    label_trg = tf.random.shuffle(label_org)

    return img, label_org, label_trg


def initialize_loss_trackers():
    d_losses = ("d_loss_real",
                "d_loss_fake",
                "d_loss_gp",
                "d_loss_cls",
                "d_loss")

    g_losses = ("g_loss_fake",
                "g_loss_rec",
                "g_loss_cls",
                "g_loss")

    d_loss_list = []
    g_loss_list = []
    dl_list = store_loss_tracker(d_loss_list, d_losses)
    gl_list = store_loss_tracker(g_loss_list, g_losses)

    return dl_list, gl_list


def define_train_loop(use_mp):
    if use_mp:
        return train_disc_mp, train_gen_mp
    else:
        return train_disc, train_gen


def store_loss_tracker(loss_list, losses):
    for name in losses:
        loss_list.append(define_loss_tracker(name))

    return loss_list


def define_loss_tracker(name):
    return tf.keras.metrics.Mean(name=name)


def reset_loss_trackers(loss_list):
    for loss in loss_list:
        loss.reset_states()


def update_loss_trackers(loss_tracker_list, losses):
    for tracker, loss in zip(loss_tracker_list, losses):
        tracker(loss)


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
        out_src, _ = discriminator(x_hat, training=True)

    gradients = tape.gradient(out_src, x_hat)
    l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
    gp_loss = tf.reduce_mean((l2_norm - 1.0) ** 2)
    return gp_loss


def get_classification_loss(target, logits):
    target = tf.cast(target, dtype=tf.float32)
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


def get_lr_decay_factor(epoch, max_epoch, init_lr=0.0001):
    return tf.cast(2.0 * init_lr * (- tf.cast(epoch, tf.float32) / tf.cast(max_epoch, tf.float32) + 1.0), dtype=tf.float32)


def get_lr_decay_factor_by_iter(iteration, diff_iter, init_lr=0.0001):
    return tf.cast(init_lr * (- tf.cast(iteration, tf.float32) / tf.cast(diff_iter, tf.float32) + 2.0), dtype=tf.float32)


@tf.function
def update_lr(gen_opt, disc_opt, max_epoch, epoch, g_lr=0.0001, d_lr=0.0001):
    if g_lr != d_lr:
        decayed_lr = get_lr_decay_factor(epoch, max_epoch, g_lr)
        gen_opt.lr.assign(decayed_lr)
        disc_opt.lr.assign(decayed_lr)
        # to debug
        tf.print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))
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
def update_lr_by_iter(gen_opt, disc_opt, iteration, diff_iter, g_lr=0.0001, d_lr=0.0001):
    if g_lr != d_lr:
        decayed_lr = get_lr_decay_factor_by_iter(iteration, diff_iter, g_lr)
        gen_opt.lr.assign(decayed_lr)
        disc_opt.lr.assign(decayed_lr)
        # to debug
        #tf.print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))
    else:
        g_decayed_lr =  get_lr_decay_factor_by_iter(iteration, 
                                                    diff_iter,
                                                    g_lr)
        d_decayed_lr =  get_lr_decay_factor_by_iter(iteration, 
                                                    diff_iter,
                                                    d_lr)
        gen_opt.lr.assign(g_decayed_lr)
        disc_opt.lr.assign(d_decayed_lr)
        # to debug
        #print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))


"""
Need to split the update for generator and discriminator according to the issues below.
https://github.com/tensorflow/tensorflow/issues/34983#issuecomment-743702919
So I do not use the 'train_step' function.
"""
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

    return d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp, d_loss, g_loss_fake, g_loss_rec, g_loss_cls, g_loss


@tf.function
def predict_before_update(x_real, label_trg, gen, disc):
    x_fake = gen(x_real, label_trg, training=False)
    gen_out_src, gen_out_cls = disc(x_fake, training=False)
    return x_fake, gen_out_src, gen_out_cls


@tf.function
def train_disc(disc,
               gen,
               x_real,
               label_org,
               label_trg,
               lambda_cls,
               lambda_gp,
               opt):

    with tf.GradientTape() as tape:
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

    # Calculate the gradients and update params for the discriminator and the generator
    disc_gradients = tape.gradient(d_loss, disc.trainable_variables)
    opt.apply_gradients(zip(disc_gradients, disc.trainable_variables))

    return d_loss_real, d_loss_fake, d_loss_gp, d_loss_cls, d_loss


@tf.function
def train_disc_mp(disc,
                  gen,
                  x_real,
                  label_org,
                  label_trg,
                  lambda_cls,
                  lambda_gp,
                  opt):

    with tf.GradientTape() as tape:
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
        scaled_d_loss = opt.get_scaled_loss(d_loss)

    # Calculate the gradients and update params for the discriminator and the generator
    scaled_d_gradients = tape.gradient(scaled_d_loss, disc.trainable_variables)
    d_gradients = opt.get_unscaled_gradients(scaled_d_gradients)
    opt.apply_gradients(zip(d_gradients, disc.trainable_variables))

    return d_loss_real, d_loss_fake, d_loss_gp, d_loss_cls, d_loss


@tf.function
def train_gen(disc,
              gen,
              x_real,
              label_org,
              label_trg,
              lambda_cls,
              lambda_rec,
              opt):

    with tf.GradientTape() as tape:
        # Compute loss for original-to-target domain
        x_fake = gen(x_real, label_trg, training=True)
        gen_out_src, gen_out_cls = disc(x_fake, training=False)
        g_loss_fake = - get_mean_for_loss(gen_out_src)
        g_loss_cls = get_classification_loss(label_trg, gen_out_cls)
        # Compute loss for target-to-original domain
        x_rec = gen(x_fake, label_org, training=True)
        g_loss_rec = get_l1_loss(x_real, x_rec)
        # Compute the total loss for the generator
        g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls

    gen_gradients = tape.gradient(g_loss, gen.trainable_variables)
    opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    return g_loss_fake, g_loss_rec, g_loss_cls, g_loss


@tf.function
def train_gen_mp(disc,
                 gen,
                 x_real,
                 label_org,
                 label_trg,
                 lambda_cls,
                 lambda_rec,
                 opt):

    with tf.GradientTape() as tape:
        # Compute loss for original-to-target domain
        x_fake = gen(x_real, label_trg, training=True)
        gen_out_src, gen_out_cls = disc(x_fake, training=False)
        g_loss_fake = - get_mean_for_loss(gen_out_src)
        g_loss_cls = get_classification_loss(label_trg, gen_out_cls)
        # Compute loss for target-to-original domain
        x_rec = gen(x_fake, label_org, training=True)
        g_loss_rec = get_l1_loss(x_real, x_rec)
        # Compute the total loss for the generator
        g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls
        scaled_g_loss = opt.get_scaled_loss(g_loss)

    scaled_g_gradients = tape.gradient(scaled_g_loss, gen.trainable_variables)
    g_gradients = opt.get_unscaled_gradients(scaled_g_gradients)
    opt.apply_gradients(zip(g_gradients, gen.trainable_variables))

    return g_loss_fake, g_loss_rec, g_loss_cls, g_loss


def print_log(epoch, start, end, d_losses, g_losses):
    tf.print("\nTime taken for epoch {} is {:.3f} sec\n".format(epoch,
                                                                round(end - start)))
    d_log = "d_loss: {:.3f} (d_loss_real: {:.3f}, d_loss_fake: {:.3f}, d_loss_gp: {:.3f}, d_loss_cls: {:.3f})"
    g_log = "g_loss: {:.3f} (g_loss_fake: {:.3f}, g_loss_rec: {:.3f}, g_loss_cls: {:.3f})"
    tf.print(d_log.format(d_losses[4], d_losses[0], d_losses[1], d_losses[2], d_losses[3]))
    tf.print(g_log.format(g_losses[3], g_losses[0], g_losses[1], g_losses[2]))


def preprocess_for_testing(img, c_dim):
    multi_imgs = []

    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)
    for i in range(c_dim):
        multi_imgs.append(img)
    imgs = tf.concat(multi_imgs, axis=0)

    return imgs


def save_img(tensor, fpath):
    bstr = tf.io.encode_jpeg(tensor)
    with open(fpath, "wb") as f:
        f.write(bstr.numpy())


def make_img_horizontal(tensor):
    tensor_list = tf.raw_ops.Unpack(value=tensor, num=tensor.shape[0])
    
    return tf.concat(tensor_list, axis=1)


def postprocess_to_plot(results):
    tensor = tf.concat(results, axis=0)
    h, w, _ = tensor.shape
    tensor = denormalize(tensor)
    tensor = tf.cast(tensor * 255, dtype=tf.uint8)
    #tensor = tf.image.resize(tensor, [h//2, w//2], method="nearest")

    return tensor


def save_test_results(model, img_list, trg_list, fpath, do_center_crop=True, do_normalize=True):
    results = []
    c_dim = len(trg_list)
    trg_tensor = tf.convert_to_tensor(trg_list)
    for i, img_path in enumerate(img_list):
        img = read_and_decode_img(img_path)
        img = preprocess_img(img, use_aug=False, do_center_crop=do_center_crop, do_normalize=do_normalize)
        x = preprocess_for_testing(img, c_dim)
        result = model(x, trg_tensor[:, i, :])
        horizontal_img = make_img_horizontal(result)
        results.append(horizontal_img)
    tensor = postprocess_to_plot(results)
    save_img(tensor, fpath)


def get_models_for_testing(attr_path="data/celeba/list_attr_celeba.txt",
                           selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                           num_test=10,
                           c_dim=5,
                           g_lr=0.0001,
                           d_lr=0.0001,
                           beta1=0.5,
                           beta2=0.999,
                           ckpt_dir="ckpts/train/"):

    # Build the generator and discriminator
    gen, disc = build_model(c_dim, False)

    # Define the optimizers for the generator and the discriminator
    gen_opt = tf.keras.optimizers.Adam(g_lr, beta1, beta2)
    disc_opt = tf.keras.optimizers.Adam(d_lr, beta1, beta2)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(gen=gen,
                            disc=disc,
                            gen_opt=gen_opt,
                            disc_opt=disc_opt)

    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              ckpt_dir
                                              ,max_to_keep=5)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint is restored!")

    return gen, disc


def test_image(model, fpath, to_black_hair, to_blond_hair, to_brown_hair, to_male, to_young):
    img = read_and_decode_img(fpath)
    img = tf.image.resize(img, [128, 128], method="nearest")
    org = img.numpy() / 255.0
    plot_image(121, "original image", org)
    img = normalize(img)
    img = tf.expand_dims(img, axis=0)
    c = tf.constant([[int(to_black_hair), 
                      int(to_blond_hair), 
                      int(to_brown_hair), 
                      int(to_male), 
                      int(to_young)]])
    result = model(img, c)
    result = tf.squeeze(result, axis=0)
    result = denormalize(result)
    result = result.numpy()
    plot_image(122, "generated image", result)


def plot_image(coord, title, img):
    plt.subplot(coord)
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")


def save_input_imgs(inp_imgs, save_path):
    input_imgs = []
    for ipath in inp_imgs:
        img = read_and_decode_img(ipath)
        img = preprocess_img(img, use_aug=False, do_normalize=False)
        input_imgs.append(img)

    inputs = tf.concat(input_imgs, axis=0)
    save_img(inputs, save_path)
