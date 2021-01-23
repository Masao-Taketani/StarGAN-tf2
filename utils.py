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
    return img / 2 + 0.5


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


def preprocess_for_training(img_path, label_org):
    center_crop_size=178
    size=128

    ## for images
    img, _ = read_and_decode_img(img_path)
    img = random_horizontal_flip(img)
    img = center_crop(img, center_crop_size)
    img = resize(img, size)
    img = normalize(img)

    ## for labels
    label_trg = tf.random.shuffle(label_org)

    return img, label_org, label_trg


def preprocess_for_testing(img):
    center_crop_size=178
    size=128

    # for images
    img = random_horizontal_flip(img)
    img = center_crop(img, center_crop_size)
    img = resize(img, resize)
    img = normalize(img)

    return img


def get_gradient_penalty(x, x_gen, discriminator):
    """
    for the implementation of the gradient penalty, I referred to the links below.
    https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
    https://qiita.com/triwave33/items/72c7fceea2c6e48c8c07
    """
    # shape=[x.shape[0], 1, 1, 1] to generate a random number for every sample
    epsilon = tf.random_uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as tape:
        # to get a gradient w.r.t x_hat, we need to record the value on the tape
        t.watch(x_hat)
        d_hat = discriminator(x_hat)
    
    gradients = tape.gradient(d_hat, x_hat)
    l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
    gp = tf.reduce_mean((l2_norm - 1.0) ** 2)
    return gp


def get_classification_loss(loss_func, logits, target):
    # Compute binary or softmax cross entropy loss.
    bce_with_logits = loss_func(target, logits)
    return bce_with_logits