import tensorflow as tf


IMG_SIZE = 256


def count_data_size(dataset):
    ct = 0
    for _ in dataset.as_numpy_iterator():
        ct += 1

    return ct


def random_crop(img):
    return tf.image.random_crop(img,
                                size=[IMG_SIZE, IMG_SIZE, 3])


def center_crop(img, crop_size=178):
    return tf.image.central_crop(img, crop_size / IMG_SIZE)


def normalize(img):
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img