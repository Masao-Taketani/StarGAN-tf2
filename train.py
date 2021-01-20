from absl import app
from absl import flags

import tensorflow as tf

from data_loader import get_data
from utils import *


FLAGS = flags.FLAGS

flags.DEFINE_string("attr_path", 
                    "data/celeba/list_attr_celeba.txt", 
                    "path to the attribute label")
flags.DEFINE_list("selected_attrs",
                  "Black_Hair, Blond_Hair, Brown_Hair, Male, Young",
                  "attributes for training")
flags.DEFINE_integer("c_dim", 5, "dimension of domain labels")
flags.DEFINE_integer("batch_size", 16, "mini-batch size")


def main(argv):
    train_imgs, train_lbls, test_imgs, test_lbls = get_data(FLAGS.attr_path, 
                                                            FLAGS.selected_attrs)
    # prepare for training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    train_dataset = tf.data.Dataset.map()
    for img, lbl in train_dataset.take(5):
        print(img, lbl)

    # Get fixed inputs for debugging
    label_org
    label_trg = tf.random.shuffle(label_org)
    targ_label = orig_label[rand_idxes]

    gen, disc = model.build_model(FLAGS.c_dim)



if __name__ == "__main__":
    app.run(main)