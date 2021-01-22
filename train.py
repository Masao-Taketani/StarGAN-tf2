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
flags.DEFINE_string("ckpt_path", "ckpts/train/", "path to the checkpoints")


def main(argv):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_imgs, train_lbls, test_imgs, test_lbls = get_data(FLAGS.attr_path, 
                                                            FLAGS.selected_attrs)
    train_buffer_size = count_data_size(train_imgs)
    # Prepare the dataset for training and testing
    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    train_dataset = train_dataset.map(
                        preprocess_for_training,
                        num_parallel_calls=AUTOTUNE).shuffle(train_buffer_size)\
                                                    .repeat()\
                                                    .batch(FLAGS.batch_size)\
                                                    .prefetch(buffer_size=AUTOTUNE)
    for img, lbl_org, lbl_trg in train_dataset.take(15):
        print(lbl_org, lbl_trg)
    #for img, lbl_org in train_dataset.take(5):
    #    print(img, lbl_org)

    """
    # Get fixed inputs for debugging.
    label_org
    label_trg = tf.random.shuffle(label_org)

    # Define losses
    classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0),
                               )
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              FLAGS.ckpt_path,
                                              max_to_keep=5)
    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint is restored!")

    gen, disc = model.build_model(FLAGS.c_dim)
    """


if __name__ == "__main__":
    app.run(main)