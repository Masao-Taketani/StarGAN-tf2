import os

from absl import app
from absl import flags

import tensorflow as tf

from data_loader import *
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
flags.DEFINE_string("ckpt_dir", "ckpts/train/", "path to the checkpoint dir")
flags.DEFINE_string("tfrecord_dir", "data/celeba/tfrecords/", "path to the tfrecord dir")


def main(argv):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_imgs, train_lbls, test_imgs, test_lbls = get_data(FLAGS.attr_path, 
                                                            FLAGS.selected_attrs)

    # Prepare the dataset for training and testing
    train_dir = os.path.join(FLAGS.tfrecord_dir, "train")
    test_dir = os.path.join(FLAGS.tfrecord_dir, "test")

    # For training
    train_dataset = tf.data.Dataset.list_files(os.path.join(train_dir, "*.tfrecord"))
    train_dataset = train_dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=AUTOTUNE,
                                         deterministic=False)
    train_dataset = train_dataset.map(parse_tfrecords)
    train_dataset = train_dataset.map(preprocess_for_training,
                                  num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size=FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    """
    # For Testing
    test_dataset = tf.data.Dataset.list_files(os.path.join(test_dir, "*.tfrecord"))
    test_dataset = test_dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=AUTOTUNE,
                                         deterministic=False)
    test_dataset = test_dataset.map(parse_tfrecords)
    test_dataset = test_dataset.map(preprocess_for_training,
                                  num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size=1)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    """

    # Get fixed inputs for debugging.
    c_fixed_trg_list = create_labels(test_lbls[:FLAGS.batch_size], 
                                     FLAGS.c_dim, 
                                     FLAGS.selected_attrs)
    c_fixed_trg_list = tf.convert_to_tensor(c_fixed_trg_list)
    print("test\n", c_fixed_trg_list)
    """
    label_trg = tf.random.shuffle(label_org)

    # Define losses
    classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0),
                               )
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              FLAGS.ckpt_dir,
                                              max_to_keep=5)
    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint is restored!")

    gen, disc = model.build_model(FLAGS.c_dim)
    """


if __name__ == "__main__":
    app.run(main)