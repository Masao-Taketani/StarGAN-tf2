import os
import time
import datetime

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
flags.DEFINE_string("log_dir", "logs/", "path to the log dir")
flags.DEFINE_integer("num_epochs", 200, "number of epopchs to train")
flags.DEFINE_integer("num_epochs_decay", 100, "number of epochs to start lr decay")
flags.DEFINE_float("lambda_cls", 1.0, "weight for domain classification loss")
flags.DEFINE_float("lambda_rec", 10.0, "weight for reconstruction loss")
flags.DEFINE_float("lambda_gp", 10.0, "weight for gradient penalty loss")
flags.DEFINE_integer("model_save_epoch", "10", "to save model every specified epochs")
flags.DEFINE_integer("num_critic_updates", 5, "number of a Discriminator updates "
                                              "every time a generator updates")
flags.DEFINE_float("g_lr", 0.0001, "learning rate for the generator")
flags.DEFINE_float("d_lr", 0.0001, "learning rate for the discriminator")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFNIE_float("beta2", 0.999, "beta2 for Adam optimizer")


def main(argv):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    os.makedirs(FLAGS.log_dir, exist_ok=True)

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
    # Get fixed inputs for testing and debugging.
    c_fixed_trg_list = create_labels(test_lbls[:FLAGS.batch_size], 
                                     FLAGS.c_dim, 
                                     FLAGS.selected_attrs)
    c_fixed_trg_list = tf.convert_to_tensor(c_fixed_trg_list)

    # Build the generator and discriminator
    gen, disc = model.build_model(FLAGS.c_dim)

    # Define the optimizers for the generator and the discriminator
    gen_opt = tf.keras.optimizers.Adam(FLAGS.g_lr, FLAGS.beta1, FLAGS.beta2)
    disc_opt = tf.keras.optimizers.Adam(FLAGS.d_lr, FLAGS.beta1, FLAGS.beta2)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0),
                               gen=gen,
                               disc=disc,
                               gen_opt=gen_opt,
                               disc_opt=disc_opt)

    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              FLAGS.ckpt_dir,
                                              max_to_keep=5)
    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint is restored!")
        
    # Create a summary writer to track the losses 
    summary_writer = tf.summary.create_file_writer(
                                    os.path.join(FLAGS.log_dir,
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                                )

    # Train the discriminator and the generator
    while ckpt.epoch < FLAGS.num_epochs:
        ckpt.epoch.assign_add(1)
        step = 0

        if ckpt.epoch > FLAGS.num_epochs_decay:
            update_lr(gen_opt, disc_opt, FLAGS.num_epochs, ckpt.epoch)

        start = time.time()

        for real_x, label_org, label_trg in train_dataset:
            step += 1
            losses = train_step(step, 
                                gen, 
                                disc, 
                                x_real, 
                                label_org, 
                                label_trg, 
                                FLAGS.lambda_cls, 
                                FLAGS.lambda_gp, 
                                FLAGS.lambda_rec, 
                                FLAGS.num_critic_updates, 
                                disc_opt, 
                                gen_opt)

        if step % 1000 == 0:
                print(".", end="")

    print("\nTime taken for epoch {} is {} sec\n".format(ckpt.epoch.numpy(), 
                                                            time.time()-start))
    print("d_loss_real: {}, d_loss_fake: {}, d_loss_cls: {}, d_loss_gp: {}, " +\
          "d_loss: {}, g_loss_fake: {}, g_loss_cls: {}, g_loss: {}".format(losses[0],
                                                                           losses[1],
                                                                           losses[2],
                                                                           losses[3],
                                                                           losses[4],
                                                                           losses[5],
                                                                           losses[6],
                                                                           losses[7]))
    # keep the log for the losses
    with summary_writer.as_default():
        tf.summary.scalar("d_loss_real", losses[0], step=ckpt.epoch)
        tf.summary.scalar("d_loss_fake", losses[1], step=ckpt.epoch)
        tf.summary.scalar("d_loss_cls", losses[2], step=ckpt.epoch)
        tf.summary.scalar("d_loss_gp", losses[3], step=ckpt.epoch)
        tf.summary.scalar("d_loss", losses[4], step=ckpt.epoch)
        tf.summary.scalar("g_loss_fake", losses[5], step=ckpt.epoch)
        tf.summary.scalar("g_loss_cls", losses[6], step=ckpt.epoch)
        tf.summary.scalar("g_loss", losses[7], step=ckpt.epoch)    


if __name__ == "__main__":
    app.run(main)