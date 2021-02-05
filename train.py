import os
import time
import datetime

from tqdm import tqdm
from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import mixed_precision

from data_loader import *
from utils import *
from model import build_model


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
flags.DEFINE_string("test_result_dir", "test_results/", "path to the test result dir")
flags.DEFINE_string("logdir", "logs/", "path to the log dir")
flags.DEFINE_integer("num_epochs", 20, "number of epopchs to train")
flags.DEFINE_integer("num_epochs_decay", 10, "number of epochs to start lr decay")
flags.DEFINE_integer("num_iters", 200000, "number of total iterations for training D")
flags.DEFINE_integer("num_iters_decay", 100000, "number of iterations for decaying lr")
flags.DEFINE_float("lambda_cls", 1.0, "weight for domain classification loss")
flags.DEFINE_float("lambda_rec", 10.0, "weight for reconstruction loss")
flags.DEFINE_float("lambda_gp", 10.0, "weight for gradient penalty loss")
flags.DEFINE_integer("model_save_epoch", 1, "to save model every specified epochs")
flags.DEFINE_integer("num_critic_updates", 5, "number of a Discriminator updates "
                                              "every time a generator updates")
flags.DEFINE_float("g_lr", 0.0001, "learning rate for the generator")
flags.DEFINE_float("d_lr", 0.0001, "learning rate for the discriminator")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_float("beta2", 0.999, "beta2 for Adam optimizer")
flags.DEFINE_integer("num_test", 10, "number of test examples")
flags.DEFINE_bool("use_mp", True, "whether to use mixed precision for training")


#tf.config.experimental.enable_tensor_float_32_execution(enabled=False)


def main(argv):
    if FLAGS.use_mp:
        mixed_precision.set_global_policy('mixed_float16')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(FLAGS.test_result_dir, exist_ok=True)

    _, _, test_imgs, test_lbls = get_data(FLAGS.attr_path, 
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
    train_dataset = train_dataset.batch(batch_size=FLAGS.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # Get fixed inputs for testing and debugging.
    c_fixed_trg_list = create_labels(test_lbls[:FLAGS.num_test], 
                                     FLAGS.c_dim, 
                                     FLAGS.selected_attrs)

    # Build the generator and discriminator
    gen, disc = build_model(FLAGS.c_dim, FLAGS.use_mp)

    # Define the optimizers for the generator and the discriminator
    gen_opt = tf.keras.optimizers.Adam(FLAGS.g_lr, FLAGS.beta1, FLAGS.beta2)
    disc_opt = tf.keras.optimizers.Adam(FLAGS.d_lr, FLAGS.beta1, FLAGS.beta2)
    if FLAGS.use_mp:
        gen_opt = mixed_precision.LossScaleOptimizer(gen_opt)
        disc_opt = mixed_precision.LossScaleOptimizer(disc_opt)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0, dtype=tf.int64),
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
                                    os.path.join(FLAGS.logdir,
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                                )

    d_loss_list, g_loss_list = initialize_loss_trackers()
    train_d, train_g = define_train_loop(FLAGS.use_mp)

    iters_per_epoch = FLAGS.num_iters // FLAGS.num_epochs
    diff_iter = FLAGS.num_iters - FLAGS.num_iters_decay

    # Train the discriminator and the generator
    while ckpt.epoch < FLAGS.num_epochs:
        ckpt.epoch.assign_add(1)
        step = tf.constant(0)
        reset_loss_trackers(d_loss_list)
        reset_loss_trackers(g_loss_list)

        #if ckpt.epoch > FLAGS.num_epochs_decay:
        #    update_lr(gen_opt, disc_opt, FLAGS.num_epochs, ckpt.epoch, FLAGS.g_lr, FLAGS.d_lr)

        start = time.time()
        for x_real, label_org, label_trg in tqdm(train_dataset):
            step += 1
            if step.numpy() > FLAGS.num_iters_decay:
                update_lr_by_iter(gen_opt, disc_opt, step, diff_iter, FLAGS.g_lr, FLAGS.d_lr)

            d_losses = train_d(disc,
                               gen,
                               x_real,
                               label_org, 
                               label_trg, 
                               FLAGS.lambda_cls,
                               FLAGS.lambda_gp, 
                               disc_opt) 
                                
            update_loss_trackers(d_loss_list, d_losses)

            if step.numpy() % FLAGS.num_critic_updates == 0:
                g_losses = train_g(disc,
                                   gen, 
                                   x_real,
                                   label_trg, 
                                   FLAGS.lambda_cls,
                                   FLAGS.lambda_rec,
                                   gen_opt)

                update_loss_trackers(g_loss_list, g_losses)

            if step.numpy() == iters_per_epoch:
                break

            #if step.numpy() % 100 == 0:
            #    fpath = os.path.join(FLAGS.test_result_dir, "{}-images.jpg".format(step.numpy()))
            #    save_test_results(gen, test_imgs[:FLAGS.num_test], c_fixed_trg_list, fpath)

        end = time.time()
        print_log(ckpt.epoch.numpy(), start, end, d_losses, g_losses)

        # keep the log for the losses
        with summary_writer.as_default():
            tf.summary.scalar("d_loss_real", d_loss_list[0].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_fake", d_loss_list[1].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_gp", d_loss_list[2].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_cls", d_loss_list[3].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss", d_loss_list[4].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_fake", g_loss_list[0].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_rec", g_loss_list[1].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_cls", g_loss_list[2].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss", g_loss_list[3].result(), step=ckpt.epoch)

        # test the generator model and save the results for each epoch
        fpath = os.path.join(FLAGS.test_result_dir, "{}-images.jpg".format(ckpt.epoch.numpy()))
        save_test_results(gen, test_imgs[:FLAGS.num_test], c_fixed_trg_list, fpath)

        if (ckpt.epoch) % FLAGS.model_save_epoch == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving a checkpoint for epoch {} at {}".format(ckpt.epoch.numpy(), ckpt_save_path))


if __name__ == "__main__":
    app.run(main)