import random
import os
import sys
import math

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import *


IMG_DIR = "data/celeba/images/"
#ATTR_PATH = "data/celeba/list_attr_celeba.txt"
#SELECTED_ATTRS = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']


def get_data(attr_path, selected_attrs):
    train_imgs = []
    train_lbls = []
    test_imgs = []
    test_lbls = []

    lines = [line.rstrip() for line in open(attr_path, 'r')]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    random.seed(1234)
    random.shuffle(lines)

    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            if values[idx] == "1":
                label.append(1)
            else:
                label.append(0)
            
        if (i+1) < 2000:
            test_imgs.append(os.path.join(IMG_DIR, filename))
            test_lbls.append(label) 
        else:
            train_imgs.append(os.path.join(IMG_DIR, filename))
            train_lbls.append(label)

    assert len(train_imgs) == len(train_lbls), "train images and lebels length doesn't match"
    assert len(test_imgs) == len(test_lbls), "test images and lebels length doesn't match"
    print("train len:", len(train_imgs))
    print("test len:", len(test_imgs))

    return train_imgs, train_lbls, test_imgs, test_lbls


def create_labels(c_org, c_dim=5, selected_attrs=None):
    """Generate each target label from a original label"""
    # Get hair colo indices
    c_org = np.array(c_org)
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.copy()
        # Set one hair color to 1 and the rest to 0.
        if i in hair_color_indices:
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            # Reverse each attribute for non-hair-color class
            c_trg[:, i] = (c_trg[:, i] == 0)

        c_trg_list.append(c_trg)
    
    return c_trg_list


## For TFRecords
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(list_val):
    features = [tf.train.Feature(
        int64_list=tf.train.Int64List(value=[val])) for val in list_val]
    return tf.train.FeatureList(feature=features)


def convert_data_to_tfrecord(imgs, labels, num_split, out_dir):
    """Convert and split the data into some TFRecord files
    Referred to https://bit.ly/3pfPYv5
    """
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout.write("Start converting data into TFRecords.\n")
    num_data = len(imgs)
    num_per_shard = math.ceil(num_data / num_split)

    for shard_id in tqdm(range(num_split)):
        out_fname = os.path.join(out_dir, 
                                 "celeb_a-{:02d}-of-{:02d}.tfrecord".format(shard_id,
                                                                        num_split))
        
        with tf.io.TFRecordWriter(out_fname) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_data)
            for i in range(start_idx, end_idx):
                example = image_label_example(imgs[i], labels[i])
                writer.write(example.SerializeToString())

    sys.stdout.write("Finished converting data into TFRecords.")


def image_label_example(img_path, label):
    img_string = open(img_path, 'rb').read()
    height, width, channel = tf.image.decode_jpeg(img_string).shape
    black_h, blond_h, brown_h, male, young = label

    feature = {
        "image": _bytes_feature(img_string),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channel": _int64_feature(channel),
    }
    """
    "black_h": _int64_feature(black_h),
    "blond_h": _int64_feature(blond_h),
    "brown_h": _int64_feature(brown_h),
    "male": _int64_feature(male),
    "young": _int64_feature(young),
    """

    feature_list = {
        "label": _int64_feature_list(label),
    }

    return tf.train.SequenceExample(context=tf.train.Features(feature=feature),
                                    feature_lists=tf.train.FeatureLists(
                                                    feature_list=feature_list))


def parse_tfrecords(example_proto):
    # Parse the input tf.train.Example proto using the dictionaries below
    feature_desc = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
    }
    """
    "black_h": tf.io.FixedLenFeature([], tf.int64),
    "blond_h": tf.io.FixedLenFeature([], tf.int64),
    "brown_h": tf.io.FixedLenFeature([], tf.int64),
    "male": tf.io.FixedLenFeature([], tf.int64),
    "young": tf.io.FixedLenFeature([], tf.int64),
    """

    feature_list_desc = {
        "label": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    inp, trg = tf.io.parse_single_sequence_example(example_proto, 
                                                   context_features=feature_desc,
                                                   sequence_features=feature_list_desc)

    img = tf.io.decode_jpeg(inp["image"])
    label = trg["label"]

    return img, label


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE

    attr_path = "data/celeba/list_attr_celeba.txt"
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    tfrecords_dir =  "data/celeba/tfrecords/"

    train_imgs, train_lbls, test_imgs, test_lbls = get_data(attr_path, selected_attrs)
    dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    for img, lbl in dataset.take(5):
        print(img, lbl)

    print("\nShape of train_lbls:", len(train_lbls), len(train_lbls[0]))

    orgs_to_test = train_lbls[:5]
    print("\nOriginal test labels\n", orgs_to_test)
    trgs_to_test = create_labels(orgs_to_test, 5, selected_attrs)
    print("\nTarget test labels\n", trgs_to_test)

    print("\nTFRecord write test")
    print("\nFor training data")
    train_dir = os.path.join(tfrecords_dir, "train")
    #convert_data_to_tfrecord(train_imgs, train_lbls, 10, train_dir)
    print("\nFor testing data")
    test_dir = os.path.join(tfrecords_dir, "test")
    #convert_data_to_tfrecord(test_imgs, test_lbls, 1, test_dir)

    print("\nTFRecord read test")
    tfr_dataset = tf.data.Dataset.list_files(os.path.join(train_dir, "*.tfrecord"))
    tfr_dataset = tfr_dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=AUTOTUNE,
                                         deterministic=False)
    tfr_dataset = tfr_dataset.map(parse_tfrecords)
    tfr_dataset = tfr_dataset.map(preprocess_for_training,
                                  num_parallel_calls=AUTOTUNE)
    tfr_dataset = tfr_dataset.batch(batch_size=16)
    tfr_dataset = tfr_dataset.prefetch(buffer_size=AUTOTUNE)

    for img, label_org, label_trg in tfr_dataset.take(1):
        print("img.shape", img.shape)
        print("img\n", img[0])
        print("label_org.shape", label_org.shape)
        print("label_org", label_org[0])
        print("label_trg.shape", label_trg.shape)
        print("label_trg", label_trg[0])

    print("\nFinished testing the CelebA dataset!")