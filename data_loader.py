import tensorflow as tf
import random
import os
import numpy as np


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


if __name__ == "__main__":
    attr_path = "data/celeba/list_attr_celeba.txt"
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    train_imgs, train_lbls, test_imgs, test_lbls = get_data(attr_path, selected_attrs)
    dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    for img, lbl in dataset.take(5):
        print(img, lbl)

    print("shape of train_lbls:", len(train_lbls), len(train_lbls[0]))

    orgs_to_test = train_lbls[:5]
    print("original test labels\n", orgs_to_test)
    trgs_to_test = create_labels(orgs_to_test, 5, selected_attrs)
    print("target test labels\n", trgs_to_test)

    print("Finished testing the CelebA dataset!")