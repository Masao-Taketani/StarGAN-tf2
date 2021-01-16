import tensorflow as tf
import random
import os


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
            label.append(values[idx] == "1")
            
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


if __name__ == "__main__":
    attr_path = "data/celeba/list_attr_celeba.txt"
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    train_imgs, train_lbls, test_imgs, test_lbls = get_data(attr_path, selected_attrs)
    dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    for img, lbl in dataset.take(5):
        print(img, lbl)


    print("Finished testing the CelebA dataset!")