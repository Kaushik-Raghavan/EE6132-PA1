from matplotlib import pyplot as plt
import download_mnist as dm
from mnist import MNIST
from skimage import feature
import numpy as np
import sys
import cv2
import os
import scipy


thismodule = sys.modules[__name__]


def one_hot(x, size):
    if not hasattr(x, "__len__"):  # x is a scalar
        x = np.array([x])
    ret = np.zeros((len(x), size))
    for i in range(len(x)):
        ret[i, int(x[i])] = 1.0
    return ret


def plot_from_files(list_of_files, title_name=None):
    """
    Reads each file present in the `list_of_files`, extracts a 1D vector, and plots them in a single figure.
    If each array read from file had different sizes, the x-axis is scaled so as to match the maximum sized array's plot
    :param list_of_files: list of strings denothing path of files to be read
    :param title_name: Title of the plot
    """

    plt.figure()
    if title_name is not None:
        plt.title(title_name)

    for filepath in list_of_files:
        file_lines = open(filepath, "r").readlines()
        data = []
        for line in file_lines:
            curr_line = []
            for num in line.split(" "):
                curr_line.append(float(num))
            data.append(curr_line)
        plt.plot(data[:, 0], data[:, 1])

    plt.show()


def read_from_file(filepath):
    file_lines = open(filepath, "r").readlines()
    data = []
    for line in file_lines:
        curr_line = []
        for num in line.split(" "):
            curr_line.append(float(num))
        data.append(curr_line)
    return np.squeeze(np.array(data))


def read_mnist():
    dm.downloadMNIST('./data/')
    mndata = MNIST("./data/")

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_data = np.zeros((train_images.shape[0], train_images.shape[1] + 1))
    train_data[:, :-1] = train_images
    train_data[:, -1] = train_labels

    test_data = np.zeros((test_images.shape[0], test_images.shape[1] + 1))
    test_data[:, :-1] = test_images
    test_data[:, -1] = test_labels
    return train_data, test_data


def confusion_matrix(predicted_labels, ground_truth_labels, num_classes):
    mat = np.zeros((num_classes, num_classes))
    for x, y in zip(predicted_labels, ground_truth_labels):
        mat[int(y), int(x)] += 1
    return mat


def metrics(predicted_labels, ground_truth_labels, num_classes):
    tp = np.zeros((num_classes,))
    tn = np.zeros((num_classes,))
    fp = np.zeros((num_classes,))
    fn = np.zeros((num_classes,))
    all_tp = 0.0

    for x, y in zip(predicted_labels, ground_truth_labels):
        if x == y:
            all_tp += 1
            for i in range(num_classes):
                if i == int(x):
                    tp[i] += 1
                else:
                    tn[i] += 1
        else:
            for i in range(num_classes):
                if i == int(x):
                    fp[i] += 1
                elif i == int(y):
                    fn[i] += 1
                else:
                    tn[i] += 1

    total_accuracy = all_tp / len(ground_truth_labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2.0 * precision * recall / (precision + recall)

    return total_accuracy, np.mean(precision), np.mean(recall), np.mean(f1_score)


""" Data Augmentation """


def random_deskew(image, height, width):
    # change 1D image to 2D image representation
    original_shape = image.shape
    image = image.reshape(original_shape[:-1] + (height,  width))

    center = (width // 2, height // 2)
    angle = (np.random.rand() - 0.5) * 60.0
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated.reshape(original_shape)


def add_noise(img, std=0.01):
    noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
    return img + noise


def augment_batch(batch):
    h, w = 28, 28
    ret = np.array([random_deskew(img, h, w) for img in batch]) + np.random.normal(0, 5.0, batch.shape)
    # ret = np.array([add_noise(img) for img in ret])
    # ret = batch.copy()
    return ret


def sift_features(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)


def hog_features(img):
    h, w = 28, 28
    H = feature.hog(img.reshape(h, w), orientations=9, pixels_per_cell=(7, 7), block_norm='L2')
    return H