import numpy as np
from mnist import MNIST
from model import Model


def one_hot(x, size):
    if not hasattr(x, "__len__"):  # x is a scalar
        x = np.array([x])
    ret = np.zeros((len(x), size))
    for i in range(len(x)):
        ret[i, int(x[i])] = 1.0
    return ret


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

"""
train_data_lines = open("./inp.txt", "r").readlines()
train_data = []
for line in train_data_lines:
    curr_line = []
    for num in line.split(" "):
        curr_line.append(float(num))
    train_data.append(curr_line)

train_data = np.array(train_data)
"""

model = Model(layers=[train_data.shape[1] - 1, 500, 500, 10],
              output_activation_name="softmax",
              loss_fn_name="cross_entropy")
# model = Model(layers=[train_data.shape[1] - 1, 500, 500, 10])

model.fit(train_data, 50, 10, l2_weight=0.0, l1_weight=0.0, momentum_factor=0.05, learning_rate=0.1, lr_decay=0.01)
