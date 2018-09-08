from matplotlib import pyplot as plt
import download_mnist as dm
from mnist import MNIST
from model import Model
import numpy as np
import argparse
import joblib
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_activation", type=str, default="sigmoid")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--l1_weight', type=float, default=0.0)
parser.add_argument('--l2_weight', type=float, default=0.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--model_dir", type=str, default=0.0)
parser.add_argument("--cross_validate", type=bool, default=False)
args = parser.parse_args()


def one_hot(x, size):
    if not hasattr(x, "__len__"):  # x is a scalar
        x = np.array([x])
    ret = np.zeros((len(x), size))
    for i in range(len(x)):
        ret[i, int(x[i])] = 1.0
    return ret


def cross_validation(network, train_data, K=5):
    random.shuffle(train_data)
    N = len(train_data)
    rem = N % K
    folds = np.split(train_data[:N - rem], K)
    folds[-1] = np.concatenate((folds[-1], train_data[N - rem:]), axis=0)
    [print(fld.shape) for fld in folds]
    avg_train_cost = []
    avg_test_cost = []
    avg_test_accuracy = []

    for i in range(1, K):
        print("Fold %i of %i" % (i, K))
        validation_data = np.array(folds[i])
        if i is not 0 and i is not K - 1:
            lst = folds[:i].copy()
            for fold in folds[i + 1:]:
                lst.append(fold)
            training_data = np.concatenate(lst, axis=0)
        elif i == 0:
            training_data = np.concatenate(folds[1:], axis=0)
        else:
            training_data = np.concatenate(folds[:-1], axis=0)

        print("Training data shape = {} ; Validation data shape = {}".
              format(training_data.shape, validation_data.shape))

        network_copy = network.copy()

        trained_model, train_cost, test_cost, test_accuracy = \
            network_copy.fit(train_data=train_data,
                             test_data=test_data,
                             mini_batch_size=args.batch_size,
                             num_epochs=args.epochs,
                             l2_weight=args.l2_weight,
                             l1_weight=args.l1_weight,
                             momentum_factor=args.momentum,
                             learning_rate=args.lr,
                             lr_decay=0.00,
                             save_model=True,
                             model_name=os.path.join(args.model_dir, args.hidden_activation))

        print("Saving model and results")
        suffix = "cross_validation_5"
        joblib.dump(trained_model, "./models/" + suffix + "_model_" + str(i))
        np.savetxt("./results/" + suffix + "_train_cost_" + str(i) + ".csv", train_cost)
        np.savetxt("./results/" + suffix + "_test_cost_" + str(i) + ".csv", test_cost)
        np.savetxt("./results/" + suffix + "_test_accuracy_" + str(i) + ".csv", test_accuracy)

        if len(avg_train_cost) == 0:
            avg_train_cost = train_cost.copy()
        else:
            avg_train_cost += train_cost
        if len(avg_test_cost) == 0:
            avg_test_cost = test_cost.copy()
        else:
            avg_test_cost += test_cost
        if len(avg_test_accuracy) == 0:
            avg_test_accuracy = test_accuracy.copy()
        else:
            avg_test_accuracy += test_accuracy

    avg_train_cost /= K
    avg_test_cost /= K
    avg_test_accuracy /= K

    plt.title("Train and Test error convergence")
    plt.xlabel("# iterations")
    plt.ylabel("Average cross entropy error")
    plt.plot(np.arange(len(avg_train_cost)), avg_train_cost, label="Training error")
    plt.plot(np.arange(len(avg_test_cost)) * 200., avg_test_cost, label="Test error")
    plt.plot(np.arange(len(avg_test_accuracy)) * 200., avg_test_accuracy, label="Test accuracy")
    plt.legend()
    plt.savefig("./results/cross_validation_plots.png")
    plt.show()


if __name__ == "__main__":

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

    model = Model(layers=[train_data.shape[1] - 1, 1000, 500, 250, 10],
                  hidden_activation_name=args.hidden_activation,
                  output_activation_name="softmax",
                  loss_fn_name="cross_entropy")

    if args.cross_validate:
        cross_validation(model, train_data, K=5)
    else:
        trained_model, train_cost, test_cost, test_accuracy = \
            model.fit(train_data=train_data,
                      test_data=test_data,
                      mini_batch_size=args.batch_size,
                      num_epochs=args.epochs,
                      l2_weight=args.l2_weight,
                      l1_weight=args.l1_weight,
                      momentum_factor=args.momentum,
                      learning_rate=args.lr,
                      lr_decay=0.00,
                      save_model=True,
                      model_name=os.path.join(args.model_dir, args.hidden_activation))

        plt.title("Training cost and test cost")
        plt.xlabel("# iterations")
        plt.ylabel("cross-entropy loss")
        plt.plot(np.arange(0, len(train_cost)), train_cost, label="Training cost")
        plt.plot(np.arange(0, len(test_cost)) * 200, test_cost, label="Test cost")
        plt.legend()
        plt.show();

        joblib.dump(trained_model, "./models/model_" + args.hidden_activation)

    """
    joblib.dump(trained_model, "./models/model1")
    joblib.dump(test_cost, "./np_objects/test_cost")
    joblib.dump(train_cost, "./np_objects/train_cost")
    joblib.dump(test_accuracy, "./np_objects/test_accuracy")
    """
