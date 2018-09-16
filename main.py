from matplotlib import pyplot as plt
from model import Model
import numpy as np
import argparse
import joblib
import random
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action='store_true', default=False)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--cross_validate", action='store_true', default=False)
parser.add_argument("--extract_features", action='store_true', default=False)
parser.add_argument("--top_predictions", type=int, default=0)
parser.add_argument("--hidden_activation", type=str, default="sigmoid")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--l1_weight', type=float, default=0.0)
parser.add_argument('--l2_weight', type=float, default=0.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--augment_data", action='store_true', default=False)
parser.add_argument("--model_dir", type=str, default="./models/")
parser.add_argument("--splname", type=str, default="")
args = parser.parse_args()


def cross_validation(network, train_data, K=5):
    print("Performing {}-fold cross-validation...\n".format(K))
    random.shuffle(train_data)
    N = len(train_data)
    rem = N % K
    folds = np.split(train_data[:N - rem], K)
    folds[-1] = np.concatenate((folds[-1], train_data[N - rem:]), axis=0)

    print("Shape of each fold")
    [print(fld.shape) for fld in folds]

    test_accuracies = []

    joblib.dump(network, "temp_network_storage_" + args.splname)

    for i in range(0, K):
        print("\nFold %i of %i" % (i, K))
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

        network_copy = joblib.load("temp_network_storage_" + args.splname)
        suffix = args.hidden_activation + "_" + args.splname + "_cross_validation_"

        trained_model, train_cost, test_cost, test_accuracy = \
            network_copy.fit(train_data=training_data,
                             test_data=validation_data,
                             mini_batch_size=args.batch_size,
                             num_epochs=args.epochs,
                             l2_weight=args.l2_weight,
                             l1_weight=args.l1_weight,
                             momentum_factor=args.momentum,
                             learning_rate=args.lr,
                             lr_decay=0.00,
                             save_model=False,
                             model_name=os.path.join(args.model_dir, suffix))

        test_accuracies.append(test_accuracy[-1, 1])
        print("Validation test accuracy for current fold = %0.4f"%(test_accuracy[-1, 1]))
        evaluate_model_performance(network_copy, validation_data)
        _, cost_mean, cost_std = network_copy.score(validation_data[:, :-1], validation_data[:, -1])
        print("Mean Validation Error = {:.4f}".format(cost_mean))
        print("Standard Deviation of Validation Error = {:.4f}\n".format(cost_std))

        print("Saving model and results...\n")
        joblib.dump(trained_model, "./models/" + suffix + "_model_fold_" + str(i))
        np.savetxt("./results/" + suffix + "_train_cost_fold_" + str(i) + ".csv", train_cost)
        np.savetxt("./results/" + suffix + "_test_cost_fold_" + str(i) + ".csv", test_cost)
        np.savetxt("./results/" + suffix + "_test_accuracy_fold_" + str(i) + ".csv", test_accuracy)

    print("Average validation accuracy = {:.4f}".format(np.mean(test_accuracies)))


def evaluate_model_performance(model, test_data):
    print("Predicting outputs...")
    predictions = model.predict(test_data[:, :-1])
    predictions = np.squeeze(predictions)
    ground_truth = test_data[:, -1]
    num_classes = np.unique(ground_truth).shape[0]
    print("Evaluating metrics...")
    confusion_mat = utils.confusion_matrix(predictions, ground_truth, num_classes)
    total_accuracy, precision, recall, f1_score = utils.metrics(predictions, ground_truth, num_classes)
    np.set_printoptions(precision=4, suppress=True)
    print("\nOverall accuracy = {}\n".format(total_accuracy))
    print("Confusion Matrix:\n{}\n".format(confusion_mat))
    print("Precision wrt each class:\n{}\n".format(precision))
    print("Recall wrt each class:\n{}\n".format(recall))
    print("F1-score wrt each class:\n{}\n".format(f1_score))


def evaluate_model_on_images(model, test_data, k=3):
    idx = utils.read_from_file("./RandomIdx.txt").astype(int)
    test_samples = test_data[idx]
    prediction = model.predict(test_samples[:, :-1], one_hot_output=True)
    prediction = np.squeeze(prediction)
    top_k = np.array([(pred.argsort()[-k:][::-1], ind) for pred, ind in zip(prediction, idx)])
    return top_k, test_samples[:, -1]


if __name__ == "__main__":

    train_data, test_data = utils.read_mnist()
    # train_data is a 2D array of shape (60000, 785), the last column being class label
    # test_data is a 2D array of shape (10000, 785), the last column being class label

    if args.train:
        if args.extract_features:
            print("Extracting features from training image data")
            features = np.array([utils.hog_features(img[:-1]) for img in train_data])
            train_data = np.concatenate((features, np.expand_dims(train_data[:, -1], axis=-1)), axis=1)

        model = Model(layers=[train_data.shape[1] - 1, 1000, 500, 250, 10],
                      hidden_activation_name=args.hidden_activation,
                      output_activation_name="softmax",
                      loss_fn_name="cross_entropy")

        if args.cross_validate:
            cross_validation(model, train_data, K=5)
        else:
            if args.extract_features:
                print("Extracting features from test image data")
                features = np.array([utils.hog_features(img[:-1]) for img in test_data])
                test_data = np.concatenate((features, np.expand_dims(test_data[:, -1], axis=-1)), axis=1)

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
                          data_augmentation=args.augment_data,
                          save_model=False,
                          model_name=os.path.join(args.model_dir, args.hidden_activation + '_' + args.splname))

            print("Saving model...")
            joblib.dump(trained_model, "./models/model_" + args.hidden_activation + "_" + args.splname)

            plt.title("Training cost and test cost")
            plt.xlabel("# iterations")
            plt.ylabel("cross-entropy loss")
            plt.plot(train_cost[:, 0], train_cost[:, 1], label="Training cost")
            plt.plot(test_cost[:, 0], test_cost[:, 1], label="Test cost")
            plt.legend()
            plt.savefig("./results/model_" + args.hidden_activation + "_" + args.splname)
            plt.show()

    else:
        if args.extract_features:
            print("Extracting features from test image data")
            features = np.array([utils.hog_features(img[:-1]) for img in test_data])
            test_data = np.concatenate((features, np.expand_dims(test_data[:, -1], axis=-1)), axis=1)
        if args.top_predictions is not 0:
            top_k, true_pred = \
                evaluate_model_on_images(joblib.load(args.model_path), test_data, k=args.top_predictions)
            print("Top {} predictions:\n{}\nTrue values\n{}".format(args.top_predictions, top_k, true_pred))
        else :
            print("Starting evaluation of model stored at ", args.model_path)
            evaluate_model_performance(joblib.load(args.model_path), test_data)