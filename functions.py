import numpy as np


class FunctionLibrary:

    def __init__(self):
        pass

    @staticmethod
    def relu(x):
        """
        :type x: float or vector of floats
        """
        ret = np.where(x > 0, x, 0)
        return ret.reshape(x.shape)

    @staticmethod
    def relu_grad(x):
        ret = np.where(x > 0, 1, 0)
        return ret.reshape(x.shape)

    @staticmethod
    def sigmoid(x):
        """
        :type x: float or vector of floats
        """
        ret = 1.0 / (1.0 + np.exp(-x))
        return ret.reshape(x.shape)

    def sigmoid_grad(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod
    def softmax(x):
        exponents = np.exp(x - np.max(x))
        return exponents / np.sum(exponents)

    @staticmethod
    def softmax_grad():
        pass

    @staticmethod
    def l2_loss(x):
        return np.mean(x ** 2)

    @staticmethod
    def l2_loss_grad(x):
        return 2 * np.mean(x)

    @staticmethod
    def l1_loss(x):
        return np.mean(np.abs(x))

    @staticmethod
    def l1_loss_grad(x):
        ret = np.where(x > 0, 1, -1)
        arr_x, arr_y = np.where(x == 0)
        for i, j in zip(arr_x, arr_y):
            ret[i, j] = 0.0
        return ret.reshape(x.shape)

    @staticmethod
    def mse(x, y):
        """
        Calculates the error of x with respect to y, i.e, x is the output obtained, y is the ground truth output
        :param x: float or vector of floats
        :param y: float or vector of floats
        :return: scalat float value
        """
        return np.mean((x - y) ** 2)

    @staticmethod
    def mse_grad(x, y):
        """
        Calculates the gradient of MSE cost wrt estimate x. Vector y is assumed to be ground truth
        :param x: float or vector of floats
        :param y: float or vector of floats
        :return: float or vector of gradient os MSE cost wrt x
        """
        return 2.0 * (x - y)

    @staticmethod
    def cross_entropy(x, y):
        """
        Suitable for classification problem where all elements in x and y lies in range [0, 1]
        :param x: Estimated probability distribution -- float or vector of floats
        :param y: Target probability distribution -- float or vector of floats
        :return: a scalar float value equal to the cross entropy cost between the probability distributions x and y
        """
        return np.sum(y * -np.log(x))

    @staticmethod
    def cross_entropy_grad(x, y):
        """
        Suitable for classification problem where all elements in x and y lies in range [0, 1]
        :param x: Estimated probability distribution -- float or vector of floats
        :param y: Target probability distribution -- float or vector of floats
        :return: vector of float having same shape as input
        """
        return -y / x


class Activation:

    def __init__(self, activation_name):
        library = FunctionLibrary()
        self.name = activation_name
        self.func = getattr(library, activation_name)
        self.grad = getattr(library, activation_name + "_grad")
