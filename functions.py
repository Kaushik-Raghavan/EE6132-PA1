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
        sgm = self.sigmoid(x)
        return sgm * (1 - sgm)

    @staticmethod
    def softmax(x):
        exponents = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
        ret = exponents / np.expand_dims(np.sum(exponents, axis=1), axis=1)
        assert (not (ret == 0).any()), "Error in softmax\n{}".format(x)
        return ret

    def softmax_grad(self, x):
        """
        Evaluates gradient matrix of softmax function
        :param x: A vector with shape (len,) or (len, 1), wrt which softmax gradient matrix need to be computed
        :return: A matrix with shape (len, len). Element at ith row and jth column corresponds to derivative of
                 softmax(x)_j wrt x_i
        """
        p = self.softmax(x)
        if p.ndim == 1:
            np.expand_dims(p, axis=-1)
        ret = np.fill_diagonal(np.zeros((p.shape[0], p.shape[0])), p) - np.matmul(p, p.T)
        return ret

    @staticmethod
    def l2_loss(x):
        return np.sum(x ** 2)

    @staticmethod
    def l2_loss_grad(x):
        return 2 * x

    @staticmethod
    def l1_loss(x):
        return np.sum(np.abs(x))

    @staticmethod
    def l1_loss_grad(x):
        ret = np.where(x > 0, 1, -1) * np.where(x == 0, 0, 1)
        return ret.reshape(x.shape)

    @staticmethod
    def mse(x, y):
        """
        Calculates the squared error of x with respect to y, i.e, x is the output obtained, y is the ground truth output
        :param x: float or vector of floats
        :param y: float or vector of floats
        :return: scalar float value
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
        return 2.0 * (x - y) / x.size

    @staticmethod
    def mae(x, y):
        """
        Calculates the absolute error of x with respect to y, where, x : output obtained, y : the ground truth labels
        :param x: float or vector of floats
        :param y: float or vector of floats
        :return: scalar float value
        """
        return np.mean(np.abs(x - y))

    @staticmethod
    def mae_grad(x, y):
        """
        Calculates the gradient of MAE cost wrt estimate x. Vector y is assumed to be ground truth
        :param x: float or vector of floats
        :param y: float or vector of floats
        :return: float or vector of gradient os MSE cost wrt x
        """
        diff = x - y
        ret = np.where(diff > 0, 1, -1) * np.where(diff == 0, 0, 1)
        return ret.reshape(x) / x.size

    @staticmethod
    def cross_entropy(x, y):
        """
        Suitable for classification problem where all elements in x and y lies in range [0, 1]
        :param x: Estimated probability distribution -- float or vector of floats
        :param y: Target probability distribution -- float or vector of floats
        :return: a scalar float value equal to the cross entropy cost between the probability distributions x and y
        """
        assert not (x == 0).any(), "x is 0 somewhere {}".format(x)
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
