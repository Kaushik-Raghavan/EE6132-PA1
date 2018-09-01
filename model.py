import numpy as np
import random
import functions


def one_hot(x, size):
    if not hasattr(x, "__len__"):  # x is a scalar
        x = np.array([x])
    ret = np.zeros((len(x), size))
    for i in range(len(x)):
        ret[i, int(x[i])] = 1.0
    return ret


class Model:

    def __init__(self, layers, hidden_activation_name="sigmoid", output_activation_name="sigmoid", loss_fn_name="mse",
                 l2_weight=None, l1_weight=None):
        """

        :param layers: A list containing the description of layers of the architecture. ith layer of the network will
                        have layers[i] neurons
        :param hidden_activation_name: A string specifying the activation function to be used for hidden layers neurons
        :param output_activation_name: A string specifying the activation function to be used for output layer neurons
        :param loss_fn_name: A string specifying the loss function to be used to compare output with ground truth
        :param l2_weight:
        :param l1_weight:
        """
        assert len(layers) >= 2, "There should be atleast 2 layers in a Neural Network"

        self.layers = layers
        self.num_layers = len(layers)
        self.weights = np.array([np.random.normal(loc=0.0, scale=0.08, size=(layers[i], layers[i - 1]))
                                 for i in range(1, self.num_layers)])
        self.bias = np.array([np.random.normal(loc=0.0, scale=0.08, size=(layers[i], 1))
                              for i in range(1, self.num_layers)])
        self.delta_w = np.array([np.zeros((layers[i], layers[i - 1])) for i in range(1, self.num_layers)])
        self.delta_b = np.array([np.zeros((layers[i], 1)) for i in range(1, self.num_layers)])

        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.loss_fn_name = loss_fn_name
        self.activation = functions.Activation(hidden_activation_name)
        self.output_activation = functions.Activation(output_activation_name)
        self.loss_fn = functions.Activation(loss_fn_name)
        self.l2_loss = functions.Activation("l2_loss")
        self.l1_loss = functions.Activation("l1_loss")
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight

        self.Z = []
        self.A = []

    def feed_forward(self, x, save_activations=False):
        """
        Performs feed forward process of feeding the data at input and generating activations in hidden layer and
        outputs in final layer.
        :param x: A column vector of shape (self.layers[0], 1), where self.layers[0] is the size of first (input) layer
        :param save_activations: A boolean flag indicating whether to copy the activations obtained in the feed forward
        process to a model variable
        :return: returns the estimated output activations
        """
        assert x.shape == (self.layers[0], 1), "Input dimension for feed forward does not match"
        if save_activations:
            self.Z.clear()
            self.A.clear()

        self.A.append(x.copy())
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            x = np.matmul(w, x) + b
            if save_activations:
                self.Z.append(x.copy())
            x = self.activation.func(x)
            if save_activations:
                self.A.append(x.copy())

        # When output activation is different from activations of hidden layers
        x = np.matmul(self.weights[-1], x) + self.bias[-1]
        if save_activations:
            self.Z.append(x.copy())
        x = self.output_activation.func(x)
        if save_activations:
            self.A.append(x.copy())

        return x

    def cost_gradient(self, y, y_gt):
        if self.output_activation_name == "softmax":
            assert self.loss_fn_name == "cross_entropy", "softmax needs cross entropy as the cost measure"
            return y - y_gt
        return self.loss_fn.grad(y, y_gt) * self.output_activation.grad(self.Z[-1])  # Hadamard product of 2 vectors

    def get_gradients(self, x, y_gt):
        assert (self.layers[0], 1) == x.shape, "Input shape does not match. Expected shape: (len, 1)"

        y = self.feed_forward(x, True)  # self.A[-1] == y

        # Calculating gradients at output layer
        y_gt = y_gt.reshape(y.shape)
        cost = self.loss_fn.func(y, y_gt)
        delta = self.cost_gradient(y, y_gt)
        grad_weights = []
        grad_biases = []

        '''Back propagation (Layers are 0 indexed)'''
        for i in np.arange(self.num_layers - 2, -1, -1):  # i runs from (num_layers - 2) to 0
            # delta is the gradient of linear accumulation of values at layer i + 1 wrt cost function
            grad_weights.append(np.matmul(delta, self.A[i].T))  # gradient of weights connecting layer i to i + 1
            if self.l2_weight > 0:
                grad_weights[-1] += self.l2_weight * self.l2_loss.grad(self.weights[i])
            if self.l1_weight > 0:
                grad_weights[-1] += self.l1_weight * self.l1_loss.grad(self.weights[i])
            grad_biases.append(delta)  # gradient of bias of layer i + 1
            if i > 0:
                delta = np.matmul(self.weights[i].T, delta) * self.activation.grad(self.Z[i - 1])
                # delta is the gradient corresponding to ith layer accumulations now

        grad_weights = list(reversed(grad_weights))
        grad_biases = list(reversed(grad_biases))
        return np.array(grad_weights), np.array(grad_biases), cost

    def fit(self, train_data, mini_batch_size=10, num_epochs=10, learning_rate=1e-3, momentum_factor=0.1,
            l2_weight=0.0, l1_weight=0.0):
        """
        :param train_data: 2D vector of float with each row representing a separate input data
                            and the last column denoting the ground truth output
        :param mini_batch_size: integer denoting the size of mini batch for training
        :param num_epochs: integer denoting number of epochs to train
        :param learning_rate: float value denoting learning rate
        :param momentum_factor: float value denoting the factor by which change in parameters from previous iterations
                                need to be discounted
        :param l2_weight:
        :param l1_weight:
        :return: None
        """
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        num_data = train_data.shape[0]
        num_mini_batches = int(num_data / mini_batch_size)
        if num_data % mini_batch_size != 0:
            num_mini_batches += 1
        for epoch in range(num_epochs):
            random.shuffle(train_data)
            costs = []
            for batch_idx in range(num_mini_batches):
                start_index = batch_idx * mini_batch_size
                end_index = min(start_index + mini_batch_size, num_data)
                batch = train_data[start_index: end_index]
                batch_cost: float = 0.0
                for data in batch:
                    grad_weights, grad_biases, cost = self.get_gradients(data[:-1].reshape(data[:-1].shape[0], 1),
                                                                         one_hot([data[-1]], 10))
                    self.delta_w = momentum_factor * self.delta_w - learning_rate * grad_weights
                    self.delta_b = momentum_factor * self.delta_b - learning_rate * grad_biases
                    batch_cost += cost

                batch_size = float(end_index - start_index)
                costs.append(batch_cost / batch_size)
                if batch_idx % 10 == 0:
                    print("batch %i : %0.8f" % (batch_idx, batch_cost / batch_size))
                self.weights += self.delta_w / batch_size
                self.bias += self.delta_b / batch_size

    def predict(self, test_input):
        """
        :param test_input: 2D vector of float with each row representing a separate input data
        :return: A column vector of dtype int with shape (len(test_data), 1) denoting the predicted outputs
        """
        prediction = []
        for data in test_input:
            out = self.feed_forward(data)
            prediction.append(np.argmax(out, axis=0))
        return np.array(prediction)
