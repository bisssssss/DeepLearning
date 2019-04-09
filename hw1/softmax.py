import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, fwd_fun=relu_forward, bwd_fun=relu_backward):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.fwd_fun = fwd_fun
        self.bwd_fun = bwd_fun
        input_dim_vec = np.prod(input_dim)

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        
        if hidden_dim == None:

            W1 = np.random.normal(0, weight_scale, (input_dim_vec, num_classes))
            b1 = np.zeros((num_classes,))
            self.params = {'W1': W1, 'b1': b1}

        else:

            W1 = np.random.normal(0, weight_scale, (input_dim_vec, hidden_dim))
            b1 = np.zeros((hidden_dim,))
            W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
            b2 = np.zeros((num_classes,))
            self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        W1 = self.params['W1']
        b1 = self.params['b1']
        out_fc1, cache_fc1 = fc_forward(X, W1, b1)
        scores = out_fc1

        if len(self.params) > 2:
            out_fc1, cache_fc1 = fc_forward(X, W1, b1)
            out_fc_act1, cache_fc_act1 = self.fwd_fun(out_fc1) # ReLU ???????????????????????????
            W2 = self.params['W2']
            b2 = self.params['b2']
            out_fc2, cache_fc2 = fc_forward(out_fc_act1, W2, b2)
            scores = out_fc2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dLdscores = softmax_loss(scores, y)

        for p, w in self.params.items():
            loss += 1/2 * self.reg * np.sum(self.params[p]**2)

        if len(self.params) == 2:

            dLdout1 = dLdscores
            dLdX, dLdW1, dLdb1 = fc_backward(dLdout1, cache_fc1)
            dLdW1 += self.reg * np.reshape(W1, dLdW1.shape)
            grads = {'W1': dLdW1, 'b1': dLdb1}

        else:

            dLdout2 = dLdscores
            dLdout1_, dLdW2, dLdb2 = fc_backward(dLdout2, cache_fc2)
            dLdout1 = self.bwd_fun(dLdout1_, cache_fc_act1) # ReLU ???????????????????????????
            dLdX, dLdW1, dLdb1 = fc_backward(dLdout1, cache_fc1)
            dLdW1 += self.reg * np.reshape(W1, dLdW1.shape)
            dLdW2 += self.reg * np.reshape(W2, dLdW2.shape)
            grads = {'W1': dLdW1, 'b1': dLdb1, 'W2': dLdW2, 'b2': dLdb2}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
