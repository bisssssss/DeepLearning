import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, dropout=0, normalization=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    maxpool_dim = int((((input_dim[1]-filter_size+1) - 2) / 2) + 1)
    maxpool_dim_vec = num_filters * maxpool_dim * maxpool_dim
    W1 = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
    b1 = np.zeros((1))
    W2 = np.random.normal(0, weight_scale, (maxpool_dim_vec, hidden_dim))
    b2 = np.zeros((hidden_dim,))
    W3 = np.random.normal(0, weight_scale, (hidden_dim, num_classes)) # dropout
    b3 = np.zeros((num_classes,))
    self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    self.drop = 0

    # Normalization and dropout
    if normalization == True:
        gamma = np.random.normal(0, weight_scale, (hidden_dim,))
        beta = np.random.normal(0, weight_scale, (hidden_dim,))
        self.params['gamma'] = gamma
        self.params['beta'] = beta
    if dropout != 0:
        self.drop = dropout

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    drop = self.drop
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    # architecture: 
    # conv-(relu-maxpool)-fc-(relu)-fc-(softmax)

    xshape = X.shape
    X = np.reshape(X, (xshape[0], 1, xshape[1], xshape[2]))
    out_conv1, cache_conv1 = conv_forward(X, W1)        # (N, C, HH, WW)
    '''
    if normalization != None:
        if y is None:
            out_conv1_norm, cache_conv1_norm = batchnorm_forward(out_conv1, 1, 0, {'mode': 'test'})
        else:
            out_conv1_norm, cache_conv1_norm = batchnorm_forward(out_conv1, 1, 0, {'mode': 'train'})
        out_relu1, cache_relu1 = relu_forward(out_conv1_norm)    # (N, C, HH, WW)
    else:
        out_relu1, cache_relu1 = relu_forward(out_conv1)    # (N, C, HH, WW)
    '''
    out_relu1, cache_relu1 = relu_forward(out_conv1)    # (N, C, HH, WW)
    out_maxpool1, cache_maxpool1 = max_pool_forward(out_relu1, pool_param)    #(N, C, HH, WW)
    
    if drop != 0:
        if y is None:
            out_maxpool1_drop, _ = dropout_forward(out_maxpool1, {'p': drop, 'mode': 'test'})
        else:
            out_maxpool1_drop, cache_maxpool1_drop = dropout_forward(out_maxpool1, {'p': drop, 'mode': 'test'})
        out_intmat = np.reshape(out_maxpool1_drop, (out_maxpool1_drop.shape[0], W2.shape[0])) #(N, D1=C*HH*WW)
    else:
        out_intmat = np.reshape(out_maxpool1, (out_maxpool1.shape[0], W2.shape[0])) #(N, D1=C*HH*WW)
    
    out_fc1, cache_fc1 = fc_forward(out_intmat, W2, b2) # (N, D2)
    
    if len(self.params) > 6: # normalization
        beta = self.params['beta']; gamma = self.params['gamma']
        if y is None:
            out_fc1_norm, _ = batchnorm_forward(out_fc1, gamma, beta, {'mode': 'test'})
        else:
            out_fc1_norm, cache_fc1_norm = batchnorm_forward(out_fc1, gamma, beta, {'mode': 'train'})
        out_relu2, cache_relu2 = relu_forward(out_fc1_norm)
    else:
        out_relu2, cache_relu2 = relu_forward(out_fc1)      # (N, D2)

    if drop != 0:
        if y is None:
            out_relu2_drop, _ = dropout_forward(out_relu2, {'p': drop, 'mode': 'test'})
        else:
            out_relu2_drop, cache_relu2_drop = dropout_forward(out_relu2, {'p': drop, 'mode': 'train'})
        out_fc2, cache_fc2 = fc_forward(out_relu2_drop, W3, b3)
    else:
        out_fc2, cache_fc2 = fc_forward(out_relu2, W3, b3)  # (N, D3=10)
    
    scores = out_fc2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dLdscores = softmax_loss(scores, y)

    for p, w in self.params.items():
        loss += 1/2 * self.reg * np.sum(self.params[p]**2)

    dout_fc2 = dLdscores

    if drop != 0:
        dout_relu2_drop, dW3, db3 = fc_backward(dout_fc2, cache_fc2)
        dout_relu2 = dropout_backward(dout_relu2_drop, cache_relu2_drop)
    else:
        dout_relu2, dW3, db3 = fc_backward(dout_fc2, cache_fc2)

    # normalization
    if len(self.params) > 6:
        dout_fc1_norm = relu_backward(dout_relu2, cache_relu2)
        dout_fc1, dgamma, dbeta = batchnorm_backward(dout_fc1_norm, cache_fc1_norm)
    else:
        dout_fc1 = relu_backward(dout_relu2, cache_relu2)
  
    dout_intmat, dW2, db2 = fc_backward(dout_fc1, cache_fc1)

    if drop != 0:
        dout_maxpool1_drop = np.reshape(out_intmat, out_maxpool1_drop.shape)
        dout_maxpool1 = dropout_backward(dout_maxpool1_drop, cache_maxpool1_drop)
    else:
        dout_maxpool1 = np.reshape(dout_intmat, out_maxpool1.shape)
    
    dout_relu1 = max_pool_backward(dout_maxpool1, cache_maxpool1)
    dout_conv1 = relu_backward(dout_relu1, cache_relu1)
    dX, dW1 = conv_backward(dout_conv1, cache_conv1)

    # regularization
    dW1 += self.reg * np.reshape(W1, dW1.shape)
    dW2 += self.reg * np.reshape(W2, dW2.shape)
    dW3 += self.reg * np.reshape(W3, dW3.shape)
    db1 = np.zeros((1))

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    if len(self.params) > 6:
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'gamma': dgamma, 'beta': dbeta}

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
