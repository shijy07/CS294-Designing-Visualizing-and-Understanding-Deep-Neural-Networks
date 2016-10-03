import numpy as np

from cs294_129.layers import *
from cs294_129.fast_layers import *
from cs294_129.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
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
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size
    stride = 1
    pad = (filter_size - 1) / 2
    H2 = (1 + (H + 2 * pad - HH) / stride) / 2
    W2 = (1 + (W + 2 * pad - WW) / stride) / 2

    self.params['W1'] = np.random.normal(0, weight_scale,(num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(0, weight_scale, (H2 * W2 * num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
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
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    N, num_filters, HH, WW = out1.shape
    out1_flattened = out1.reshape(N, num_filters * HH * WW)
    out2, cache2 = affine_relu_forward(out1_flattened, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)
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
    loss, dout3 = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

    dout2, dw3, db3 = affine_backward(dout3, cache3)
    dw3 += self.reg * W3

    grads['W3'] = dw3
    grads['b3'] = db3

    dout1, dw2, db2 = affine_relu_backward(dout2, cache2)
    dw2 += self.reg * W2

    grads['W2'] = dw2
    grads['b2'] = db2

    dout1_reshaped = dout1.reshape(N, num_filters, HH, WW)
    dx, dw1, db1 = conv_relu_pool_backward(dout1_reshaped, cache1)
    dw1 += self.reg * W1

    grads['W1'] = dw1
    grads['b1'] = db1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

#########################################################################
#            CONVOLUTIONAL NEURAL NETWORK                               #
#########################################################################
class ConvNet(object):
    """
    Net Archetectures:

      [conv-(sbn)-relu-pool]XN - [affine-(bn)- relu]X(M - 1) - affine  - [softmax]

    """
    def __init__(self, num_filters, filter_sizes, conv_strides, pool_sizes, pool_strides, affine_dims,
                   input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-4, reg=0.0, use_sbn = False, use_bn = True,
                 dtype=np.float32):

        M = len(affine_dims)+1
        N = len(filter_sizes)
        self.M = M
        self.N = N
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_layer = N + M + 1
        self.use_batchnorm = use_bn
        self.use_spatial_batchnorm = use_sbn
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.pool_sizes = pool_sizes
        # W_conv_i, b_conv_i convlution layer parameter
        # W_affine_i, b_affine_i affine layer parameter
        input_dim_conv = input_dim
        for i in range(N):
            C, H, W = input_dim_conv
            HH = filter_sizes[i]
            WW = filter_sizes[i]
            stride = conv_strides[i]
            num_filter_layer = num_filters[i]
            self.params['W_conv' + str(i + 1)] = np.random.normal(0, weight_scale,(num_filter_layer, C, HH, WW))
            self.params['b_conv' + str(i + 1)] = np.zeros(num_filter_layer)
            if use_sbn:
                self.params['gamma_sbn'+ str(i + 1)] =  np.ones(C)
                self.params['beta_sbn' + str(i + 1)] =  np.zeros(C)
            C = num_filter_layer

            pad = (HH - 1) / 2
            input_dim_pool = num_filter_layer,(H +2 * pad - HH)/stride + 1, (W + 2 * pad - WW)/stride + 1
            HHP = pool_sizes[i]
            WWP = pool_sizes[i]
            CP, HP, WP = input_dim_pool 
            stride_pool = pool_strides[i]              
            input_dim_conv = num_filter_layer, (HP - HHP)/ stride_pool + 1, (WP - WWP)/ stride_pool + 1
        
        input_dim_affine = input_dim_conv
        CA, HA, WA = input_dim_affine 
        dims = [CA * HA * WA] + affine_dims
        for i in range(M - 1):
            self.params['W_affine' + str(i + 1)] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params['b_affine' + str(i + 1)] = np.zeros((dims[i+1])) 
            if use_bn:
                self.params['gamma_bn' + str(i + 1)] = np.ones((dims[i + 1]))
                self.params['beta_bn' + str(i + 1)] = np.zeros((dims[i + 1]))

        self.params['W_affine' + str(M)] = weight_scale * np.random.randn(dims[M-1], num_classes)
        self.params['b_affine' + str(M)] = np.zeros((num_classes))

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(M - 1)]
        
        self.sbn_params = []
        if self.use_spatial_batchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(N)]
    
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        M = self.M 
        N = self.N
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        if self.use_spatial_batchnorm:
            for sbn_param in self.sbn_params:
                sbn_param[mode] = mode
        
        scores = None
        cache = {}

        out_forward = X
        for i in range(self.N):
            index = str(i+ 1)

            W_conv, b_conv = self.params['W_conv'+index], self.params['b_conv'+index]
            
            filter_size = W_conv.shape[2]
            conv_stride = self.conv_strides[i]
            pool_size = self.pool_sizes[i]
            pool_stride = self.pool_strides[i]

            conv_param = {'stride': conv_stride, 'pad': (filter_size - 1) / 2}
            pool_param = {'pool_height': pool_size, 'pool_width': pool_size, 'stride': pool_stride}
            w = self.params['W_conv' + index]
            b = self.params['b_conv' + index]
            if self.use_spatial_batchnorm:

                gamma = self.params['gamma_sbn' +index]
                beta = self.params['beta_sbn' + index] 
                   
                out_forward, cache['conv'+ index] = \
                     conv_sbn_relu_pool_forward(out_forward, w, b, conv_param, pool_param, gamma, beta, self.sbn_params[i])
            else:
                out_forward, cache['conv'+ index] = \
                     conv_relu_pool_forward(out_forward, w, b, conv_param, pool_param )
        
        for i in range(self.M - 1):
            index = str(i+1)
            w = self.params['W_affine' + index]
            b = self.params['b_affine' + index]
            out_forward, cache['affine' + index ] = affine_forward(out_forward, w, b)
            # bachnorm
            if self.use_batchnorm:
                gamma = self.params['gamma_bn' + index]
                beta = self.params['beta_bn' + index]
                out_forward, cache['bn' + index ] = batchnorm_forward(out_forward, gamma,beta , self.bn_params[i])
      
            #relu_affine
            out_forward, cache['relu_affine' + index] = relu_forward(out_forward)
        scores, cache['affine'+ str(M)] = \
            affine_forward(out_forward, self.params['W_affine' + str(M)], self.params['b_affine' + str(M)])

        if mode == 'test':
            return scores
        loss, grads = 0.0, {}

        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W_affine' + str(M)] * self.params['W_affine' + str(M)]))
        index = str(M)
        dout, grads['W_affine'+index], grads['b_affine'+index] = \
            affine_backward(dout, cache['affine'+index])
        grads['W_affine'+ index] += self.reg * self.params['W_affine'+index]
        for indx in range(M - 2, -1, -1):
            index = str(indx + 1)
            loss += 0.5 * self.reg * (np.sum(self.params['W_affine' + index] * self.params['W_affine' + index]))
            # relu backward
            dout = relu_backward(dout, cache['relu_affine' + index])
            # batchnorm backward
            if self.use_batchnorm:
                dout, grads['gamma_bn'+ index], grads['beta_bn' + index] = \
                      batchnorm_backward_alt(dout, cache['bn' + index])
             # affine backward
            dout, grads['W_affine'+ index], grads['b_affine' + index] = \
                affine_backward(dout, cache['affine' + index])
            grads['W_affine' + index] += self.reg * self.params['W_affine' + index]

        for i in range(N-1, -1, -1):
            index = str(i+1)
            loss += 0.5 *self.reg * (np.sum(self.params['W_conv' + index] * self.params['W_conv' + index]))
            if self.use_spatial_batchnorm:
                dout, grads['W_conv' + index], grads['b_conv'+index] = conv_sbn_relu_pool_backward(dout, cache['conv'+index])
            else:
                dout, grads['W_conv' + index], grads['b_conv'+index] = conv_relu_pool_backward(dout, cache['conv'+index])
            grads['W_conv'+ index] += self.reg * self.params['W_conv'+index]
        return loss, grads

