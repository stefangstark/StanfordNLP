#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def extract(params, Dx, H, Dy):
    ofs = 0
    
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    import ipdb; ipdb.set_trace()
    return


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    M = data.shape[0]

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1) + b1
    hidden = sigmoid(z1)
    z2 = hidden.dot(W2) + b2
    output = softmax(z2)
    cost = -np.sum(labels * np.log(output))
    ### END YOUR CODE

    assert z1.shape == (M, H)
    assert hidden.shape == (M, H)
    assert z2.shape == (M, Dy)
    assert output.shape == (M, Dy)

    ### YOUR CODE HERE: backward propagation
    if False:
        gradW2 = 0
        gradb2 = 0
        gradW1 = 0
        gradb1 = 0
        for indx in range(data.shape[0]):
            gradz2 = output[indx] - labels[indx]
            gradz2 = gradz2[np.newaxis,:]
            gradW2 += np.outer(hidden[indx], gradz2)
            gradb2 += gradz2
            
            gradz1 = gradz2.dot(W2.T) * sigmoid_grad(sigmoid(z1[indx]))
            gradW1 += np.outer(data[indx], gradz1)
            gradb1 += gradz1
        gradW2 = gradW2 / data.shape[0]
        gradb2 = gradb2 / data.shape[0]
        gradW1 = gradW1 / data.shape[0]
        gradb1 = gradb1 / data.shape[0]
    else:
        gradz2 = (output - labels) / data.shape[0]
        gradW2 = hidden.T.dot(gradz2)
        gradb2 = gradz2.sum(axis=0, keepdims=True)

        gradz1 = gradz2.dot(W2.T) * sigmoid_grad(sigmoid(z1))
        gradW1 = data.T.dot(gradz1)
        gradb1 = gradz1.sum(axis=0, keepdims=True)

    assert gradW2.shape == W2.shape
    assert gradb2.shape == b2.shape
    assert gradW1.shape == W1.shape
    assert gradb1.shape == b1.shape
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    assert grad.size == params.size

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."
    N = 1
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    reldiffs = gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    Dx, H, Dy = dimensions
    extract(reldiffs, Dx, H, Dy)
    return

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
