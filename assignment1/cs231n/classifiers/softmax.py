from math import log
from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(X.shape[0]):
        scores = X[i].dot(W)
        regularized_scores = scores - np.max(scores)
        exp_scores = np.exp(regularized_scores)

        loss -= log(exp_scores[y[i]]/np.sum(exp_scores))

        for j in range(W.shape[1]):
            S_j = exp_scores[j]/np.sum(exp_scores)
            if j==y[i]:
                dW[:,j] += X[i] * (S_j-1)
            else:
                dW[:,j] += S_j*X[i]
   
    loss /= X.shape[0]
    dW /= X.shape[0]
    
    loss += reg * np.sum(W * W)
    dW  += reg*2*W
   # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X, W)
    maxes = np.amax(scores, axis=1)
    scores = np.exp(scores - maxes[:,None])
    probs = scores/np.sum(scores, axis = 1).reshape(-1,1)

    loss -= np.sum(np.log(probs[range(X.shape[0]),y]))

    gradient_q = (scores.T/np.sum(scores, axis = 1)).T
    gradient_q[range(X.shape[0]),y] -= 1
    dW = np.matmul(X.T, gradient_q)

    loss /= X.shape[0]
    dW /= X.shape[0]
    
    loss += reg * np.sum(W * W)
    dW  += reg*2*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
