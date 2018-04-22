import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):  # go through all the data points
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # all the classes expect the one which is correct for the current data points
    # if the estimated class score is bigger for some other class (+ delta) than for the right one,
    # increment the loss and adjust the weight matrix.
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :]
        dW[:, y[i]] -= X[i, :]  # sum because we are iterating over all the data points

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)

  corr_class_scores = scores[range(len(y)), y].reshape(-1, 1)
  margins = np.maximum(0, scores - corr_class_scores + 1)
  margins[np.arange(len(y)), y] = 0  # do not take into account the loss for the correct class
  loss = sum(sum(margins))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ''' old way
  coeffs = np.zeros_like(margins)
  coeffs[margins > 0] = 1
  dW += np.dot(X.T, coeffs)  # dW updated based on incorrectly classified samples

  coeffs = np.zeros_like(margins)
  # count the instances where a sample did not meet the margin
  coeffs[np.arange(len(y)), y] = (-1) * np.sum(coeffs, axis=1)
  dW += np.dot(X.T, coeffs)
  '''
  coeffs = np.zeros_like(margins)
  coeffs[margins > 0] = 1
  # count the instances where a sample did not meet the margin
  coeffs[np.arange(len(y)), y] = (-1) * np.sum(coeffs, axis=1)
  dW += np.dot(X.T, coeffs)


  loss /= X.shape[0]
  dW /= X.shape[0]

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization to the gradient
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
