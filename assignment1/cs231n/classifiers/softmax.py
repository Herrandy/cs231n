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

  num_classes = W.shape[1]
  for i in xrange(X.shape[0]):
    f = X[i].dot(W)
    f -= np.max(f) #  to prevent overflow
    f_corr = f[y[i]]
    sum_exp = np.sum(np.exp(f))
    loss += -1 * f_corr + np.log(sum_exp)

    for j in xrange(num_classes):
      if y[i] == j:
        dW[:, y[i]] += ((np.exp(f_corr) / sum_exp) - 1.0) * X[i]
        continue
      dW[:, j] += (np.exp(f[j]) / sum_exp * X[i])

  loss /= X.shape[0]
  loss += reg * np.sum(W * W)

  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  dW = np.zeros_like(W)
  F = X.dot(W)
  F -= np.array([np.max(F, axis=1)] * num_classes).transpose()
  f_corr = F[xrange(F.shape[0]), y]
  f_exp = np.exp(F)
  f_sum_exp = np.sum(f_exp, axis=1)

  ### loss function
  loss = -1 * f_corr + np.log(f_sum_exp)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  loss = sum(loss)

  # dw_corr_class =  (((np.exp(f_corr) / f_sum_exp) - 1.0 * np.ones(f_corr.shape[0])) * X.transpose())

  ### gradient
  # case y != k
  dW_temp = f_exp / (f_sum_exp.reshape(f_sum_exp.shape[0], 1))
  # case y == k
  dW_temp[range(dW_temp.shape[0]), y] = ((np.exp(f_corr) / f_sum_exp) - 1.0)
  dW = np.dot(X.transpose(), dW_temp)
  dW /= X.shape[0]
  dW += 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

