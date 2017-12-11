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
  dW = np.zeros_like(W) # D x C
  num_train = X.shape[0] # N
  num_class = W.shape[1] # C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for sample_i in xrange(num_train):
    exp_scores = np.exp(X[sample_i].dot(W)) #1xC
    sum_deno = 0.0
    
    for class_j in xrange(num_class):
      sum_deno += exp_scores[class_j]
    
    for class_j in xrange(num_class):
      dW[:,class_j] += exp_scores[class_j]*X[sample_i]/sum_deno
      if y[sample_i] == class_j:
        dW[:, y[sample_i]] -= X[sample_i]


    
    loss += -np.log(exp_scores[y[sample_i]] / sum_deno)



  loss = loss/num_train
  loss += 0.5*reg*np.sum(W*W)
  
  dW = dW/num_train
  dW += W*reg

  
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
  num_train = X.shape[0] # N
  num_class = W.shape[1] # C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass\
  scores = X.dot(W)
  exp_scores = np.exp(X.dot(W)) #NxC
  sum_j = np.sum(exp_scores, axis = 1).reshape(num_train,1)
  log_sum_j = np.log(sum_j) # Nx1
  log_part_sum = np.sum(log_sum_j)
  
  correct_class_scores = scores[range(num_train), y].reshape(num_train,1) #Nx1 #1xC !!remember range(num_train)!!
  correct_class_sum = np.sum(correct_class_scores)
  
  loss = -correct_class_sum + log_part_sum
  loss = loss/num_train
  loss += 0.5*reg*np.sum(W*W)
  
  #gradient dW: DxC X:NxD X':DxN
  correct_flag_matrix = np.zeros(exp_scores.shape) #NxC
  correct_flag_matrix[range(num_train), y] = 1
  
  dW = X.T.dot(exp_scores/sum_j) - X.T.dot(correct_flag_matrix)
  dW = dW/num_train + reg*W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

