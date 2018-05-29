# Function to calculate hidden layer from data
# 
# @keyword internal
# Function for calculating hidden layer:
VisToHid <- function(vis, weights, y, y.weights) {
  # Function for calculating a hidden layer.
  #
  # Args:
  #   vis: Visual layer, or hidden layer from previous layer in DBN
  #   weights: Trained weights including the bias terms (use RBM)
  #   y: Label vector if only when training an RBM for classification
  #   y.weights: Label weights and bias matrix, only neccessary when training a RBM for classification
  #
  # Returns:
  #   Returns a hidden layer calculated with the trained RBM weights and bias terms.
  #
  # Initialize the visual, or i-1 layer
  V0 <- vis
  if ( is.null(dim(V0))) {
    # If visual is a vector create matrix
    V0 <- matrix(V0, nrow= length(V0))
  }
  if(missing(y) & missing(y.weights)) {
    # Calculate the hidden layer with the trained weights and bias
    H <- 1/(1 + exp(-( V0 %*% weights))) 
  } else {
    Y0 <- y
    H <- 1/(1 + exp(- ( V0 %*% weights + Y0 %*% y.weights))) 
  }
  return(H)
}

# Function for reconstructing data from a hidden layer
# 
# @keyword internal
# Function for reconstructing visible layer:
HidToVis <- function(inv, weights, y.weights) {
  # Function for reconstructing a visible layer.
  #
  # Args:
  #   inv: Invisible layer
  #   vis.bias: Trained visible layer bias (use RBM)
  #   weights: Trained weights (use RBM)
  #   y.weights: Label weights, only nessecessary when training a classification RBM.
  #
  # Returns:
  #   Returns a vector with reconstructed visible layer or reconstructed labels.
  #
  if(missing(y.weights)) {
    # Reconstruct only the visible layer when y.weights is missing
    V <- 1/(1 + exp(-(   inv %*% t(weights)) ))
    return(V)
  } else {
    # Reconstruct visible and labels if y.weights
    Y <- 1/(1 + exp(-( inv %*% t(y.weights)))) 
    return(Y)
  }
}

# Logistic function
# 
# @keyword internal
# Logistic function
logistic <- function(x) {
  1/(1+exp(-x))
}

# Function to calculate the energy of a RBM 
# 
# @keyword internal
# Function for calculating the energy of the machine:
Energy <- Energy <- function(vis, inv, weights, y, y.weights) {
  # Function for calculating the energy of a trained RBM
  #
  # Args:
  #   vis: visible layer
  #   weights: the weights matrix including the bias terms
  #   inv: invisible layer
  #   y: label vector (binary)
  #   y.weights: trained label weights (use RBM), including bias terms
  #
  # Returns:
  #   The energy of the RBM machine for label y
  #
  # Calculate the energy if supervised
  if(!missing(y) & !missing(y.weights)){
    E <- -(vis %*% weights %*% t(inv)) - (y %*% y.weights %*% t(inv))
  } else {
    # Calculate the energy if unsupervised
    E <- -(vis %*% weights %*% t(inv))
  }
  # Return the energy:
  return(E)
  
}

# Function for binarizing label data
# 
# @keyword internal
# Function for binarizing labels:
LabelBinarizer <- function(labels) {
  # This function takes as input the labels of the trainset.
  # Args:
  #   Labels: has to be numerical data vector from 1 to 9.
  #
  # Returns:
  #   Matrix with binarized vectors for the labels that can be used in the RBM function
  #
  # Initialize matrix to save label vectors:
  y <- matrix(0, length(labels), 10)
  for (i in 1:length(labels)) {
    # Put a one on position of the number in vector:
    y[i, labels[i] + 1] <- 1
  }
  return(y)
}
