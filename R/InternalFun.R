# Function to calculate hidden layer from data
# 
# @keyword internal
#
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

# Function for doing contrastive divergence CD
#
# @keyword internal
CD <- function(vis, weights, y, y.weights) {
  # Function for doing k=1 contrastive divergence
  # 
  # Args:
  #   vis: visible layer values vector of shape n_features * 1
  #   weights: weights vector of shape n_features * n_hidden
  #   vis.bias: bias of the visible layer
  #   inv.bias: bias of the invisible layer
  #   y: labels, only used when provided
  #   y.weigths: label weights of shape n_labels * n_hidden, only used when provided
  #   y.bias: bias term for the labels of shape n_features * 1, only used when provided
  #
  # Returns:
  #   A list with all gradients for the bias and weights; adds label bias and weights if y is provided
  #
  # Start positive phase
  if (missing(y) & missing(y.weights)) {
    # Calculate hidden layer
    H0 <- VisToHid(vis, weights)
    H0[,1] <- 1
  } else {
    # Add a layer with labels if y is provided
    H0 <- VisToHid(vis, weights, y, y.weights)
    H0[,1] <- 1
  }
  # Binarize the hidden layer:
  unif  <- runif(nrow(H0) * (ncol(H0)))
  H0.states <- H0 > matrix(unif, nrow=nrow(H0), ncol= ncol(H0))
  
  # Calculate positive phase, we always use the probabilities for this
  pos.phase <- t(vis) %*% H0
  if (!missing(y)) {
    pos.phase.y <- t(y) %*% H0
  }
  # Start negative  phase
  # Reconstruct visible layer
  V1 <- HidToVis(H0.states, weights)
  # Set the bias unit to 1
  V1[,1] <- 1
  
  if (missing(y) & missing(y.weights) ) {
    # Reconstruct hidden layer unsupervised, no need to fix the bias anymore
    H1 <- VisToHid(V1, weights)
  } else {
    # Reconstruct labels if y is provided
    Y1 <- HidToVis(H0, weights,  y.weights )
    # Set the bias unit to 1
    Y1[,1] <- 1
    # Reconstruct hidden layer supervised, no need to fix the bias anymore
    H1 <- VisToHid(V1, weights, Y1, y.weights)
  }
  # Calculate negative associations, we alway use the probabilities for this:
  neg.phase <- t(V1) %*% H1
  if (!missing(y) & !missing(y.weights)) {
    # Calculate negative phase y
    neg.phase.y <- t(Y1) %*% H1
  }
  ## Calculate the gradients
  # Calculate gradients for the weights:
  grad.weights <- pos.phase - neg.phase
  
  if (!missing(y) & !missing(y.weights)) {
    # Calculate gradients for y.weigths
    grad.y.weights <- pos.phase.y - neg.phase.y
    
    # Return list with  gradients supervised
    return(list('grad.weights' = grad.weights,'grad.y.weights' = grad.y.weights))
  } else {
    # Return list with gradients unsupervised
    return(list('grad.weights' = grad.weights  ))
  }
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


