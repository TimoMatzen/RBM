##############################################################
###############Restricted Boltzmann Machine###################
##############################################################

# TODO: Make possibility for stacking boltzmann machines.

## Initialize train-data MNIST
# Load in the train-data of the MNIST data-set:
train1 <- read.csv('train.csv', nrows = 50000)
labels <- train1$label

# Converting labels to binary feature vectors:
y <- matrix(0,length(labels), 10)

# Loop over all labels and binarize:
for(i in 1:length(labels)){
  y[i, labels[i]+1] <- 1
}

# Put the data in a matrix of shape features * samples:
train <- matrix(unlist(train1[1:20000,-1]), nrow =784,ncol = 20000, byrow = T)/255


## Initialize test_data MNIST
test <- matrix(unlist(train1[20001:21000,-1]), nrow =784, ncol = 1000, byrow = T)/255
test_y <- labels[20001:21000]

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

# Function for calculating hidden layer:
VisToHid <- function(vis, weights, y, y.weights) {
  # Function for calculating a hidden layer.
  #
  # Args:
  #   vis: Visual layer, or hidden layer from previous layer in DBN
  #   inv.bias: Trained invisible bias (use RBM)
  #   weights: Trained weights (use RBM)
  #   y: Label vector if only when training an RBM for classification
  #   y.weights: Label weights matrix, only neccessary when training an RBM for classification.
  #
  # Returns:
  #   Returns a hidden layer calculated with the trained RBM weights and bias terms.
  #
  # Initialize the visual, or i-1 layer
  V0 <- vis
  if(missing(y) & missing(y.weights)) {
    # Calculate the hidden layer with the trained weights and bias
    H <- 1/(1 + exp(-( t(weights) %*% V0))) 
  } else {
    Y0 <- y
    H <- 1/(1 + exp(- (t(weights) %*% V0 + t(y.weights) %*% Y0))) 
  }
  return(H)
}

# Function for reconstructing visible layer:
HidToVis <- function(inv, weights, y, y.weights) {
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
    V <- 1/(1 + exp(-(  weights %*% inv)) )
    return(V)
  } else {
    # Reconstruct visible and labels if y.weights
    Y <- 1/(1 + exp(-(y.weights %*% inv))) 
    return(Y)
  }
}

# Function for calculating the energy of the machine:
Energy <- Energy <- function(vis, inv, weights, y, y.weights) {
  # Function for calculating the energy of a trained RBM
  #
  # Args:
  #   vis: visible layer
  #   vis.bias: trained visible bias (use RBM)
  #   inv: invisible layer
  #   inv.bias: trained invisible bias (use RBM)
  #   y: label vector (binary)
  #   y.weights: trained label weights (use RBM)
  #   y.bias: trained label bias (use RBM)
  #
  # Returns:
  #   The energy of the RBM machine for label y
  #
  # Calculate the energy with the parameters
  E <- -(t(vis)%*% weights %*% inv) - (y %*% y.weights %*% inv)
  # Return the energy:
  return(E)
  
}

# Function for doing contrastive divergence
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
  } else {
    # Add a layer with labels if y is provided
    H0 <- VisToHid(vis, weights, y, y.weights)
  }
  # Binarize the hidden layer:
  unif  <- runif(nrow(H0)*(ncol(H0)))
  H0 <- H0 > matrix(unif, nrow=nrow(H0), ncol= ncol(H0))
  # Calculate positive phase
  pos.phase <- vis %*% t(H0)
  if (!missing(y)) {
    pos.phase.y <- H0 %*% y
  }
  # Start negative  phase
  # Reconstruct visible layer
  V1 <- HidToVis(H0, weights)
  # Set the bias unit to 1
  V1[1,] <- 1
  
  if (missing(y) & missing(y.weights) ) {
    # Reconstruct hidden layer unsupervised
    H1 <- VisToHid(V1, weights)
  } else {
    # Reconstruct labels if y is provided
    Y1 <- HidToVis(H0, weights, y,  y.weights )
    # Set the bias unit to 1
    Y1[1,] <- 1
    # Reconstruct hidden layer supervised
    H1 <- VisToHid(V1, weights, t(Y1), y.weights)
  }
  # Calculate negative phase:
  neg.phase <- V1 %*% t(H1)
  if (!missing(y) & !missing(y.weights)) {
    # Calculate negative phase y
    neg.phase.y <- H1 %*% t(Y1)
  }
  ## Calculate the gradients
  # Calculate gradients for the weights:
  grad.weights <- pos.phase - neg.phase
  
  if (!missing(y) & !missing(y.weights)) {
    # Calculate gradients for y.weigths
    grad.y.weights <- pos.phase.y - neg.phase.y
    
    # Return list with  gradients supervised
    return(list('grad.weights' = grad.weights,'grad.y.weights' = t(grad.y.weights)))
  } else {
    # Return list with gradients unsupervised
    return(list('grad.weights' = grad.weights  ))
  }
}

## Initialize RBM function
RBM <- function (x, y, n.iter, n.hidden, learning.rate = 0.1, 
          plot = FALSE, size.minibatch = 10 ) {
  # Trains a Restricted Boltzmann Machine.
  #
  # Args:
  #   train: Training dataset of shape features * samples.
  #   y: Labels of the data (vector)
  #   n.hidden: Desired size of the hidden layer.
  #   learning.rate: Desired learning rate, default is 0.1.
  #   n.iter: Number of iterations for training the system. More iterations 
  #   generally results in a better model but could take very long depending on the size of 
  #   the data.
  #   plot: Plot of the progress of learning of the hidden layers during learning. 
  #   mom: Is the momentum or velocity for training the system. See 
  #   "A Practical Guide to Training Restricted Boltzmann Machines (Hinton, 2010)"
  # 
  # Returns:
  #   A list with: A weights matrix of the learned system; the weights for the labels and label bias;
  #   the bias of the invisible and visible layers.
  # 
  # Check whether n.iter is devicable by ten and if so initialize plot.epoch:
  if (plot == TRUE) {
    if ((n.iter %% 10) == 0) {
      plot.epoch <- n.iter/10
    } else{
      print ('Number of iterations was not devicable by ten: plots are turned off')
      plot <- FALSE
      plot.epoch <- 0
    }
  } 
  if (plot == FALSE) {
    plot.epoch <- FALSE
  }
  # Initialize the weights, n.features * n.hidden with values from gaussian distribution
  weights <- matrix(rnorm(nrow(train)*n.hidden, 0, .01), nrow = nrow(train), ncol = n.hidden)
  # Add bias to weights
  weights <- cbind(0, weights)
  weights <- rbind(0, weights)
  # Make sparse matrix
  #weights <- Matrix(weights, spars = TRUE)
  # Add 1 for the bias to x
  x <- rbind(1, x)
  # Make sparse matrix
  #x <- Matrix(x, sparse = TRUE)
  
  # Initialize the labels, weights and bias for the labels if supervised = TRUE
  if (!missing(y)) {
    # Make binarized vectors of the labels
    y <- LabelBinarizer(y)
    # Initialize number of labels:
    n.labels <- ncol(y)
    # Add one term for the bias
    y <- cbind(1, y)
    # Make y sparse
    #y <- sparseMatrix(y, sparse = TRUE)
    # Create the y weights matrix
    y.weights <- matrix(rnorm(n.labels * n.hidden, 0, 01), nrow = n.labels, ncol = n.hidden)
    # add bias to weights
    y.weights <- cbind(0, y.weights)
    y.weights <- rbind(0, y.weights)
    # Make sparse matrix
    #y.weights <- sparseMatrix(y.weights, sparse = TRUE)
  }
  # PLot the original data:
  if(plot == TRUE){
    par(mfrow = c(3,10), mar = c(3,1,1,1))
    for(i in 1:n.hidden) {
      image(matrix(weights[-1, i], nrow = sqrt(nrow(x)-1)), col=grey.colors(255))
      title(main = paste0('Hidden node ', i), font.main = 4)
    }
  }
  # Initialize counter for the plotting:
  plot.counter <- 0
  #TODO: Create seperate function for contrastive divergence
  # Start contrastive divergence, k = 1:
  for (i in 1:n.iter){
    # Sample minibatch from x
    samp <- sample(1:ncol(x), size.minibatch, replace = TRUE)
    # Update plot counter
    plot.counter <- plot.counter + 1
    # At iteration set visible layer to random sample of train:
    V0 <- x[, samp, drop = FALSE]
    if (missing(y)) {
      grads <- CD(V0, weights)
    } else {
      grads <- CD(V0, weights, y[samp,,drop = FALSE], y.weights)
    }
    # Update bias and weights:
    weights <- weights + (learning.rate * (grads$grad.weights/ size.minibatch)) 
    # Update y bias and weights if supervised = T
    if (!missing(y)) {
      y.weights <- y.weights + (learning.rate * (grads$grad.y.weights/ size.minibatch))
    }
    # Plot learning of hidden nodes at every plot.epoch:
    if(plot.counter == plot.epoch & plot == TRUE) {
      par(mfrow = c(3,10), mar = c(3,1,1,1))
      for(i in 1:n.hidden) {
        image(matrix(weights[-1, i], nrow = sqrt(nrow(x)-1)), col=grey.colors(255))
        title(main = paste0('Hidden node ', i), font.main = 4)
      }
      # Reset the plot counter:
      plot.counter <- 0
    }
  }
  # Return list with all the trained variables:
  if (!missing(y)) {
    return(list('trained.weights' = weights,'trained.y.weights' = y.weights))
  } else {
    return(list('trained.weights' = weights))
  }
}

# Wrapper function to create a deep belief net:
DeepBelief <- function(train, y, n.iter, n.hidden.first,n.hidden.second,n.hidden.third, learning.rate = 0.1) {
  # Trains a 3 layer Deep Restricted Boltzmann Machine
  #
  # Always 3 layers, last layer is for classification.
  n.layers <- 3
  DBN <- list()
  for (i in 1: n.layers) {
    name <- paste('RBM', i, sep='')
    if (i == 1) {
      # Calculating first layer:
      DBN[[name]] <- RBM(train = train, y = labels, n.hidden = n.hidden.first, n.iter = n.iter,
                         learning.rate = learning.rate, plot = FALSE, supervised = FALSE)
    } else if (i != n.layers) {
      # Create samples
      sample <- matrix(apply(train[, 1:n.iter], 2, VisToHid, DBN[[i-1]]$trained.inv.bias, DBN[[i-1]]$trained.weights),
                   nrow = n.hidden.first, ncol = n.iter, byrow = FALSE)
      
      # Learning all the other layers inbetween:
      DBN[[name]] <- RBM(sample, y = labels, n.hidden = n.hidden.second,  n.iter = n.iter,  
                         learning.rate = learning.rate, plot = FALSE, 
                         bias = DBN[[i-1]]$trained.inv.bias, supervised = FALSE)
    } else {
      sample2 <- matrix(apply(sample, 2, VisToHid, DBN[[i-1]]$trained.inv.bias, DBN[[i-1]]$trained.weights),
                        nrow = n.hidden.second, ncol = n.iter, byrow = FALSE)
      
      # Learning last layer supervised:
      DBN[[name]] <- RBM(sample2, y = labels, n.hidden = n.hidden.third, n.iter = n.iter, 
                         learning.rate = learning.rate, plot = FALSE, 
                         bias = DBN[[i-1]]$trained.inv.bias, supervised = TRUE)
      
    }
  }
  return(DBN)
}

# Standalone function to create a deep belief network
DBN <- function(x, y, n.iter = 100, layers = c(100,100,30), learning.rate = 0.1) {
  # Initialize list for the model
  weights <- vector("list", length(layers))
  
  # Binarize y if provided
  if (!missing(y)){
    y <- LabelBinarizer(y)
  }
  # Initialize the weight matrices for each layer
  for (i in 1:(length(layers))) {
    if (i == 1) { 
      weights[[i]]$weights <- matrix(rnorm(nrow(train)* layers[i], 0, .01),
                             nrow = nrow(train), ncol = layers[i])
      # Add terms for bias 
      weights[[i]]$weights <- cbind(0, weights[[i]]$weights)
      weights[[i]]$weights <- rbind(0, weights[[i]]$weights)
      # Add label weights if y is provided
      } else if (i < length(layers)) {
        # Initialize weights
        weights[[i]]$weights <- matrix(rnorm(layers[i-1] * layers[i], 0, .01),
                                       nrow = layers[i-1], ncol = layers[i])
        # Add terms for bias 
        weights[[i]]$weights <- cbind(0, weights[[i]]$weights)
        weights[[i]]$weights <- rbind(0, weights[[i]]$weights)
    } else if (!missing(y)) {
      # Initialize weights
      weights[[i]]$weights <- matrix(rnorm(layers[i-1] * layers[i], 0, .01),
                                     nrow = layers[i-1], ncol = layers[i])
      # Add terms for bias 
      weights[[i]]$weights <- cbind(0, weights[[i]]$weights)
      weights[[i]]$weights <- rbind(0, weights[[i]]$weights)
      
      # Initialize y weights
      weights[[i]]$y.weights <- matrix(rnorm(ncol(y)* layers[i], 0, .01),
                                       nrow = ncol(y), ncol = layers[i])
      # Add terms bias for y
      weights[[i]]$y.weights <- cbind(0, weights[[i]]$y.weights)
      weights[[i]]$y.weights <- rbind(0, weights[[i]]$y.weights)
      
    } else {
      # Initialize weights
      weights[[i]]$weights <- matrix(rnorm(layers[i-1] * layers[i], 0, .01),
                                     nrow = layers[i-1], ncol = layers[i])
      # Add terms for bias 
      weights[[i]]$weights <- cbind(0, weights[[i]]$weights)
      weights[[i]]$weights <- rbind(0, weights[[i]]$weights)
    }
  } 
  # Add bias to x and y if provided
  x <- rbind(1, x)
  if (!missing(y)) {
    y <- cbind(1, y)
  }
  for (i in 1:n.iter) {
    # Set visible layer to training example
    samp <- sample(1:ncol(x), 1, replace = TRUE)
    V0 <- x[, samp, drop = FALSE]
    for (j in 1:length(layers)) {
      if (j < length(layers)) {
        # Run CD
        grads <- CD(V0, weights[[j]]$weights)
        weights[[j]]$weights <- weights[[j]]$weights + (learning.rate * grads$grad.weights) 
        # Set new visible layer to hidden layer
        V0 <- VisToHid(V0, weights[[j]]$weights)
        # Fix the bias term again
        V0[1, ] <- 1
      } else if (j == length(layers) & !missing(y)) {
        grads <- CD(V0, weights[[j]]$weights, y[samp,, drop = FALSE], weights[[j]]$y.weights )
        weights[[j]]$weights <- weights[[j]]$weights + (learning.rate * grads$grad.weights) 
        weights[[j]]$y.weights <- weights[[j]]$y.weights + (learning.rate * grads$grad.y.weights) 
      } else {
        grads <- CD(V0, weights[[j]]$weights)
        weights[[j]]$weights <- weights[[j]]$weights + (learning.rate * grads$grad.weights) 
      }
    }
  }
  # Return the learned model
  return(weights)
}


# Create the predict function:
PredictRBM <- function(test, labels, model, layers) {
  # Function to predict on test-data given trained RBM weights and bias terms for the hidden and visible layer
  # 
  # Args:
  #   test: Is the test-data on which predictions are to be made of shape n_features * samples
  #   labels: Is a vector of possible labels for the data-set
  #   trained.weights: A weight matrix for the data of shape n_features * n_hidden
  #   trained.y.weights: The trained label weights of shape n_labels * n_hidden
  #   trained.y.bias: The trained label bias of shape n_labels * 1
  #   trained.inv.bias: The trained invisible layer bias of shape n_hidden * 1
  #   trained.vis.bias: The trained visible layer bias of shape n_features * 1
  #
  # Returns:
  #   List: containing a dataframe of the predicted labels and the actual labels & accuracy
  #
  # Create dataframe to save predictions and actual labels
  result.dat <- data.frame('y' = labels, 'y.pred'= rep(0,length(labels)))
  
  # Creating binarized matrix of all the possible labels and bind to bias term
  y <- cbind(1, LabelBinarizer(unique(labels)))
  
  # Name the rows after the possible labels:
  rownames(y) <- unique(labels)
  
  # Add a column to save the energies:
  y <- cbind(y,rep(0,nrow(y)))
  # Loop over all the test data and calculate model predictions
  for (i in 1:ncol(test)) {
    # Initialize visible unit:
    V <- test[, i, drop = FALSE]
    # Add a 1 for the bias
    V <- rbind(1, V)
    if (missing(layers)) {
      for (j in 1:nrow(y)) {
        # Calculate the hidden units for each class:
        H <- VisToHid(V, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
        # Calculate energy for each class:
        y[j, 12] <- Energy(V, H, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
      }
    } else {
      for (j in 1:nrow(y)) {
        # Initialize visible unit:
        V <- test[, i, drop = FALSE]
        # Add a 1 for the bias
        V <- rbind(1, V)
        for (l in 1:layers){
          if (l < layers) {
            V <- VisToHid(V, model[[l]]$weights)
            # Fix the bias term
            #V[1,] <- 1
          } else {
            H <- VisToHid(V, model[[l]]$weights, y[j, 1:11, drop = FALSE], model[[l]]$y.weights)
            y[j, 12] <- Energy(V, H, model[[l]]$weights, y[j, 1:11, drop = FALSE], model[[l]]$y.weights)
          }
        }
      }
    }
    # Predict the label with the highest energy
    result.dat[i,2] <- as.numeric(rownames(y)[y[, 12] == min(y[, 12])])
  }
  # Calculate the accuracy of the classifier
  acc <- mean(result.dat[, 1] == result.dat[, 2])
  # Return list with predictions and accuracy
  return(list('Preds' = result.dat, 'Acc' = acc))
}



PredictRBM(test, labels = test_y, model = par)
