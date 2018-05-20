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
Energy <- function(vis, inv, weights, y, y.weights) {
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
  E <- -(t(vis)%*% weights %*% inv) - (t(y) %*% y.weights %*% inv)
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
  H0 <- ifelse(H0 > runif(nrow(H0)),1,0)
  # Calculate positive phase
  pos.phase <- vis %*% t(H0)
  if (!missing(y)) {
    pos.phase.y <- y %*% t(H0)
  }
  # Start negative  phase
  # Reconstruct visible layer
  V1 <- HidToVis(H0, weights)
  # Set the bias unit to 1
  V1[1,] <- 1
  
  if (missing(y) & missing(y.weights) ) {
    # Reconstruct hidden layer unsupervised
    H1 <- VisToHid(V1, inv.bias, weights)
  } else {
    # Reconstruct labels if y is provided
    Y1 <- HidToVis(H0, weights, y,  y.weights )
    # Reconstruct hidden layer supervised
    H1 <- VisToHid(V1, weights, Y1, y.weights)
  }
  # Calculate negative phase:
  neg.phase <- V1 %*% t(H1)
  if (!missing(y) & !missing(y.weights)) {
    # Calculate negative phase y
    neg.phase.y <- Y1 %*% t(H1)
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
    return(list('grad.weights' = grad.weights,  ))
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
  # Add 1 for the bias to x
  x <- rbind(1, x)
  
  # Initialize the labels, weights and bias for the labels if supervised = TRUE
  if (!missing(y)) {
    # Make binarized vectors of the labels
    y <- LabelBinarizer(y)
    # Initialize number of labels:
    n.labels <- ncol(y)
    # Add one term for the bias
    y <- cbind(1, y)
    # Create the y weights matrix
    y.weights <- matrix(rnorm(n.labels * n.hidden, 0, 01), nrow = n.labels, ncol = n.hidden)
    # add bias to weights
    y.weights <- cbind(0, y.weights)
    y.weights <- rbind(0, y.weights)
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
    # Update plot counter
    plot.counter <- plot.counter + 1
    # At iteration set visible layer to random sample of train:
    V0 <- matrix( x[, i], ncol = 1)
    if (missing(y)) {
      grads <- CD(V0, weights)
    } else {
      grads <- CD(V0, weights, matrix(y[i,], ncol = 1), y.weights)
    }
    # Update bias and weights:
    weights <- weights + (learning.rate * grads$grad.weights) 
    # Update y bias and weights if supervised = T
    if (!missing(y)) {
      y.weights <- y.weights + (learning.rate * grads$grad.y.weights)
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

# Standalone function to create a deep belief net:
DeepBel <- function(train, y, n.iter, n.hidden, layers, learning.rate = 0.1) {
  # Initialize the weights matrix, tie all layers
  weights <- vector("list", layers)
  weights[[1]] <- matrix(rnorm(nrow(train)*n.hidden, 0, .01),
                         nrow = nrow(train), ncol = n.hidden)
  weights[[2]] <- weights[[3]] <-  matrix(rnorm(n.hidden*n.hidden, 0, .01),
                         nrow = n.hidden, ncol = n.hidden)
  # Initialize visible bias
  # Taking a uniform sample with size train for visible bias:
  samp.unif <- matrix(runif(dim(train)[1] * dim(train)[2]), nrow = dim(train)[1], ncol = dim(train)[2])
  # Turn on when train > uniform sample:
  train.bin <- ifelse(train > samp.unif, 1, 0)
  vis.bias <- vector("list", layers)
  vis.bias[[1]] <- log(rowMeans(train.bin) / (1 - rowMeans(train.bin)) )
  # Make bias 0 when -infinity:
  vis.bias[[1]] <- ifelse(vis.bias[[1]] == -Inf, 0,vis.bias[[1]])
  vis.bias[[2]] <- vis.bias[[3]] <-  matrix(0,nrow = n.hidden,ncol = 1)
  # Initialize invisible bias
  inv.bias <- vector("list", layers)
  inv.bias[[1]] <- inv.bias[[2]] <- inv.bias[[3]]<- matrix(0, nrow = n.hidden, ncol = 1)
  # Initialize invisible layers
  inv.layer <- vector("list", layers)
  inv.layer[[1]] <- inv.layer[[2]] <- inv.layer[[3]] <- matrix(0, nrow = n.hidden, ncol = 1)
  # Preparing the labels
  # Make binarized vectors of the labels
  y <- LabelBinarizer(y)
  # Initialize number of labels:
  n.labels <- ncol(y)
  y.weights <- matrix(rnorm(n.labels * n.hidden, 0, 01), nrow = n.labels, ncol = n.hidden)
  # Initialize label bias:
  y.bias <- log(colMeans(y) / (1- colMeans(y)))
  # Do CD for n.iter
  for (i in 1:n.iter) {
    # Loop over all the layers
    # Set visible layer to first training example
    V0 <- matrix(train[, i], nrow = nrow(train), ncol = 1 )
    for (j in 1:layers) {
      # Lower layers CD unsupervised
      if (j < layers) {
        grads <- CD(V0, weights[[j]], vis.bias[[j]], inv.bias[[j]])
        # Update bias and weights for layer j:
        weights[[j]] <- weights[[j]] + (learning.rate * grads$grad.weights) 
        vis.bias[[j]] <- vis.bias[[j]] + (learning.rate * grads$grad.vis.bias) 
        inv.bias[[j]] <- inv.bias[[j]] + (learning.rate * grads$grad.inv.bias) 
        # Set new visible layer to hidden layer
        V0 <- VisToHid(V0, inv.bias[[j]], weights[[j]])
        
      } else { # If last layer create supervised RBM
        grads <- CD(V0, weights[[j]], vis.bias[[j]], inv.bias[[j]], y[i,], y.weights, y.bias)
        # Update bias and weights for layer j:
        weights[[j]] <- weights[[j]] + (learning.rate * grads$grad.weights) 
        vis.bias[[j]] <- vis.bias[[j]] + (learning.rate * grads$grad.vis.bias) 
        inv.bias[[j]] <- inv.bias[[j]] + (learning.rate * grads$grad.inv.bias) 
        y.bias <- y.bias + (learning.rate * grads$grad.y.bias)
        y.weights <- y.weights + (learning.rate * grads$grad.y.weights)
        
      }
    }
  }
  # Return the learned model
  RBM1 <- list('trained.weights' = weights[[1]], 'trained.vis.bias' = vis.bias[[1]],
               'trained.inv.bias' = inv.bias[[1]])
  RBM2 <- list('trained.weights' = weights[[2]], 'trained.vis.bias' = vis.bias[[2]], 
               'trained.inv.bias' = inv.bias[[2]])
  RBM3 <- list('trained.weights' = weights[[3]], 'trained.vis.bias' = vis.bias[[3]], 
               'trained.inv.bias' = inv.bias[[3]], 'trained.y.bias' = y.bias, 'trained.y.weights' = y.weights)
  return(list('RBM1'=RBM1, 'RBM2' = RBM2, 'RBM3' = RBM3))
  
}

# Test the function:
par <- RBM(train = train, y = labels, n.hidden = 100, n.iter = 10000, learning.rate = .1, plot = TRUE,
           supervised = TRUE)

DBN <- DeepRBN(train = train, y = labels, n.hidden = 100, n.iter = 1000, n.layers = 3, learning.rate = .1)

# Create the predict function:
PredictRBM <- function(test, labels, model, deep = FALSE) {
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
  # Creating binarized matrix of all the possible labels:
  y <- cbind(1, LabelBinarizer(unique(labels)))
  # Name the rows after the possible labels:
  rownames(y) <- unique(labels)
  # Add a column to save the energies:
  y <- cbind(y,rep(0,nrow(y)))
  # Loop over all the test data and calculate model predictions
  for (i in 1:ncol(test)) {
    # Initialize visible unit:
    V <- matrix(test[, i], ncol = 1)
    # Add a 1 for the bias
    V <- rbind(1, V)
    if (deep == FALSE) {
      for (j in 1:nrow(y)) {
        # Calculate the hidden units for each class:
        H <- VisToHid(V, model$trained.weights, y[j, 1:11], model$trained.y.weights)
        # Calculate energy for each class:
        y[j, 12] <- Energy(V, H, model$trained.weights, matrix(y[j, 1:11], ncol = 1), model$trained.y.weights)
      }
    } else {
      # Calculate first hidden:
      H1 <- VisToHid(V, model$RBM1$trained.inv.bias, model$RBM1$trained.weights)
      # Calculate second hidden:
      H2 <- VisToHid(H1, model$RBM2$trained.inv.bias, model$RBM2$trained.weights )
      for (j in 1:nrow(y)) {
        # Calculate third hidden layer for each class:
        H3 <- VisToHid(H2, model$RBM3$trained.inv.bias, model$RBM3$trained.weights, 
                       y[j, 1:10], model$RBM3$trained.y.weights)
        # Calculate the energy for each class:
        y[j, 11] <- Energy(H2, model$RBM3$trained.vis.bias, H3, model$RBM3$trained.inv.bias, model$RBM3$trained.weights,
               y[j, 1:10], model$RBM3$trained.y.weights, model$RBM3$trained.y.bias)
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
