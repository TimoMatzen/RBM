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
train <- matrix(unlist(train1[1:40000,-1]), nrow =784, ncol = 40000, byrow = T)/255
train <- t(train)
labels_train <- labels[1:40000]

## Initialize test_data MNIST
test <- matrix(unlist(train1[40001:41000,-1]), nrow =784, ncol = 1000, byrow = T)/255
test <- t(test)
test_y <- labels[40001:41000]

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

d## Initialize RBM function
RBM <- function (x, y, n.iter = 100, n.hidden = 30, learning.rate = 0.1, 
          plot = FALSE, size.minibatch = 10, momentum = 0.5, lambda = 0.001) {
  # Trains a Restricted Boltzmann Machine.
  #
  # Args:
  #   x: Training dataset of shape features * samples.
  #   y: Labels of the data (vector)
  #   n.hidden: Desired size of the hidden layer, default is 30
  #   learning.rate: Desired learning rate, default is 0.1.
  #   n.iter: Number of iterations for training the system. More iterations 
  #   generally results in a better model but could take very long depending on the size of 
  #   the data. Default = 100.
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
      # Plot at each n.iter/10
      plot.epoch <- n.iter/10
    } else {
      # Warn user and turn plotting off
      print ('Number of iterations was not devicable by ten: plots are turned off')
      plot <- FALSE
      plot.epoch <- 0
    }
  } else {
    # Set plot.epoch to FALSE
    plot.epoch <- FALSE
  }

  # Some error messages
  if (!is.matrix(x)) {
    warning('Data was not in a matrix, converted data to matrix')
    x <- as.matrix(x)
  }
  if (any(!is.numeric(train))) {
    stop('Sorry the data has non-numeric values, the function is executed')
  }
  if (!missing(y)) {
    if (any(!is.numeric(y))) {
      stop('Sorry the albels have non-numeric values, the function is executed')
    }
    if (any(!is.finite(y))) {
      stop('Sorry this function cannot handle NAs or non-finite label values')
    }
    #if (length(y) != ncol(train)) {
      #stop('Labels and data should be equal for supervised RBM: try training an unsupervised RBM')
   # }
  }
  if (any(!is.finite(x))) {
    stop('Sorry this function cannot handle NAs or non-finite data')
  }
  if (size.minibatch > 100) {
    warning('Sorry the size of the minibatch is too long: resetting to 10')
    size.minibatch <- 10
  } 
  if (min(x) < 0 | max(x) > 1) {
    stop('Sorry the data is out of bounds, should be between 0 and 1')
  }
  if( length(dim(x)) < 2 ) {
    stop("Dimensions of the data were not right, should be of shape n.features * n.samples")
  }
  if(nrow(x) > ncol(x)) {
    warning('Less data than features, this will probably result in a bad model fit')
  }
  
  # Initialize the weights, n.features * n.hidden with values from gaussian distribution
  weights <- matrix(rnorm(ncol(x) * n.hidden, 0, .01), nrow = ncol(x), ncol = n.hidden)
  # Initialize the momentum_speed matrix
  momentum_speed_x <- matrix(0, nrow = ncol(x) + 1, ncol = n.hidden + 1)
  
  # Add bias to weights
  weights <- cbind(0, weights)
  weights <- rbind(0, weights)

  # Add 1 for the bias to x
  x <- cbind(1, x)

  # Initialize the labels, weights and bias for the labels if supervised = TRUE
  if (!missing(y)) {
    # Get all the unique labels in y
    labels <- unique(y)
    # Get the indexes of each unique label in y
    idx <- vector('list', length = length(labels))
    # Save indexes
    for (i in 1:length(labels)) {
      idx[[i]]<- which(y == labels[i])
    }
    # Make binarized vectors of the labels
    y <- LabelBinarizer(y)
    # Add one term for the bias
    y <- cbind(1, y)
   
    # Create the y weights matrix
    y.weights <- matrix(rnorm(length(labels) * n.hidden, 0, 01), nrow = length(labels), ncol = n.hidden)
    # Add momentum speed matrix
    momentum_speed_y <- matrix(0, nrow = length(labels) + 1, ncol = n.hidden + 1)
    
    # add bias to weights
    y.weights <- cbind(0, y.weights)
    y.weights <- rbind(0, y.weights)
    
  }
  # PLot the untrained weights
  if(plot == TRUE){
    # Set plotting margins
    par(mfrow = c(3,10), mar = c(3,1,1,1))
    plot.weights <- weights[-1, -1]
    if (n.hidden > 30) {
      # Warn user that only a sample of the hidden nodes will plotted
      print('n.hidden > 30, only plotting a sample of the invisible nodes')
      # Take sample
      samp.plot <- sample(1:n.hidden, 30)
      # Remove weights for plotting
      for(i in samp.plot) {
        # Plot weights
        image(matrix(plot.weights[, i], nrow = sqrt(ncol(x)-1)), col=grey.colors(255))
        title(main = paste0('Hidden node ', i), font.main = 4)
        # Initialize counter for the plotting:
        plot.counter <- 0
      }
    } else {
        for(i in 1:n.hidden) {
          # Plot weights
          image(matrix(plot.weights[, i], nrow = sqrt(ncol(x)-1)), col=grey.colors(255))
          title(main = paste0('Hidden node ', i), font.main = 4)
          # Initialize counter for the plotting:
          plot.counter <- 0
        }
    }
  }
  plot.counter <- 0
  # Start contrastive divergence, k = 1
  for (i in 1:n.iter){
    if (missing(y)) {
      # Sample minibatch from x, unsupervised
      samp <- sample(1:nrow(x), size.minibatch, replace = TRUE)
     } else {
      # Pick balanced labels
      samp <- rep(0,size.minibatch)
      p <- 1
      for (i in 1 : size.minibatch){
        samp[p]<- sample(idx[[p]], 1)
        p <- p + 1
        if (p == length(labels) +1) {
          # Reset counter
          p <- 1
        }
      }
     }
    plot.counter <- plot.counter + 1
    # At iteration set visible layer to random sample of train:
    V0 <- x[samp, ,drop = FALSE]
    if (missing(y)) {
      # Calculate gradients
      grads <- CD(V0, weights)
    } else {
      # Calculate gradients
      grads <- CD(V0, weights, y[samp,,drop = FALSE], y.weights)
    }
    # Update the momentum speed
    momentum_speed_x <- momentum * momentum_speed_x + ((grads$grad.weights - (lambda * weights))/ size.minibatch)
   
    # Update weights and bias
    weights <- weights + (learning.rate * momentum_speed_x) 
    
    if (!missing(y)) {
      # Update momentum speed
      momentum_speed_y <- momentum * momentum_speed_y + ((grads$grad.y.weights - (lambda * y.weights))/ size.minibatch)
     
      
      # Update weights and bias
      y.weights <- y.weights + (learning.rate * momentum_speed_y) 
    }
    # Plot learning of hidden nodes at every plot.epoch:
    if(plot.counter == plot.epoch & plot == TRUE) {
      # Create margins
      par(mfrow = c(3,10), mar = c(3,1,1,1))
      # Remove bias for plottingun
      plot.weights <- weights[-1, -1]
      if (n.hidden > 30) {
        for(i in samp.plot) {
          image(matrix(plot.weights[, i], nrow = sqrt(ncol(x)-1)), col=grey.colors(255))
          title(main = paste0('Hidden node ', i), font.main = 4)
        }
      } else {
          for(i in 1:n.hidden) {
            image(matrix(plot.weights[, i], nrow = sqrt(ncol(x)-1)), col=grey.colors(255))
            title(main = paste0('Hidden node ', i), font.main = 4)
          }
      }
      # Reset the plot counter:
      plot.counter <- 0
    }
  }
  # Return list with the matrices of trained weights and bias terms
  if (!missing(y)) {
    return(list('trained.weights' = weights,'trained.y.weights' = y.weights))
  } else {
    return(list('trained.weights' = weights))
  }
}

# DBN function based around the RBM function
PretrainGreedy <- function(x, y, n.iter = 100, layers = c(100,100,30), learning.rate = 0.1, 
                           size.minibatch = 10, lambda = 0.1) {
  # Function to train a deep belief network by using stacked RBM's
  # 
  # Args:
  #   x: Is the train data set of shape n.features * n.samples
  #   y: Is a vector of labels for the data-set
  #   n.iter: The number of maximum iterations for training each RBM layer
  #   layers: Vector of the hidden nodes in each layer
  #   learning.rate Learning rate for the stochastic gradient descent
  #   size.minibatch: the size of the minibatches to used for learning the system
  #
  # Returns:
  #   List containing the learned weight and biases for each layer
  #
  #
  # Initialize list for the model
  weights <- vector("list", length(layers))
  # Some checks
  if (length(layers) == 1) {
    stop('system has only one layer and is basically an RBM, use RBM function')
  }
  
  # Train all layers except last layer
  for (j in 1:(length(layers)-1)) {
    # Train first RBM
    if (j == 1){
      # Save trained weights
      weights[[j]] <- RBM(x, n.iter = n.iter, n.hidden = layers[j], size.minibatch = size.minibatch, lambda = lambda)
      # create hidden layer in one go with matrix algebra (improved running time)
     
      H.probs <- 1/(1 + exp(-( cbind(1, x) %*% weights[[j]]$trained.weights   ))) 
      #H.probs <- matrix(apply(rbind(1, x), 2, VisToHid, weights = weights[[j]]$trained.weights ), ncol = ncol(x))
      # Sample states
      H.states <- H.probs > matrix(runif(dim(H.probs)[1] * dim(H.probs)[2]), ncol = dim(H.probs)[2] )
      # Fix the bias 
      H.states[,1] <- 1
    } else { # train in between layers with las hidden layer states
      weights[[j]] <- RBM(H.states[, -1 ], n.iter = n.iter, n.hidden = layers[j], size.minibatch = size.minibatch, lambda = lambda) #Delete bias term from states
      # Use matrix algebra to calcalate next layer
      H.probs <- 1/(1 + exp(-( H.states %*% weights[[j]]$trained.weights  ))) 
      # Create new hidden layers with states of last iteration
      #H.probs <- matrix(apply(H.states, 2, VisToHid, weights = weights[[j]]$trained.weights ), ncol = ncol(x))
      # Sample all the node states
      H.states <- H.probs > matrix(runif(dim(H.probs)[1] * dim(H.probs)[2]), ncol = dim(H.probs)[2])
      # Fix the bias 
      H.states[, 1] <- 1
    } 
    if(!missing(y)) {
    # Then train the last classification layer with the hidden states of the last layer
    weights[[length(layers)]] <- RBM(H.states[, -1], y, n.iter, 
                                     n.hidden = layers[[length(layers)]], size.minibatch = size.minibatch, lambda = lambda)# Delete bias terms
    } else {
      weights[[length(layers)]] <- RBM(H.states[, -1], n.iter= n.iter, 
                                       n.hidden = layers[[length(layers)]], size.minibatch = size.minibatch, lambda = lambda)
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
  # Add one for bias to data
  test <- cbind(1, test)
  # Loop over all the test data and calculate model predictions
  for (i in 1:nrow(test)) {
    y[,12] <- 0
    # Initialize visible unit:
    V <- test[i , , drop = FALSE]
    # Make the predictions 
    if (missing(layers)) {
      for (j in 1:nrow(y)) {
        # Calculate the hidden units for each class:
        H <- VisToHid(V, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
        # Calculate energy for each class:
        y[j, 12] <- Energy(V, H, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
      }
    } else {
      if (length(model) != layers) {
        stop('The model object has more layers than defined in the predict function')
      }
      for (j in 1:nrow(y)) {
        # Initialize visible unit:
        V <- test[i,, drop = FALSE]
        for (l in 1:layers){
          if (l < layers) {
            V <- VisToHid(V, model[[l]]$trained.weights)
            # Fix the bias term
            V[, 1] <- 1
         
          } else {
            H <- VisToHid(V, model[[l]]$trained.weights, y[j, 1:11, drop = FALSE], model[[l]]$trained.y.weights)
            y[j, 12] <- Energy(V, H, model[[l]]$trained.weights, y[j, 1:11, drop = FALSE], model[[l]]$trained.y.weights)
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




PredictRBM(test,  test_y, model = mod, layers = 3)

vis <- matrix(c(1,3,2), nrow = 1, ncol = 3)
weight <- matrix(c(.1,.1,.1,.1,.3,.2,.1,.3,.3,.1,.3,.3,.1,.3,.3), nrow = 3, byrow = F)

logistic <- function(x) {
  1/(1+exp(-x))
}
