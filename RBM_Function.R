##############################################################
###############Restricted Boltzmann Machine###################
##############################################################

# TODO: Make possibility for stacking boltzmann machines.

## Initialize train-data MNIST
# Load in the train-data of the MNIST data-set:
train1 <- read.csv('train.csv', nrows = 10000)
labels <- train1$label

# Converting labels to binary feature vectors:
y <- matrix(0,length(labels), 10)

# Loop over all labels and binarize:
for(i in 1:length(labels)){
  y[i, labels[i]+1] <- 1
}

# Put the data in a matrix of shape features * samples:
train <- matrix(unlist(train1[1:9000,-1]), nrow =784,ncol = 9000, byrow = T)/255
train_y <- y[1:9000,]

## Initialize test_data MNIST
test <- matrix(unlist(train1[9001:10000,-1]), nrow =784, ncol = 1000, byrow = T)/255
test_y <- labels[9001:10000]

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
VisToHid <- function(vis, inv.bias, weights, y, y.weights){
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
  V0 <- matrix(vis, nrow = length(vis), ncol = 1 )
  Y0 <- matrix(y, nrow = length(y), ncol = 1)
  if(missing(y) & missing(y.weights)) {
    # Calculate the hidden layer with the trained weights and bias
    H <- 1/(1 + exp(-(inv.bias + t(t(V0) %*% weights))) )
  } else {
    H <- 1/(1 + exp(-(inv.bias + t(t(V0) %*% weights) + t(t(Y0) %*% y.weights))) )
  }
  return(H)
}

# Function for reconstructing visible layer:
HidToVis <- function(inv, vis.bias, weights, y.weights, y.bias){
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
  if(missing(y.weights) & missing(y.bias)) {
    # Reconstruct only the visible layer when y.weights is missing
    V <- 1/(1 + exp(-(vis.bias + t(t(inv) %*% t(weights)))) )
    return(V1)
  } else {
    # Reconstruct visible and labels if y.weights
    Y <- 1/(1 + exp(-(y.bias + t(t(inv) %*% t(y.weights)))) )
    return(Y)
  }
}

## Initialize RBM function
RBM <- function (train, y, n.iter, n.hidden, learning.rate = 0.1, 
                plot = FALSE, supervised = FALSE, bias) {
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
  # PLot the original data:
  
  # Intialize the hidden layers (only one layer):
  inv.layer <- matrix(0, nrow = n.hidden, ncol = 1)
  # Initialize the bias terms:
  inv.bias <- matrix(0, nrow = n.hidden, ncol = 1)
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
  # Taking a uniform sample with size train for visible bias:
  samp.unif <- matrix(runif(dim(train)[1] * dim(train)[2]), nrow = dim(train)[1], ncol = dim(train)[2])
  # Turn on when train > uniform sample:
  train.bin <- ifelse(train > samp.unif, 1, 0)
  
  if (missing(bias) ) {
    # Visible bias:
    vis.bias <- log(rowMeans(train.bin) / (1 - rowMeans(train.bin)) )
    # Make bias 0 when -infinity:
    vis.bias <- ifelse(vis.bias == -Inf, 0,vis.bias)
  } else {
    vis.bias <- bias
  }
  
  # Initialize the weights, n.features * n.hidden:
  weights <- matrix(rnorm(nrow(train)*n.hidden, 0, .01), nrow = nrow(train), ncol = n.hidden)
  
  # Initialize the labels, weights and bias for the labels if supervised = TRUE
  if (supervised == TRUE) {
    # Make binarized vectors of the labels
    y <- LabelBinarizer(y)
    # Initialize number of labels:
    n.labels <- ncol(y)
    y.weights <- matrix(rnorm(n.labels * n.hidden, 0, 01), nrow = n.labels, ncol = n.hidden)
    # Initialize label bias:
    y.bias <- log(colMeans(y) / (1- colMeans(y)))
  }
  # PLot the original data:
  if(plot == TRUE){
    par(mfrow = c(3,10), mar = c(3,1,1,1))
    for(i in 1:n.hidden) {
      image(matrix(weights[,i], nrow = sqrt(nrow(train))), col=grey.colors(255))
      title(main = paste0('Hidden node ', i), font.main = 4)
    }
  }
  # Initialize counter for the plotting:
  plot.counter <- 0
  ###################################################################
  ##TODO: Add velocity to improve learning.
  # Initialize velocity at t = 0
  # vel.weights <- matrix(0, nrow = nrow(train), ncol = n.hidden)
  # vel.vis.bias <- matrix(0, nrow = nrow(train), ncol = 1)
  # vel.inv.bias <- matrix(0, nrow = n.hidden, ncol = 1)
  ##################################################################
  #TODO: Create seperate function for contrastive divergence
  # Start contrastive divergence, k = 1:
  for (i in 1:n.iter){
    # Update plot counter
    plot.counter <- plot.counter + 1
    # At iteration set visible layer to random sample of train:
    V0 <- matrix(train[,i], nrow= nrow(train))
    # At a layer with labels if supervised = TRUE
    if (supervised == TRUE) {
      Y0 <- matrix(y[i,], nrow = n.labels)
      H0 <- VisToHid(V0, inv.bias, weights, y.weights)
    } else {
      H0 <- VisToHid(V0, inv.bias, weights)
    }
    # Binarize the hidden layer:
    H0 <- ifelse(H0 > runif(nrow(H0)),1,0)
    # Calculate positive phase
    pos.phase <- V0 %*% t(H0)
    if (supervised == TRUE) {
      pos.phase.y <- Y0 %*% t(H0)
    }
    # Negative  phase CD:
    # Reconstruct visible layer
    V1 <- HidToVis(H0, vis.bias, weights)
    if (supervised == TRUE) {
      # Reconstruct labels
      Y1 <- HidToVis(H0, vis.bias, weights, y.weights, y.bias)
      # Reconstruct hidden layer supervised
      H1 <- VisToHid(V1, inv.bias, weights, Y1, y.weights)
    } else {
      # Reconstruct hidden layer unsupervised
      H1 <- VisToHid(V1, inv.bias, weights)
    }
    # Calculate negative phase:
    neg.phase <- V1 %*% t(H1)
    if (supervised == TRUE) {
      # Calculate negative phase y:
      neg.phase.y <- Y1 %*% t(H1)
    }
    ## Calculate the gradients
    # Calculate gradients for the weights:
    grad.weights <- pos.phase - neg.phase
    if (supervised == TRUE) {
      # Calculate gradients for y.weigths:
      grad.y.weights <- pos.phase.y - neg.phase.y
      # Calculate gradients for y.bias
      grad.y.bias <- Y0 - Y1
    }
    # Calculate gradients for the bias terms
    grad.vis.bias <- V0 - V1
    grad.inv.bias <- H0 - H1
    ##################################################################
    ## TODO: Make function work with momentum
    # Trying to use velocity instead of the gradient en learning_rate:
    #vel_weights <- (mom * vel_weights) + (grad_weights)
    #vel_vis_bias <- (mom * vel_vis_bias) + (grad_vis_bias)
    #vel_inv_bias <- (mom * vel_inv_bias) + (grad_inv_bias)
    ##################################################################
    # Update bias and weights:
    weights <- weights + (learning.rate * grad.weights) 
    vis.bias <- vis.bias + (learning.rate * grad.vis.bias) 
    inv.bias <- inv.bias + (learning.rate * grad.inv.bias) 
    # Update y bias and weights if supervised = T
    if (supervised == TRUE) {
      y.bias <- y.bias + (learning.rate * grad.y.bias)
      y.weights <- y.weights + (learning.rate * grad.y.weights)
    }
    # Plot learning of hidden nodes at every plot.epoch:
    if(plot.counter == plot.epoch & plot == TRUE) {
      par(mfrow = c(3,10), mar = c(3,1,1,1))
      for(i in 1:n.hidden) {
        image(matrix(weights[,i], nrow = sqrt(nrow(train))), col=grey.colors(255))
        title(main = paste0('Hidden node ', i), font.main = 4)
      }
      # Reset the plot counter:
      plot.counter <- 0
    }
  }
  # Return list with all the trained variables:
  if (supervised == TRUE) {
    return(list('trained.weights' = weights,'trained.y.weights' = y.weights, 'trained.y.bias' = y.bias, 
              'trained.inv.bias' = inv.bias, 'trained.vis.bias' = vis.bias))
  } else {
    return(list('trained.weights' = weights,
                'trained.inv.bias' = inv.bias, 'trained.vis.bias' = vis.bias))
    }
}

# Wrapper function to create a deep belief net:
DeepRBN <- function(train, y, n.iter, n.hidden.first,n.hidden.second,n.hidden.third, learning.rate = 0.1) {
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
      sample <- matrix(apply(train[, 1:n.iter], 2, VisToHidden, DBN[[i-1]]$trained.inv.bias, DBN[[i-1]]$trained.weights),
                   nrow = n.hidden.first, ncol = n.iter, byrow = FALSE)
      
      # Learning all the other layers inbetween:
      DBN[[name]] <- RBM(sample, y = labels, n.hidden = n.hidden.second,  n.iter = n.iter,  
                         learning.rate = learning.rate, plot = FALSE, 
                         bias = DBN[[i-1]]$trained.inv.bias, supervised = FALSE)
    } else {
      sample2 <- matrix(apply(sample, 2, VisToHidden, DBN[[i-1]]$trained.inv.bias, DBN[[i-1]]$trained.weights),
                        nrow = n.hidden.second, ncol = n.iter, byrow = FALSE)
      
      # Learning last layer supervised:
      DBN[[name]] <- RBM(sample2, y = labels, n.hidden = n.hidden.third, n.iter = n.iter, 
                         learning.rate = learning.rate, plot = FALSE, 
                         bias = DBN[[i-1]]$trained.inv.bias, supervised = TRUE)
      
    }
  }
  return(DBN)
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
  y <- LabelBinarizer(unique(labels))
  # Name the rows after the possible labels:
  rownames(y) <- unique(labels)
  # Add a column to save the energies:
  y <- cbind(y,rep(0,nrow(y)))
  
  for (i in 1:ncol(test)) {
    # Initialize visible unit:
    V <- matrix(test[, i], ncol = 1)
    if (deep == FALSE) {
      for (j in 1:nrow(y)) {
        # Calculate the hidden units for each class:
        H <- 1/(1 + exp(-(trained.inv.bias + t(t(V) %*% trained.weights) + t(y[j,1:10] %*% trained.y.weights))) )
        # Calculate energy for each class:
        y[j,11] <- ( (-t(H) %*% t(trained.weights) %*% V) -(t(trained.vis.bias) %*% V) -(t(trained.inv.bias) %*% H) -
          (t(trained.y.bias) %*% y[j,1:10]) - (t(H) %*% t(trained.y.weights) %*% y[j,1:10]) )
      }
    } else {
      # Calculate first hidden:
      H1 <- VisToHidden(V, model$RBM1$trained.inv.bias, model$RBM1$trained.weights)
      # Calculate second hidden:
      H2 <- VisToHidden(H1, model$RBM2$trained.inv.bias, model$RBM2$trained.weights )
      for (j in 1:nrow(y)) {
        # Calculate third hidden layer for each class:
        H3 <- 1/(1 + exp(-(model$RBM3$trained.inv.bias + t(t(H2) %*% model$RBM3$trained.weights) 
                           + t(y[j,1:10] %*% model$RBM3$trained.y.weights))) )
        # Calculate the energy for each class:
        y[j,11] <- ( (-t(H3) %*% t(model$RBM3$trained.weights) %*% H2) -(t(model$RBM3$trained.vis.bias) %*% H2) -(t(model$RBM3$trained.inv.bias) %*% H3) -
                       (t(model$RBM3$trained.y.bias) %*% y[j,1:10]) - (t(H3) %*% t(model$RBM3$trained.y.weights) %*% y[j,1:10]) )
      }
    }
    # Predict the label with the highest energy
    result.dat[i,2] <- as.numeric(rownames(y)[y[, 11] == min(y[, 11])])
  }
  # Calculate the accuracy of the classifier
  acc <- mean(result.dat[, 1] == result.dat[, 2])
  # Return list with predictions and accuracy
  return(list('Preds' = result.dat, 'Acc' = acc))
}


# Test performance:
Performance <- PredictRBM(test = test, labels = test_y, par[[1]], par[[2]], par[[3]], par[[4]], par[[5]])

PredictRBM(test[,100:200], labels = test_y[100:200], model = DBN, deep = TRUE)
