##############################################################
###############Restricted Boltzmann Machine###################
##############################################################

# TODO: Make possibility for stacking boltzmann machines.
# Use greedy pretraining.

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

# Deep RBM function:
DeepRBM <- function(train, y, n.iter, n.hidden, lerning.rate = 0.1) {
  # Trains a 3 layer Deep Restricted Boltzmann Machine
  #
  
}

## Initialize RBM function
RBM <- function(train,y, n.iter, n.hidden, learning.rate = 0.1, 
                plot = FALSE, supervised = FALSE){
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
  # Taking a uniform sample with size train for visible bias:
  samp.unif <- matrix(runif(dim(train)[1] * dim(train)[2]), nrow = dim(train)[1], ncol = dim(train)[2])
  # Turn on when train > uniform sample:
  train.bin <- ifelse(train > samp.unif, 1, 0)
  # Visible bias:
  vis.bias <- log(rowMeans(train.bin) / (1 - rowMeans(train.bin)) )
  # Make bias 0 when -infinity:
  vis.bias <- ifelse(vis.bias == -Inf, 0,vis.bias)
  
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
      image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
      title(main = paste0('Hidden node ', i), font.main = 4)
    }
  }
  
  # Initialize counter for the plotting:
  plot.counter <- 0
  
  # Initialize velocity at t = 0
  vel.weights <- matrix(0, nrow = nrow(train), ncol = n.hidden)
  vel.vis.bias <- matrix(0, nrow = nrow(train), ncol = 1)
  vel.inv.bias <- matrix(0, nrow = n.hidden, ncol = 1)
  
  
  # Start contrastive divergence, k = 1:
  for (i in 1:n.iter){
    # Update plot counter
    plot.counter <- plot.counter + 1
    
    # Take a sample at each iteration
    samp <- sample((1:ncol(train)),1)
    
    # At iteration set visible layer to random sample of train:
    V0 <- matrix(train[,samp], nrow= nrow(train))
    
    # At a layer with labels if supervised = TRUE
    if (supervised == TRUE) {
      Y0 <- matrix(y[samp,], nrow = n.labels)
    
    ## Contrastive Divergence (k = 1)
    # Positive phase CD:
    H0 <- 1/(1 + exp(-(inv.bias + t(t(V0) %*% weights) + t(t(Y0) %*% y.weights))) )
    } else {
      H0 <- 1/(1 + exp(-(inv.bias + t(t(V0) %*% weights))) )
    }
    # Binarize the hidden layer:
    H0 <- ifelse(H0 > runif(nrow(H0)),1,0)
    
    # Calculate positive phase
    pos.phase <- V0 %*% t(H0)
    if (supervised == TRUE) {
      pos.phase.y <- Y0 %*% t(H0)
    }
    
    # Negative  phase CD:
    V1 <- 1/(1 + exp(-(vis.bias + t(t(H0) %*% t(weights)))) )
    if (supervised == TRUE) {
      Y1 <- 1/(1 + exp(-(y.bias + t(t(H0) %*% t(y.weights)))) )
      H1 <- 1/(1 + exp(-(inv.bias + t(t(V1) %*% weights) + t(t(Y1) %*% y.weights))) )
    } else {
      H1 <- 1/(1 + exp(-(inv.bias + t(t(V1) %*% weights))) )
    }
    
    # Calculate negative phase:
    neg.phase <- V1 %*% t(H1)
    if (supervised == TRUE) {
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
    
    
    # TODO: Make function work with momentum
    #
    # Trying to use velocity instead of the gradient en learning_rate:
    #vel_weights <- (mom * vel_weights) + (grad_weights)
    #vel_vis_bias <- (mom * vel_vis_bias) + (grad_vis_bias)
    #vel_inv_bias <- (mom * vel_inv_bias) + (grad_inv_bias)
    
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
    if(plot.counter == plot.epoch & plot == TRUE){
      par(mfrow = c(3,10), mar = c(3,1,1,1))
      for(i in 1:n.hidden){
        image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
        title(main = paste0('Hidden node ', i), font.main = 4)
      }
      # Reset the plot counter:
      plot.counter <- 0
      
    }
  }
  if (supervised == TRUE) {
    return(list('trained.weights' = weights,'trained.y.weights' = y.weights, 'trained.y.bias' = y.bias, 
              'trained.inv.bias' = inv.bias, 'trained.vis.bias' = vis.bias))
  } else {
    return(list('trained.weights' = weights,
                'trained.inv.bias' = inv.bias, 'trained.vis.bias' = vis.bias))
    }
}



# Test the function:
par <- RBM(train = train, y = labels, n.hidden = 100, n.iter = 1000, learning.rate = .1, plot = TRUE,
           supervised = FALSE)


# Create the predict function:
PredictRBM <- function(test, labels, trained.weights, trained.y.weights,
                       trained.y.bias, trained.inv.bias, trained.vis.bias) {
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
    for (j in 1:nrow(y)) {
      # Calculate the hidden units for each class:
      H <- 1/(1 + exp(-(trained.inv.bias + t(t(V) %*% trained.weights) + t(y[j,1:10] %*% trained.y.weights))) )
      # Calculate energy for each class:
      y[j,11] <- ( (-t(H) %*% t(trained.weights) %*% V) -(t(trained.vis.bias) %*% V) -(t(trained.inv.bias) %*% H) -
        (t(trained.y.bias) %*% y[j,1:10]) - (t(H) %*% t(trained.y.weights) %*% y[j,1:10]) )
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


