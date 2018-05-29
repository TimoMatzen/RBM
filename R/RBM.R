#' Restricted Boltzmann Machine
#' 
#' Trains a Restricted Boltzmann Machine on binary data, either supervised 
#' or unsupervised. This function uses contrastive diversion with k = 1 for training the system.
#'  
#'@param x A matrix with binary features of shape samples * features.
#'@param y A matrix with labels for the data, only when training a classification RBM. (Optional)
#'@param n.iter Defines the number of epochs to run contrastive diversion. 
#'@param n.hidden The number of nodes in the hidden layer.
#'@param learning.rate The learning rate, alpha, for training the system. 
#'@param plot Whether to plot the learning progress of the weights
#'@param size.minibatch The size of the minibatches used for training. 
#'@param momentum Speeds up the gradient descent learning. 
#'@param lambda The sparsity penalty lambda to prevent the system from overfitting. 
#'  
#'@return A list with the trained weights of the RBM that can be used for the predict RBM function when supervised learning was applied 
#'  or the ReconstructRBM function to reconstruct data with the model.
#'  
#'@export
#'  
## Initialize RBM function
RBM <- function (x, y, n.iter = 100, n.hidden = 30, learning.rate = 0.1, 
                 plot = FALSE, size.minibatch = 10, momentum = 0.5, lambda = 0.001) {
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
      stop('Sorry the labels have non-numeric values, the function is executed')
    }
    if (any(!is.finite(y))) {
      stop('Sorry this function cannot handle NAs or non-finite label values')
    }
    if (length(y) != nrow(train)) {
      stop('Labels and data should be equal for supervised RBM: try training an unsupervised RBM')
    }
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
  if(ncol(x) > nrow(x)) {
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

