# TODO: Add regularisation.
# TODO: Add momentum.
# TODO: Make function faster (RCPP)
# TODO: Make option for Wake Sleep algorithem Hinton et al. (1995)

#' Deep Belief Network
#' 
#' Trains a deep belief network starting with a greedy pretrained stack of RBM's (unsupervised) using the function
#' StackRBM and then DBN adds a supervised output layer. The stacked RBM is then finetuned on the supervised criterion
#' by using backpropogation.
#' 
#'@param x A matrix with binary features of shape samples * features.
#'@param y A matrix with labels for the data. (Always needs to be provided for training the DBN)
#'@param n.iter The number of epochs to run backpropogation.
#'@param nodes A vector with the number of hidden nodes at each layer
#'@param learning.rate Learning rate for supervised finetuning of Stacked RBM.
#'@param size.minibatch The size of the minibatches used for training. 
#'@param n.iter.pre Past on to the StackRBM function, defines how many epochs are used to pretrain each
#'RBM layer.
#'@param learning.rate.pre The pretraining learning rate, passed on to the StackRBM function. 
#'@param verbose Whether to print th training error at each epoch, printing will slow down the fitting.
#'
#'@return Returns the finetuned DBN model that can be used in the PredictDBN function.
#'
#'@export
#'
#'@examples
#'# Load the MNIST dat
#'data(MNIST)
#'
#'# Train the DBN model
#'modDBN <- DBN(MNIST$trainX, MNIST$trainY,n.iter = 500, nodes = c(500, 300, 150), learning.rate = 0.5,
#'size.minibatch = 10, n.iter.pre = 300, learning.rate.pre = 0.1, verbose = FALSE)
#'
#'# Turn Verbose on to check the learning progress
#'modDBN <- DBN(MNIST$trainX, MNIST$trainY,n.iter = 500, nodes = c(500, 300, 150), learning.rate = 0.5,
#'size.minibatch = 10, n.iter.pre = 300, learning.rate.pre = 0.1, verbose = TRUE)
#'
# Initialize the DBN function
DBN <- function(x, y, n.iter = 300, nodes = c(30,40,30),
                      learning.rate = 0.5, size.minibatch = 10, n.iter.pre = 30, learning.rate.pre = .1,
                verbose = FALSE) {
  if (length(nodes) > 3) {
    print("training a very large system, model will take longer to converge")
  }
  if (missing(y)) {
    stop("Please provide the labels for training a DBN or train a unsupervised stacked RBM.")
  }
  if (!is.matrix(x)) {
    print('Data was not in a matrix, converted data to a matrix')
    x <- as.matrix(x)
  }
  if (any(!is.numeric(x))) {
    stop('Sorry the data has non-numeric values, the function is terminated')
  }
  if (!missing(y)) {
    if (any(!is.numeric(y))) {
      stop('Sorry the labels have non-numeric values, the function is terminated')
    }
    if (any(!is.finite(y))) {
      stop('Sorry this function cannot handle NAs or non-finite label values')
    }
    if (length(y) != nrow(x)) {
      stop('Labels and data should be equal for supervised RBM: try training an unsupervised RBM')
    }
  }
  if (any(!is.finite(x))) {
    stop('Sorry this function cannot handle NAs or non-finite data')
  }
  if (size.minibatch > 100) {
    print('Sorry the size of the minibatch is too long: resetting to 10')
    size.minibatch <- 10
  } 
  if (size.minibatch > 20) {
    print('Large minibatch size, could take a long time to fit model')
  } 
  if (min(x) < 0 | max(x) > 1) {
    stop('Sorry the data is out of bounds, should be between 0 and 1')
  }
  if( length(dim(x)) < 2 ) {
    stop("Dimensions of the data were not right, should be of shape n.features * n.samples")
  }
  if(ncol(x) > nrow(x)) {
    print('Less data than features, this will probably result in a bad model fit')
  }
  if (n.iter.pre > 10000) {
    print("Number of epochs for each RBM > 10000, could take a while to fit")
  }
  if (n.iter > 10000) {
    print('Number of epochs for finetuning > 10000, could take a while to fit')
  }
  
  # Initialize weights with the pretrain algorithm
  print(paste0('Starting greedy pretraining with ', n.iter.pre, ' epochs for each RBM layer....'))
  weights <- StackRBM(x,  n.iter= n.iter.pre, layers = nodes, 
                            learning.rate = learning.rate.pre, size.minibatch = size.minibatch )
  
  # Remove bias from pretrained weights in opposite directions
  for (l in 1:length(nodes)) {
    weights[[l]]$trained.weights <- t(weights[[l]]$trained.weights)[-1,]
  }
  # Get all the indexes for the unique labels
  labels <- unique(y)
  idx <- vector('list', length = length(labels))
  
  # Initialize the y weights
  y.weights <- matrix(rnorm(length(labels) * nodes[length(nodes)], 0, 01), 
                                    nrow = length(labels) , ncol = nodes[length(nodes)])
  # Add term for the bias
  y.weights <- cbind(rnorm(length(labels)), y.weights)
  
  # Add y.weights to the pretrained weights
  weights[[length(nodes) + 1]] <-   list('trained.y.weights' = y.weights)
  
  # Save indexes
  for (i in 1:length(labels)) {
    idx[[i]]<- which(y == labels[i])
  }
  y <- LabelBinarizer(y)
  
  # Attach bias to data
  x <- cbind(1, x)
  # Start gradient descent
  print(paste0('Starting gradient descent with ', n.iter, ' epochs.....'))
  
  # Add output layer to nodes
  nodes <- append(nodes, length(labels))
  # Initialize all the hidden layers
  H <- vector('list', length(nodes))
  # Initialize all the error gradients
  Err <- vector('list', length(nodes))
  # Copy the weights matrix to make a gradients matrix
  Grad <- weights
  
  # Initialize all the hidden layers
  for (i in 1:(length(nodes))) {
    H[[i]] <- matrix(0, nrow = nodes[i], ncol = size.minibatch)
    # Initialize the error gradients
    Err[[i]] <- matrix(0, nrow = nodes[i], ncol = size.minibatch)
  }
  
  # Start backpropogation
  for (j in 1:n.iter) {
    # Pick balanced labels
    samp <- rep(0,size.minibatch)
    p <- 1
    for (i in 1 : size.minibatch){
      # Sample from each label
      samp[p]<- sample(idx[[p]], 1)
      # Update counter
      p <- p + 1
      if (p == length(labels) +1) {
        # Reset counter
        p <- 1
      }
    }
    # First perform one feed-forward pass
    
    # Initialize feed-forward pass by first sampling from data
    V <- x[ samp, ,drop = FALSE]
    
    # Then sample each next layer
    for (l in 1:(length(nodes))) {
      if (l == 1) {
        H[[l]] <- logistic(weights[[l]]$trained.weights %*% t(V))
        # Fix bias
        H[[l]] <- rbind(1, H[[l]])
      } else if (l < length(nodes)){
        H[[l]] <- logistic(weights[[l]]$trained.weights %*% H[[l-1]])
        # Fix bias
        H[[l]] <- rbind(1, H[[l]])
      } else {
        # Sample label layer, don't fix the bias
        H[[l]] <- logistic(weights[[l]]$trained.y.weights %*% H[[l-1]])
      }
    }
    
    # Calculate the cost
    J <- 1/size.minibatch* (sum(-t(y[samp,,drop = FALSE]) * 
                                  log(H[[length(nodes)]]) - ((1 - t(y[samp,,drop = FALSE])) 
                                  * log(1 - H[[length(nodes)]]))))
    J <- sum(J)
    
    if (verbose == TRUE) {
      # Print cost at current epoch
      print(paste0('Cost at epoch ', j, ' = ', J))
    }
    # Compare label to actual label and backpropogate
    for (l in length(nodes):1) {
      if(l == length(nodes)) {
        Err[[l]] <- logistic(H[[l]]) * 
          (1 - logistic(H[[l]])) * -(t(y[samp,,drop = FALSE]) -  H[[l]])
        # Calculate next layer
        back <- t(weights[[l]]$trained.y.weights[, -1]) %*% Err[[l]]
      } else {
        Err[[l]] <- logistic(H[[l]][-1,]) * (1 - logistic(H[[l]][-1,])) * back
        back <- t(weights[[l]]$trained.weights[,-1]) %*% Err[[l]]
      }
      
    }
    
    # Calculate the gradients
    for (l in 1:length(nodes)) {
      if (l == 1) {
        # Calculate gradient
        Grad[[l]]$trained.weights <- (1/size.minibatch * (Err[[l]] %*% V)) 
        # Adjust the weights
        weights[[l]]$trained.weights <- weights[[l]]$trained.weights - (learning.rate  * Grad[[l]]$trained.weights)  
      } else if (l < length(nodes)) {
        # Calculate gradient
        Grad[[l]]$trained.weights <- (1/size.minibatch * (Err[[l]] %*% t(H[[l-1]]))) 
        # Adjust the weights
        weights[[l]]$trained.weights <- weights[[l]]$trained.weights - (learning.rate  * Grad[[l]]$trained.weights)
      } else { # Adjust the label weights
        # Calculate gradient
        Grad[[l]]$trained.y.weights <- (1/size.minibatch * (Err[[l]] %*% t(H[[l-1]]))) 
        # Adjust the weights
        weights[[l]]$trained.y.weights <- weights[[l]]$trained.y.weights - (learning.rate  * Grad[[l]]$trained.y.weights)
      }
    }
  }
  # Return the finetuned weights
  return(weights)
}

