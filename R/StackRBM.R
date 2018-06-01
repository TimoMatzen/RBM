# TODO: Make function faster (RCPP?)

#' Stacked Restricted Boltzmann Machine
#' 
#' Function to stack several Restricted Boltzmann Machines, trained greedily by training a RBM (using the RBM function) at each layer and
#' then using the output of that RBM to train the next layer RBM.
#' 
#'@param x A matrix with binary features of shape samples * features.
#'@param y A matrix with labels for the data, only when the last layer is a classification RBM. (Optional)
#'@param n.iter Number of epochs for training each RBM layer.
#'@param layers Vector with the number of hidden nodes for each RBM layer.
#'@param learning.rate The learning rate for training each RBM layer.
#'@param size.minibatch The size of the minibatches used for training. 
#'@param momentum Speeds up the gradient descent learning. 
#'@param lambda The sparsity penalty lambda to prevent the system from overfitting.
#'
#'@return A list with the trained weights of the stacked RBM that can be used for the predict RBM function when a classification RBM is at
#' the top layer of the ReconstructRBM function to reconstruct data with the model.
#' 
#'@export
#'@examples
#'# Load MNIST data
#'data(MNIST)
#'
#'# Train a unsupervised Stack of 3 RBMs
#'mod <- StackRBM(MNIST$trainX, layers = c(100,100,100))
#'
#'# Classification RBM as top layer
#'modSup <- StackRBM(MNIST$trainX, MNIST$trainY, layers = c(100,100,100))
#' 
StackRBM <- function(x, y, n.iter = 100, layers = c(100,100,30), learning.rate = 0.1, 
                     size.minibatch = 10, lambda = 0.1, momentum = 0.5) {
  
  # Some checks
  if (!is.matrix(x)) {
    print('Data was not in a matrix, converted data to a matrix')
    x <- as.matrix(x)
  }
  if (length(layers) == 1) {
    stop('system has only one layer and is basically an RBM: please use the RBM function')
  }
  if (size.minibatch > 20) {
    print("minibatch size is very large, it could take very long to fit the model depending on the system.")
  }
  if (length(layers) > 3) {
    print("training a very large system, model will take longer to converge")
  }
  if (n.iter > 10000) {
    print("Number of epochs for each RBM > 10000, could take a while to fit")
  }
  if (any(!is.numeric(x))) {
    stop('Sorry the data has non-numeric values, the function is executed')
  }
  if (any(!is.finite(x))) {
    stop('Sorry this function cannot handle NAs or non-finite data')
  }
  if (!missing(y)) {
    if (any(!is.numeric(y))) {
      stop('Sorry the labels have non-numeric values, the function is not executed')
    }
    if (any(!is.finite(y))) {
      stop('Sorry this function cannot handle NAs or non-finite label values')
    }
    if (length(y) != nrow(x)) {
      stop('Labels and data should be equal for supervised RBM: try training an unsupervised RBM')
    }
  }
  if(ncol(x) > nrow(x)) {
    print('Less data than features, this will probably result in a bad model fit')
  }
  if (size.minibatch > 20) {
    print('Large minibatch size, could take a long time to fit model')
  } 
    
  # Initialize list for the model parameters
  weights <- vector("list", length(layers))
  
  # Train all layers except last layer
  for (j in 1:(length(layers)-1)) {
    
    # Train first level RBM
    if (j == 1){
      
      # Save trained weights
      weights[[j]] <- RBM(x, n.iter = n.iter, n.hidden = layers[j], 
                          size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)
      # create hidden layer in one go with matrix algebra (improved running time)
      
      H.probs <- logistic(cbind(1, x) %*% weights[[j]]$trained.weights)
      # Sample states
      H.states <- H.probs > matrix(runif(dim(H.probs)[1] * dim(H.probs)[2]), ncol = dim(H.probs)[2])
      # Fix the bias 
      H.states[,1] <- 1
    } else { 
      # train in between layers with las hidden layer states
      weights[[j]] <- RBM(H.states[, -1], n.iter = n.iter, n.hidden = layers[j], 
                          size.minibatch = size.minibatch, lambda = lambda, momentum = momentum) #Delete bias term from states
     
       # Use matrix algebra and logistic function to calcalate next layer
      H.probs <- logistic(H.states %*% weights[[j]]$trained.weights )
      
      # Sample all the node states
      H.states <- H.probs > matrix(runif(dim(H.probs)[1] * dim(H.probs)[2]), ncol = dim(H.probs)[2])
      
      # Fix the bias 
      H.states[, 1] <- 1
    } 
    if(!missing(y)) {
      
      # Then train the last classification layer with the hidden states of the last layer
      weights[[length(layers)]] <- RBM(H.states[, -1], y, n.iter, 
                                       n.hidden = layers[[length(layers)]], 
                                       size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)# Delete bias terms
    } else {
      # Train a unsupervised last layer
      weights[[length(layers)]] <- RBM(H.states[, -1], n.iter= n.iter, 
                                       n.hidden = layers[[length(layers)]],
                                       size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)
    }
  }
  # Return the learned model
  return(weights)
}

