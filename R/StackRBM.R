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
#' the top layeror the ReconstructRBM function to reconstruct data with the model.
#' 
#'@export
#' 
#' 
StackRBM <- function(x, y, n.iter = 100, layers = c(100,100,30), learning.rate = 0.1, 
                     size.minibatch = 10, lambda = 0.1, momentum = 0.5) {
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
      weights[[j]] <- RBM(x, n.iter = n.iter, n.hidden = layers[j], 
                          size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)
      # create hidden layer in one go with matrix algebra (improved running time)
      
      H.probs <- 1/(1 + exp(-( cbind(1, x) %*% weights[[j]]$trained.weights   ))) 
      #H.probs <- matrix(apply(rbind(1, x), 2, VisToHid, weights = weights[[j]]$trained.weights ), ncol = ncol(x))
      # Sample states
      H.states <- H.probs > matrix(runif(dim(H.probs)[1] * dim(H.probs)[2]), ncol = dim(H.probs)[2])
      # Fix the bias 
      H.states[,1] <- 1
    } else { # train in between layers with las hidden layer states
      weights[[j]] <- RBM(H.states[, -1], n.iter = n.iter, n.hidden = layers[j], 
                          size.minibatch = size.minibatch, lambda = lambda, momentum = momentum) #Delete bias term from states
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
                                       n.hidden = layers[[length(layers)]], 
                                       size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)# Delete bias terms
    } else {
      weights[[length(layers)]] <- RBM(H.states[, -1], n.iter= n.iter, 
                                       n.hidden = layers[[length(layers)]],
                                       size.minibatch = size.minibatch, lambda = lambda, momentum = momentum)
    }
  }
  # Return the learned model
  return(weights)
}

