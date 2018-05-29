# TODO: Add regularisation.
# TODO: Add momentum.
# TODO: Make function faster (RCPP)

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
#'@param n.iter.pre Past on to the StackRBM function, defines the how many epochs are used to pretrain each
#'RBM layer.
#'@param learning.rate.pre The pretraining learning rate, passed on to the StackRBM function. 
#'
#'@return Returns the finetuned DBN model that can be used in the PredictDBN function.
#'
#'
# Initialize the DBN function
DBN <- function(x, y, n.iter = 300, nodes = c(30,40,30),
                      learning.rate = 0.5, size.minibatch = 10, n.iter.pre = 30, learning.rate.pre = .1) {

 
  # Initialize weights with the pretrain algorithm
  print(paste0('Starting greedy pretraining with ', n.iter.pre, ' epochs for each RBM layer....'))
  weights <- StackRBM(x,  n.iter= n.iter.pre, layers = nodes, 
                            learning.rate = learning.rate.pre, size.minibatch = size.minibatch )
  
  # Remove bias weights in opposite directions
  weights[[1]]$trained.weights <- t(weights[[1]]$trained.weights)[-1,]
  weights[[2]]$trained.weights <- t(weights[[2]]$trained.weights)[-1,]
  weights[[3]]$trained.weights <- t(weights[[3]]$trained.weights)[-1,]
  
  # Get all the indexes for the unique labels
  labels <- unique(y)
  idx <- vector('list', length = length(labels))
  
  # Initialize the y weights
  y.weights <- matrix(rnorm(length(labels) * nodes[3], 0, 01), 
                                    nrow = length(labels) , ncol = nodes[3])
  # Add term for the bias
  y.weights <- cbind(rnorm(length(labels)), y.weights)
  # Save indexes
  for (i in 1:length(labels)) {
    idx[[i]]<- which(y == labels[i])
  }
  y <- LabelBinarizer(y)
  
  # Attach bias to data
  x <- cbind(1, x)
  # Start gradient descent
  print(paste0('Starting gradient descent with ', n.iter, 'epochs.....'))
  for (j in 1:n.iter) {
    # Pick balanced labels
    #samp <- sample(1:nrow(x),size.minibatch)
    
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
    
    # First perform one feed-forward pass
    # Sample vis from the data
    vis <- x[ samp, ,drop = FALSE]
    # Go to first hidden layer
    hid <- logistic(weights[[1]]$trained.weights %*% t(vis))
    # Fix bias
    hid <- rbind(1, hid)
    # Go to second hidden layer, first hidden layer as input
    pen <- logistic(weights[[2]]$trained.weights %*% hid)
  
    # Fix the bias
    pen <- rbind(1, pen)
    
    # Go to third hidden layer, second hidden layer as input
    top <- logistic(weights[[3]]$trained.weights %*% pen)
    # Fix the bias
    top <- rbind(1, top)

    # Now the last hidden to label output
    lab <- logistic(y.weights %*% top)
    
    # Calculate the cost
    J <- 1/size.minibatch* (sum(-t(y[samp,,drop = FALSE]) * 
                                  log(lab) - ((1 - t(y[samp,,drop = FALSE])) 
                                  * log(1 - lab))))
   
    J <- sum(J)
    
    # Print cost
    print(paste0('Cost at epoch ', j, ' = ', J))
    
    # Compare label to actual label and backpropogate
    er.grad.lab <- logistic(lab) * (1 - logistic(lab)) * -(t(y[samp,,drop = FALSE]) -  lab)
    
    # Backpropogate the error
    top.back <- t(y.weights[,-1]) %*% er.grad.lab
    
    # Calculate gradient top layer
    er.grad.top <- logistic(top[-1,]) * (1 - logistic(top[-1,])) * top.back
    
    # Back to pentative layer
    pen.back <- t(weights[[3]]$trained.weights[,-1]) %*% er.grad.top
    # calculate gradient pentative layer
    er.grad.pen <- logistic(pen[-1, ]) * (1 - logistic(pen[-1,])) * pen.back
   
    # Back to hidden layer 
    hid.back <- t(weights[[2]]$trained.weights[,-1]) %*% er.grad.pen
    # Calculate gradient hidden layer
    er.grad.hid <- logistic(hid[-1,]) * (1 - logistic(hid[-1,])) * hid.back
  
    # Adjust weights
    # hidden layer
    grad.hid <-  (1/size.minibatch * (er.grad.hid %*% vis)) 
    
    # Add regularisation
    #grad.hid[, -1] <- grad.hid[, -1] + (lambda/size.minibatch * weights[[1]]$trained.weights[, -1])
                                   
    # adjust weights hidden
    weights[[1]]$trained.weights <- weights[[1]]$trained.weights - (learning.rate  * grad.hid)  
    
    # Pentative layer
    grad.pen <- (1/size.minibatch * ( er.grad.pen %*% t(hid))) 
    #grad.pen[, -1] <- grad.pen[, -1] + (lambda/size.minibatch * weights[[2]]$trained.weights[, -1])    
    
    # Adjust weights pentative layer
    weights[[2]]$trained.weights <- weights[[2]]$trained.weights - (learning.rate * grad.pen)
    
    # Top layer
    grad.top <- (1/size.minibatch * (er.grad.top %*% t(pen))) 
    # Add regularisation
    #grad.top[, -1] <- grad.top[, -1] + (lambda/size.minibatch * weights[[3]]$trained.weights[, -1])
    # Adjust weights top layer
    weights[[3]]$trained.weights <- weights[[3]]$trained.weights - (learning.rate * grad.top)
    
    # Label weights
    grad.lab <- ( 1/size.minibatch * (er.grad.lab %*% t(top)))
    # add regularisation
    #grad.lab[, -1] <- grad.lab[, -1] + (lambda/size.minibatch * y.weights[, -1])
    y.weights <- y.weights - (learning.rate * grad.lab)
    
  }
  # Add the label weights to the weights
  weights[[4]] <- y.weights
  return(weights)
}



