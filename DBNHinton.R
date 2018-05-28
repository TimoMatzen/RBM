DBN <- function(x, y, n.iter.pre = 30, n.iter = 300, nodes = c(30,40,30),
                      learning.rate = 0.5, size.minibatch = 1, learning.rate.pre = .1) {

 
  # Initialize weights with the pretrain algorithm
  weights <- PretrainGreedy(x,  n.iter= n.iter.pre, layers = nodes, 
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
  
  
  # Start the wake sleep algorithm
  for (j in 1:n.iter) {
    # Take a sample of each class balanced
    samp <- sample(1:nrow(x), size.minibatch)
    
    # First perform one feed firward pass
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
    J <- 1/size.minibatch* (sum(-t(y[samp,,drop = FALSE]) * log(lab) - ((1 - t(y[samp,,drop = FALSE])) * log(1 - lab))))
    J <- sum(J)
    # Stopping criterion
    if (J < 1) break
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
    grad.hid <- learning.rate  * (  er.grad.hid %*% vis)
    # adjust weights hidden
    weights[[1]]$trained.weights <- weights[[1]]$trained.weights - grad.hid
    
    # Pentative layer
    grad.pen <- learning.rate * ( er.grad.pen %*% t(hid))
    # Adjust weights pentative layer
    weights[[2]]$trained.weights <- weights[[2]]$trained.weights - grad.pen
    
    # Top layer
    grad.top <- learning.rate * (er.grad.top %*% t(pen))
    # Adjust weights top layer
    weights[[3]]$trained.weights <- weights[[3]]$trained.weights - grad.top
    
    # Label weights
    grad.lab <- learning.rate * (er.grad.lab %*% t(top))
    y.weights <- y.weights - grad.lab
    
    
    
  }
  # Add the label weights to the weights
  weights[[4]] <- y.weights
  return(weights)
}

# Take a sample
samp <- sample(1 : nrow(test),1)
# Sample label 
y <- test_y[samp]
# Sample visible
vis <- cbind(1, test[samp,,drop = FALSE])

H <- logistic(mod[[1]]$trained.weights %*% t(vis))
H <- rbind(1, H)


H <- logistic(mod[[2]]$trained.weights %*% H)
H <- rbind(1, H)

H <- logistic(mod[[3]]$trained.weights %*% H)
H <- rbind(1, H)

lab <- logistic(mod[[4]] %*% H)
lab
y
