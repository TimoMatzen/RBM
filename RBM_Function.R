##############################################################
###############Restricted Boltzmann Machine###################
##############################################################

## Initialize train-data MNIST
# Load in the train-data of the MNIST data-set:
train1 <- read.csv('train.csv', nrows = 1000)
labels <- train1$label

# Converting labels to binary feature vectors:
y <- matrix(0,length(labels), 10)

# Loop over all labels and binarize:
for(i in 1:length(labels)){
  y[i, labels[i]+1] <- 1
}

# Put the data in a matrix of shape features * samples:
train <- matrix(unlist(train1[1:900,-1]), nrow =784,ncol = 900, byrow = T)/255
train_y <- y[1:900,]
## Initialize test_data MNIST
test <- matrix(unlist(train1[901:1000,-1]), nrow =784, ncol = 100, byrow = T)/255
test_y <- y[901:1000, ]

## Initialize RBM function
RBM <- function(train,y, n_hidden, learning_rate = 0.1, n_iter, 
                plot = FALSE, mom = 0.9){
  
  # PLot the original data:
  if(plot == TRUE){
    #image(matrix(train[,10], nrow = 28),col=grey.colors(255))
    #title(main = "Original data", font.main = 4)
  }
  
  # Initialize number of labels:
  n_labels <- ncol(y)
  
  # Intialize the hidden layers (only one layer):
  inv_layer <- matrix(0, nrow = n_hidden, ncol = 1)
  # Initialize the bias terms:
  inv_bias <- matrix(0, nrow = n_hidden, ncol = 1)
  
  # Taking a uniform sample with size train for visible bias:
  samp_unif <- matrix(runif(dim(train)[1] * dim(train)[2]), nrow = dim(train)[1], ncol = dim(train)[2])
  # Turn on when train > uniform sample:
  train_bin <- ifelse(train > samp_unif, 1, 0)
  # Visible bias:
  vis_bias <- log(rowMeans(train_bin) / (1 - rowMeans(train_bin)) )
  # Make bias 0 when -infinity:
  vis_bias <- ifelse(vis_bias == -Inf, 0,vis_bias)
  # Initialize label bias:
  y_bias <- log(colMeans(y) / (1- colMeans(y)))
  
  # Initialize the weights, n_features * n_hidden:
  weights <- matrix(rnorm(nrow(train)*n_hidden, 0, .01), nrow = nrow(train), ncol = n_hidden)
  
  # Initialize the weights for the labels, n_labels * n_hidden
  y_weights <- matrix(rnorm(n_labels * n_hidden, 0, 01), nrow = n_labels, ncol = n_hidden)
  
  # PLot the original data:
  if(plot == TRUE){
    #image(matrix(train[,10], nrow = 28),col=grey.colors(255))
    #title(main = "Original data", font.main = 4)
    par(mfrow = c(3,10), mar = c(3,1,1,1))
    for(i in 1:n_hidden){
      image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
      title(main = paste0('Hidden node ', i), font.main = 4)
    }
  }
  
  # Initialize counter for the plotting:
  d <- 0
  # Initialize velocity at t = 0
  vel_weights <- matrix(0, nrow = nrow(train), ncol = n_hidden)
  vel_vis_bias <- matrix(0, nrow = nrow(train), ncol = 1)
  vel_inv_bias <- matrix(0, nrow = n_hidden, ncol = 1)
  # Start contrastive divergence, k = 1:
  for (i in 1:n_iter){
    d <- d+1
    
    # Take a sample at each iteration
    samp <- sample((1:ncol(train)),1)
    # At iteration set visible layer to random sample of train:
    V0 <- matrix(train[,samp], nrow= nrow(train))
    Y0 <- matrix(y[samp,], nrow = n_labels)
    
    ## Contrastive Divergence (k = 1)
    # Positive phase CD:
    H0 <- 1/(1 + exp(-(inv_bias + t(t(V0) %*% weights) + t(t(Y0) %*% y_weights))) )
    # Binarize the hidden layer:
    H0 <- ifelse(H0 > runif(nrow(H0)),1,0)
    # Calculate positive phase
    pos_phase <- V0 %*% t(H0)
    pos_phase_y <- Y0 %*% t(H0)
    
    # Negative  phase CD:
    V1 <- 1/(1 + exp(-(vis_bias + t(t(H0) %*% t(weights)))) )
    Y1 <- 1/(1 + exp(-(y_bias + t(t(H0) %*% t(y_weights)))) )
    
    H1 <- 1/(1 + exp(-(inv_bias + t(t(V1) %*% weights) + t(t(Y1) %*% y_weights))) )
    # Calculate negative phase:
    neg_phase <- V1 %*% t(H1)
    neg_phase_y <- Y1 %*% t(H1)
    
    ## Calculate the gradients
    # Calculate gradients for the weights:
    grad_weights <- pos_phase - neg_phase
    grad_y_weights <- pos_phase_y - neg_phase_y
    # Calculate gradients for the bias terms
    grad_vis_bias <- V0 - V1
    grad_inv_bias <- H0 - H1
    grad_y_bias <- Y0 - Y1
    
    ### Still under development
    ## Trying to use velocity instead of the gradient en learning_rate:
    #vel_weights <- (mom * vel_weights) + (grad_weights)
    #vel_vis_bias <- (mom * vel_vis_bias) + (grad_vis_bias)
    #vel_inv_bias <- (mom * vel_inv_bias) + (grad_inv_bias)
    ###
    
    # Update bias and weights:
    weights <- weights + (learning_rate * grad_weights) 
    y_weights <- y_weights + (learning_rate * grad_y_weights)
    vis_bias <- vis_bias + (learning_rate * grad_vis_bias) 
    inv_bias <- inv_bias + (learning_rate * grad_inv_bias) 
    y_bias <- y_bias + (learning_rate * grad_y_bias)
    
    # Adding plots of learning if plot = TRUE:
    #if(i == 1 & plot == TRUE){
     # image(matrix(V1, nrow = 28),col=grey.colors(255))
      #title(main = paste0('Data at iteration ', i), font.main = 4)
    #}
    if(d == 1000 & plot == TRUE){
      # Plot figure at each 100 epoc
      #image(matrix(V1, nrow = 28),col=grey.colors(255))
      #title(main = paste0('Data at iteration ', i), font.main = 4)
      par(mfrow = c(3,10), mar = c(3,1,1,1))
      for(i in 1:n_hidden){
        image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
        title(main = paste0('Hidden node ', i), font.main = 4)
      }
      d <- 0
      
    }
    
  }
  
  return(list(weights, y_weights, y_bias, inv_bias, vis_bias))
}

# Test the function:
par <- RBM(train = train,y = y, n_hidden = 500, n_iter = 1000, learning_rate = .01, plot = FALSE
                 )


# Create the predict function:
predict_RBM <- function(test, trained_weights, trained_y_weights,
                         trained_y_bias, trained_inv_bias, trained_vis_bias){
  # Names for the matrix with predictions:
  names <- seq(0,9,1)
  y <- matrix(0,length(names), 11)
  for(i in 1:length(names)){
    y[i, names[i]+1] <- 1
  }
  rownames(y) <- names
  # Initialize visible unit:
  V <- matrix(test,nrow = length(test), ncol = 1)
  
  for(i in 1:nrow(y)){
    # Calculate the hidden units for each class:
    H <- 1/(1 + exp(-(trained_inv_bias + t(t(V) %*% trained_weights) + t(y[i,1:10] %*% trained_y_weights))) )
    # Add energy for each class:
    y[i,11] <- ( (-t(H) %*% t(trained_weights) %*% V) -(t(trained_vis_bias) %*% V) -(t(trained_inv_bias) %*% H) -
      (t(trained_y_bias) %*% y[i,1:10]) - (t(H) %*% t(trained_y_weights) %*% y[i,1:10]) )

  }
  
  # Predict the label with the highest energy
  prediction <- rownames(y)[y[,11] == min(y[,11])]
  print(prediction)
  
}

# Test the predict function:
predict_RBM(test = test[,2], par[[1]], par[[2]], par[[3]], par[[4]], par[[5]])



weights0 <-  matrix(rnorm(nrow(train)*n_hidden, 0, .01), nrow = nrow(train), ncol = n_hidden)

par(mfrow = c(3,10), mar = c(3,1,1,1))
for(i in 1:n_hidden){
  image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
  title(main = paste0('Hidden node ', i), font.main = 4)
}
