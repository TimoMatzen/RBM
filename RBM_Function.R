##############################################################
###############Restricted Boltzmann Machine###################
##############################################################

# Load in the train-data of the MNIST data-set
train <- read.csv('train.csv', nrows = 200)
labels <- train$label

# Converting labels to binary feature vectors
y <- matrix(0,length(labels), 10)

# Loop over all labels and binarize:
for(i in 1:length(labels)){
  y[i, labels[i]+1] <- 1
}

# Put the data in a matrix of shap features * samples
train <- matrix(unlist(train[,-1]), nrow =784,ncol = 200, byrow = T)/255

## Initialize RBM function
RBM <- function(train, n_hidden, learning_rate = 0.1, n_iter, 
                plot = FALSE, mom = 0.9){
  
  # PLot the original data:
  if(plot == TRUE){
    #image(matrix(train[,10], nrow = 28),col=grey.colors(255))
    #title(main = "Original data", font.main = 4)
  }
  
  # Intialize the hidden layers (only one layer):
  inv_layer <- matrix(0, nrow = n_hidden, ncol = 1)
  # Initialize the bias terms:
  inv_bias <- matrix(0, nrow = n_hidden, ncol = 1)
  
  # Taking a uniform sample with size train:
  samp_unif <- matrix(runif(dim(train)[1] * dim(train)[2]), nrow = dim(train)[1], ncol = dim(train)[2])
  # Turn on when train > uniform sample:
  train_bin <- ifelse(train > samp_unif, 1, 0)
  # Visible bias:
  vis_bias <- log(rowMeans(train_bin) / (1 - rowMeans(train_bin)) )
  # Make bias 0 when -infinity:
  vis_bias <- ifelse(vis_bias == -Inf, 0,vis_bias)
  # Initialize the weights, n_features * n_hidden:
  weights <- matrix(rnorm(nrow(train)*n_hidden, 0, .01), nrow = nrow(train), ncol = n_hidden)
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
    
    # At iteration set visible layer to random sample of train:
    V0 <- matrix(train[,sample((1:ncol(train)),1)], nrow= nrow(train))
    
    ## Contrastive Divergence (k = 1)
    # Negative phase CD:
    H0 <- 1/(1 + exp(-(inv_bias + t(t(V0) %*% weights))) )
    # Binarize the hidden layer:
    H0 <- ifelse(H0 > runif(nrow(H0)),1,0)
    # Calculate negative phase
    neg_phase <- V0 %*% t(H0)
    
    # Positive phase CD:
    V1 <- 1/(1 + exp(-(vis_bias + t(t(H0) %*% t(weights)))) )
    H1 <- 1/(1 + exp(-(inv_bias + t(t(V1) %*% weights))) )
    pos_phase <- V1 %*% t(H1)
    
    # Calculate gradients:
    grad_weights <- neg_phase - pos_phase
    grad_vis_bias <- V0 - V1
    grad_inv_bias <- H0 - H1
    
    ## Trying to use velocity instead of the gradient en learning_rate:
    vel_weights <- (mom * vel_weights) + (grad_weights)
    vel_vis_bias <- (mom * vel_vis_bias) + (grad_vis_bias)
    vel_inv_bias <- (mom * vel_inv_bias) + (grad_inv_bias)
  
    # Update bias and weights:
    weights <- weights + (learning_rate * vel_weights) # Use velocity instead (learning_rate * grad_weights)
    vis_bias <- vis_bias + (learning_rate * vel_vis_bias) #(learning_rate * grad_vis_bias)
    inv_bias <- inv_bias + (learning_rate * vel_inv_bias) #(learning_rate * grad_inv_bias)
    
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
  
  return(weights)
}

# Test the function:
weights <- RBM(train = train, n_hidden = 30, n_iter = 10000, learning_rate = .005, plot = TRUE,
               mom = 0.5)


weights0 <-  matrix(rnorm(nrow(train)*n_hidden, 0, .01), nrow = nrow(train), ncol = n_hidden)

par(mfrow = c(3,10), mar = c(3,1,1,1))
for(i in 1:n_hidden){
  image(matrix(weights[,i], nrow = 28),col=grey.colors(255))
  title(main = paste0('Hidden node ', i), font.main = 4)
}
