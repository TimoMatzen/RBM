##############################################################
###############Restricted Boltzmann Machine###################
##############################################################
install.packages('tensorflow')
library(tensorflow)
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)
# Load in the train-data of the MNIST data-set
train <- read.csv('train.csv', nrows = 200)
# Put the data in a matrix of shap features * samples
m <- matrix(unlist(train[,-1]), nrow =784,ncol = 200, byrow = T)/255

image(matrix(m[,10], nrow = 28),col=grey.colors(255))

# Initialize RBM function

RBM <- function(train,n_hidden, learning_rate, n_iter){
  # Intialize the hidden layers (only one layer):
  inv_layer <- matrix(0,nrow = n_hidden, ncol = 1)
  # Initialize the bias terms:
  inv_bias <- matrix(0, nrow = n_hidden, ncol = 1)
  # Taking a uniform sample with size train:
  samp_unif <- matrix(runif(dim(m)[1]*dim(m)[2]), nrow = dim(m)[1], ncol = dim(m)[2])
  # Turn on when train > uniform sample:
  train_bin <- ifelse( m > samp_unif,1,0)
  # Visible bias:
  vis_bias <- log(rowMeans(train_bin)/(1-rowMeans(train_bin)))
  # Make bias 0 when -infinity:
  vis_bias <- ifelse(vis_bias == -Inf, 0,vis_bias)
}
