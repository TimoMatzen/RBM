#' MNIST digit recognizer data
#'
#' This is the famous MNIST handwritten digits dataset. It has been seperated in a train 
#' and test set and train and test labels. It was downloaded from Kaggle. The pixels have been normalized so that they lie between
#' 0 and 1 and can be used in the RBM() function. The shape of the data, trainX and testX, is of shape (samples * features). The labels, testY and trainY are a vector with
#' the labels.
#' 
#'
#' @docType data
#'
#' @usage data(MNIST)
#' 
#'
#' 
#'
#' @keywords datasets
#'
#' @source \href{https://www.kaggle.com/c/mnist-digits/data}{Kaggle}
#' 
#' @examples 
#' # Load the data
#' data(MNIST)
#' # Use MNIST train set (x) for fitting RBM model
#' modelRBM <- RBM(MNIST$trainX)
#'
"MNIST"