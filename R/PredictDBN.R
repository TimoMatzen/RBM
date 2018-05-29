#' Predict DBN
#' 
#' Function for predicting on a test set with a Deep Belief Network (Trained with DBN)
#' 
#'@param test Is the test-data (matrix) on which the user wants to make predictions.
#'@param labels Is a matrix with the corresponding labels for test-data.
#'@param model Is the trained DBN model.
#'@param layers Define the number of layers.
#'
#'@return A list with a confusion matrix and the accuracy of the predictions.
#'
#'@export
#'
PredictDBN <- function(test, labels, model, layers ) {
  if (missing(layers)) {
    stop('Please specify the number of layers of the DBN model.')
  }
  if (missing(labels)) {
    stop('The labels argument is missing: PredictDBN can only make predictions with labels')
  }
  if (nrow(test) != length(labels)) {
    stop('The test data and labels are not of equal size.')
  }
  # Add bias term to data
  V <- cbind(1, test)
  # Make the predictions with a feed-forward pass
  for (i in 1:layers) {
    if (i == 1) {
      H <- logistic(model[[i]]$trained.weights %*% t(V))
      # Fix bias
      H <- rbind(1, H)
    } else if (i < layers) {
      H <- logistic(model[[i]]$trained.weights %*% H)
      # Fix bias 
      H <- rbind(1, H)
    } else {
      H <- logistic(model[[i]]$trained.y.weights %*% H)
    }
  }
  # Calculate accuracy
  acc <- mean(apply(H, 2, which.max)-1 == labels)
  # Create confusion matrix
  conf <- table('Preds'=apply(H, 2, which.max)-1,'truth'=labels)
  # Create list with the output variables
  output <- list('ConfusionMatrix' = conf, 'Accuracy' = acc)
  # Return the output
  return(output)
}
