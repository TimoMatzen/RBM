#' Predict RBM
#' 
#' Function for predicting on a test set with either a classification RBM (trained with the RBM function) or a 
#' stacked RBM with a classification RBM on top (trained with StackRBM).
#' 
#'@param test Is the test-data (matrix) on which the user wants to make predictions.
#'@param labels Is a matrix with the corresponding labels for test-data.
#'@param model Is the trained RBM or StackRBM model.
#'@param layers Only needed with StackRBM to define the number of layers.
#'
#'@return A list with a confusion matrix and the accuracy of the predictions.
#'
#'@export
#'
# 
PredictRBM <- function(test, labels, model, layers) {

  # Create dataframe to save predictions and actual labels
  result.dat <- data.frame('y' = labels, 'y.pred'= rep(0,length(labels)))
  
  # Creating binarized matrix of all the possible labels and bind to bias term
  y <- cbind(1, LabelBinarizer(unique(labels)))
  
  # Name the rows after the possible labels:
  rownames(y) <- unique(labels)
  # Add a column to save the energies:
  y <- cbind(y,rep(0,nrow(y)))
  # Add one for bias to data
  test <- cbind(1, test)
  # Loop over all the test data and calculate model predictions
  for (i in 1:nrow(test)) {
    y[,12] <- 0
    # Initialize visible unit:
    V <- test[i , , drop = FALSE]
    # Make the predictions 
    if (missing(layers)) {
      for (j in 1:nrow(y)) {
        # Calculate the hidden units for each class:
        H <- VisToHid(V, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
        # Calculate energy for each class:
        y[j, 12] <- Energy(V, H, model$trained.weights, y[j, 1:11, drop = FALSE], model$trained.y.weights)
      }
    } else {
      if (length(model) != layers) {
        stop('The model object has more layers than defined in the predict function')
      }
      for (j in 1:nrow(y)) {
        # Initialize visible unit:
        V <- test[i,, drop = FALSE]
        for (l in 1:layers){
          if (l < layers) {
            V <- VisToHid(V, model[[l]]$trained.weights)
            # Fix the bias term
            V[, 1] <- 1
            
          } else {
            H <- VisToHid(V, model[[l]]$trained.weights, y[j, 1:11, drop = FALSE], model[[l]]$trained.y.weights)
            y[j, 12] <- Energy(V, H, model[[l]]$trained.weights, y[j, 1:11, drop = FALSE], model[[l]]$trained.y.weights)
          }
        }
      }
    }
    # Predict the label with the highest energy
    result.dat[i,2] <- as.numeric(rownames(y)[y[, 12] == min(y[, 12])])
  }
  
  # Calculate the accuracy of the classifier
  acc <- mean(result.dat[, 1] == result.dat[, 2])
  # make the confusion matrix of the classifier
  conf <- table('pred' = result.dat[, 2], 'truth' = result.dat[, 1])
  # Make an output list
  output <- list('ConfusionMatrix' = conf, 'Accuracy' = acc)
  # Return list with predictions and accuracy
  return(output)
}
