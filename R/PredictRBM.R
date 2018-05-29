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
#'@return A list with the actual labels and predictions side by side and the accuracy score on the test-data.
#'
#'@export
#'
# Create the predict function:
PredictRBM <- function(test, labels, model, layers) {
  # Function to predict on test-data given trained RBM weights and bias terms for the hidden and visible layer
  # 
  # Args:
  #   test: Is the test-data on which predictions are to be made of shape n_features * samples
  #   labels: Is a vector of possible labels for the data-set
  #   model: Is the trained RBM or StackRBM model.
  #   layers: Only needed with StackRBM to define the number of layers.
  # Returns:
  #   List: containing a dataframe of the predicted labels and the actual labels & accuracy
  #
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
  # Return list with predictions and accuracy
  return(list('Preds' = result.dat, 'Acc' = acc))
}