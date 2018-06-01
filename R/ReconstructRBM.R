#' Reconstruct RBM
#' 
#' Reconstructs an input image with a trained RBM or stacked RBM.
#' 
#'@param test A test example to reconstruct, works only with one example.
#'@param model A model trained with either the RBM or StackRBM function.
#'@param layers The number of layers that was used for training the StackRBM function.
#'
#'@return A plot of the original and reconstructed image (model) side by side.
#'
#'@export
#'
#'@examples
#'# Load MNIST data
#'data(MNIST)
#'
#'# Fit unsupervised RBM 
#'mod <- RBM(MNIST$trainX, n.iter = 1000, n.hidden = 100)
#'
#'# Reconstruct a training image
#'ReconstructRBM(MNIST$testX[6, ], model = mod)
#'
#'

ReconstructRBM <- function(test, model, layers = 1) {
  if (!missing(layers)) {
    if ( layers == 1) {
      print('Layers is 1, will treat the model as a regular RBM. 
            If the model is a stacked RBM only the first layer weights will be used')
    }
  }
  if (!is.null(dim(test))) {
    stop('It is only possible te reconstruct one training image at a time')
  }
  if (any(!is.numeric(test))) {
    stop('Sorry the data has non-numeric values, the function is executed')
  }
  if (any(!is.finite(test))) {
    stop('Sorry this function cannot handle NAs or non-finite data')
  }
  
  if (length(model)  != layers) {
    stop("Number of layers is unequal to the number of weight matrices in the model")
  } 
  
  
  test <- matrix(test, nrow = 1)
  
  # Add bias term to visible layer
  V <- cbind(1, test[1,, drop = FALSE])
  
  # Reconstruct the image
  if (missing(layers)) {
    # Calculate hidden
    H <- VisToHid(V, model$trained.weights)
    # Reconstruct the visible layer
    V.rec <- HidToVis(H, model$trained.weights)
  } else {
    # First do one forward pass
    for (i in 1:layers) {
      V <- VisToHid(V, model[[i]]$trained.weights)
    }
    # Set last sampled layer to start of backward pass
    H <- V
    # Perform backward pass to get reconstruction
    for (i in layers:1) {
      H <- HidToVis(H, model[[i]]$trained.weights)
    }
    # Set last sampled layer to reconstruction
    V.rec <- H
    # Set V back to input 
    V <- cbind(1, test[1,, drop = FALSE])
  }
  
  # Set plotting margins
  par(mfrow = c(1,2))
  # Plot original
  image(matrix(V[, -1], nrow = sqrt(ncol(test))))
  title(main = 'Original Image', font.main = 4)
  # Plot reconstructed image
  image(matrix(V.rec[, -1], nrow = sqrt(ncol(test))))
  title(main = 'Reconstruction Model', font.main = 4)
}

