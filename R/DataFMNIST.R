#' Fashion MNIST dataset from Zalando
#'
#'This is the fashion MNIST dataset collected by Zalando. The datset consists of pictures of fashion items with labels. It is in the same format as the MNIST
#'digit recognizer dataset and can be used for all the functions in the RBM package. 
#'
#'The labels are as follows: \cr
#'0: T-shirt/tops \cr
#'1: Trouser \cr
#'2: Pullover \cr
#'3: Dress \cr
#'4: Coat \cr
#'5: Sandal \cr
#'6: Shirt \cr
#'7: Sneaker \cr
#'8: Bag \cr
#'9: Ankle Boot \cr
#'
#' @docType data
#'
#' @usage data(Fashion)
#' 
#'
#' 
#' @keywords datasets
#'
#' @source \href{https://www.kaggle.com/zalando-research/fashionmnist}{Kaggle}
#' @source \href{https://github.com/zalandoresearch/fashion-mnist}{Zalando}
#' 
#' @examples 
#' # Load the data
#' data(Fashion)
#' # Use MNIST train set (x) for fitting RBM model
#' modelRBM <- RBM(Fashion$trainX)
#'
"Fashion"