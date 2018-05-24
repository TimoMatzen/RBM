DBNHinton <- function(x, y, n.iter.pre = 100, n.iter = 440, n.CD.iters = 1 , nodes = c(30,40,30), learning.rate = 0.1, size.mini.batch = 10) {
  # Initialize the weights
  # First create list for weights, size 3
  # Pretrain the system
  weights <- pretrain(x,y,n.iter= n.iter.pre, layers = nodes, learning.rate = learning.rate, size.minibatch = size.mini.batch )
  labels <- unique(y)
  idx <- vector('list', length = length(labels))
  # Save indexes
  for (i in 1:length(labels)) {
    idx[[i]]<- which(y == labels[i])
  }
  y <- LabelBinarizer(y)
  
  # Initialize all the weights from a gaussian
  
  #for (n in 1:length(nodes)) {
    # Initialize the weights, features * hidden
   # if (n == 1){
    #  weights[[n]]$weights <- matrix(rnorm(nrow(x)*nodes[n], 0, .01), nrow = nrow(x), ncol = nodes[n])
      
    #} else if (n == length(nodes)) {
     # weights[[n]]$weights <- matrix(rnorm(nodes[n-1]*nodes[n], 0, .01), nrow = nodes[n-1], ncol = nodes[n])
      #weights[[n]]$y.weights <- matrix(rnorm(ncol(y)*nodes[n], 0, .01), nrow = ncol(y), ncol = nodes[n])
      #weights[[n]]$y.weights <- cbind(0, weights[[n]]$y.weights)
      #weights[[n]]$y.weights <- rbind(0, weights[[n]]$y.weights)
    #} else {
      #weights[[n]]$weights <- matrix(rnorm(nodes[n-1]*nodes[n], 0, .01), nrow = nodes[n-1], ncol = nodes[n])
    #}
    
    ## Add term for the bias
    #weights[[n]]$weights <- cbind(0, weights[[n]]$weights)
    #weights[[n]]$weights <- rbind(0, weights[[n]]$weights)
  #}
  # Attach bias to data
  x <- rbind(1, x)
  # Add one term for the bias
  y <- cbind(1, y)
  
  for (j in 1:n.iter) {
    # Take a sample of each class balanced
    samp <- rep(0,size.minibatch)
    p <- 1
    for (i in 1 : size.minibatch){
      samp[p]<- sample(idx[[p]], 1)
      p <- p + 1
      if (p == length(labels)+1) {
        # Reset counter
        p <- 1
      }
    }
    # First perform one bottom up pass to get weights
    # Sample vis 
    vis <- x[, samp, drop = FALSE]
    # Calculate first hidden 
    wake.hid.probs <- VisToHid(vis, weights[[1]]$trained.weights)
    wake.hid.states <- wake.hid.probs > runif(length(wake.hid.probs))
    # Fix bias
    wake.hid.states[1, ] <- 1
  
    # Calculate second hidden 
    wake.pen.probs <- VisToHid(wake.hid.states, weights[[2]]$trained.weights)
    wake.pen.states <- wake.pen.probs > runif(length(wake.pen.probs))
    # Fix bias
    wake.pen.states[1, ] <- 1
    
    # Calculate last hidden
    wake.top.probs <- VisToHid(wake.pen.states, weights[[3]]$trained.weights, 
                              y[samp,,drop = FALSE], weights[[3]]$trained.y.weights)
    wake.top.states <- wake.top.probs > runif(length(wake.top.probs))
    
    wake.top.states[1,] <- 1
    # Positive phase 
    pos.lab.top.statistics <- wake.top.states %*% y[samp,,drop = FALSE]
    pos.pen.top.statistics <- wake.top.states %*% t(wake.pen.states )
    
    neg.top.states <- wake.top.states
    
    for (i in 1:n.CD.iters) {
      neg.pen.probs <- HidToVis(neg.top.states, weights[[3]]$trained.weights) 
      neg.pen.states <- neg.pen.probs > runif( nodes[2]+1)
      neg.pen.states[1, ] <- 1
      neg.lab.probs <- HidToVis(neg.top.states, weights[[3]]$trained.weights, weights[[3]]$trained.y.weights)
      neg.lab.probs[ 1,] <- 1
      neg.top.probs <-  VisToHid(neg.pen.states, weights[[3]]$trained.weights, 
                                 t(neg.lab.probs), weights[[3]]$trained.y.weights)
      neg.top.probs[1, ] <- 1
      neg.top.states <- neg.top.probs > runif(length(neg.top.probs))
      neg.top.states[1,] <- 1
    }
    
    # NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
    neg.pen.top.statistics <- neg.pen.states %*% t(neg.top.states)
    neg.lab.top.statistics <- neg.lab.probs %*% t(neg.top.states)
    
    # STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
    # TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE PROBABILITIES
    # AND SAMPLE STATES
    sleep.pen.states <- neg.pen.states
    sleep.hid.probs <- HidToVis(sleep.pen.states, weights[[2]]$trained.weights)
    sleep.hid.states <- sleep.hid.probs > runif(nodes[1]+1)
    sleep.hid.states[1,] <- 1
    sleep.vis.probs <- HidToVis(sleep.hid.states, weights[[1]]$trained.weights )
    sleep.vis.probs[1, ] <- 1
    
    # PREDICTIONS
    psleep.pen.states <- VisToHid(sleep.hid.states, weights[[2]]$trained.weights)
    psleep.pen.states[1, ] <- 1 
    psleep.hid.states <- VisToHid(sleep.vis.probs, weights[[1]]$trained.weights )
    psleep.hid.states[1, ] <- 1
    pvis.probs <- VisToHid(wake.hid.states, t(weights[[1]]$trained.weights))
    pvis.probs[1,] <- 1
    phid.probs <- VisToHid(wake.pen.states, t(weights[[2]]$trained.weights))
    phid.probs[1, ] <- 1
    
    # UPDATES TO GENERATIVE PARAMETERS
    weights[[1]]$trained.weights <- weights[[1]]$trained.weights + (learning.rate * (vis-pvis.probs) %*% t(wake.hid.states))
    weights[[2]]$trained.weights <- weights[[2]]$trained.weights + (learning.rate * (wake.hid.states-phid.probs) %*% t(wake.pen.states))
    
    # UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
    weights[[3]]$trained.y.weights <- weights[[3]]$trained.y.weights + (learning.rate *(t(pos.lab.top.statistics)-neg.lab.top.statistics))
    weights[[3]]$trained.weights <- weights[[3]]$trained.weights + (learning.rate* (t(pos.pen.top.statistics) - neg.pen.top.statistics))
    
    # UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
    weights[[2]]$trained.weights <- weights[[2]]$trained.weights + ( learning.rate * sleep.hid.states %*% t(sleep.pen.states-psleep.pen.states))
    weights[[1]]$trained.weights <- weights[[1]]$trained.weights + (learning.rate *  sleep.vis.probs %*% t(sleep.hid.states-psleep.hid.states))
  }
  return(weights)
}
