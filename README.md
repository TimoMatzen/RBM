# Restricted Boltzmann Machine 

The original purpose of this project was to create a working implementation of the Restricted Boltzmann Machine (RBM). However, after creating a working RBM function my interest moved to the classification RBM. After creating the classification RBM I got interested in stacking RBM's and Deep Belief Networks (DBN). As the project kept expanding I decided to turn my work into a package. This Readme serves as a description of my package and a small introduction to RBM's, classification RBM's, stacked RBM's and DBN's. 


## Table of contents

  1. [Getting Started](#installation)
      1. Install Package
      2. Data
  2. [Restricted Boltzmann Machine](#RBM)
      1. Small Intro
      2. Using `RBM()` 
  3. [Classification Restricted Boltzmann Machine](#ClassRBM)
      1. Small Intro
      2. Using `RBM()` for classification
  4. [Stacking Restricted Boltzmann Machines](#Stack)
      1. Small Intro
      2. Using `StackRBM()` 
  5. [Deep Belief Model](#DBN)
      1. Small Intro 
      2. Using `DBN()` 

<a name="installation"/>

## Getting Started

### Install Package


Lets start by installing my package from GitHub, this can be done with the following code snippet:

```R
install_github("TimoMatzen/RBM")
```

### Data

I have included the MNIST dataset as an example dataset in my package to get you started. The MNIST dataset will be used throughout this document. The MNIST dataset is a hand-written numbers dataset and was downloaded from [Kaggle](https://www.kaggle.com/c/mnist-digits/data). The data is in the format of a list with a train set, train labels, test set and test labels. Lets load in the data:

```R
# Load the MNIST data
MNIST <- data(MNIST)
```

Lets also check for a moment whether the data looks like it is supposed to look:

```R
# Lets plot a data example of the train set
image(matrix(MNIST$train.x[2, ], nrow = 28), col = grey(seq(0, 1, length = 256)))
```

If the data is loaded in correctly it should look something like this:

![](Zero.jpeg)

Oke now we are ready to go on, lets start with the (easiest) model: the Restricted Boltzmann Machine.

<a name="RBM"/>

## Restricted Boltzmann Machine

### Small Intro
The restricted Boltzmann Machine Iets over CD ook

### Using `RBM()`
After installing the RBM package from Github you can start using the RBM function. I will provide you with a short example of how to use the RBM function. For information about this function and the arguments that it takes you can also just type `?RBM` in your R console.

First start by training your RBM and save it as a model:

```R
# First get the train data from MNIST
train <- MNIST$train.x
# Then fit the model
modelRBM <- RBM(x = train, n.iter = 1000, n.hidden = 100, size.minibatch = 10)
```

To train a RBM you need to provide the function with train data, which should be a matrix of the shape (samples * features) other parameters have default settings. The number of iterations defines the number of training epochs, at each epoch `RBM()` will sample a new minibatch. When you have enough data it is recommended to set the number of iterations to a high value as this will improve your model; the downside is that the function will also take longer to train. The n.hidden argument defines how many hidden nodes the RBM will have and size.minibatch is the number of training samples that will be used at every epoch.

You can also turn the `plot` argument on to see what the hidden nodes are learning:

```R
# Turn plot on
RBM(x = train, plot = TRUE, n.iter = 1000, n.hidden = 30, size.minibatch = 10)
```
I made a gif of the plots so you can see what it should look like:

![](https://github.com/TimoMatzen/RBM/blob/master/learning_RBM.gif)


After training the RBM model you can check how well it reconstructs the data with the `ReconstructRBM()` function:

```R
# Get the test-data from MNIST
test <- MNIST$test.x
# Reconstruct the image with modelRBM
ReconstructRBM(test = test[6, ], model = modelRBM)
```

The function will then output the original image with the reconstructed image next to it. If the model is any good the reconstructed image should look similar or even better than the original:

![](ReconThree.jpeg)

Congratulations you trained a good generative model on the MNIST data-set! The model reconstruction looks even more like a three than the original image :)

Lets go on and see if we can train a RBM that is not only good at reconstructing data but can actually make predictions on new data with the *classification RBM*.

<a name="ClassRBM"/>

## Classification Restricted Boltzmann Machine

### Small Intro
Link naar het paper van Larochelle

### Using `RBM()` for classification
In the previous part we looked at the RBM as a generative model, now we are going to try and classify labels of the MNIST data-set by training a classification RBM. We can use the `RBM()` function again, the only difference is that we now also provide the labels as the *y* argument:

```R
# First get the train labels of MNIST
TrainY <- MNIST$train.y
# This time we add the labels as the y argument
modelClassRBM <- RBM(x = train, y = TrainY, n.iter = 1000, n.hidden = 100, size.minibatch = 10)
```
Now that we have trained our classification RBM we can use it to predict the labels on some unseen test-data with the `PredictRBM()` function:

```R
# First get the test labels of MNIST
TestY <- MNIST$test.y
# Give our ClassRBM model as input
PredictRBM(test = test, labels = TestY, model = modelClassRBM)
```
Which should output a confusion matrix and the accuracy score on the test set:

![](OutClassRBM.png)

Not bad for a first try with the RBM! An accuracy of 85%. We could further improve our classification by performing hyper-parameter tuning on the (1) regularisation, (2) momentum, (3) number of epochs and the size of the minibatches (for more about these terms `?RBM`). However for the minibatch size it is recommended to make this the same size as the number of classes in the data as the RBM function always takes a balanced sample, for the current example this means that the RBM function takes a random sample of each digit (0:9).


<a name="Stack"/>

## Stacking Restricted Boltzmann Machines

### Small intro

### Using `StackRBM()`
Now lets try and stack some layers of RBM's with the `StackRBM()` function, this function calls the `RBM()` function for training each layer and so the arguments are not much different, except for the added *layers* argument. With the layers argument you can define how many RBM's you want to stack and how many hidden nodes each hidden layer should have. Lets try it out:

```R
# Train a stacked RBM with three levels
modStack <- StackRBM(x = train, layers = c(100, 100, 100), n.iter = 1000, size.minibatch = 10)

```


Now that we have trained our stacked RBM with three layers (each of size 100), we can see how well it reconstructs data, again with the `ReconstructRBM()` function. Only this time we should also define the number of layers that were used to train the model:

```
# Add the layers argument when you have multiple layers of RBM's
ReconstructRBM(test = test[6, ], model = modStack, layers = 3)
```
The reconstruction looks quite reasonable:

![](StackRec.jpeg)

We can also use the StackRBM function for a supervised model, `StackRBM()` does this by makking the last RBM layer a classification RBM as we discussed earlier. Lets fit a supervised stacked RBM:

```R
# Add the labels as the y argument
modStackSup <- StackRBM(x = train,y = TrainY, layers = c(100, 100, 100), n.iter = 1000, size.minibatch = 10)
```
Lets again use the `PredictRBM()` function to see how well our stacked RBM predicts on test data, again we need to provide the layers argument:

```R
# Add layers = 3 
PredictRBM(test = test, labels = TestY, model = modStackSup, layers = 3)
```
The output:

![](StackOut.png)

Stacking RBM's improves our classification performance with 1% as compared to the normal RBM. Stacking RBM's is not a very elegant method though as each RBM layer is trained on the output of the last layer and all the other RBM weights are frozen. It is a greedy method that will not give us the most optimal results for classification. That is why it is normally only used as a greedy pretrain algorithm to get good starting parameters. In the next part we will fit a Deep Belief Network, which finetunes a stacked RBM model with backpropogation using the labels as the criterion. 


<a name="DBN"/>

## Deep Belief Model

### Small Intro

### Using `DBN()`

Now lets try to fit a Deep Belief Network with the `DBN()` function. This could will take a bit longer but should normally not take more than a couple of minutes:

```R
# Fit the DBN model
modDBN <- DBN(x = train, y = TrainY, n.iter.pre = 300, n.iter = 1000, nodes = c(500, 300, 150), size.minibatch = 10)
```
As can be seen above running the DBN model takes some more parameters and I would highly recommend first looking at the help page (`?DBN`) to see what each argument does. Not only do you need to define the arguments for fitting the DBN but you also need to define the arguments for pretraining that are passed on to `StackRBM()`. First the function trains a stack of unsupervised RBM's to get weights in the right ball park. After that `DBN()` adds a layer of weights at the end of the model and finetunes the entire system with backpropagation and batch gradient descent. Because the `DBN()` adds a layer of weights the end model is always the number of layers that were defined in the nodes argument + 1, so in this example we end up with a 4 layer DBN. 

We can now check how well our model classifies on unseen data. To do this we run the `PredictDBN()` function remember when inputting the number of layers that the DBN now has a label layer on top:

```R
# Predicting DBN, layers is length(nodes)+1(label layer).
PredictDBN(test = test, labels = TestY, model = modDBN, layers = 4)
```
The output:

![](OutDBN.png)

Alright we improved our classification performance with almost 2% compared to the stacked RBM! Hopefully this readme document helped with getting an understanding of what the package does and how it can be used, you can always just check the help files of the functions in the package if you forgot something or want to know all the possibilities. Feel free to play around with the parameters of the model and see whether you can improve the classification performance in this readme! 



### Ergens ook een linkje naar handige filmpjes voor RBM








