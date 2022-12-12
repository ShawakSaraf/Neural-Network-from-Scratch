# Building a Neural Network from scratch

You know the rite of passage everyone has to go through when they first step foot into the world of programming?  
The iconic, print("Hello World!").

This is that but for the world of Neural Networks. This small project come to existence when I first started doing research about neural networks and found the absolutly incredible book, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen and the amazing YouTube series, ["Neural Networks"](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3blue1brown. They did such an amazing job of explaining the magic behind these nerworks in easy to follow language. And most importantly in the book, Michael Nielsen provided the code so you don't get stuck. I used it as a refrence to do my own implementation of neural networks, built solely using NumPy library.

But what actually is a "Neural Network"?

## What is a Neural Network?
Machine learning is everywhere these days, but "machine leaning" is an umbrella term, a tree with many branches of different algorithms, and a neural network is a branch of this huge tree, designed to mimic the way the human brain processes information. It does this by simulating the behavior of neurons, which are the basic building blocks of the brain.

![NN_2 2 7](https://user-images.githubusercontent.com/74816223/206890368-1160deda-70f2-4297-8cbf-d7d83435820b.png)

A basic neural network is composed of multiple layers of interconnected nodes called "neurons", where each layer is fully connected to the next, i.e. each neuron in one layer is connected with every neuron in next layer. So, Each neuron in a layer receives input from every neuron in the previous layer, performs a simple calculation on that input, and then sends the result to the next layer. By passing the input data through many layers of neurons, a neural network is able to learn to recognize complex patterns and make predictions or decisions based on that data.

<img src="https://user-images.githubusercontent.com/74816223/206891743-1553aec1-001b-424c-a2d8-a68abbc54dd7.gif" width="360" height="360"/> <img src="https://user-images.githubusercontent.com/74816223/206890857-c01f8aa7-211a-4187-b965-bf4fbab4d22e.gif" width="360" height="202.5"/>

You can see there's an input layer with neurons, connected to the neurons in the hidden layers, those hidden layers themselves are connected to each other and the neurons of the output layer. The arrows are the connections and the thickness of the arrow shows the weight or importance of that connection. Excluding input layer, each layer also have an activation funtion, which depending on the input, tells the neurons to be more active or less active. The more active a neuron, the bigger its output to the next layer, and that helps the network make some decision rather than the other, from an output space. 

So we input the data, the hidden layers apply some transformation to the data and output the result.

But what does it mean to train a network?    
Well, we train the network to find the values of the weights of the connections so that we get the desired output.  

And how do we do that?  
In our dataset we've got two things, the data - eg. images the cats, and the labels - what color the perticular cat is. The labels are the desired output we want from a network, so when a network outputs a prediction, a funtion compares the labels to the pardiction, and tells the network how to change its weights so that the prediction gets closer to the actual labels. We call it Loss or Cost function. The closer the output of the loss function is to zero, the closer the predictions are to the labels.

We can boil down the training of a neural network into two algorithms, forward and backward propagation.

### • Forward-Propagation 
In the forward propagation step, the input data is passed through the network, layer by layer, and the predicted output is produced. This is done by taking the dot product of the input data and the weights of the connections between the input and the first hidden layer, and then applying an activation function to the result. This process is repeated for each subsequent layer, until the final output is produced.

### • Back-Propagation 
In the backward propagation step, the error between the predicted output and the true label is calculated, and this error is then used to update the weights of the connections in the network in a way that reduces the error. This is done through the calculation of gradients, which indicate the direction in which the weights should be adjusted in order to minimize the error.

Theoretically these basic neural networks are capable of approximating any funtion given enough neurons, but due to computing limitations the network I made is capable of performing basic tasks such as image classification, as shown below, and can serve as a good starting point for more advanced deep learning projects.
 
https://user-images.githubusercontent.com/74816223/206852868-7cf65399-7504-4640-910c-1b9d97597dab.mp4

I used the famous MNIST dataset, which consists of 70,000 images of handwritten digits (0-9) and their corresponding labels for training. This dataset is commonly used as a benchmark for evaluating the performance of different machine learning algorithms.

By implementing a neural network using only the Numpy library, I was able to understand the core principles and methematics of how a neural network works without the complexities and bells and whistles of mordern machine learning frameworks. Numpy is a powerful library for numerical computing in Python, but it does not provide high-level abstractions for building and training neural networks. As a result, I had a lot of fun implementing the core components of neural networks from scratch.

If you want a more methematical and deeper sense of the inner-workings of neural networks, I highly recommend you to go thought the [book](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen and also the YouTube series, ["Neural Networks"](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3blue1brown. It's Chef's kiss.
