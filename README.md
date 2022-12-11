# Building a Neural Network from scratch

You know the rite of passage everyone has to go through when they first step foot into the world of programming?  
The iconic, print("Hello World!").

This is that but for the world of Neural Networks. This small project come to existence when I first started doing research about neural networks and found the absolutly incredible book, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. They did such an amazing job of explaining the magic behind these nerworks with code examples, I used it as a refrence to do my own implementation, built solely using NumPy library, of neural networks.

But what actually is a "Neural Network"?

## What is a Neural Network?
Machine learning is everywhere these days, but "machine leaning" is an umbrella term, a tree with many branches of different algorithms, and a neural network is a branch of this huge tree of algorithms, designed to mimic the way the human brain processes information. It does this by simulating the behavior of neurons, which are the basic building blocks of the brain.

![NN_2 2 7](https://user-images.githubusercontent.com/74816223/206890368-1160deda-70f2-4297-8cbf-d7d83435820b.png)

A basic neural network is composed of multiple layers of interconnected processing units called "neurons", where each layer is fully connected to the next. Each "neuron" receives input from other neurons, performs a simple calculation on that input, and then sends the result to other neurons in the next layer. By passing the input data through many layers of neurons, a neural network is able to learn to recognize complex patterns and make predictions or decisions based on that data.

![Neural_Network_AdobeExpress](https://user-images.githubusercontent.com/74816223/206890857-c01f8aa7-211a-4187-b965-bf4fbab4d22e.gif)
![Neural_Network2_AdobeExpress](https://user-images.githubusercontent.com/74816223/206891743-1553aec1-001b-424c-a2d8-a68abbc54dd7.gif)

These are the examples of a dense fully connected neural network. "Dense" and "fully connected" just means that each neuron in one layer is connected with every neuron in next layer. You can see there's an input layer with neurons, connected to the neurons in the hidden layers, and those hidden layers are connected to each other and the neurons of output layer. The arrows are the connections and the thickness of the arrow shows the weight or importance of that connection.

So we input the data, the hiddeb layers apply some transformation to the data and output the result. But what does it mean to train a network?  
Well, we train the network to find the values of the weights of hidden layers so that we get the desired output.  

We can boil down the training of a neural network into two algorithms, forward and backward propagation.

### • Forward-Propagation 
In the forward propagation step, the input data is passed through the network, layer by layer, and the predicted output is produced. This is done by taking the dot product of the input data and the weights of the connections between the input and the first hidden layer, and then applying an activation function to the result. This process is repeated for each subsequent layer, until the final output is produced.

### • Back-Propagation 
In the backward propagation step, the error between the predicted output and the true label is calculated, and this error is then used to update the weights of the connections in the network in a way that reduces the error. This is done through the calculation of gradients, which indicate the direction in which the weights should be adjusted in order to minimize the error.

This dense, fully connected neural network — a more technical term would be, a Multi-Layer Perceptron (MLP) — is capable of performing basic tasks such as image classification, as shown below, and can serve as a good starting point for more advanced deep learning projects.
 
https://user-images.githubusercontent.com/74816223/206852868-7cf65399-7504-4640-910c-1b9d97597dab.mp4

I used the famous MNIST dataset, which consists of 70,000 images of handwritten digits (0-9) and their corresponding labels for training. This dataset is commonly used as a benchmark for evaluating the performance of different machine learning algorithms.

By implementing an MLP using only the Numpy library, I was able to understand the core principles and methametics of how a neural network works. Numpy is a powerful library for numerical computing in Python, but it does not provide high-level abstractions for building and training neural networks. As a result, I had a lot of fun implementing the core components of an MLP from scratch.

This project turned out to be a great exercise to learn the fundamental concepts and principles of neural networks and deep learning. I was able to gain hands-on experience in building and training a neural network without the bells and whistles of state of the art frameworks like Tensorflow and Pytorch.

