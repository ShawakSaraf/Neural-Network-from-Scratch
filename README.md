# Understanding Neural Networks

You know the rite of passage everyone has to go through when they first set foot into the world of programming.  
The iconic, print("Hello World!").

This is that but for the world of Neural Networks.

Let's first go through what's inside the repository.    
Prerequisites for libraries - Python 3 or above, NumPy, Json, Pickle, Gzip, and Tqdm.

• Model.py and Optimizer.py consist of the model architecture and the training algorithm.  
• Run.py is where you can train the model and do inference.

Everything else is just trained model variables, and a script to load training data from the mnist.pkl.gz file.

This neural network is competent in tasks such as image classification (as shown below), and can serve as a good starting point for more advanced deep learning projects.
 
https://user-images.githubusercontent.com/74816223/206852868-7cf65399-7504-4640-910c-1b9d97597dab.mp4

I used the famous MNIST dataset for training, which consists of 70,000 images of handwritten digits (0-9) and their corresponding labels.

By implementing a neural network using only the Numpy library, I was able to understand the core principles and mathematics of how a neural network works without the complexities of modern machine learning frameworks. Numpy is a powerful library for numerical computing in Python, but it does not provide high-level abstractions for building and training neural networks. As a result, I had a lot of fun implementing the core components of neural networks from scratch.

Ok, let's move on.

This small project came into existence when I first started doing research about neural networks and found the absolutely incredible book, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen, and the amazing YouTube series, ["Neural Networks"](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3blue1brown. They did such an amazing job of explaining the magic behind these networks in easy-to-follow language. And most importantly in the book, Michael Nielsen provided the code, so you don't get stuck. I used it as a reference to do my own implementation of neural networks.

**But what actually is a "Neural Network"?**

## What makes a Neural Network tick?
Machine learning (ML) is everywhere these days, from how google assistant answers you, to how youtube recommends a video it thinks you'd like to watch. From note-taking softwares to medical diagnosis. Instagram uses ML to choose what posts to show you first. As I said, everywhere. I used to be really intimidated by the the term "ML", I used to think this is something way ahead on my league, but when I finally exposed myself to it, I found the idea of a machine-learning really simplistic in some sense. Let's dive deeper into it. 

*"Machine Learning"* is an umbrella term, a tree with sprawling branches of different algorithms, and a neural network is one of them, a very important one I might add. Designed to mimic the way the human brain processes information. It does this by simulating the behavior of the neurons, which are the basic building blocks of the brain.

<p align="center">
<img src="https://i.imgur.com/XvGNrA1.png" width=85% height=85%/>
</p>

A neural network is a function approximator. Just like a mathematical function, it takes an input, does some calculations on it, and outputs the result. On some level, your brain does the same thing. It takes pressure waves as input, and what comes out is sound. Electromagnetic radiation goes in, an image comes out. So in a sense, the brain is a function approximator too! This is where the term *"Neural Network"* comes in, I'll just leave it at that.  

A more technical definition would be, a basic neural network is composed of multiple layers of interconnected nodes called "neurons", where each layer is fully connected to the next, i.e. each neuron in one layer is connected with every neuron in the next layer. So, Each neuron in a layer receives input from every neuron in the previous layer, performs a simple calculation on that input, and passes the result to the next layer. By flowing the input data through many layers of multiple neurons, a neural network can learn to recognize complex patterns and make predictions or decisions based on that data.

<p align="center">
<img src="https://user-images.githubusercontent.com/74816223/206891743-1553aec1-001b-424c-a2d8-a68abbc54dd7.gif" width="360" height="360"/> <img src="https://user-images.githubusercontent.com/74816223/206890857-c01f8aa7-211a-4187-b965-bf4fbab4d22e.gif" width="360" height="202.5"/>
</p>

You can see there's an input layer with neurons, connected to the neurons in the hidden layers (we call them hidden layers because their internal structure is not visible when we examine the neural network), those hidden layers themselves are connected to each other and the neurons of the output layer. The arrows are the connections and the thickness of the arrow shows the weight or importance of that connection. Excluding the input layer, each layer also has an activation function, which, depending on the input, tells the neurons to be more active or less active. The more active a neuron is, the bigger its output to the next layer, and that helps the network make some decision rather than the other, from an output space.  

- Here's the mathematical expression for calculating the activation of the neurons in a layer (I won't mind if you want to skip this part. I'd rather you not engage with math right now than stop reading.) -  
   - let's denote *Layer* by ' **$l$** ',  
   *weights* by ' **$w$** ',  
   *bias* by ' **$b$** ' (this is another variable a neuron has that helps it learn),  
   *activation function* by ' **$\sigma$** ' (greek letter sigma),  
   and the *activation* of the neuron with ' **$a$** '.
</p>

- Activation of the neurons in the layer ( **$a^l$** ) would be -

**$$a^{\ l}\ =\ \sigma\left(\sum_{ }^{ }w^{\ l}\ a^{\ l-1}+b^{\ l}\right)$$**
<p>
<img align="left" src="https://i.imgur.com/s8gfGz3.png" width=50% height=50%/>
</p>
As you can see, the activation of a neuron in layer (L) is the weighted sum of all activations in the previous layer (L-1, the output of the neuron N of L-1 is  its activation) plus the bias term, passed through an activation function which normalizes it, i.e. squeeze it down to a range between 0 and 1 or -1 and 1, depending upon what activation function we use.

This means the activation of a neuron in a layer is dependent upon the activations of all neurons in the previous layer.  
And that is the connection.

P.s. I'm using vector notation to avoid confusing indices as subscripts.

So we input the data, the hidden layers apply some transformation to that data, and output the result. 

This is how you and I learn, As a kid when you saw some color, you heard "Red", and your brain did the necessary changes to your neural circuit's activation pattern, and now you know what "Red" means. This is how you're learning right now, but I digress.  

#### But what does it mean to train a network?    
Well, we train the network to find the values of the weights of the connections and biases so that we get the desired output.  

#### And how do we do that?  
In our dataset, we've got two things, the data (eg. images of cats) and the labels (color of the cats). The labels are the desired output we want from a network, the ground truth. When we input a piece of data and label from the dataset, the network outputs a prediction, a function then compares the label to the prediction, and tells the network how to change its weights and biases so that the prediction gets closer to the actual label. We call it the Cost or Loss function. The closer the output of the loss function is to zero, the closer the predictions are to the labels.  
Let's explore the job of the Loss function.

### • Gradient Descent

<p>
<img align="right" src="https://user-images.githubusercontent.com/74816223/207776057-b5fddc2b-535b-4962-9bae-548c69cf21ed.gif" width=40% height=40%/>
</p>

Think of our loss function as a mountain range, and you're a hiker traveling through it. You want to reach the lowest point of a valley, so you follow the steepest downhill path. In the neural network world, this is called Gradient Descent.  
While training, the network tries to find the steepest downhill path on the valley of the loss function. To make this computationally more efficient, for each epoch (training session) we choose some random points on the mountain range and try to find the valley with the lowest point, i.e. take random batches out of the training data, feed it into the optimizer, which tries to find the lowest point in the loss function, which then helps network change its variables in the right direction.  

And this is what optimization means, finding the best weights and biases of the network relative to the dataset.

We can boil down the training into two algorithms, forward and backward propagation.

### • Forward-Propagation 

In the forward propagation step, the input data is passed through the network, layer by layer, and the predicted output is produced. This is done by calculating the activation of the first hidden layer, where the activation of the input layer is the data itself. This process is repeated for each subsequent layer passing the activations forward in the network until the final output is produced.  
Basically, it calculates the activations of the neurons in a layer and passes them to the next layer.

### • Back-Propagation  

Here lies the meat and potatoes of the training of a neural network. In the backward propagation step, the error between the predicted output and the true label is calculated using the loss function, and this error is then used to update the weights of the connections and biases of the neurons in the network in a way that reduces the error, i.e. reaching the lowest point of the loss function. This is done through the calculation of gradients, which indicate the direction in which the hiker must go to reach the lowest point in the valley, or in the network's case, the weights and biases should be adjusted to minimize the error.  

The calculation of gradients is done by using the multivariable calculus chain rule which I'm not gonna explain here.  

All you need to know is this, the gradient of any function relative to a variable tells us how that function changes (increasing or decreasing) if we nudge the variable a little bit to the negative or positive side.  
So when we calculate the gradient of the loss function relative to all the variables in the network, we are trying to know whether to subtract or add some value to the variables, i.e. weights and biases, so that the loss function is minimized.

- For the aficionados out there, here are the necessary equations used in the back-propagation algorithm (again, if you don't feel comfortable with the math, just skip it for now) -

   - Error delta ( $\delta$ ) of loss function relative to last layer ( $L$ ),
   **$$\delta^{L} = \bigtriangledown_a C\odot\sigma'(z^l)$$**  

   - Error delta ( $\delta$ ) of loss function relative to other layer ( $l$ ),
   **$$\delta^{l} = \Big(\ (\ w^{l+1}\ )^T \delta^{l+1}\ \Big)\odot\sigma'(z^l)$$**  

   - Partial derivative of Loss function ( $\partial C$ ) relative to the bias of the neuron in index, $j$, of layer, $l$ ( $\partial b^l_j$ ),
   **$$\frac{\partial C}{\partial b^l_j} = \delta^{l}_j $$**

   - Partial derivative of Loss function ( $\partial C$ ) relative to the weight of the neuron in index, $jk$, of layer, $l$ ( $\partial w^l_{jk}$ ),
   **$$\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k\ \delta^{l}_j $$**

I know I just threw a lot of cryptic symbols on you, but these equations are the beating heart of any neural network.  
If you want a deeper sense of the mathematics of neural networks, I highly recommend going through the book I mentioned earlier, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen, and the YouTube series, ["Neural Networks"](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3blue1brown. It's Chef's kiss.

## The Finish Line
To me, It's beautiful how a neural network can perform complex tasks such as Image Recognition, Natural Language Processing, Generative Art, and many more. But at the heart of it are just these elegant algorithms doing their job.  
Theoretically, these basic neural networks are capable of approximating any function given enough neurons, but due to computing implications, as a problem gets more complex, we've to come up with smart and efficient architectures to solve it.

I hope you had a great time reading through my blabber, and most importantly, I hope I was able to pull you into the beautiful world of neural networks.  
Happy learning.
