Download Link: https://assignmentchef.com/product/solved-comp4901l-homework-assignment-6-digit-recognition-with-convolutional-neural-networks
<br>
<h1>Overview</h1>

In this assignment you will implement a convolutional neural network (CNN). You will be building a numeric character recognition system trained on the MNIST dataset. This assignment has both theory and programming components. You are expected to answer the theory questions in your write up.

We begin with a brief description of the architecture and the functions<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. A typical convolutional neural network has four different types of layers.

<h1>Fully Connected Layer/ Inner Product Layer (IP)</h1>

The fully connected or the inner product layer is the simplest layer which makes up neural networks. Each neuron of the layer is connected to all the neurons of the previous layer (See Figure 1). Mathematically it is modelled by a matrix multiplication and the addition of a bias term. For a given input <strong>x </strong>the output of the fully connected layer is given by the following equation,

<em>f</em>(<strong>x</strong>) = <em>W</em><strong>x </strong>+ <em>b</em>

<em>W,b </em>are the weights and biases of the layer. <em>W </em>is a two dimensional matrix of <em>m </em>× <em>n </em>size where <em>n </em>is the dimensionality of the previous layer and <em>m </em>is the number of neurons in this layer. <em>b </em>is a vector with size <em>m </em>× 1.

<h1>Convolutional Layer</h1>

This is the fundamental building block of CNNs.

Before we delve into what a convolution <em>layer </em>is, let’s do a quick recap of convolution.

Like we saw in homework 1 of this course, convolution is performed using a <em>k </em>× <em>k </em>filter/kernel and a <em>W </em>× <em>H </em>image. The output of the convolution operation is a feature map. This feature map can bear different meanings according to the filters being used – for example, using a Gaussian filter will lead to a blurred version of the image. Using the Sobel filters in the x and y direction give us the corresponding edge maps as outputs.

<strong>Terminology</strong>: Each number in a filter will be referred to as a filter weight. For example, the 3×3 gaussian filter has the following 9 filter weights.

Figure 1: Fully connected layer

When we perform convolution, we decide the exact type of filter we want to use and accordingly decide the filter weights. CNNs try to learn these filter weights and biases from the data. We attempt to learn a set of filters for each convolutional layer.

In general there are two main motivations for using convolution layers instead of fullyconnected (FC) layers (as used in neural networks).

<ol>

 <li>A reduction in parameters</li>

</ol>

In FC layers, every neuron in a layer is connected to every neuron in the previous layer. This leads to a large number of parameters to be estimated – which leads to over-fitting. CNNs change that by sharing weights (the same filter is translated over the entire image).

<ol start="2">

 <li>It exploits spatial structure</li>

</ol>

Images have an inherent 2D spatial structure, which is lost when we unroll the image into a vector and feed it to a plain neural network. Convolution by its very nature is a 2D operation which operates on pixels which are spatially close.

<strong>Implementation details</strong>

The general convolution operation can be represented by the following equation:

<em>f</em>(<em>X,W,b</em>) = <em>X </em>∗ <em>W </em>+ <em>b</em>

where <em>W </em>is a fiter of size <em>k </em>× <em>k </em>× <em>C<sub>i </sub></em>, <em>X </em>is an input volume of size <em>N<sub>i </sub></em>× <em>N<sub>i </sub></em>× <em>C<sub>i </sub></em>and <em>b </em>is 1 × 1 element. The meanings of the individual terms are shown below.

Figure 2: Input and output of a convolutional layer (Image source: Stanford CS231n)

In the following example the subscript <em>i </em>refers to the input to the layer and the subscript <em>o </em>refers to the output of the layer. <em>N<sub>i </sub></em>– width of the input image

<em>N<sub>i </sub></em>– height of the input image

<em>C<sub>i </sub></em>– number of channels in the input image <em>k<sub>i </sub></em>– width of the filter <em>s<sub>i </sub></em>– stride of the convolution <em>p<sub>i </sub></em>– number of padding pixels for the input image num – number of convolution filters to be learnt

In assignment 1, we performed convolution on a grayscale image – this had 1 channel. This is basically the depth of the image volume. For an image with <em>C<sub>i </sub></em>channels – we will learn num filters of size <em>k<sub>i </sub></em>× <em>k<sub>i </sub></em>× <em>C<sub>i </sub></em>. The output of convolving with each filter is a feature map with height and width <em>N<sub>o </sub></em>where

If we stack the num feature maps, we can treat the output of the convolution as another 3D volume/ image with <em>C<sub>o </sub></em>= num channels.

In summary, the input to the convolutional layer is a volume with dimensions <em>Ni</em>×<em>N<sub>i </sub></em>×<em>C<sub>i </sub></em>and the output is a volume of size <em>N<sub>o </sub></em>× <em>N<sub>o </sub></em>× num. Figure 2 shows a graphical picture. <strong>Pooling layer</strong>

A pooling layer is generally used after a convolutional layer to reduce the size of the feature maps. The pooling layer operates on each feature map separately and replaces a local region of the feature map with some aggregating statistic like max or average. In addition to reducing the size of the feature maps, it also makes the network invariant to small translations. This means that the output of the layer doesnt change when the object moves a little.

In this assignment we will use only a MAX pooling layer shown in figure 3. This operation is performed in the same fashion as a convolution, but instead of applying a filter, we find the max value in each kernel. Let <em>k </em>represent the kernel size, <em>s </em>represent the stride and <em>p </em>represent the padding. Then the output of a pooling function <em>f </em>applied to a padded feature map <em>X </em>is given by:

Figure 3: Example MAX pooling layer

<h1>Activation layer – ReLU – Rectified Linear Unit</h1>

Activation layers introduce the non-linearity in the network and give the power to learn complex functions. The most commonly used non-linear function is the ReLU function defined as follows,

<em>f</em>(<em>x</em>) = max(<em>x,</em>0)

The ReLU function operates on each output of the previous layer.

<h1>Loss layer</h1>

The loss layer has a fully connected layer with the same number of neurons as the number of classes. And then to convert the output to a probability score, a softmax function is used. This operation is given by, <em>p </em>= softmax(<em>Wx </em>+ <em>b</em>)

where, <em>W </em>is of size <em>C </em>× <em>n </em>where <em>n </em>is the dimensionality of the previous layer and <em>C </em>is the number of classes in the problem.

This layer also computes a loss function which is to be minimized in the training process. The most common loss functions used in practice are cross entropy and negative log-likelihood. In this assignment, we will just minimize the negative log probability of the given label.

<h1>Architecture</h1>

In this assignment we will use a simple architecture based on a very popular network called the LeNet<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. The exact architecture is as follows:

<ul>

 <li>Input – 1 × 28 × 28</li>

 <li>Convolution – <em>k </em>= 5<em>,s </em>= 1<em>,p </em>= 0<em>,</em>20 filters</li>

 <li>ReLU</li>

 <li>MAX Pooling – <em>k </em>= 2<em>,s </em>= 2<em>,p </em>= 0</li>

 <li>Convolution – <em>k </em>= 5<em>,s </em>= 1<em>,p </em>= 0<em>,</em>50 filters</li>

 <li>ReLU</li>

 <li>MAX Pooling – <em>k </em>= 2<em>,s </em>= 2<em>,p </em>= 0</li>

 <li>Fully Connected layer – 500 neurons</li>

 <li>ReLU</li>

 <li>Loss layer</li>

</ul>

<h1>Part 1: Theory</h1>

<strong>Q1.1 – 5 Pts </strong>We have a function which takes a two-dimensional input <em>x </em>= (<em>x</em>1<em>,x</em>2) and has two parameters <em>w </em>= (<em>w</em>1<em>,w</em>2) given by <em>f</em>(<em>x,w</em>) = <em>σ</em>(<em>σ</em>(<em>x</em><sub>1</sub><em>,w</em><sub>1</sub>)<em>w</em><sub>2 </sub>+ <em>x</em><sub>2</sub>) where. We want to estimate the parameters that minimize a L-2 loss by performing gradient descent. We initialize both the parameters to 0. Assume that we are given a training point <em>x</em><sub>1 </sub>= 1<em>,x</em><sub>2 </sub>= 0<em>,y </em>= 5, where <em>y </em>is the true value at (<em>x</em><sub>1</sub><em>,x</em><sub>2</sub>). Based on this answer the following questions:

<ul>

 <li>What is the value of ?</li>

 <li>If the learning rate is 0.5, what will be the value of <em>w</em><sub>2 </sub>after one update using SGD?</li>

</ul>

<strong>Q1.2 – 5 Pts </strong>All types of deep networks use non-linear activation functions for their hidden layers. Suppose we have a neural network (not a CNN) with input dimension <em>N </em>and output dimension <em>C </em>and <em>T </em>hidden layers. Prove that if we have a linear activation function <em>g</em>, then the number of hidden layers has no effect on the actual network.

<strong>Q1.3 – 5 Pts </strong>In training deep networks ReLU activation function is generally preferred to sigmoid, comment why?

<strong>Q1.4 – 5 Pts </strong>Why is it not a good idea to initialize a network with all zeros. How about all ones, or some other constant value?

<strong>Q1.5 – 5 Pts </strong>There are a lot of standard Convolutional Neural Network architectures used in the literature. In this question we will analyse the complexity of these networks measured in terms of the number of parameters. For each of the following networks calculate the total number of parameters.

<ul>

 <li>AlexNet<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a></li>

 <li>VGG-16<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a></li>

 <li>GoogLeNet<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a></li>

</ul>

Compare these numbers, and comment about despite being deeper, why GoogLeNet has fewer parameters.

<h1>Programming</h1>

Most of the basic framework to implement a CNN has been provided. You will need to fill in a few functions. Before going ahead into the implementations, you will need to understand the data structures used in the code.

<h2>Data structures</h2>

We define four main data structures to help us implement the Convolutional Neural Network which are explained in the following section.

Each <strong>layer </strong>is defined by a data structure, where the field type determines the type of the layer. This field can take the values of DATA, CONV, POOLING, IP, RELU, LOSS which correspond to data, convolution, max-pooling layers, inner-product/ fully connected, ReLU and Loss layers respectively. The fields in each of the layer will depend on the type of layer.

The <strong>input </strong>is passed to each layer in a structure with the following fields.

<ul>

 <li>height – height of the feature maps</li>

 <li>width – width of the feature maps</li>

 <li>channel – number of channels / feature maps</li>

 <li>batch size – batch size of the network. In this implementation, you will be implementing the mini-batch stochastic gradient descent to train the network. The idea behind this is very simple, instead of computing gradients and updating the parameters after each image, we doing after looking at a batch of images. This parameter batch size determines how many images it looks at once before updating the parameters.</li>

 <li>data – stores the actual data being passed between the layers. This is always supposed to be of the size [height × width × channel<em>,</em>batchsize]. You can resize this structure during computations, but make sure to revert it to a two-dimensional matrix.</li>

 <li>diff – Stores the gradients with respect to the data, it has the same size as data. Each layer’s parameters are stored in a structure param</li>

 <li>w – weight matrix of the layer</li>

 <li>b – bias</li>

</ul>

param grad is used to store the gradients coupled at each layer with the following properties:

<ul>

 <li>w – stores the gradient of the loss with respect to <em>w</em>.</li>

 <li>b – stores the gradient of the loss with respect to the bias term.</li>

</ul>

<h1>Part 2: Forward Pass</h1>

Now we will start implementing the forward pass of the network. Each layer has a very similar prototype. Each layers forward function takes input, layer, param as argument. The input stores the input data and information about its shape and size. The layer stores the specifications of the layer (eg. for a conv layer, it will have <em>k,s,p</em>). The params is an optional argument passed to layers which have weights and this contains the weights and biases to be used to compute the output. In every forward pass function, you expected to use the arguments and compute the output. You are supposed to fill in the height, width, channel, batch size, data fields of the output before returning from the function. Also make sure that the data field has been reshaped to a 2D matrix.

<strong>Q2.1 Inner Product Layer – 5 Pts </strong>The inner product layer of the fully connected layer should be implemented with the following definition

[output] = inner product forward(input, layer, param)

<strong>Q2.2 Pooling Layer – 10 Pts </strong>Write a function which implements the pooling layer with the following definition.

[output] = pooling layer forward(input, layer)

input and output are the structures which have data and the layer structure has the parameters specific to the layer. This layer has the following fields,

<ul>

 <li>pad – padding to be done to the input layer</li>

 <li>stride – stride of the layer</li>

 <li>k – size of the kernel (Assume square kernel)</li>

</ul>

<strong>Q2.3 Convolution Layer – 10 Pts </strong>Implement a convolution layer as defined in the following definition.

[output] = conv layer forward(input, layer, param)

The layer for a convolutional layer has the same fields as that of a pooling layer and param has the weights corresponding to the layer.

<strong>Q2.4 ReLU – 5 Pts </strong>Implement the ReLU function with the following defintion.

[output] = relu forward(input, layer)

<h1>Part 3: Back propagation</h1>

After implementing the forward propagation, we will implement the back propagation using the chain rule. Let us assume layer <em>i </em>computes a function <em>f<sub>i </sub></em>with parameters of <em>w<sub>i </sub></em>then final loss can be written as the following.

<em>l </em>= <em>f<sub>i</sub></em>(<em>w<sub>i</sub>,f<sub>i</sub></em>−<sub>1</sub>(<em>w<sub>i</sub></em>−<sub>1</sub><em>,</em>···))

To update the parameters we need to compute the gradient of the loss w.r.t. to each of the parameters.

where, <em>h<sub>i </sub></em>= <em>f<sub>i</sub></em>(<em>w<sub>i</sub>,h<sub>i</sub></em>−<sub>1</sub>).

Each layer’s back propagation function takes input, output, layer, param as input and return param grad and input od. output.diff stores the . . You are to use this to compute  and store it in param grad.w and  to be stored in param grad.b. You are also expected to

return  which is the gradient of the loss w.r.t the input layer.

<strong>Q3.1 ReLU – 10 Pts </strong>Implement the backward pass for the Relu layer in relu backward.m file. This layer doesnt have any parameters so, you dont have to return the param grad structure.

<strong>Q3.2 Inner Product layer – 10 Pts </strong>Implement the backward pass for the Inner product layer in inner product backward.m.

<h1>Putting the network together</h1>

This part has been done for you and is available in the function convnet forward. This function takes the parameters, layers and input data and generates the outputs at each layer of the network. It also returns the probabilities of the image belonging to each class. You are encouraged to look into the code of this function to understand how the data is being passed to perform the forward pass.

<h1>Part 4: Training</h1>

The function conv net puts both the forward and backward passes together and trains the network. This function has also been implemented.

<strong>Q4.1 Training – 5 Pts </strong>The script train lenet.m defines the optimization parameters and performs the actual updates on the network. This script loads a pretrained network and trains the network for 2000 iterations. Report the test accuracy obtained in your write-up after training for 3000 more iterations.

<strong>Q4.2 Test the network – 10 Pts </strong>The script test lenet.m has been provided which runs the test data through the network and obtains the predictions probabilities. Modify this script to generate the confusion matrix and comment on the top two confused pairs of classes.

<h1>Part 5: Visualization</h1>

<strong>Q5.1 – 5 Pts </strong>Write a script vis data.m which can load a sample image from the data, visualize the output of the second and third layers. Show 20 images from each layer on a single figure file (use subplot and organize them in 4 × 5 format – like in Figure 4).

<strong>Q5.2 – 5 Pts </strong>Compare the feature maps to the original image and explain the differences.

<strong>Extra credit</strong>

<h1>Part 6: Image Classification – 20 pts</h1>

We will now try to use the fully trained network to perform the task of Optical Character Recognition. You are provided a set of real world images in the images folder. Write a script ec.m which

Figure 4: Feature maps of the second layer

will read these images and recognize the handwritten numbers. The network you trained requires a binary image with a single digit in each image. There are many ways to obtain this given a real image. Here is an outline of a possible approach:

<ol>

 <li>Classify each pixel as foreground or background pixel by performing simple operations like thresholding.</li>

 <li>Find connected components and place a bounding box around each character.</li>

 <li>Take each bounding box, pad it if necessary and resize it to 28 × 28 and pass it through the network.</li>

</ol>

There might be errors in the recognition, report the output of your network in the report.

<h1>Appendix: List of all files in the project</h1>

<ul>

 <li>col2im conv.m Helper function, you can use this if needed</li>

 <li>col2im conv matlab.m Helper function, you can use this if needed</li>

 <li>conv layer backward.m – Do not modify</li>

 <li>conv layer forward.m – To implement</li>

 <li>conv net.m – Do not modify</li>

 <li>convnet forward.m – To implement</li>

 <li>get lenet.m – Do not modify. Has the architecture.</li>

 <li>get lr.m – Gets the learning rate at each iterations</li>

 <li>im2col conv.m Helper function, you can use this if needed</li>

 <li>im2col conv matlab.m Helper function, you can use this if needed</li>

 <li>init convnet.m Initialise the network weights</li>

 <li>inner product backward.m – To implement</li>

 <li>inner product forward.m – To implement</li>

 <li>load nist.m – Loads the training data.</li>

 <li>m – Implements the loss layer</li>

 <li>poolinglayer backward.m Implemented, do not modify</li>

 <li>poolinglayer forward.m – To implement</li>

 <li>relu backward.m – To implement</li>

 <li>relu forward.m – To implement</li>

 <li>sgd m – Do not modify. Has the update equations</li>

 <li>test network.m – Test script</li>

 <li>train lenet.m – Train script</li>

 <li>vis m – Add code to visualise the filters</li>

 <li>lenet pretrained.mat – Trained weights</li>

 <li>mnist all.mat – Dataset</li>

</ul>

<h1>Notes</h1>

Here are some points which you should keep in mind while implementing:

<ul>

 <li>All the equations above describe the functioning of the layers on a single data point. Your implementation would have to work on a small set of inputs called a “batch” once.</li>

 <li>Always ensure that the data of each layer has been reshaped to a 2-D matrix.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> This is meant to be a short introduction, you are encouraged to read resources online like http://cs231n.

stanford.edu/ to understand further.

<a href="#_ftnref2" name="_ftn2">[2]</a> http://ieeexplore.ieee.org/abstract/document/726791/

<a href="#_ftnref3" name="_ftn3">[3]</a> http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks. pdf

<a href="#_ftnref4" name="_ftn4">[4]</a> https://arxiv.org/pdf/1409.1556.pdf(pg-3ArchitectureD)

<a href="#_ftnref5" name="_ftn5">[5]</a> https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf