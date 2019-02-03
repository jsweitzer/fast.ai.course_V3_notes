ConvNet(CNNs) architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

**Architecture Overview**
Recall: Regular Neural Nets. As we saw in the previous chapter, Neural Networks receive an input (a single vector), and transform it through a series of hidden layers. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the “output layer” and in classification settings it represents the class scores.

3D volumes of neurons. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. (Note that the word depth here refers to the third dimension of an activation volume, not to the depth of a full Neural Network, which can refer to the total number of layers in a network.) For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). As we will soon see, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension. Here is a visualization:

![alt text](http://cs231n.github.io/assets/nn1/neural_net2.jpeg "2D architecture visualization")

![alt text](http://cs231n.github.io/assets/cnn/cnn.jpeg "3D architecture visualization")

Top: A regular 3-layer Neural Network. Bottom: A ConvNet arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a ConvNet transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).

A ConvNet architecture is in the simplest case a list of Layers that transform the image volume into an output volume (e.g. holding the class scores)
There are a few distinct types of Layers (e.g. CONV/FC/RELU/POOL are by far the most popular)
Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function
Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)
Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)

**Convolutional Layer**

The Conv layer is the core building block of a Convolutional Network that does most of the computational heavy lifting.

The CONV layer's parameters consist of a set of learnable filters. Every filter is small spatially, but extends through the full depth of the input volume. A typical filter on a first layer of a ConvNet might have size 5x5x3 (5x5 width/height and 3 depth because images have a depth of three for the color channels). Each filter is convolved across the width and height of the input volume, creating a 2D activation map that represents the response of the filter. The network will learn filters that activate when they see some type of visual features such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Each CONV layer will have an entire set of filters production their own 2D activation map. The activation maps will be stacked along the depth dimension to produce the output volume.

Local Connectivity:

When dealing with high-dimensional inputs such as images, as we saw above it is impractical to connect neurons to all neurons in the previous volume. Instead, we will conenct each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently this is the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume. The connections are localized along the cartesian plane, but always full along the depth of the input volume.

Example 1. For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of 5*5*3 = 75 weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

Example 2. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3*3*20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

Spatial arrangement:

Three hyperparameters control the size of the output volume: depth, stride and zero-padding.

1. Depth - A hyperparameter that corresponds to the number of filters we would like to use, each learning to look for something different in the input. The different neurons along the depth dimension may activate in the presence of various oriented edges, or blobs of color. The set of neurons looking at the same region of the input volume will be referred to as *the depth column*.

2. Stride - A hyperparemter that adjusts the movement of the filters along the input volume. When stride is 1 the filters move one pixel at a time, when 2 or three, it will move 2 or 3 pixels at a time. 2 and 3 or higher are uncommon and will produce smaller output volumes.

3. Zero-padding - Sometimes it's convenient to pad the input volume with zeros to control the spatial size of the output volume. Zero-padding is a hyper parameter that controls this.

The spatial size of the output volume can be computed by letting W = input volume size, F = the receptive field size of the CONV layer neurons, S = the stride with which they're applied, and P = the zero padding used.

(W - F + 2P)/S + 1 = Output volume size
