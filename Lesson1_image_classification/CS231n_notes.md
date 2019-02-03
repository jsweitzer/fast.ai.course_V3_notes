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

In general, setting the zero padding to be P = (F - 1)/2 when the strice is S = 1 ensures that the input volume and the output volume will have the same spatial dimensions.

Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a convolution of the neuron’s weights with the input volume (Hence the name: Convolutional Layer). This is why it is common to refer to the sets of weights as a filter (or a kernel), that is convolved with the input.

Summary. To summarize, the Conv Layer:

Accepts a volume of size W1×H1×D1
Requires four hyperparameters:
Number of filters K,
their spatial extent F,
the stride S,
the amount of zero padding P.
Produces a volume of size W2×H2×D2 where:
W2=(W1−F+2P)/S+1
H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
D2=K
With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.
A common setting of the hyperparameters is F=3,S=1,P=1. However, there are common conventions and rules of thumb that motivate these hyperparameters. See the ConvNet architectures section below.

-- Go to http://cs231n.github.io/convolutional-networks/ and CTRL + F "Convolution Demo." for a nice visual demo

**Pooling Layer**

The function of the pooling layer is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. It is commonly inserted between successive CONV layers in a CNN architecture. The pooling layer operates independently on every depth slice and resizes it spatially using the MAX operation.

Generally the pooling layer:

Accepts a volume of size W1×H1×D1
Requires two hyperparameters:
their spatial extent F,
the stride S,
Produces a volume of size W2×H2×D2 where:
W2=(W1−F)/S+1
H2=(H1−F)/S+1
D2=D1
Introduces zero parameters since it computes a fixed function of the input
For Pooling layers, it is not common to pad the input using zero-padding.

![alt text](http://cs231n.github.io/assets/cnn/maxpool.jpeg "Max pooling example")

Pooling layers may be on their way out and "it seems likely that future architectures will feature very few to no pooling layers."

**Normilzation Layer**

Many types of normalization layers have been proposed for use in ConvNet architectures, sometimes with the intentions of implementing inhibition schemes observed in the biological brain. However, these layers have since fallen out of favor because in practice their contribution has been shown to be minimal, if any. For various types of normalizations, see the discussion in Alex Krizhevsky’s cuda-convnet library API.

**Fully-connected layer**

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

**Layer patterns**

The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern:

INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

where the * indicates repetition, and the POOL? indicates an optional pooling layer. Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). For example, here are some common ConvNet architectures you may see that follow this pattern:

INPUT -> FC, implements a linear classifier. Here N = M = K = 0.
INPUT -> CONV -> RELU -> FC
INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC. Here we see that there is a single CONV layer between every POOL layer.
INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

**In practice: use whatever works best on ImageNet.** If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. **You should rarely ever have to train a ConvNet from scratch or design one from scratch.** I also made this point at the Deep Learning school.

**Layer sizing patterns**

The input layer should be divisible by 2 many times

The conv layers should be using small filters - 3x3 or at most 5x5, using a stride if S=1 and crucially, padding the input volume with zeros in such a way that the conv layer does not alter the spatial dimensions of the input.

Zero-padding generally improves performance and prevents the borders of the image from being "washed away" by downsampling and stuff.

A stride of 1 works better in practice.

**Case studies**

ResNet. [Residual Network](http://arxiv.org/abs/1512.03385) developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special skip connections and a heavy use of batch normalization. The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming’s presentation ([video](https://www.youtube.com/watch?v=1PGLj-uKT1w), [slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)), and some [recent experiments](https://github.com/gcr/torch-residual-networks) that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from Kaiming He et al. Identity Mappings in Deep Residual Networks (published March 2016).
