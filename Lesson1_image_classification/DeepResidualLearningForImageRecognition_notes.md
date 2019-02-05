
[Source](https://arxiv.org/pdf/1512.03385.pdf)

**TL;DR**
The source shows successfully trained models with 100 or even 1000 layers. On the ImageNet classification dataset they obtained a 3.57% error rate whil maintaining a lower complexity than VGG nets. This method won 1st place in the ILSVRC 2015 classification competition, the ImageNet detection, ImageNet localization, COCO detections, COCO segmentation in ILSVRC & COCO 2015 competitions. It generalizes very well. It also claims that the residual nets are generally easier to set up and train than other methods.

Deep convolutional neural networks have led to a series of breakthroughs for image classification. Deep networks naturally integration low/mid/high level features and calssifiers in an end-to-end multi-layer fashion, and the "levels" of features can be enriched by the number of stacked layers (depth).

Recent evidence reveals that network depth is of crucial importance, and the leading results on the challenging ImageNet dataset all exploit "very deep" models, with a depth of sixteen to thirty.

*Is learning better networks as easy as stacking more layers?*

An obstacle to answering this question is the problem of [vanishing/exploding gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

In the worst case, this may completely stop the neural network from further training. As one example of the problem cause, traditional activation functions such as the hyperbolic tangent function have gradients in the range (0, 1), and backpropagation computes gradients by the chain rule. This has the effect of multiplying n of these small numbers to compute gradients of the "front" layers in an n-layer network, meaning that the gradient (error signal) decreases exponentially with n while the front layers train very slowly.

One of the newest and most effective ways to resolve the vanishing gradient problem is with residual neural networks, ResNets,[12] not to be confused with recurrent neural networks.[13] It was noted prior to ResNets that a deeper network would actually have higher training error than the shallow network. 

Note that ResNets are an ensemble of relatively shallow Nets and do not resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network – rather, they avoid the problem simply by constructing ensembles of many short networks together. (Ensemble by Construction[16])

When deeper networks are able to start converging, a degredation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. This degredation is *not caused by overfitting*, and adding more layers to suitably deep models leads to a higher training-error.

The degredation problem can be addressed by introducing a deep residual learning architecture. In this architecture, instead of hoping each few stacked layers directly fit a desired underlying mapping, they are allowed to explicitly fit a residual mapping (wut?).

"Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x) := H(x)−x. The original mapping is recast into F(x)+x" 

I think F(x) is the original mapping and H(x) - x is the residual. The original is recast and the residual is 'optimized' by moving the
-x. Dunno why this optimizes it. Maybe it's what prevents the vanishing gradient problem?

The formulation of F(x) + x can be realized by [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_network) with "shortcut connections". Shortcut connection skip one or more layers. In the case of a deep residual neural network, the shortcut connections perform identity mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcut connections do not add extra parameters or computational complexity. A large deep residual neural network can still be trained end-to-end by [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation).

