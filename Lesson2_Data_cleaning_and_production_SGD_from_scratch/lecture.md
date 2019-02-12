"Most of the parameters used don't matter that much in detail. Just use what you see here and it will probably work." :)

FileDeleter is a nice widget that lets you browse by top loss and delete bad/mislabeled images

Checkout **ipywidgets.** It's a nice way to make little applications that function inside of jupyter notebooks.

Checkout **Starlette**. It sounds like Flask with async support.

Checkout **computational linear algebra for coders**

Use CPU for production inference systems except at massive scale.

?? before or after a function displays the source code - nice!

3e-3 is generally a good learning rate to start with

**What to do if things go wrong**

* Check the learning rate. High loss can indicate high LR. Slow decrease in loss can indicate low LR.
* Check the number of epochs.
* You never want a model where your training loss is higher than your validation loss. It means that you havent fitted enough. Either your   learning rate is too low or your number of epochs is too low.
* Too many epochs can cause overfitting.
* An indication of overfitting is if the error rate improves for a while, and then starts to increase.
* **It is not true that train loss is lower than you validation loss then you are overfitting.** Any model that is trained correctly will always have train loss lower than validation loss.


Predictors are functions of pixel values

The sum of two products is a dot product  
[**Awesome matrix multiplication animation**](http://www.matrixmultiplication.xyz)

**SGD**
Stochastic Gradient Descent

**tensor**
An array of a regular shape (without jagged edges)
Rank is how many axis the tensor has - a vector is a rank 1 tensor

From generating a random'ish scatter plot @ ~1:24

#Creates a collection of 100 length 2 vectors
x = torch.ones(100,2)
#Fiils the first column with **uniformly distributed** random numbers between -1 and 1
x[:,0].uniform_(-1.,1)
#Creates a rank 1 tensor
a = tensor(3.,2)
#Gets the product of x and a with some random numbers added for noise
y = x@a + torch.rand(100)
#Mean squared error is the mean of (the difference of y predicted and y actual squared)
def mse(y_hat, y): return ((y_hat-y)**2).mean()
