"Most of the parameters used don't matter that much in detail. Just use what you see here and it will probably work." :)

FileDeleter is a nice widget that lets you browse by top loss and delete bad/mislabeled images

Checkout **ipywidgets.** It's a nice way to make little applications that function inside of jupyter notebooks.

Use CPU for production inference systems except at massive scale.

Checkout **Starlette**. It sounds like Flask with async support.

What to do if things go wrong:

* Check the learning rate. High loss can indicate high LR. Slow decrease in loss can indicate low LR.
* Check the number of epochs.
* You never want a model where your training loss is higher than your validation loss. It means that you havent fitted enough. Either your   learning rate is too low or your number of epochs is too low.
* Too many epochs can cause overfitting.
* An indication of overfitting is if the error rate improves for a while, and then starts to increase.
* **It is not true that train loss is lower than you validation loss then you are overfitting**


