**from fastai import * needs to be added to execute past the third input**

External Links:

* https://arxiv.org/pdf/1512.03385.pdf - Deep Residual Learning for Image Recognition
* http://cs231n.github.io/convolutional-networks/ - CS231n Convolutional Neural Networks for Visual Recognition

The Oxford-IIIT PET Dataset has several mislabeled images that may show up in top losses:

https://imgur.com/a/YtmIqmq

http://www.robots.ox.ac.uk/~vgg/data/pets/getCategory.php?category=saint_bernard

#Auto reload imports in jupyter notebook

%reload_ext autoreload

%autorepload 2

Similar multi-class classification - fine grained categorization

FastAi package stuff:

* path.ls() - list contents from path
* get_image_files(path) - list image files from path
* ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224) - returns operable image data object
  * dataBunch.show_batch(rows=n, figsize=(x,y)) - preview images
  * dataBunch.classes() - list classes
  * dataBunch.c - returns the number of classes in the bunch (sort of...it'll do for now)
* ConvLearner(dataBunch, models.resnet34, metrics=error_rate)
  * 'resnet works very very well almost all the time' The number indicates the size of the model
  * ConvLearner downloads a pretrained resnet model trained on image net **This is transferred learning** and can allow for training accurate models with 1/100th the time and with 1/100th the data
  * convLearner.fit vs convLearner.fit_one_cycle - just use fit_one_cycle. It's a new method that is much faster and more accurate in most cases
  
**Spend time running the code!**
