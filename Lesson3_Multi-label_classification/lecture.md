[data block api](https://docs.fast.ai/data_block.html) provides a workflow pipline

[nice data block explanation](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4)

[deforestation kaggle comp](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

Image segmentation involves labeling each pixel of an image. The [camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) provides some pre labeled. Others are available 
at [https://course.fast.ai/datasets](https://course.fast.ai/datasets)

tfm_y transform parameter needs to be true when using image masks or the mask and source image wont match

use create_unet for segmentation models

[list of transforms](https://docs.fast.ai/vision.transform.html#List-of-transforms)

[**deploying to render**](https://course.fast.ai/deployment_render.html)

Gradually increasing LR early on can prevent getting stuck in loss function surface bumps. Gradually decreasing later on
can help get to the bottom of a tight loss function curve.

*Slightly* increasing losses early in training can indicate a good LR if it is overall going down.

[IMDB datasets](https://datasets.imdbws.com/) and [documentation](https://www.imdb.com/interfaces/)







