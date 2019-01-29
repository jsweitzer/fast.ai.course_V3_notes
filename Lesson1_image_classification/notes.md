#Auto reload imports in jupyter notebook
%reload_ext autoreload
%autorepload 2

Similar multi-class classification - fine grained categorization

FastAi methods:

*path.ls() - list contents from path
*get_image_files(path) - list image files from path
*ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224) - returns operable image data object
