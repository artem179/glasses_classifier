# glasses_classifier
This repository contains the scripts for training simple glasses classifier.

### Dataset
The well-known [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset 
(in-the-wild images) was used to train and test the model.
You need to run the following script to preprocess images 
obtained in the wild to train a simple classifier.
```
$ python scripts/extract_eyes_region.py 
$ python scripts/extract_eyes_region.py --help #For more information about parameters
```
