# glasses_classifier
This repository contains the scripts for training simple glasses classifier.

### Prepare environment
Create new environment via conda and inside it run:
```
$ pip install -r requirements.txt
```

### Dataset
The well-known [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset 
(in-the-wild images) was used to train and test the model.

### Training
#### Dataset preprocessing
You need to run the following script to preprocess images 
obtained in the wild to train a simple classifier.
```
$ python scripts/extract_eyes_region.py --img_folder=dataset/img_celeba --output_folder=cropped_images
$ python scripts/extract_eyes_region.py --help #For more information about parameters
```
#### Train neural network
```
$ python train.py
```

### Evaluation
```
$ python scripts/inference_on_images.py
$ python scripts/inference_on_images.py --help #For more information about parameters
$ python scripts/inference_on_images.py --time #If you want check inference time
```

### Summary
There are two trained models in this repository. 

- Simple small VGG-like (it classifies cropped image which obtains via landmark detector)
+ MobileNetV3 (it classifies whole image)

| Networks      |classifier_score (CelebA/intheWild(100 images))| weights  | time (GTX1080-ti) |
| ------------- |:-------------:| -----:|  -----:|
| small VGG-like      | 0.9987/1.0 | 2.9MB |  5ms (<100ms)|
| MobileNet-v3  | 0.98/0.99      |  6.7MB |   70ms (<100ms)|

For MobileNet-v3 you can look at `MobileNetV3/mobile-net.ipynb`