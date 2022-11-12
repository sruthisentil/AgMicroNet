# TOMicro_Classification
Talk about what TOMicro
[
input the image of example microimages
under it put saliency maps
]

## Dependencies
## Contents

This library contains Tensorflow/Keras code for:

the LEAF frontend, as well as mel-filterbanks, SincNet and Time-Domain Filterbanks
Keras models for PANN, PCEN and SpecAugment
an example training loop using gin, to train models with various frontends and architectures on tensorflow datasets.

## Creating a Custom Dataset
talk about pickling the data
```
python3 data_pickler.py --img_size 320 --dataset "/Users/skanda/Downloads/CompleteDataset/Fruit" --type_of_crop "tomato"
```

## Training and Customizing the TOMicroCNN Architecture
[put the pipeline diagram where you see the entire thing]
- with or without kfold

## Training a Transfer Learning Architecture
[give them the options]
[also include xgboost]

## Producing Validation Metrics
[kfold graphs, scores themselves, confusion matrices]

[how to run inference on a trained model]
- show the example of manoj's validation

## Reference


