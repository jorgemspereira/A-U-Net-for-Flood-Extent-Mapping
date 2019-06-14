# A U-Net Model Leveraging Multiple Remote Sensing Data Sources for Flood Extent Mapping

This repository contains the implementation for a U-Net model that leverages multiple remote sensing data
for flood extent mapping using the dataset from the FDSI sub-task from the Multimedia Satellite Task of the 
MediaEval 2017. The presented implementation uses the [keras.io](http://keras.io/) deep learning library 
(combined with [scikit-learn](https://scikit-learn.org/stable/), and other machine learning libraries). 

### Files needed not in this repository

The dataset files are too big to put on a github repository, so it is necessary to download them from
[this](https://drive.google.com/drive/folders/1gUzU0cNzAxlPd3czLv9GBWS1kAHrDxwA) Google Drive folder and place them in the following hierarchy:

```
project
│   README.md
│   main.py
│   ...
└─── flood-data
│   │    devset_01_elevation_and_slope
│   │    devset_01_gdacs_water_and_magnitude
│   │    devset_01_imperviousness
│   │    devset_01_NVDI
│   │    devset_01_satellite_images
│   │    devset_01_segmentation_masks
│   │    devset_01_elevation_and_slope
│   │    ... 
```

### How to use  

The code was developed and tested in Python 3.6.7 with Keras 2.2.4, using Tensorflow as backend. 
The code supports re-training and model loading from a previous saved model. To run the script simply execute:

```console
$ python3 rgb2lab.py                      # Convert the RGB images to LAB
$ python3 gen_patches.py                  # Generate patches from the LAB images
$ python3 main.py --mode {train, load}    # Train/evaluate the U-Net model
```

### Acknowledgments

- [Lovász-Hinge Loss](https://github.com/bermanmaxim/LovaszSoftmax)
- [Squeeze and Excitation Convolutions](https://github.com/titu1994/keras-squeeze-excite-network)
