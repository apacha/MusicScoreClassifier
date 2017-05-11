# Music Score Classifier

This repository is the model trainer part of the Mobile Music Score Classifier, which is a mobile Android application that takes the live camera-feed and classifies the image in real-time into either music scores, or something else and displays the result in the application.
It is part of a set of three tools:

* **Model Trainer**: https://github.com/apacha/MusicScoreClassifier for the training of a classifier that uses Deep Learning to train a model to automatically classify images into scores or something else.
* **Mobile App**: https://github.com/apacha/MobileMusicScoreClassifier for the mobile Android application that uses a trained model to perform real-time classification on a mobile device.
* **Manual Classifier**: https://github.com/apacha/ManualMusicScoreClassifier for a small C#/WPF application that can be used manually classify images, used during evaluation.

|Model Trainer|Mobile App|Manual Classifier|
|:----:|:-----:|:-----:|
|[![Build Status](https://travis-ci.org/apacha/MusicScoreClassifier.svg?branch=master)](https://travis-ci.org/apacha/MusicScoreClassifier)|[![Build Status](https://travis-ci.org/apacha/MobileMusicScoreClassifier.svg?branch=master)](https://travis-ci.org/apacha/MobileMusicScoreClassifier)|[![Build status](https://ci.appveyor.com/api/projects/status/4715vyioa98eje0k?svg=true)](https://ci.appveyor.com/project/apacha/manualmusicscoreclassifier)|
|[![codecov](https://codecov.io/gh/apacha/MusicScoreClassifier/branch/master/graph/badge.svg)](https://codecov.io/gh/apacha/MusicScoreClassifier)|||



# Running the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Requirements

- Python 3.5 (3.6 can be tricky, as of May 2017 [tensorflow does not yet officially support it](https://github.com/tensorflow/tensorflow/issues/6999))
- Keras 2.0.4
- Tensorflow 1.1.0 (or optionally tensorflow-gpu 1.1.0)

Optional: If you want to print the graph of the model being trained, install GraphViz on Windows via http://www.graphviz.org/Download_windows.php and add /bin to the PATH or run `sudo apt-get install graphviz` on Ubuntu (see https://github.com/fchollet/keras/issues/3210)

Note that installing Tensorflow and Keras can be quite a hassle, so we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

## Training the model

`python TrainModel.py` can be used to training the convolutional neural network. 
It will automatically download and prepare three separate datasets for training with
Keras and Tensorflow (MUSCIMA dataset of handwritten music scores, 
Pascal VOC dataset of general purpose images and an additional dataset that 
was created for this project, containing 1000 realistic score images and 1000 
images of other documents and objects). 

The result of this training is a .h5 (e.g. vgg.h5) file that contains the trained model.

_Troubleshooting_: If for some reason the download of any of the datasets fails, stop the script, remove the partially
downloaded file and restart the script.

## Exporting the Model for being used in Tensorflow

Since the Android App only uses Tensorflow, the resulting Keras model (despite having a tensorflow model inside)
has to be exported into a Protobuf file. This is a bit cumbersome, because Tensorflow separates between
the model description and the actual weights. To get both of them into one file, one has to freeze the model.

`python ExportModelToTensorflow.py --path_to_trained_keras_model vgg.h5` will take the file `vgg.h5` and create
 a file called `output_graph.pb` that is ready to be used in the Android application.

# Additional Dataset
If you are just interested in the additional dataset that was created for this project,
it can be downloaded from [here](https://owncloud.tuwien.ac.at/index.php/s/JHzEMlwCSw8lTFp).

# License

Published under MIT License,

Copyright (c) 2017 [Alexander Pacha](http://my-it.at), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
