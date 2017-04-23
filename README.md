# Music Score Classifier

This repository is the model training part of the Mobile Music Score Classifier, which is a mobile Android application that takes the live camera-feed and classifies the image in real-time into either music scores, or something else and displays the result in the application.

See https://github.com/apacha/MobileMusicScoreClassifier for the mobile Android application that uses the model trained with the scripts from this repository.

# Building the application
The application contains scripts for automatically downloading and preparing the datasets used for training of the convolutional neural network. The additional dataset that was created for this project can be downloaded from [here](https://owncloud.tuwien.ac.at/index.php/s/JHzEMlwCSw8lTFp)

[![Build Status](https://travis-ci.org/apacha/MusicScoreClassifier.svg?branch=master)](https://travis-ci.org/apacha/MusicScoreClassifier)

## Model Generator
The Model generator is a python script that downloads the MUSCIMA dataset (handwritten music scores) and the Pascal VOC dataset (general purpose images), extracts them and uses them as two distinct sets of images for training a Convolutional Neural Network with Keras and Tensorflow.

## Authors
Alexander Pacha, Technical University of Vienna

## License

Published under MIT License,

Copyright (c) 2017 Alexander Pacha

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
