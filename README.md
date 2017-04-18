# Music Score Classifier

This repository is the model training part of the Mobile Music Score Classifier, which is a mobile Android application that takes the live camera-feed and classifies the image in real-time into either music scores, or something else and displays the result in the application.

See https://github.com/apacha/MobileMusicScoreClassifier for the mobile Android application that uses the model trained with the scripts from this repository.

# Building the application

## Model Generator
The Model generator is a python script that downloads the MUSCIMA dataset (handwritten music scores) and the Pascal VOC dataset (general purpose images), extracts them and uses them as two distinct sets of images for training a Convolutional Neural Network with Keras and Tensorflow.


## Mobile Classifier

The Mobile classifier is an Android application that takes the model created by the model generator and uses it to classify the live video-feed to either Music Scores or something else.

## Authors
Alexander Pacha, Technical University of Vienna

## Contributing

## License