import os

import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

mobilenet_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
dense = Dense(2, activation='softmax', name='predictions')(mobilenet_model.output)
mobilenet_model = Model(mobilenet_model.input, dense, name='xception')
optimizer = Adadelta(lr=1.0)
mobilenet_model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
estimator_mobilenet_v2 = tf.keras.estimator.model_to_estimator(keras_model=mobilenet_model)

train_generator = ImageDataGenerator(horizontal_flip=True, )
training_data_generator = train_generator.flow_from_directory(os.path.join("data", "training"),
                                                              target_size=(224, 224),
                                                              batch_size=16)
dataset = Dataset.from_generator(training_data_generator)
value = dataset.make_one_shot_iterator().get_next()

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
estimator_mobilenet_v2.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
estimator_mobilenet_v2.train(input_fn=train_input_fn, steps=2000)
