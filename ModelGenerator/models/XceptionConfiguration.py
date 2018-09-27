from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adadelta

from models.TrainingConfiguration import TrainingConfiguration


class XceptionConfiguration(TrainingConfiguration):
    """ A rudimentary configuration for starting """

    def __init__(self):
        super().__init__(data_shape = (299, 299, 3), learning_rate=1.0, training_minibatch_size=4)

    def classifier(self) -> Model:
        """ Returns the classifier of this configuration """
        model = Xception(include_top=False, weights='imagenet', input_shape=self.data_shape, pooling='avg')
        dense = Dense(2, activation='softmax', name='predictions')(model.output)
        model = Model(model.input, dense, name='xception')

        # optimizer = SGD(lr=self.learning_rate, momentum=self.nesterov_momentum, nesterov=True)
        optimizer = Adadelta(lr=self.learning_rate)
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "xception"
