import datetime
import os
from time import time

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from TrainingHistoryPlotter import TrainingHistoryPlotter
from datasets.AdditionalDataset import AdditionalDataset
from datasets.DatasetSplitter import DatasetSplitter
from datasets.MuscimaDataset import MuscimaDataset
from datasets.PascalVocDataset import PascalVocDataset
from models.ConfigurationFactory import ConfigurationFactory

print("Downloading and extracting datasets...")

dataset_directory = "data"
additional_dataset_directory = "C:\\Users\\Alex\\Dropbox\\Doktorat\\MusicScoresDataset"

pascalVocDataset = PascalVocDataset(dataset_directory)
pascalVocDataset.download_and_extract_dataset()
muscimaDataset = MuscimaDataset(dataset_directory)
muscimaDataset.download_and_extract_dataset()
additionalDataset = AdditionalDataset(dataset_directory, additional_dataset_directory)
additionalDataset.download_and_extract_dataset()

dataset_splitter = DatasetSplitter(dataset_directory, dataset_directory)
dataset_splitter.split_images_into_training_validation_and_test_set()

print("Training on datasets...")
start_time = time()

model_name = "simple3"
training_configuration = ConfigurationFactory.get_configuration_by_name(model_name)
img_rows, img_cols = training_configuration.data_shape[0], training_configuration.data_shape[1]
number_of_pixels_shift = training_configuration.number_of_pixel_shift

train_generator = ImageDataGenerator(horizontal_flip=True,
                                     rotation_range=10,
                                     width_shift_range=number_of_pixels_shift / img_rows,
                                     height_shift_range=number_of_pixels_shift / img_cols,
                                     )
training_data_generator = train_generator.flow_from_directory(os.path.join(dataset_directory, "training"),
                                                              target_size=(img_cols, img_rows),
                                                              batch_size=training_configuration.training_minibatch_size,
                                                              # save_to_dir="train_data"
                                                              )
training_steps_per_epoch = np.math.ceil(training_data_generator.samples / training_data_generator.batch_size)

validation_generator = ImageDataGenerator()
validation_data_generator = validation_generator.flow_from_directory(os.path.join(dataset_directory, "validation"),
                                                                     target_size=(img_cols, img_rows),
                                                                     batch_size=training_configuration.training_minibatch_size)
validation_steps_per_epoch = np.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)

test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(os.path.join(dataset_directory, "test"),
                                                         target_size=(img_cols, img_rows),
                                                         batch_size=training_configuration.training_minibatch_size)
test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

model = training_configuration.classifier()
model.summary()

print("Model {0} loaded.".format(training_configuration.name()))
print(training_configuration.summary())

best_model_path = "{0}.h5".format(training_configuration.name())

model_checkpoint = ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_acc',
                           patience=training_configuration.number_of_epochs_before_early_stopping,
                           verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=training_configuration.number_of_epochs_before_reducing_learning_rate,
                                            verbose=1,
                                            factor=training_configuration.learning_rate_reduction_factor,
                                            min_lr=training_configuration.minimum_learning_rate)
history = model.fit_generator(
        generator=training_data_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_configuration.number_of_epochs,
        callbacks=[model_checkpoint, early_stop, learning_rate_reduction],
        validation_data=validation_data_generator,
        validation_steps=validation_steps_per_epoch
)

print("Loading best model from check-point and testing...")
# For some models, loading the model directly does not work, but loading the weights does
# (see https://github.com/fchollet/keras/issues/4044#issuecomment-254921595)
# best_model = keras.models.load_model(best_model_path)
best_model = training_configuration.classifier()
best_model.load_weights(best_model_path)


evaluation = best_model.evaluate_generator(test_data_generator, steps=test_steps_per_epoch)

print(best_model.metrics_names)
print("Loss : ", evaluation[0])
print("Accuracy : ", evaluation[1])
print("Error : ", 1 - evaluation[1])

TrainingHistoryPlotter.plot_history(history,
                                    "Results-{0}-{1}.png".format(training_configuration.name(), datetime.date.today()))

endTime = time()
print("Execution time: %.1fs" % (endTime - start_time))
