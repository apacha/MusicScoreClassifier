import argparse
import datetime
import os
import shutil
from math import ceil
from typing import Tuple

import torch
from time import time

from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adadelta
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.AdditionalDataset import AdditionalDataset
from datasets.DatasetSplitter import DatasetSplitter
from datasets.MuscimaDataset import MuscimaDataset
from datasets.PascalVocDataset import PascalVocDataset
from models.SimpleNetwork import SimpleNetwork
import torch.optim as optim


def delete_dataset_directory(dataset_directory):
    print("Deleting dataset directory and creating it anew")

    if os.path.exists(dataset_directory):
        shutil.rmtree(dataset_directory)


def download_datasets(dataset_directory):
    pascal_voc_dataset = PascalVocDataset(dataset_directory)
    pascal_voc_dataset.download_and_extract_dataset()
    muscima_dataset = MuscimaDataset(dataset_directory)
    muscima_dataset.download_and_extract_dataset()
    additional_dataset = AdditionalDataset(dataset_directory)
    additional_dataset.download_and_extract_dataset()


def print_model_architecture_and_parameters(network):
    print(network)
    summary(network, (3, 224, 224))


def get_dataset_loaders(dataset_directory, minibatch_size) -> Tuple[DataLoader, DataLoader]:
    data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    number_of_workers = 6
    training_dataset = ImageFolder(root=os.path.join(dataset_directory, "training"), transform=data_transform)
    training_dataset_loader = DataLoader(training_dataset, batch_size=minibatch_size, shuffle=True,
                                         num_workers=number_of_workers)
    validation_dataset = ImageFolder(root=os.path.join(dataset_directory, "training"), transform=data_transform)
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=minibatch_size, shuffle=False,
                                           num_workers=number_of_workers)

    return training_dataset_loader, validation_dataset_loader


def train_model(dataset_directory: str,
                model_name: str,
                delete_and_recreate_dataset_directory: bool,
                minibatch_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Downloading and extracting datasets...")

    if delete_and_recreate_dataset_directory:
        delete_dataset_directory(dataset_directory)
        download_datasets(dataset_directory)
        dataset_splitter = DatasetSplitter(dataset_directory, dataset_directory)
        dataset_splitter.split_images_into_training_validation_and_test_set()

    print("Training on dataset...")

    model = SimpleNetwork()
    model.to(device)
    print_model_architecture_and_parameters(model)

    training_dataset_loader, validation_dataset_loader = get_dataset_loaders(dataset_directory, minibatch_size)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = Adadelta(model.parameters())
    cross_entropy_loss = CrossEntropyLoss()

    trainer = create_supervised_trainer(model, optimizer, cross_entropy_loss, device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(), 'cross-entropy': Loss(cross_entropy_loss)},
                                            device=device)

    iteration_between_progress_bar_updates = 1
    progress_description_template = "ITERATION - loss: {:.2f} - acc: {:.2f}"
    progress_bar = tqdm(
        initial=0, leave=False, total=len(training_dataset_loader),
        desc=progress_description_template.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        iter = (trainer.state.iteration - 1) % len(training_dataset_loader) + 1
        if iter % iteration_between_progress_bar_updates == 0:
            progress_bar.desc = progress_description_template.format(trainer.state.output)
            progress_bar.update(iteration_between_progress_bar_updates)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        progress_bar.refresh()
        evaluator.run(training_dataset_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['cross-entropy']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(validation_dataset_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['cross-entropy']))
        progress_bar.n = progress_bar.last_print_n = 0

    trainer.run(training_dataset_loader, max_epochs=10)
    progress_bar.close()

    print('Finished Training')

    return

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
    best_model = keras.models.load_model(best_model_path)

    evaluation = best_model.evaluate_generator(test_data_generator, steps=test_steps_per_epoch)

    print(best_model.metrics_names)
    print("Loss : ", evaluation[0])
    print("Accuracy : ", evaluation[1])
    print("Error : ", 1 - evaluation[1])
    end_time = time()
    print("Execution time: %.1fs" % (end_time - start_time))

    TrainingHistoryPlotter.plot_history(history,
                                        "Results-{0}-{1}.png".format(training_configuration.name(),
                                                                     datetime.date.today()),
                                        show_plot=show_plot_after_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_name", type=str, default="mobilenetv2",
                        help="The model used for training the network. "
                             "Currently allowed values are \'simple\', \'vgg\', \'xception\', \'mobilenetv2\'")
    parser.add_argument("--delete_and_recreate_dataset_directory", dest="delete_and_recreate_dataset_directory",
                        action="store_true",
                        help="Whether to delete and recreate the dataset-directory (by downloading the appropriate "
                             "files from the internet) or simply use whatever data currently is inside of that "
                             "directory")
    parser.set_defaults(delete_and_recreate_dataset_directory=False)

    flags, unparsed = parser.parse_known_args()

    train_model(flags.dataset_directory,
                flags.model_name,
                flags.delete_and_recreate_dataset_directory)
