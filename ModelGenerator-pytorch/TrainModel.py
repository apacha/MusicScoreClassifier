import argparse
import datetime
import os
import shutil
from typing import Tuple

import torch
from time import time

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events, Engine
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.models import VGG, vgg11_bn, SqueezeNet
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.AdditionalDataset import AdditionalDataset
from datasets.DatasetSplitter import DatasetSplitter
from datasets.MuscimaDataset import MuscimaDataset
from datasets.PascalVocDataset import PascalVocDataset
from models.MobileNetV2 import MobileNetV2
from models.SimpleNetwork import SimpleNetwork


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


def get_dataset_loaders(dataset_directory, minibatch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    number_of_workers = 8
    training_dataset = ImageFolder(root=os.path.join(dataset_directory, "training"), transform=data_transform)
    training_dataset_loader = DataLoader(training_dataset, batch_size=minibatch_size, shuffle=True,
                                         num_workers=number_of_workers)
    validation_dataset = ImageFolder(root=os.path.join(dataset_directory, "validation"), transform=data_transform)
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=minibatch_size, shuffle=False,
                                           num_workers=number_of_workers)
    testing_dataset = ImageFolder(root=os.path.join(dataset_directory, "test"), transform=data_transform)
    testing_dataset_loader = DataLoader(testing_dataset, batch_size=minibatch_size, shuffle=False,
                                        num_workers=number_of_workers)

    return training_dataset_loader, validation_dataset_loader, testing_dataset_loader


def get_model_by_name(model_name) -> Module:
    if model_name == "vgg":
        return VGG(vgg11_bn().features, num_classes=2)
    if model_name == "mobilenetv2":
        return MobileNetV2(num_classes=2)
    if model_name == "simple":
        return SimpleNetwork()
    if model_name == "squeezenet":
        return SqueezeNet(version=1.1,num_classes=2)

    raise Exception(f"Invalid model name: {model_name}.")


def train_model(dataset_directory: str,
                model_name: str,
                delete_and_recreate_dataset_directory: bool,
                minibatch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Downloading and extracting datasets...")

    if delete_and_recreate_dataset_directory:
        delete_dataset_directory(dataset_directory)
        download_datasets(dataset_directory)
        dataset_splitter = DatasetSplitter(dataset_directory, dataset_directory)
        dataset_splitter.split_images_into_training_validation_and_test_set()

    print("Training on dataset...")

    model = get_model_by_name(model_name)
    model.to(device)
    print_model_architecture_and_parameters(model)
    training_dataset_loader, validation_dataset_loader, testing_dataset_loader = get_dataset_loaders(dataset_directory,
                                                                                                     minibatch_size)
    optimizer = Adadelta(model.parameters())
    learning_rate_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=8, verbose=True)

    trainer = create_supervised_trainer(model, optimizer, CrossEntropyLoss(), device=device)
    validation_evaluator = create_supervised_evaluator(model,
                                                       metrics={'accuracy': Accuracy(),
                                                                'cross-entropy': Loss(CrossEntropyLoss())},
                                                       device=device)

    iteration_between_progress_bar_updates = 1
    progress_description_template = "Training - loss: {:.2f}"
    progress_bar = tqdm(
        initial=0, leave=False, total=len(training_dataset_loader),
        desc=progress_description_template.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer: Engine):
        iter = (trainer.state.iteration - 1) % len(training_dataset_loader) + 1
        if iter % iteration_between_progress_bar_updates == 0:
            progress_bar.desc = progress_description_template.format(trainer.state.output)
            progress_bar.update(iteration_between_progress_bar_updates)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer: Engine):
        progress_bar.refresh()
        validation_evaluator.run(validation_dataset_loader)
        metrics = validation_evaluator.state.metrics
        learning_rate_scheduler.step(metrics['accuracy'])
        print(f"\nValidation - Epoch: {trainer.state.epoch} - "
              f"Avg accuracy: {metrics['accuracy']:.2f} - "
              f"Avg loss: {metrics['cross-entropy']:.2f}")
        progress_bar.n = progress_bar.last_print_n = 0

    def score_function(evaluator: Engine):
        validation_accuracy = evaluator.state.metrics['accuracy']
        return validation_accuracy

    checkpoint_directory = "checkpoints"
    checkpoint_handler = ModelCheckpoint(checkpoint_directory, model_name, score_function=score_function)
    validation_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'mymodel': model})
    early_stopping_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    trainer.run(training_dataset_loader, max_epochs=50)
    progress_bar.close()

    print('Finished Training')

    print("Loading best model from check-point and testing...")
    best_model_name = os.listdir(checkpoint_directory)[0]
    best_model = torch.load(os.path.join(checkpoint_directory, best_model_name))  # type: Module
    testing_evaluator = create_supervised_evaluator(best_model, metrics={'accuracy': Accuracy()}, device=device)
    testing_evaluator.run(testing_dataset_loader)
    metrics = testing_evaluator.state.metrics
    print(f"\nTesting - Avg Accuracy {metrics['accuracy']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_name", type=str, default="squeezenet",
                        help="The model used for training the network. "
                             "Currently allowed values are \'simple\', \'vgg\', \'squeezenet\', \'mobilenetv2\'")
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
