import argparse
import datetime
import os
import shutil
import torch
from time import time

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

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


def train_model(dataset_directory: str,
                model_name: str,
                delete_and_recreate_dataset_directory: bool):
    print("Downloading and extracting datasets...")

    if delete_and_recreate_dataset_directory:
        delete_dataset_directory(dataset_directory)
        download_datasets(dataset_directory)
        dataset_splitter = DatasetSplitter(dataset_directory, dataset_directory)
        dataset_splitter.split_images_into_training_validation_and_test_set()

    print("Training on dataset...")
    start_time = time()

    data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    minibatch_size = 1
    music_scores_dataset = ImageFolder(root=os.path.join(dataset_directory, "training"), transform=data_transform, )
    training_dataset_loader = DataLoader(music_scores_dataset, batch_size=minibatch_size, shuffle=True, num_workers=4)

    network = SimpleNetwork()
    print(network)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summary(network.to(device), (3, 128, 128))

    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adadelta(network.parameters())

    for epoch in range(10):  # loop over the dataset multiple times
        for i, data in enumerate(training_dataset_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients before computing the next batch
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, (i + 1) * minibatch_size, loss.item()))

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
