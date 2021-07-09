from pytorch_metric_learning import testers
from torchvision import transforms
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchvision import transforms, datasets
import CIFAR10
import CIFAR100
import FashionMNIST
import numpy as np

# Transformations for different uses
normal_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023))
    ])
first_augmented_train_transform = transforms.Compose([
        transforms.transforms.RandomHorizontalFlip(p=1),
        transforms.transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
        transforms.transforms.RandomAffine(degrees = 10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023))
    ])
second_augmented_train_transform = transforms.Compose([
        transforms.transforms.RandomVerticalFlip(p=0.5),
        transforms.transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0)),
        transforms.transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023))
    ])
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023))
    ])
# The dictionary stores the corresponding augmentation straregies
transfrom_dic = {0:first_augmented_train_transform,
                1:second_augmented_train_transform}
# The dictionary stores the corresponding parameters of each dataset
param_dic = {"CIFAR10":[32,32,3],
                "CIFAR100":[32,32,3],
                "FashionMNIST":[28,28,1]}

# Function from pytorch-metric-learning to the embeddings
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

# Train the model witin one epoch and returns the corresponding learning information
def train(model, loss_func, mining_func, train_loader, optimizer, epoch, device):
    # The information recorded from one epoch
    epoch_data = list()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)#(batch size, embedding_size)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        # The information recorded from one iteration
        iteration_data=[batch_idx, loss.cpu().data.numpy().tolist(), mining_func.num_triplets]
        epoch_data.append(iteration_data)
        if batch_idx % 60 == 0:
            iteration_data = [batch_idx, loss, mining_func.num_triplets]
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))    
    return epoch_data

# The test function
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    # Compute accuracy using AccuracyCalculator from pytorch-metric-learning
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                np.squeeze(test_labels),
                                                np.squeeze(train_labels),
                                                True)
    print(accuracies.keys())
    print(accuracies.values())
    return accuracies.values()

# Return the corresponding dataset based on the parameters
# The Normal Sample Rate is applied here to prodcue a combined dataset
def get_dataset(dataset_name,normal_sample_rate,train,trans_index):
    if train:
        augmented_train_transform = transfrom_dic[trans_index]
        if normal_sample_rate is not 1:
            if dataset_name is "CIFAR10":
                part_normal_dataset = CIFAR10.SampleRateCIFAR10(path = "./data",transforms = normal_train_transform,sampleRate=normal_sample_rate,normal = True)
                part_augmented_dataset = CIFAR10.SampleRateCIFAR10(path = "./data",transforms = augmented_train_transform,sampleRate=normal_sample_rate,normal = False)
            if dataset_name is "CIFAR100":
                part_normal_dataset = CIFAR100.SampleRateCIFAR100(path = "./data",transforms = normal_train_transform,sampleRate=normal_sample_rate,normal = True)
                part_augmented_dataset = CIFAR100.SampleRateCIFAR100(path = "./data",transforms = augmented_train_transform,sampleRate=normal_sample_rate,normal = False)
            if dataset_name is "FashionMNIST":
                part_normal_dataset = FashionMNIST.SampleRateFashionMNIST(path = "./data",transforms = normal_train_transform,sampleRate=normal_sample_rate,normal = True)
                part_augmented_dataset = FashionMNIST.SampleRateFashionMNIST(path = "./data",transforms = augmented_train_transform,sampleRate=normal_sample_rate,normal = False)
            return torch.utils.data.ConcatDataset([part_normal_dataset,part_augmented_dataset])
        else:
            if dataset_name is "CIFAR10":
                return datasets.CIFAR10("./data",train=True,transform=normal_train_transform,download=True)
            if dataset_name is "CIFAR100":
                return datasets.CIFAR100("./data",train=True,transform=normal_train_transform,download=True)
            if dataset_name is "FashionMNIST":
                return datasets.FashionMNIST("./data",train=True,transform=normal_train_transform,download=True)
    else:
        if dataset_name is "CIFAR10":
            return datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
        if dataset_name is "CIFAR100":
            return datasets.CIFAR100('./data', train=False, transform=test_transform, download=True)
        if dataset_name is "FashionMNIST":
            return datasets.FashionMNIST('./data', train=False, transform=test_transform, download=True)

# Return the corresponding parameter of each dataset        
def get_param(data_name):
    return param_dic[data_name]
