from torchvision import transforms, datasets
from collections import defaultdict, deque
import itertools
# The modification of dataset is inspried from 
#   Author: Olof Harrysson
#   Type: source code
#   Web address: https://discuss.pytorch.org/t/creating-custom-dataset-from-inbuilt-pytorch-datasets-along-with-data-transformations/58270/2

class SampleRateCIFAR100(datasets.CIFAR100):
    def __init__(self,sampleRate, normal,path, transforms, train=True):
        super().__init__(path, train, download=True)
        # Number of images per class
        n_data_per_class = 500
        # The Normal Sample Rate
        self.sampleRate = sampleRate
        # If the dataset is for normal transformation use
        self.normal = normal
        # Transformations used for this dataset
        self.transforms = transforms
        # Number of normal samples in this dataset, given the normal sample rate
        self.n_images_per_class_normal = int(n_data_per_class*self.sampleRate)
        if self.normal:
            self.new2old_indices = self.create_idx_mapping(self.n_images_per_class_normal)
        else:
            self.n_images_per_class_aug = int(n_data_per_class*(1-self.sampleRate))
            self.new2old_indices = self.create_idx_mapping(self.n_images_per_class_aug)

    # Return the new index
    def create_idx_mapping(self,n_images_per_class):
        # Create the dictionary that have a max length of the number of images per class
        label2idx = defaultdict(lambda: deque(maxlen=n_images_per_class))
        # Add the key-value(label and index) pairs to the dictionary
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)
        # If we are making the augmentation dataset, replace the index of samples that exceed the
        # the highest normal sample index to the corresponding index by taking the modulo of the
        # number of images per class with the index
        if not self.normal:
            for label in label2idx.keys():
                for index in range(len(label2idx[label])) :
                    if index+1 > self.n_images_per_class_normal:
                        correct_index = (index+1 % self.n_images_per_class_normal)-1
                        label2idx[label][index] = label2idx[label][correct_index]
        # Get the dictionary mapping from new index to the corresponding old index 
        old_idxs = set(itertools.chain(*label2idx.values()))
        new2old_indices = {}
        for new_idx, old_idx in enumerate(old_idxs):
            new2old_indices[new_idx] = old_idx

        return new2old_indices

    # Overwrite the get length method
    def __len__(self):
        return len(self.new2old_indices)

    # Overwrite the get item method
    def __getitem__(self, index):
        index = self.new2old_indices[index]
        im, label = super().__getitem__(index)
        return self.transforms(im), label