import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pathlib import Path 
import warnings
warnings.filterwarnings("ignore")



class NCSISTMNISTDataset(Dataset):
    """Kaggle dataset."""

    def __init__(self, root, transform=None, cache=None, resize_scale=None, dbtype='training'):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root        = root
        self.transform   = transform
        self.resize_scale= resize_scale
        self.dbtype      = dbtype
        self.ids         = []
        self.labels      = []
        print('root ' + root)
        
        for i in range(10):
            mnist_class_path = Path(root , dbtype, str(i))
            jpg_items = mnist_class_path.glob('*jpg')
            jpg_names = list(map(lambda x: x.name, jpg_items))
            self.ids.extend(sorted(jpg_names))
            self.labels.extend([i]*len(jpg_names))
            
        self.cache = cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            uid = self.ids[idx]
            label = self.labels[idx]
        except:
            raise IndexError()

        if self.cache is not None and uid in self.cache:
            sample = self.cache[uid]
        #self.ids.sort()

        else:

            img_name = Path(self.root , self.dbtype , str(label) , uid)
            image = Image.open(img_name)
            # ignore alpha channel if any, because they are constant in all training set
            #if image.mode != 'RGB' and image.mode != 'GRAY':
            #    image = image.convert('RGB')
            # resize image for model
            if self.resize_scale is not None:
                image = image.resize(self.resize_scale, Image.ANTIALIAS)
            # overlay masks to single mask
            # w, h = image.size

            sample = {'image': image,
                      'label': label}
            
            if self.cache is not None:
                self.cache[uid] = sample
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class Compose():
    def __init__(self, augment=False, padding=False, tensor=True):
        self.tensor = tensor
        self.augment = augment
        self.padding = padding


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Ignore skimage convertion warnings
        # perform ToTensor()
        if self.tensor:
            image = tx.to_tensor(image)
            label = label

        # prepare a shadow copy of composed data to avoid screwup cached data
        x = sample.copy()
        x['image'], x['label'] = image, label
        
        return x



if __name__ == '__main__':
    # over sampling testing
    dataset = NCSISTMNISTDataset('./data/MNIST_JPG',transform=Compose(), dbtype='training', resize_scale=None)
    train_loader = DataLoader(dataset, sampler=None,batch_size=7,shuffle=False, num_workers=2) 
    for i, data in enumerate(train_loader, 0):
        inputs = data['image']
        labels = data['label']
        break

    print(labels)
    #plt.imshow(inputs[0][0].cpu(), cmap='gray')


