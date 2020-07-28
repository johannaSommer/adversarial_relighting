# Adapted from: https://github.com/ashrutkumar/Indoor-scene-recognition

import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset,SubsetRandomSampler,Sampler
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
import glob
from skimage import io, transform
from PIL import Image
import random
import PIL.ImageEnhance as ie
import copy
from torch.autograd import Variable
import PIL.Image as im
from math import floor

torch.cuda.empty_cache() 

seed = 249
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ImageDataset(Dataset): 
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame) - 1

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         
        image = Image.open(img_name).convert('RGB')                               
        label = np.array(self.data_frame['Category'][idx])                        
        if self.transform:            
            image = self.transform(image)                                         
        sample = (image, label)        
        return sample
        
class SubsetSampler(Sampler):
     
    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return self.num_samples
        
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

class RandomFlip(object):
    """Randomly flips the given PIL.Image with a probability of 0.25 horizontal,
                                                                0.25 vertical,
                                                                0.5 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM)
        }
    
        return dispatcher[random.randint(0,3)] #randint is inclusive

class RandomRotate(object):
    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,
                                                                 1/6 180°,
                                                                 1/6 270°,
                                                                 1/2 as is
    """
    
    def __call__(self, img):
        dispatcher = {
            0: img,
            1: img,
            2: img,            
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270)
        }
    
        return dispatcher[random.randint(0,5)] #randint is inclusive
    
class PILColorBalance(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)

class PILContrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)

class PILSharpness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)
    


class PowerPIL(RandomOrder):
    def __init__(self, rotate=True,
                       flip=True,
                       colorbalance=0.4,
                       contrast=0.4,
                       brightness=0.4,
                       sharpness=0.4):
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))
            
def train_valid_split(dataset, test_size = 0.25, shuffle = False, random_seed = 0):
    length = dataset.__len__()
    indices = list(range(1,length))
    
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    
    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]
    

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
indoor_mean = [0.48196751, 0.42010041, 0.36075131]
indoor_std = [0.23747521, 0.23287786, 0.22839358]

transform_augmented = transforms.Compose([
        transforms.RandomResizedCrop(224),
        PowerPIL(),
        transforms.ToTensor(),
        transforms.Normalize(indoor_mean, indoor_std)])

transform_raw = transforms.Compose([
                     transforms.Resize((224,224)),
                     transforms.ToTensor(),
                     transforms.Normalize(indoor_mean, indoor_std)])

test_files = set(open('IndoorTestImages.txt', 'r').read().split())

dirs = os.listdir('Images')
dirs.sort()

total_train = 0
class_freq = {}
labeldict = {label:idx for idx, label in enumerate(dirs)}
train_csvlist, test_csvlist = [], []

for dir_name in dirs:
    dir_path = os.path.join('Images', dir_name)
    images = os.listdir(dir_path)
    class_freq[dir_name] = 0
    for image in images:
        image_path = os.path.join(dir_name, image)
        if image_path not in test_files:
            train_csvlist.append([labeldict[dir_name], image_path])
            class_freq[dir_name] += 1
        else:
            test_csvlist.append([labeldict[dir_name], image_path])
        total_train += 1
        
train_df = pd.DataFrame(train_csvlist, columns=['Category', 'Id'])
test_df = pd.DataFrame(test_csvlist, columns=['Category', 'Id'])
train_df.to_csv('Train.csv')
test_df.to_csv('Test.csv')

trainset = ImageDataset(csv_file = 'Train.csv', root_dir = './Images', transform=transform_augmented)
valset   = ImageDataset(csv_file = 'Train.csv', root_dir = './Images', transform=transform_raw)
accset   = ImageDataset(csv_file = 'Test.csv', root_dir = './Images', transform=transform_raw)

train_idx, valid_idx = train_valid_split(trainset, 0.25, shuffle=True)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetSampler(valid_idx)

valid_loader = DataLoader(valset,batch_size=200, sampler=valid_sampler, num_workers=1,)
acc_loader   = DataLoader(accset,batch_size=200, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

best_acc = 0.0

for batch_size in [16, 32, 64, 128, 256]:
    for lr in np.linspace(0.004, 0.01, 7):
        for gamma in [0.95, 9.98]:
            for num_layers in [1, 2]:
                print('Batch size {}, lr {}, gamma {}, num_layers {}'.format(
                    batch_size, lr, gamma, num_layers))
                print('-' * 40)
                train_loader = DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          sampler=train_sampler,
                                          num_workers=1,)    

                model = torchvision.models.resnet18(pretrained=True)
                for param in model.parameters():
                    param.requires_grad = False

                if num_layers == 1:
                    model.fc = nn.Linear(512, 67)
                elif num_layers == 2:
                    model.fc = nn.Sequential(
                        nn.Dropout(p=0.3),
                        nn.Linear(512, 256),
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 67),
                    )
                                             
                model.fc.requires_grad = True
                model.to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.92)

                num_epochs = 10
                for epoch in range(num_epochs):    
                    print("\t Epoch  : "+str(epoch))
                    print("\t" + "-"*10)

                    #training loop
                    running_loss = 0.0
                    running_corrects=0
                    correct=0
                    wrong=0
                    model.train()
                    for inp,labels in train_loader:
                        inp=inp.to(device)
                        labels=labels.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            outputs = model(inp)            
                            loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()     

                        _, preds = torch.max(outputs.data, 1)
                        correct += torch.sum(preds == labels)
                        wrong += torch.sum(preds != labels)


                    epoch_loss = running_loss / (len(train_loader)*1.0)
                    acc = (correct.float()) / ((correct+wrong).float())
                    print('\t TRAINING SET   Loss: {} Acc: {}'.format(epoch_loss, acc))

                    # validation loop
                    if True:
                        correct=0
                        wrong=0
                        model.eval()
                        for inp,labels in valid_loader:
                            inp=inp.to(device)
                            labels=labels.to(device)
                            optimizer.zero_grad()
                            with torch.no_grad():
                                outputs = model(inp)
                            _, preds = torch.max(outputs.data, 1)
                            correct += torch.sum(preds == labels)
                            wrong += torch.sum(preds != labels)

                        acc = (correct.float()) / ((correct+wrong).float())
                        print('\t VALIDATION SET Correct: {} Wrong {} Acc {}'.format(correct,wrong,acc))
                        if acc > best_acc:
                            best_acc = acc     
                            torch.save( model.state_dict(), 
                                       "ep_{}_acc_{}_bs_{}_lr_{}_gm_{}_nl_{}.pkl".format(
                                           epoch, acc, batch_size, lr, gamma, num_layers))

                    running_loss = 0.0
                    running_correct = 0

                    my_lr_scheduler.step()
                    print('Learning rate', get_lr(optimizer))

                print('------     Finished Training    -----')
                
