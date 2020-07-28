import numpy as np
import os
import pandas as pd
import sys
import torch
import zipfile

from facenet_pytorch import MTCNN
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image
from torchvision.datasets import ImageFolder

sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))
from utils import labels_util
from classifiers.FaceNet.Facenet import crop_images_batch

import torchvision.transforms as transforms


def download(data_path, zip_name, drive_file_id):
    zip_path = os.path.join(data_path, zip_name)
    gdd.download_file_from_google_drive(file_id=drive_file_id,
                                dest_path=zip_path,
                                unzip=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    
class IndoorScenesDataset(torch.utils.data.Dataset):
    """Minimal dataset functionality for the Indoor Classification data.
    
    Example usage (in a notebook in /experiments):
    
    ds = IndoorScenesDataset('../data/indoor-scenes/Train.csv', 
                             '../data/indoor-scenes/')
                             
    ds = IndoorScenesDataset('../data/indoor-scenes/Test.csv', 
                             '../data/indoor-scenes/')                         
    """
    def __init__(self, csv_filename, data_path):   
        self.data_path = data_path
        
        # Download the data from Google Drive if not already
        # available.
        if not os.path.exists(data_path):
            zip_name =  'indoor-scenes.zip'
            drive_file_id = '19sajDHxP1YNs9IcvUJdgCI9nOyJeE8Up'
            download(data_path, zip_name, drive_file_id)

        self.df = pd.read_csv(csv_filename)

    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, idx):
        from attacks import utils # This import sometimes causes problems, so we only import it here
        im_name = self.df['Id'][idx]
        im_path = os.path.join(self.data_path, 'Images', im_name)
        img = utils.read_image(im_path) 
        
        gt_label = self.df['Category'][idx]
        
        return img, gt_label
    

class PubFigDataset(torch.utils.data.Dataset):
    """ Reduced PubFig Dataset with 10 manually selected classes.
    
    Example usage (for a notebook in /experiments):
    
    ds = PubFigDataset('../data/pubfig/', mode='train') 
    ds = PubFigDataset('../data/pubfig/', mode='test') 
    """
    def __init__(self, data_path, mode, crop=False, transform=None, crop_size=240):
        assert mode in ['train', 'test']
        
        # Download the data from Google Drive if not already
        # available.
        if not os.path.exists(data_path):
            zip_name = 'pubfig.zip'
            drive_file_id = '1hukredXUXnSNQcOjohk7INHcFnCy2KTb'
            download(data_path, zip_name, drive_file_id)
    
        idx_to_label = labels_util.load_idx_to_label('pubfig10')
        label_to_idx = {label : idx for idx, label in idx_to_label.items()}
        
        # Store the data in a list of (image path, label)
        self.data = []
        self.crop = crop
        self.crop_size = crop_size
        self.transform = transform
        
        if mode == 'train':
            data_path = os.path.join(data_path, 'PubFig_og')
        elif mode == 'test':
            data_path = os.path.join(data_path, 'testimgs')
            
        for celeb_name in os.listdir(data_path):
            dir_path = os.path.join(data_path, celeb_name)
            
            if not os.path.isdir(dir_path):
                continue
                
            if celeb_name not in label_to_idx:
                continue
                          
            gt_label = label_to_idx[celeb_name]
            for celeb_img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, celeb_img_name)
                self.data.append((img_path, gt_label))
                
        if crop:
            self.cropper = MTCNN(image_size=self.crop_size, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=False, device='cuda')

    def shuffle(self):
        np.random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        im_path = self.data[idx][0]
        gt_label = self.data[idx][1]
        
        if not self.crop:
            img = utils.read_image(im_path) 
      
        else:
            # PIL image has range 0...255. This is what the
            # cropper expects.
            img = np.array(Image.open(im_path))
            img, probs = self.cropper(img, return_prob=True)
            
            if img is None:
                return None, None
  
            img = img.detach().cpu().numpy().transpose((1, 2, 0))          
            img = img / 255.0
            
        # Add a transform (eg: EOTAttackTransform).
        if self.transform is not None:
            img = self.transform(img, gt_label)
            
        return img, gt_label
    
class PubFig83Dataset_test(torch.utils.data.Dataset):
    """ PubFig dataset with 83 identities
    """
    def __init__(self):
        crops = np.load('/home/jupyter/project-1/data/pubfig83/pubfig83_crop_test.npz')['data']
        self.crops_perm = []
        for crop in crops:
            self.crops_perm.append(torch.Tensor(crop).permute(1, 2, 0).numpy())
        self.label_names = np.load('/home/jupyter/project-1/data/pubfig83/pubfig83_crop_test.npz')['labels']    
        
    def __len__(self):
        return len(self.crops_perm)
    
    def __getitem__(self, idx):
        img = self.crops_perm[idx]
        gt_label = self.label_names[idx]
        return img, gt_label


class VGGFace2(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None, image_size_for_crop=224):
        self.data = ImageFolder(data_path)
        self.image_size_for_crop = image_size_for_crop
        self.cropper = MTCNN(image_size=image_size_for_crop, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                             factor=0.709, post_process=False, device='cuda')

        complete_to_subset = labels_util.load_idx_to_label('vggface2')
        self.subset_to_complete = {value: key for key, value in complete_to_subset.items()}

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        cropped = self.cropper(image)

        if cropped is None:
            # Previously this returned None if the cropper could not detect a face. Now we just do a random crop of the
            # right size. The advantage is, that this works with DataLoaders, whereas the DataLoaders  throw an
            # exception if None is returned. The result should otherwise be the same, as it is unlikely that the
            # classifier will classifiy a randomly cropped image correctly. If it does by chance, this is fine
            #return None, None
            img = transforms.RandomCrop(self.image_size_for_crop, pad_if_needed=True, padding_mode='edge')(image)
            cropped = transforms.ToTensor()(img)

        img = cropped.numpy().transpose(1, 2, 0) / 255

        # Add a transform (eg: EOTAttackTransform).
        if self.transform is not None:
            img = self.transform(img, label)

        return img.astype(np.float64), self.subset_to_complete[label]
