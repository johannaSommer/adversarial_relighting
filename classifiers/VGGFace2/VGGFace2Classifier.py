from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class VGGFace2Classifier(nn.Module):
    """
        Instantiate InceptionResnet and simple classifier
        :param num_classes: number of identities/classes
        :param load_model: whether to load weights for 5-celeb
    """

    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

    def forward(self, input):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #cropped = crop_image_single(input, device)
        logits = self.model(input)
        return logits

    #def to(self, device, **kwargs):
    #   self.model.to(device)


def crop_image_single(img, device):
    """
        Implementation of the MTCNN network to crop single image to only show the face as shown in the
        facenet_pytorch doc:
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        :param device: pytorch device
        :param img: single image to be cropped
        :return: cropped image
    """
    model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=False, device=device)
    x_aligned = model(img)
    return x_aligned


def crop_images_batch(device, image_folder):
    """
        Implementation of the MTCNN network to crop images to only show the face as shown in the facenet_pytorch doc:
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        :param device: pytorch device
        :param image_folder: path to images
        :return: cropped images, names of celebrities according to file structure
    """
    model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=False, device=device)

    dataset = datasets.ImageFolder(image_folder)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)
    aligned = None
    names = None
    for x, y in loader:
        x_aligned, prob = model(x, return_prob=True)
        if x_aligned is not None:
            x_aligned = x_aligned / 255
            if aligned is None and names is None:
                aligned = np.expand_dims(x_aligned, axis=0)
                names = dataset.idx_to_class[y]
            else:
                aligned = np.concatenate((aligned, np.expand_dims(x_aligned, axis=0)), axis=0)
                names = np.append(names, dataset.idx_to_class[y])
    return aligned, names


def collate_fn(x):
    return x[0]
