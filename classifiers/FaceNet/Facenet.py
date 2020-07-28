from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class FaceNet:
    """
        Instantiate InceptionResnet and simple classifier
        :param num_classes: number of identities/classes
        :param load_model: whether to load weights for 5-celeb
    """
    def __init__(self, num_classes, load_model=True):
        self.model_embedding = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
        if load_model:
            self.model_classifier = Net(num_classes=num_classes)
            self.model_classifier.load_state_dict(torch.load('weights/fiveceleb.t7'))
        else:
            self.model_classifier = Net(num_classes=num_classes)

    def train(self, X_train, y_train, num_steps, learning_rate):
        """
            Fit simple classifier to training data
            :param X_train: training data in embedding format
            :param y_train: training labels
            :param num_steps: number of optim steps in training
            :param learning_rate: learning rate for training
            :return: history of loss values during training
        """
        print('Training FaceNet on PubFig10')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_classifier.parameters(), lr=learning_rate)
        loss_history = []
        for i in range(0, num_steps):  # loop over the dataset multiple times
            optimizer.zero_grad()
            outputs = self.model_classifier(X_train)
            loss = criterion(outputs, y_train.long())
            loss.backward(retain_graph=True)
            loss_history.append(loss)
            # print('Loss: {}'.format(loss))
            optimizer.step()
        return loss_history

    def forward(self, input):
        self.model_classifier.eval()
        embedding = self.model_embedding(input)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-12, out=None)
        logits = self.model_classifier(embedding)
        
        return logits
    
    def to(self, device):
        self.model_embedding.to(device)
        self.model_classifier.to(device)
        
    def predict(self, input, log=False):
        """
            predict identity for input
            :param input: input images (have to be already cropped)
            :param log: whether to return the log-likelihood (for NLL computation)
            :return: probabilites for all classes
        """
        self.model_classifier.eval()
        embedding = self.model_embedding(input)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-12, out=None)
        logits = self.model_classifier(embedding)
        if log:
            probs = F.log_softmax(logits, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
        return probs


class Net(nn.Module):
    """
        Instantiate simple classifier to map embeddings to faces, as mentioned in
        https://arxiv.org/pdf/1801.00349.pdf
        :param num_classes: number of identities/classes
        :return: logits
    """
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.softmax(x)
        return x


def crop_image_single(img, device):
    """
        Implementation of the MTCNN network to crop single image to only show the face as shown in the facenet_pytorch doc:
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        :param device: pytorch device
        :param img: single image to be cropped
        :return: cropped image
    """
    model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=False, device=device)
    x_aligned, prob = model(img, return_prob=True)
    return x_aligned


def crop_images_batch(device, image_folder, transform=None):
    """
        Implementation of the MTCNN network to crop images to only show the face as shown in the facenet_pytorch doc:
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        :param device: pytorch device
        :param image_folder: path to images
        :return: cropped images, names of celebrities according to file structure
    """
    model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=False, device=device)

    dataset = datasets.ImageFolder(image_folder, transform=transform)        
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)
    aligned = None
    names = None
    for x, y in loader:
        print(x, y)
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


def encode_pubfig(y):
    """
        class encoding for pubfig
        :param y: labels as strings
        :return: labels as ints
    """
    y_out = []
    for x in y:
        if x == 'Aaron-Eckhart':
            y_out.append(0)
        elif x == 'Adriana-Lima':
            y_out.append(1)
        elif x == 'Angela-Merkel':
            y_out.append(2)
        elif x == 'Beyonce-Knowles':
            y_out.append(3)
        elif x == 'Brad-Pitt':
            y_out.append(4)
        elif x == 'Clive-Owen':
            y_out.append(5)
        elif x == 'Drew-Barrymore':
            y_out.append(6)
        elif x == 'Milla-Jovovich':
            y_out.append(7)
        elif x == 'Quincy-Jones':
            y_out.append(8)
        else:
            y_out.append(9)
    return y_out
