import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import sys
import torch
import torchvision
import urllib
import zipfile
from torch.utils.data import DataLoader

from facenet_pytorch import MTCNN, InceptionResnetV1
from google_drive_downloader import GoogleDriveDownloader as gdd
from skimage.transform import resize

# Import relighters and classifiers.
from torch.optim import SGD

sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))

from classifiers.FaceNet.Facenet import FaceNet
from classifiers.FaceNet.Facenet import crop_images_batch

from relighters.DPR.model.defineHourglass_512_gray_skip import HourglassNet
from relighters.DPR.face_utils import plot_face_attack, get_sh
from relighters.DPR import spherical_harmonics 

from relighters.multi_illumination import multilum
from relighters.multi_illumination.relight import model
from relighters.multi_illumination.relight.model import Relighter

from utils import datasets
from utils import kornia_lab
from utils import labels_util
from utils import attack_transforms



def load_pretrained_classification_model(model_name):
    if model_name == 'alexnet':
        classif_model = torchvision.models.alexnet(pretrained=True)
        classif_model = classif_model.float().cuda().eval()
        
    elif model_name == 'resnet18':
        classif_model = torchvision.models.resnet18(pretrained=True)
        classif_model = classif_model.float().cuda().eval()
        
    elif model_name == 'pubfig_facenet':
        # Load the pretrained model here.
        # TODO(andreea): add the pretrained model on drive and download it here.
        classif_model = FaceNet(num_classes=10, load_model=False)
        classif_model.model_classifier.load_state_dict(torch.load('../models/facenet_model_pubfig10.pth'))
        classif_model.to('cuda')

    elif model_name == 'pubfig83_facenet':
        path = 'data/PubFig/embedding'
        embeddings_old = np.load('/home/jupyter/project-1/data/pubfig83/pubfig83_embedding.npz')['X_train']
        embeddings = []
        for em in embeddings_old:
            embeddings.append(em[0])
        labels = np.load('/home/jupyter/project-1/data/pubfig83/pubfig83_embedding.npz')['y_train']
        embeddings = torch.nn.functional.normalize(torch.Tensor(embeddings), p=2, dim=1, eps=1e-12, out=None)
        classif_model = FaceNet(num_classes=83, load_model=False)
        loss_history = classif_model.train(torch.Tensor(embeddings), torch.Tensor(labels), num_steps=100, learning_rate=0.1)

    elif model_name == 'resnet_indoor':
        # Download the pretrained indoor classification model from
        # Google Drive.
        drive_file_id = '16-y2qVAWCXyaLMGmb-sAvWaB8AOH4IF7'
        dir_name = '../models'
        checkpoint_name = 'mod_ep_16_acc_0.7938144207000732'
        checkpoint_name += '_bs_16_lr_0.001_gm_0.92_nl_1.pkl'
        zip_path = os.path.join(dir_name, 'resnet_indoor_checkpoint.zip')
        
        gdd.download_file_from_google_drive(file_id=drive_file_id,
                                    dest_path=zip_path,
                                    unzip=False)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir_name)

        checkpoint_path = os.path.join(dir_name, checkpoint_name)
        
        classif_model = torchvision.models.resnet18()
        classif_model.fc = torch.nn.Linear(512, 10)
        classif_model.load_state_dict(torch.load(checkpoint_path))
        classif_model = classif_model.float().cuda().eval()

    elif model_name == 'vggface2_pretrained':
        # Downloading of model weights is done automatically in the classifier constructor
        classif_model = InceptionResnetV1(pretrained='vggface2', classify=True)
        classif_model.to('cuda')
        classif_model.eval()

    else:
        raise NotImplementedError
        
    print('Loaded pretrained classifier: {}.'.format(model_name))

    return classif_model

    
def crop_images_batch_transformed(dataset, idxes, transform=None):
    """ TODO(andreea): add description
    """
    idx_to_class = labels_util.load_idx_to_label('pubfig10')
  
    aligned = None
    names = None
    
    for idx in idxes:
        x, y = dataset[idx]

        # No face is detected in some cases.
        if x is None:
            continue

        x_aligned = x.transpose((2, 0, 1))
        if x_aligned is not None:
            if x_aligned.max() > 1:
                x_aligned = x_aligned / 255
            if aligned is None and names is None:
                aligned = np.expand_dims(x_aligned, axis=0)
                names = idx_to_class[y]
            else:
                aligned = np.concatenate((aligned, np.expand_dims(x_aligned, axis=0)), axis=0)
                names = np.append(names, idx_to_class[y])
                
    return aligned, names


def train_pubfig_classification_model(transform):
    """ Train a PubFig10 classification model. You can pass a transform that is
    creating an adversarial attack in order to perform adversarial training or no
    transform in order to perform normal training on un-perturbed images.
    
    """
    device = 'cuda'
    path = '../data/pubfig/PubFig_og'
    
    dataset = datasets.PubFigDataset('../data/pubfig/', 
                                     mode='test', 
                                     crop=True, 
                                     crop_size=240, 
                                     transform=transform)
    batch_size = 16
    num_epochs = 20
    num_batches = len(dataset) // batch_size
    
    idx_to_label = labels_util.load_idx_to_label('pubfig10')
    label_to_idx = {label : idx for idx, label in idx_to_label.items()}

    classifier = FaceNet(num_classes=10, load_model=False)
    classifier.to('cuda')
    
    for _ in range(num_epochs):
        dataset.shuffle()
        
        for batch_idx in range(num_batches):
            # Cannot use a DataLoader because the Dataset may return None when the
            # cropper doesn't detect any face.
            idxes = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            crops, labels = crop_images_batch_transformed(dataset, idxes, transform)

            # Get the face embeddings; spare memory by not storing the gradients.
            with torch.no_grad():
                labels = list(map(lambda label: label_to_idx[label], labels))

                crops = torch.Tensor(crops).to(device)
                labels = torch.Tensor(labels).to(device)

                resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                embeddings = resnet(crops)
                embeddings = torch.nn.functional.normalize(embeddings,
                                                       p=2, dim=1, eps=1e-12, out=None)

            # Train the classifier on the current batch for one epoch.
            losshistory = classifier.train(embeddings,
                                           labels,
                                           num_steps=1, 
                                           learning_rate=0.1)
    
            # Save the trained model.
            torch.save(classifier.model_classifier.state_dict(), 
                      '../classifiers/FaceNet/adversarial_facenet_model_pubfig10.pth')
    
    return classifier


def adversarially_retrain_vggface2(attack_transform, config):
    device = 'cuda'
    path = "../data/vggface2-80/"

    learning_rate = 0.02

    batch_size = 16
    num_epochs = 20

    dataset = datasets.VGGFace2(path, transform=attack_transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    resnet = InceptionResnetV1(pretrained='vggface2', classify=True).to(device)
    for param in resnet.parameters():
        param.requires_grad = False
    # resnet.last_linear.weight.requires_grad = False
    resnet.logits.weight.requires_grad = True
    # We only want to update weights of the final classifier layer
    optim = SGD([resnet.logits.weight], lr=learning_rate)

    for i in range(num_epochs):
        print(f"Epoch {i}")

        for img, gt_label in loader:

            img = preprocess_classifier_input(img.numpy(), config)
            #img = torch.from_numpy(img).float().cuda()
            #img = img.permute(0, 2, 0, 1)[None, :]

            # Train the classifier on the current batch for one epoch.
            optim.zero_grad()

            loss_fn = torch.nn.CrossEntropyLoss()

            predictions = resnet(img)

            loss = loss_fn(predictions, gt_label.to(device))

            loss.backward()

            optim.step()

            # Manual weight updates
            # grad = resnet.logits.weight.grad
            # Do simple SGD optimization step
            # resnet.logits.weight -= learning_rate * grad

            # Save the trained model.
            torch.save(resnet.state_dict(),
                       '../classifiers/VGGFace2/adversarial_vggface2_model.pth')

    return resnet.eval()


def adversarial_training(model_name):
    if model_name == 'pubfig_facenet':
        # Define the config for obtaining the adversarial attack.
        config = {
            'dataset': 'pubfig10',   # not necessary here?
            'dataset_mode': 'test',  # not necessary here?
            'classif_model_name': 'pubfig_facenet',
            'classif_mode': 'normal_pretrained',
            'relight_model_name': 'dpr',

            'relight_checkpoint_path': '../relighters/DPR/trained_model/trained_model_03.t7',
            'learning_rate': 0.02, 
            'num_iterations': 5, 
            'epses': [0.1],
            'attack_type': 'class_constrained_eot',
            'targets': [1, 3],
            'debugging': False,
        }
        
        attack_ratio = 0.5 # Perform an attack only half of the time.
        attack_transform = attack_transforms.EOTAttackTransform(attack_ratio, config)
        return train_pubfig_classification_model(attack_transform)

    elif model_name == 'vggface2_pretrained':
        # Define the config for obtaining the adversarial attack.
        config = {
            'dataset': 'vggface2',  # not necessary here?
            'dataset_mode': 'test',  # not necessary here?
            'classif_model_name': 'vggface2_pretrained',
            'classif_mode': 'normal_pretrained',
            'relight_model_name': 'dpr',

            'relight_checkpoint_path': '../relighters/DPR/trained_model/trained_model_03.t7',
            'learning_rate': 0.02,
            'num_iterations': 5,
            'epses': [0.1],
            'attack_type': 'class_constrained_eot',
            'targets': [1, 3],
            'debugging': False,
        }

        attack_ratio = 0.5  # Perform an attack only half of the time.
        attack_transform = attack_transforms.EOTAttackTransform(attack_ratio, config)
        return adversarially_retrain_vggface2(attack_transform, config)

    # TODO: implement adversarial training for other models.
    else:
        raise NotImplementedError
        
        
def load_classification_model(model_name, mode):
    """Load a pretrained PyTorch classification model.
    
    model_name: string. For now only 'alexnet', 'resnet18', 'pubfig_facenet'
          and 'resnet_indoor' are supported.
    mode: string. Should be 'normal_pretrained' when loading the model pretrained
          on the original data and 'adversarial_train' for training the model
          adversarially and then loading it.
    """
    if mode == 'normal_pretrained':
        print('Loading pretrained model!')
        classif_model = load_pretrained_classification_model(model_name)
        
    elif mode == 'adversarial_train':
        print('Starting adversarial training!')
        classif_model = adversarial_training(model_name)
        
    elif mode == 'adversarial_pretrained':
        print('Loading adversarially pretrained model!')
        # TODO: add adversarially pre-trained model when available on drive.
        raise NotImplementedError
    
    else:
        raise NotImplementedError

    return classif_model


def load_relighting_model(model_name, checkpoint_path):
    """Load a relighting model.
    
    Currently only supporting the model from Murmann, Lukas, et al., 2019.
    """
    if model_name == 'multi_illumination_murmann':
        # The model is trained to expect input illumination 
        # 0, i.e. light form behind the camera
        src = [0]
        
        # In a single forward pass, we predict the scene 
        # under 8 novel light conditions.
        tgt = [5, 7, 12, 4, 16, 6, 17, 11]
        num_lights = len(tgt)

        relight_model = Relighter(**{
          'n_in': 1,
          'n_out': 8,
          'normals': False})
        relight_model.eval()
        try:
            relight_model.cuda()
        except:
            print('no cuda available')

        multilum.ensure_checkpoint_downloaded(checkpoint_path)
        chkpt = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        relight_model.load_state_dict(chkpt["model"])

    elif model_name == 'dpr':   
        relight_model = HourglassNet()
        relight_model.load_state_dict(torch.load(checkpoint_path))
        try:
            relight_model.train(False).cuda()
        except:
            relight_model.train(False)
        
    else:
        # TODO: load other relighters here.
        raise NotImplementedError

    print('Loaded the relighter: {}.'.format(model_name))
    
    return relight_model


def load_dataset(dataset_name, mode):
    if dataset_name == 'indoor_scenes' and mode == 'train':
        dataset = datasets.IndoorScenesDataset('../data/indoor-scenes/Train.csv', 
                                               '../data/indoor-scenes/')
    elif dataset_name == 'indoor_scenes' and mode == 'test':
        dataset = datasets.IndoorScenesDataset('../data/indoor-scenes/Test.csv', 
                                               '../data/indoor-scenes/')
   
    elif dataset_name == 'pubfig10':
        dataset = datasets.PubFigDataset('../data/pubfig/', mode=mode, crop=True)

    elif dataset_name == 'vggface2':
        # TODO: change location to fit with the others
        path = "../data/vggface2-80/"
        dataset = datasets.VGGFace2(path)
        # dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    elif dataset_name == 'pubfig83':
        dataset = datasets.PubFig83Dataset_test()

    else:
        raise NotImplementedError('No dataset with that name exists')

    return dataset
        
    
def read_image(img_name, size=(224, 224)):
    img = mpimg.imread(img_name)
    img = resize(img, size)
    
    # Convert grayscale image to RGB.
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
        
    # Convert RGBA image to RGB.
    if img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    
    return img
    

def preprocess_relight_input(img_cp, config):
    """ Prepare an image or a batch of images to be relit.
    
    img_cp: torch.Tensor of shape (H, W, 3) or (N, H, W, 3)
        representing an image or a batch of images with pixels
        having values in [0, 1]
    """
    
    # Work in the log space.
    if config['relight_model_name'] == 'multi_illumination_murmann':
        pc = np.percentile(img_cp.detach().cpu().numpy(), 90)
        img = torch.clamp(img_cp, 0, 1) / pc
        
        # Batch processing.
        if len(img.shape) == 4:
            img = img.float().permute(0, 3, 1, 2)
        
        # Single image processing.
        if len(img.shape) == 3:
            img = img.float().permute(2, 0, 1)[None, :]
       
        sample = {}
        sample["input"] = img

        in_ = sample["input"]
        in_ = torch.log(config['relighter_eps'] + in_)
        mean = in_.mean()

        return sample, mean
    
    elif config['relight_model_name'] == 'dpr':
        
        # Get one random spherical harmonics vector from a list of 
        # 6 hardcoded valid spherical harmonics.
        sh_np = spherical_harmonics.get_random_spherical_harmonics()
        try:
            sh = torch.from_numpy(sh_np).float().cuda()
        except:
            sh = torch.from_numpy(sh_np).float()

        if len(img_cp.shape) == 3:
            img_cp = img_cp[None, :]
            
        # l-space transformations, batch processing.
        img_transposed = img_cp.permute(0, 3, 1, 2)
        img_lab = kornia_lab.rgb_to_lab(img_transposed)
        input_l = img_lab[:, 0, :, :] / 100. # DPR expects values between 0 and 1
        input_l = input_l.view(input_l.shape[0], 1, input_l.shape[1], input_l.shape[2])
        input_l = input_l.float()
        
        input_ab = img_lab[:, 1:, :, :]
    
        return input_l, sh, input_ab

    else:
        # TODO: add preprocessing for other relighters here.
        raise NotImplementedError


def postprocess_relight_output(out, config):
    if config['relight_model_name'] == 'multi_illumination_murmann':
        # Go back from log space.
        bs, _, h, w = out.shape

        out[out < 0] = 0

        out = torch.log(out + config['relighter_eps'])
        out += 1
        out += config['mean']
        out = torch.exp(out) - config['relighter_eps']
        out = torch.clamp(out, 0, 1)
        out = torch.pow(out, config['gamma'])
        out = torch.clamp(out, 0, 1) 
        out = out.view(-1, 3, h, w)
        return out
    
    elif config['relight_model_name'] == 'dpr':
        # The output of the relighter has always 4 dimensions, so 
        # we always do batch processing.
        out_l_scaled = out * 100.
        output_lab = torch.cat([out_l_scaled.float(), config['input_ab'].float()], dim=1)
        output_rgb = kornia_lab.lab_to_rgb(output_lab)
            
        return output_rgb
            
    else:
        # TODO: add relighting post-processing here.
        raise NotImplementedError

    
def normalize_classifier_input(imgs, config):
    """
    @param imgs: torch.Tensor of shape [N, 3, H, W],
        where N is the batch size.
    """
    
    # Customized normalization mean and std depending on
    # the dataset we are working with.
    if config['classif_model_name'] in ['alexnet', 'resnet18']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif config['classif_model_name'] == 'resnet_indoor':
        mean = [0.48196751, 0.42010041, 0.36075131]
        std = [0.23747521, 0.23287786, 0.22839358]
    elif config['classif_model_name'] == 'pubfig_facenet' or config['classif_model_name'] == 'pubfig83_facenet':
        mean = [0.5, 0.5, 0.5]
        std = [0.50196078, 0.50196078, 0.50196078]
    elif config['classif_model_name'] == 'vggface2_pretrained':
        mean = [127.5, 127.5, 127.5]
        std = [128.0, 128.0, 128.0]
    else:
        raise NotImplementedError

    mean = torch.from_numpy(np.array(mean))
    std = torch.from_numpy(np.array(std))
    
    for i in range(imgs.shape[0]):
        if config['classif_model_name'] == 'vggface2_pretrained':
            # Because vggface2 classifier expects image range [0, 255] we have to scale up _before_ we normalize
            img = imgs[i] * 255
        else:
            img = imgs[i]
        torchvision.transforms.Normalize(mean, std, inplace=True)(img)
            
    return imgs


def preprocess_classifier_input(img, config):
    """Process a raw image for feeding it into a classifier.
    
    @param img: np.array of shape [H, W, 3] representing
        the image to be classified.
    """
    
    # Preprocess a batch of images.
    if len(img.shape) == 4:
        ans = img.transpose((0, 3, 1, 2))
        try:
            ans = torch.from_numpy(ans).float().cuda()
        except:
            ans = torch.from_numpy(ans).float()
    
    # Preprocess just one image.
    if len(img.shape) == 3:
        ans = img.transpose((2, 0, 1))[None, :]
        try:
            ans = torch.from_numpy(ans).float().cuda()
        except:
            ans = torch.from_numpy(ans).float()
    
    return normalize_classifier_input(ans, config)

    
def barplot_probabilities(ax, probs, labels):
    ax.set_xticks(np.arange(len(probs)))
    ax.set_xticklabels(labels)
    ax.tick_params(rotation=80)
    ax.set_ylim([0, 1])
    ax.bar(np.arange(len(probs)), probs, 0.35, color='b')
    
    
def visualize_attack(img, result, idx_to_label):
    """Print the original image, along with the root image,
    adversarial attack, the image difference and the loss history.
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    fig.tight_layout()
    
    sorted_labels = []
    for i, key in enumerate(idx_to_label):
        sorted_labels.append(str(idx_to_label[key]) + ' ' + str(i))
        
    axes[0, 0].imshow(img)
    axes[0, 1].imshow(np.clip(result['root_img'], 0, 1))
    axes[0, 2].imshow(np.clip(result['adv_img'], 0, 1))
    axes[0, 3].imshow(np.clip(result['diff_img'], 0, 1))
    
    if len(result['orig_probs']) > 10:
        orig_probs = torch.topk(torch.Tensor(result['orig_probs']), 10)[0].tolist()
        sorted_labels_og = [idx_to_label[x] for x in torch.topk(torch.Tensor(result['orig_probs']), 10)[1].tolist()]
        root_probs = torch.topk(torch.Tensor(result['root_probs']), 10)[0].tolist()
        sorted_labels_ro = [idx_to_label[x] for x in torch.topk(torch.Tensor(result['root_probs']), 10)[1].tolist()]
        adv_probs =  torch.topk(torch.Tensor(result['adv_probs']), 10)[0].tolist()
        sorted_labels_ad = [idx_to_label[x] for x in torch.topk(torch.Tensor(result['adv_probs']), 10)[1].tolist()]

    else:
        orig_probs = result['orig_probs']
        sorted_labels_og = sorted_labels
        root_probs = result['root_probs']
        sorted_labels_ro = sorted_labels
        adv_probs =  result['adv_probs']
        sorted_labels_ad = sorted_labels

    barplot_probabilities(axes[1, 0], orig_probs, sorted_labels_og)
    barplot_probabilities(axes[1, 1], root_probs, sorted_labels_ro)
    barplot_probabilities(axes[1, 2], adv_probs, sorted_labels_ad)


    axes[0, 0].set_title('Original, %s (%d)' % 
                      (str(idx_to_label[result['orig_label']])[:20],
                       result['orig_label']))
    axes[0, 1].set_title('Root, %s (%d)' %
                      (str(idx_to_label[result['root_label']])[:20],
                       result['root_label']))
    axes[0, 2].set_title('Adversarial, %s (%d)' % 
                      (str(idx_to_label[result['adv_label']])[:20],
                       result['adv_label']))
    axes[0, 3].set_title('Image difference')
    
    if 'loss_hist' in result:
        axes[1, 3].plot(result['loss_hist'])
        axes[1, 3].set_title('Loss history')
    
    plt.show()

