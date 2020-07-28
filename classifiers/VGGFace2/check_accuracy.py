import torch
from facenet_pytorch import MTCNN

import pickle

from torchvision import transforms
from torchvision.datasets import ImageFolder
#from classifiers.VGGFace2.VGGFace2Classifier import VGGFace2Classifier
from classifiers.VGGFace2.inception_resnet import InceptionResnetV1

# This is for normal pretrained model
#model = VGGFace2Classifier()

# This is for our re-trained model
model = InceptionResnetV1(pretrained="vggface2", classify=True)


def collate_fn(x):
    return x[0]


path = "../../data/vggface2-80"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(path)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
batch_size = 1
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

max_iterations = len(loader)

num_samples = len(loader)
num_images = max_iterations * batch_size
print(f"there are {num_images} images")
test_correct = 0
model.eval()
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
              factor=0.709, post_process=True)

# For getting the index from the name
filename = 'name_to_idx.pkl'
with open(filename, 'rb') as file:
    name_to_idx = pickle.load(file)

no_face = []
count_images = 0
for i, (image, label) in enumerate(loader):
    #if i >= max_iterations:
    #    break
    #print(f"{i}/{max_iterations}")


    # This is a string of the form "n000012"
    gt_class_str = dataset.idx_to_class[label]
    gt_class = name_to_idx[gt_class_str]

    # Cropping the image:
    cropped, probability = mtcnn(image, return_prob=True)  # Squeeze, as mtcnn doesn't work with batches
    if cropped is not None:
        count_images += 1
        logits = model(cropped.unsqueeze(0))
        predicted_label = torch.argmax(logits, dim=1)
        num_correct_in_batch = (predicted_label == gt_class).sum()
        test_correct += num_correct_in_batch

    else:
        no_face.append((image, label))

# print accuracy of pretrained net

print(f"Correct: {test_correct}, total: {count_images}")
print(f"Test accuracy: {test_correct / float(count_images)}")
print(f"No face detected: {len(no_face)}")
