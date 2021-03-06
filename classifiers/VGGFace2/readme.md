# VGGFace2 pretrained classifier

- VGGFace2 is a classifier
    - [More info about the dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/data_infor.html)
- We are using the pre-trained classifier from the `facenet-pytorch` python library

## Contents

### helper_scripts

Scripts that help with various aspects of VGGFace2

- create_idx_to_label_dict.py
    - Creates a dictionary that maps from the labels of the complete VGGFace2 dataset to uor VGGFace2-80 subset
- vggface2_80_dataset_script.py
    - Used to create the VGGFace2-80 subset from the complete VGGFace2 training set

### check_accuracy

Can be used to evaluate the pertrained / re-trained classifier on the complete dataset, it will print out the accuracy

### inception_resnet.py

Model definition copied from `facenet-pytorch` so we can use our own re-trained weights

### name_to_idx.pkl & vggface2_80_to_complete.pkl

Dictionaries generated by helper scripts checked in here for convenience

### VGGFace2Classifier.py

Wrapper for the `facenet-pytorch` vggface2 classifier used in some parts of the code