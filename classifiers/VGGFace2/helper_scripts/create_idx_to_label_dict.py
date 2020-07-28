import os
import pickle

#####
# Create dictionary that maps from folder_name (i.e. label in the original dataset) to
# labels of the pretrained model (i.e. labels the pretrained model was trained on

path = "../../../datasets/vggface2/train"

# Create destination folder
if not os.path.exists(path):
    os.makedirs(path)

identity_names = sorted([entry.name for entry in os.scandir(path) if entry.is_dir()])

name_to_idx = {name: i for i, name in enumerate(identity_names)}

filename = '../name_to_idx.pkl'
with open(filename, 'wb') as file:
    pickle.dump(name_to_idx, file)

#####
# Create dictionary that maps from vggface2-80 label (i.e. the labels that are assigned
# to the different classes by the ImageFolderDataset, when used with the vggface2-80 dataset)
# to the labels used by the pretrained model

path_80 = "../../../data/vggface2-80"

identity_names = sorted([entry.name for entry in os.scandir(path_80) if entry.is_dir()])

complete_to_subset = {name_to_idx[name]: i for i, name in enumerate(identity_names)}

# Because it is possible that the prediction ends up being something that is not in our subset
# we fill the wholes with the value -1, which just stands for all the indices that are not
# in our subset
for i in range(90_000):
    if i not in complete_to_subset:
        complete_to_subset[i] = -1

filename = '../vggface2_80_to_complete.pkl'
with open(filename, 'wb') as file:
    pickle.dump(complete_to_subset, file)
