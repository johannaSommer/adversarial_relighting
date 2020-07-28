import os
from shutil import copy

path_new = "../../../data/vggface2-80"

path_old = "../../../datasets/vggface2/train"

# Create destination folder
if not os.path.exists(path_new):
    os.makedirs(path_new)

subfolders = [entry for entry in os.scandir(path_old) if entry.is_dir()][0:80]

for dir in subfolders:
    folder_name = dir.name
    copy_count = 0
    # Create new folder for identity
    identity_folder = os.path.join(path_new, folder_name)
    if not os.path.exists(identity_folder):
        os.makedirs(identity_folder)

    for entry in (os.scandir(dir)):
        if entry.is_file():
            copy(entry.path, os.path.join(path_new, folder_name, entry.name))
            copy_count += 1
            if copy_count >= 100:
                break

    if copy_count < 100:
        print(f"Identity '{folder_name}' does not have 100 images. It has {copy_count}")

print("Successfully finished copying!")
