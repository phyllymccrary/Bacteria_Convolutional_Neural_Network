import os
from os.path import join

import fnmatch

import Augmentor
from PIL import Image
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook

Image.MAX_IMAGE_PIXELS = 1000000000


# Image Preprocessing - Organization and Augmenting

def create_strain_directories(base_directory, strain_ids):
    for strain in strain_ids:
        try:
            strain_path = os.path.join(base_directory, str(strain))
            os.mkdir(strain_path)
            print("Directory Created:", strain_path)
        except FileExistsError as e:
            pass


# Moves the images into the directory
def move_images(base_directory, strain_ids):
    for strain in strain_ids:
        for file in os.listdir(base_directory):
            strain_img = "PIL-{}_*.jpg".format(int(strain))
            if fnmatch.fnmatch(file, strain_img):
                image_path = os.path.join(base_directory, file)
                mv_dir = os.path.join(base_directory, str(strain))
                mv_path = os.path.join(mv_dir, file)
                if os.path.exists(image_path):
                    os.rename(image_path, mv_path)


# Clean Up: Deletes Empty Directories
def delete_empty_directories(base_directory):
    directories = [x[0] for x in os.walk(base_directory)]
    for directory in directories:
        try:
            os.rmdir(directory)
        except OSError as e:
            pass


# Organizes the Images into their respective directories
def organize_images(base_directory, dataset):
    strain_ids = dataset["strain"]
    create_strain_directories(base_directory, strain_ids)
    move_images(base_directory, strain_ids)
    delete_empty_directories(base_directory)


# Gets all the strain directories
def get_strain_directories(base_directory):
    strain_dirs = []
    for root, dirs, files in os.walk(base_directory, topdown=False):
        for name in dirs:
            dir_path = os.path.join(base_directory, name)
            if os.path.isdir(dir_path):
                strain_dirs.append(dir_path)
    return strain_dirs


# Checks if the path is a parent to ensure we're saving the augmented images in their correct directory
def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])


# Converts to grayscale
def to_grayscale(strain_directory, output_dir):
    try:
        if not path_is_parent(strain_directory, output_dir):
            print("Not the correct subdirectory")
            return

        dirs = os.listdir(strain_directory)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for item in dirs:
            if ".jpg" in item:
                img_file = os.path.join(strain_directory, item)
                if os.path.isfile(img_file):
                    file_name, extension = os.path.splitext(img_file)
                    file_name = file_name + "_gray.jpg"
                    new_file_location = join(output_dir, item)

                    img = Image.open(img_file)
                    img_gray = img.convert('L')
                    img_gray.save(file_name)

                    os.rename(file_name, new_file_location)  # moves file to new location
    except Exception as e:
        print("Issue with directory in to_grayscale() method:", strain_directory)
        print(e)


# Resizes Images
def resize(strain_directory, output_dir):
    try:
        if not path_is_parent(strain_directory, output_dir):
            print("Not the correct subdirectory")
            return

        dirs = os.listdir(strain_directory)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for item in dirs:
            if ".jpg" in item:
                img_file = os.path.join(strain_directory, item)
                if os.path.isfile(img_file):
                    file_name, extension = os.path.splitext(img_file)
                    file_name = file_name + "_resized.jpg"
                    new_file_location = join(output_dir, item)

                    img = Image.open(img_file)
                    file_name, extension = os.path.splitext(img_file)
                    imResize = img.resize((28, 28), Image.ANTIALIAS)
                    imResize.save(file_name, 'JPEG', quality=90)

                    os.rename(file_name, new_file_location)  # moves file to new location
    except Exception as e:
        print("Issue with directory in resize() method:", strain_directory)
        print(e)


# Augments the images
def augment_strain_images(strain_dir, sample_size=20):
    try:
        p = Augmentor.Pipeline(strain_dir)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.resize(probability=1.0, width=28, height=28)
        p.sample(sample_size, multi_threaded=False)
    except Exception as e:
        print("Issue with directory in augment_strain_images() method:", strain_dir)
        print(e)


# Processes the entire images dataset
def process_images(strain_directories, gray_strain_directories, output_strain_directories):
    # Converts original images to grayscale
    for i in tqdm(range(len(strain_directories))):
        to_grayscale(strain_directories[i], gray_strain_directories[i])

    # Augments the gray images
    for image_dir in (gray_strain_directories):
        augment_strain_images(image_dir)

    # (Just for completeness) Resizes the original grayscale images without augmentations
    for i in tqdm(range(len(output_strain_directories))):
        resize(gray_strain_directories[i], output_strain_directories[i])


def split_path(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


# Create excel sheet with strain id , image location pairs
def get_strain_images_as_df(strain_directories):
    strains = []
    try:
        for directory in strain_directories:
            strain_id = split_path(directory)[-3]

            for item in os.listdir(directory):
                if ".jpg" in item:
                    image_path = os.path.join(directory, item)
                    if os.path.exists(image_path):
                        strain_dict = {"strain": strain_id, "strain_image": image_path}
                        strains.append(strain_dict)
                    else:
                        print("Doesn't exist:", image_path)
    except FileNotFoundError:
        pass
    return pd.DataFrame(strains)


def save_to_excel(excel_file, sheet_name, df):
    book = load_workbook(excel_file)
    writer = pd.ExcelWriter(excel_file)
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name, index=False)
    writer.save()


# Joins datasets
def merge_datasets(df_1, df_2):
    info = []
    for strain, row in df_1.iterrows():
        try:
            if '0' in strain[0]:
                strain_id = pd.Series(strain, index=["strain_id"])
                info.append(df_2.loc[strain].append(row).append(strain_id))
            else:
                strain_id = pd.Series(strain, index=["strain_id"])
                info.append(df_2.loc[int(strain)].append(row).append(strain_id))
        except Exception as e:
            print("error with strain", strain, ":", e)

    return pd.DataFrame(info)


base_directory = "Enter path location for directory where the images are stored"
bacteria_workbook = "Enter path location for excel file"

total_data = pd.read_excel(bacteria_workbook, dtype=object, sheet_name="Total Database")

organize_images(base_directory, total_data)

strain_directories = get_strain_directories(base_directory=base_directory)
gray_strain_directories = [join(s_dir, "gray") for s_dir in strain_directories]
output_strain_directories = [join(s_dir, "output") for s_dir in gray_strain_directories]

process_images(strain_directories, gray_strain_directories, output_strain_directories)

output_df = get_strain_images_as_df(output_strain_directories)

save_to_excel(bacteria_workbook, 'Images', output_df)

images_data = pd.read_excel(bacteria_workbook, dtype=object, sheet_name="Images")

total_data.set_index("strain", inplace=True)

images_data.set_index("strain", inplace=True)

cnn_dataset = merge_datasets(images_data, total_data)

save_to_excel(bacteria_workbook, 'CNN Dataset', cnn_dataset)
