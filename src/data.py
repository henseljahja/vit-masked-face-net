import os
import shutil
import zipfile

import splitfolders
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = CUR_DIR + "/dataset"
DATASET_SPLIT_DIR = CUR_DIR + "/dataset_split"
DATASET_CLASS_DIR = CUR_DIR + "/dataset_class"
CMFD_DIR = CUR_DIR + "/CMFD"
IMFD_DIR = CUR_DIR + "/IMFD"
DATASET_SPLIT_DIR_VAL = CUR_DIR + "/dataset_split/val"
DATASET_SPLIT_DIR_MINI = CUR_DIR + "/dataset_split_mini"
# LIST_CMFD_DIR = os.listdir(CMFD_DIR)
# LIST_IMFD_DIR = os.listdir(IMFD_DIR)

CLASS_NAMES = [
    "Mask.jpg",
    "Mask_Mouth_Chin.jpg",
    "Mask_Nose_Mouth.jpg",
    "Mask_Chin.jpg",
]

ZIP_FILES = [
    "/home/hensel/mfn/mfn_zipped/CMFD.zip",
    "/home/hensel/mfn/mfn_zipped/CMFD1.zip",
    "/home/hensel/mfn/mfn_zipped/IMFD.zip",
    "/home/hensel/mfn/mfn_zipped/IMFD1.zip",
]

EXTRACTED_ZIPPED_DIR = "/home/hensel/mfn"
MFN_EXTRACTED_DIR = "/home/hensel/mfn/mfn_extracted"
EXTRACTED_CMFD_DIR = MFN_EXTRACTED_DIR + "/cmfd"
EXTRACTED_IMFD_DIR = MFN_EXTRACTED_DIR + "/imfd"
MFN_CLASSES_DIR = "/home/hensel/mfn/mfn_class"

DATA_224_DIR = "/home/hensel/mfn/mfn_224"
DATA_224_SPLIT_DIR = "/home/hensel/mfn/mfn_224_split"

DATA_224_AUGMENT_DIR = "/home/hensel/mfn/mfn_224_augment"
DATA_224_AUGMENT_SPLIT_DIR = "/home/hensel/mfn/mfn_224_augment_split"

DATA_SPLIT_DIR = "/home/hensel/data_split"

DATA_224_AUGMENT_MINI_DIR = "/home/hensel/mfn/mfn_224_split/val"
DATA_224_SPLIT_MINI_DIR = "/home/hensel/mfn/mfn_224_split_mini"
DATA_224_AUGMENT_MINI_DIR = "/home/hensel/mfn/mfn_224_augment_split/val"
DATA_224_AUGMENT_SPLIT_MINI_DIR = "/home/hensel/mfn/mfn_224_augment_split_mini"

# for i in tqdm(range(len(LIST_CMFD_DIR))):
class Extract:
    @classmethod
    def extract_all(cls) -> None:
        os.makedirs(EXTRACTED_CMFD_DIR, exist_ok=True)
        os.makedirs(EXTRACTED_IMFD_DIR, exist_ok=True)
        for ZIP_FILE in tqdm(ZIP_FILES, desc="Unzip Files"):
            if "CMFD" in ZIP_FILE:
                zip_file_dir = ZIP_FILE
                with zipfile.ZipFile(zip_file_dir, "r") as zip_ref:
                    zip_ref.extractall(EXTRACTED_CMFD_DIR)
            else:
                zip_file_dir = ZIP_FILE
                with zipfile.ZipFile(zip_file_dir, "r") as zip_ref:
                    zip_ref.extractall(EXTRACTED_IMFD_DIR)

    @classmethod
    def create_split_dir(cls) -> None:
        for class_name in tqdm(CLASS_NAMES, desc="Creating Dir"):
            sub_dir = class_name.replace(".jpg", "")
            # if os.path.exists(DATASET_CLASS_DIR + "/" + sub_dir):
            #     print(f"Directory {DATASET_CLASS_DIR}/{sub_dir} exist")
            # else:
            os.makedirs(f"{MFN_CLASSES_DIR}/{sub_dir}")
            print(f"Directory {DATASET_CLASS_DIR}/{sub_dir} created")

    @classmethod
    def extract_unzipped(cls) -> None:
        # dir -> 00000,00001
        (
            mask_counter,
            mask_chin_counter,
            mask_mouth_chin_counter,
            mask_nose_mouth_counter,
        ) = (0, 0, 0, 0)
        for MFN_EXTRACTED_DIR in [EXTRACTED_IMFD_DIR, EXTRACTED_CMFD_DIR]:
            for dir in tqdm(
                os.listdir(MFN_EXTRACTED_DIR),
                desc="Extracting file to classes ",
            ):
                image_dir = f"{MFN_EXTRACTED_DIR}/{dir}"
                for image in os.listdir(image_dir):

                    old_image_path = f"{image_dir}/{image}"
                    replaced_name = image.split("_")[1:]

                    if "Mask.jpg" in image:
                        mask_counter += 1
                        new_image_name = (
                            str(mask_counter) + "_" + "_".join(replaced_name)
                        )
                        new_image_path = f"{MFN_CLASSES_DIR}/Mask/{new_image_name}"
                        shutil.move(old_image_path, new_image_path)

                    elif "Mask_Mouth_Chin.jpg" in image:
                        mask_mouth_chin_counter += 1
                        new_image_name = (
                            str(mask_mouth_chin_counter) + "_" + "_".join(replaced_name)
                        )
                        new_image_path = (
                            f"{MFN_CLASSES_DIR}/Mask_Mouth_Chin/{new_image_name}"
                        )
                        shutil.move(old_image_path, new_image_path)

                    elif "Mask_Chin.jpg" in image:
                        mask_chin_counter += 1
                        new_image_name = (
                            str(mask_mouth_chin_counter) + "_" + "_".join(replaced_name)
                        )
                        new_image_path = f"{MFN_CLASSES_DIR}/Mask_Chin/{new_image_name}"
                        shutil.move(old_image_path, new_image_path)

                    elif "Mask_Nose_Mouth.jpg" in image:
                        mask_nose_mouth_counter += 1
                        new_image_name = (
                            str(mask_mouth_chin_counter) + "_" + "_".join(replaced_name)
                        )
                        new_image_path = (
                            f"{MFN_CLASSES_DIR}/Mask_Nose_Mouth/{new_image_name}"
                        )
                        shutil.move(old_image_path, new_image_path)

    @classmethod
    def create_224_dir(cls) -> None:
        # data_dir = "/home/hensel/projects/skripsi/torchvision_save/animals"
        image_datasets = datasets.ImageFolder(
            MFN_CLASSES_DIR,
            transform=transforms.Compose(
                [
                    torchvision.transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    # torchvision.transforms.RandAugment(num_ops=2),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        classes = list(image_datasets.class_to_idx.keys())
        os.makedirs(DATA_224_DIR, exist_ok=True)
        for i in classes:
            os.makedirs(f"{DATA_224_DIR}/{i}", exist_ok=True)
        for index, (image, labels) in tqdm(enumerate(image_datasets), desc="224"):
            torchvision.utils.save_image(
                image,
                f"{DATA_224_DIR}/{str(classes[labels])}/{classes[labels]}_{str(index)}.png",
            )

    @classmethod
    def create_224_augment_dir(cls) -> None:
        image_datasets = datasets.ImageFolder(
            MFN_CLASSES_DIR,
            transform=transforms.Compose(
                [
                    torchvision.transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    torchvision.transforms.RandAugment(num_ops=2),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        classes = list(image_datasets.class_to_idx.keys())
        os.makedirs(DATA_224_AUGMENT_DIR, exist_ok=True)
        for i in classes:
            os.makedirs(f"{DATA_224_AUGMENT_DIR}/{i}", exist_ok=True)
        for index, (image, labels) in tqdm(
            enumerate(image_datasets), desc="224_augment"
        ):
            torchvision.utils.save_image(
                image,
                f"{DATA_224_AUGMENT_DIR}/{str(classes[labels])}/{classes[labels]}_{str(index)}.png",
            )

    @classmethod
    def split_224_dir(cls) -> None:
        splitfolders.ratio(
            DATA_224_DIR,
            output=DATA_224_SPLIT_DIR,
            seed=42,
            ratio=(0.8, 0.1, 0.1),
            # fixed=(50, 10),
        )  # default values

    @classmethod
    def split_224_augment_dir(cls) -> None:
        splitfolders.ratio(
            DATA_224_AUGMENT_DIR,
            output=DATA_224_AUGMENT_SPLIT_DIR,
            seed=42,
            ratio=(0.8, 0.1, 0.1),
            # fixed=(50, 10),
        )  # default values

    @classmethod
    def split_224_mini_dir(cls) -> None:
        os.makedirs(DATA_224_SPLIT_MINI_DIR, exist_ok=True)
        splitfolders.ratio(
            DATA_224_AUGMENT_MINI_DIR,
            output=DATA_224_SPLIT_MINI_DIR,
            seed=42,
            ratio=(0.8, 0.1, 0.1),
            # fixed=(50, 10),
        )  # default values

    @classmethod
    def split_224_augment_mini_dir(cls) -> None:
        os.makedirs(DATA_224_AUGMENT_SPLIT_MINI_DIR, exist_ok=True)
        splitfolders.ratio(
            DATA_224_AUGMENT_MINI_DIR,
            output=DATA_224_AUGMENT_SPLIT_MINI_DIR,
            seed=42,
            ratio=(0.8, 0.1, 0.1),
            # fixed=(50, 10),
        )  # default values

    # @classmethod
    # def resize_and_extract_data(cls) -> None:
    #     for i in tqdm(os.listdir(DATASET_DIR), desc="resize & extracting dataset"):
    #         for j in os.listdir(DATASET_DIR + "/" + i):
    #             try:
    #                 image = Image.open(DATASET_DIR + "/" + i + "/" + j)
    #                 new_image = image.resize((224, 224))
    #                 new_image.save(DATASET_DIR + "/" + i + "/" + j)
    #             except IOError:
    #                 pass
    #             for k in CLASS_NAMES:
    #                 if k in j:
    #                     shutil.move(
    #                         DATASET_DIR + "/" + i + "/" + j,
    #                         DATASET_CLASS_DIR + "/" + k.replace(".jpg", "") + "/" + j,
    #                     )

    # @classmethod
    # def move_extracted_files(cls) -> None:
    #     (
    #         mask_counter,
    #         mask_chin_counter,
    #         mask_mouth_chin_counter,
    #         mask_nose_mouth_counter,
    #     ) = (0, 0, 0, 0)
    #     # CMFD, IMFD
    #     for main_dir in tqdm(os.listdir(EXTRACTED_ZIPPED_DIR), desc="Moving Data"):
    #         # CMFD, IMFD
    #         sub_main_dir_up = os.path.join(EXTRACTED_ZIPPED_DIR, main_dir)
    #         # 00000, 00001
    #         # CMFD/0000
    #         for sub_main_dir in tqdm(os.listdir(sub_main_dir_up)):

    #             image_dirs = os.path.join(sub_main_dir_up, sub_main_dir)
    #             for image in tqdm(os.listdir(image_dirs), desc="Moving Images"):
    #                 # CMFD/00000/00001_Mask.jpg
    #                 image_dir = os.path.join(image_dirs, image)
    #                 # 00001_Mask.jpg -> ["000001","Mask.jpg"]
    #                 replaced_name = image.split("_")
    #                 replaced_name = replaced_name[1:]

    #                 if "Mask.jpg" in image_dir:
    #                     mask_counter += 1
    #                     new_image_name = (
    #                         str(mask_counter) + "_" + "_".join(replaced_name)
    #                     )
    #                     new_image_path = (
    #                         f"{TARGET_DIR}/Mask_Mouth_Chin/{new_image_name}"
    #                     )
    #                     shutil.move(image_dir, new_image_path)

    #                 elif "Mask_Mouth_Chin.jpg" in image_dir:
    #                     mask_mouth_chin_counter += 1
    #                     new_image_name = (
    #                         str(mask_mouth_chin_counter) + "_" + "_".join(replaced_name)
    #                     )
    #                     new_image_path = (
    #                         f"{TARGET_DIR}/Mask_Mouth_Chin/{new_image_name}"
    #                     )
    #                     shutil.move(image_dir, new_image_path)

    #                 elif "Mask_Chin.jpg" in image_dir:
    #                     mask_chin_counter += 1
    #                     new_image_name = (
    #                         str(mask_chin_counter) + "_" + "_".join(replaced_name)
    #                     )
    #                     new_image_path = f"{TARGET_DIR}/Mask_Chin/{new_image_name}"
    #                     shutil.move(image_dir, new_image_path)

    #                 elif "Mask_Nose_Mouth.jpg" in image_dir:
    #                     mask_nose_mouth_counter += 1
    #                     new_image_name = (
    #                         str(mask_nose_mouth_counter) + "_" + "_".join(replaced_name)
    #                     )
    #                     new_image_path = (
    #                         f"{TARGET_DIR}/Mask_Nose_Mouth/{new_image_name}"
    #                     )
    #                     shutil.move(image_dir, new_image_path)

    # @classmethod
    # def split_extracted_dir(cls) -> None:
    #     splitfolders.ratio(
    #         DATA_224_DIR,
    #         output=DATA_224_SPLIT_DIR,
    #         seed=42,
    #         ratio=(0.8, 0.1, 0.1),
    #         # fixed=(50, 10),
    #     )  # default values

    # @classmethod
    # def create_train_test_val_dir(cls) -> None:
    #     splitfolders.ratio(
    #         DATA_224_AUGMENT_DIR,
    #         output=DATA_224_AUGMENT_SPLIT_DIR,
    #         seed=42,
    #         ratio=(0.8, 0.1, 0.1),
    #         # fixed=(50, 10),
    #     )  # default values
    #     # splitfolders.fixed(
    #     #     DATASET_CLASS_DIR,
    #     #     output=DATASET_SPLIT_DIR,
    #     #     seed=42,
    #     #     # ratio=(0.8, 0.1, 0.1),
    #     #     fixed=(13000, 1300),
    #     # )  # default values

    # @classmethod
    # def create_train_test_val_dir_mini(cls) -> None:
    #     splitfolders.ratio(
    #         DATASET_SPLIT_DIR_VAL,
    #         output=DATASET_SPLIT_DIR_MINI,
    #         seed=42,
    #         ratio=(0.8, 0.1, 0.1),
    #     # fixed=(50, 10),
    # )  # default values
    # # splitfolders.fixed(
    # #     DATASET_CLASS_DIR,
    # #     output=DATASET_SPLIT_DIR,
    # #     seed=42,
    # #     # ratio=(0.8, 0.1, 0.1),
    # #     fixed=(13000, 1300),
    # # )  # default values


# Extract.extract_all()
Extract.create_split_dir()
Extract.extract_unzipped()


# Extract.create_224_dir()
# Extract.create_224_augment_dir()

# Extract.split_224_dir()
# Extract.split_224_augment_dir()

# Extract.split_224_mini_dir()
# Extract.split_224_augment_mini_dir()

# Extract.resize_and_extract_data()
# Extract.move_extracted_files()
# Extract.split_extracted_dir()
# Extract.create_train_test_val_dir()
# Extract.create_train_test_val_dir_mini()
