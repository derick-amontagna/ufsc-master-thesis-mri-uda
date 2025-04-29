"""
Contains functionality for mri 
"""

import os
import shutil
import glob
import gc
import uuid

import ants
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from loguru import logger
from deepbet import run_bet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
SEED = 42


def pass_step(func):
    def verify_step(*args, **kwargs):
        for k, v in kwargs.items():
            if k == "dst_path":
                if not os.path.exists(v):
                    func(*args, **kwargs)
                else:
                    logger.info("Already done... Pass to next Step")

    return verify_step


def dcm2nifti(dcm_path, nii_out_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # Added a call to PermuteAxes to change the axes of the data
    image = sitk.PermuteAxes(image, [2, 1, 0])
    sitk.WriteImage(image, nii_out_path)


@pass_step
def transform_dicom2nifti(
    origin_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Dicom"),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti"),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    for dcm_path in tqdm(
        glob.glob(
            origin_path + f"{os.path.sep}*{os.path.sep}*{os.path.sep}*{os.path.sep}*"
        )
    ):
        id_dcm = dcm_path.split(os.path.sep)[-1]
        nifti_path = os.path.join(dst_path, id_dcm + ".nii.gz")
        try:
            dcm2nifti(dcm_path, nifti_path)
        except:
            logger.error(f"Erro with image: {dcm_path}")


@pass_step
def transform_n4_bias_field_correction(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-SkullStripping"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Registered"
    ),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    for image_path in tqdm(
        glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)
    ):
        moving_image = ants.image_read(image_path, reorient="ASR")
        img_n4 = ants.n4_bias_field_correction(moving_image)
        img_n4.to_file(
            os.path.join(
                image_path.replace(
                    origin_path.split(os.path.sep)[-1], dst_path.split(os.path.sep)[-1]
                )
            )
        )


@pass_step
def transform_skull_stripping(
    origin_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti"),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-SkullStripping"
    ),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    os.system(f"hd-bet -i {origin_path} -o {dst_path}")


@pass_step
def transform_registration(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-SkullStripping"
    ),
    template_path: str = os.path.join(
        BASE_DIR, "data", "Template", "sri24_spm8_T1_brain.nii"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Registered"
    ),
    type_of_transform: str = "antsRegistrationSyN[a]",
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    logger.info(f"The template apply - {template_path}")
    logger.info(f"The Type of Transform apply - {type_of_transform}")
    for moving_image_path in tqdm(
        glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)
    ):
        moving_image = ants.image_read(moving_image_path, reorient="ASR")
        template_image = ants.image_read(template_path, reorient="ASR")
        try:
            os.mkdir(os.path.join(BASE_DIR, "cache"))
        except FileExistsError:
            pass
        transformation = ants.registration(
            fixed=template_image,
            moving=moving_image,
            type_of_transform=type_of_transform,
            outprefix=os.path.join("cache", f"{uuid.uuid4().hex}"),
            verbose=False,
        )
        registered_img_ants = transformation["warpedmovout"]
        registered_img_ants.to_file(
            os.path.join(
                moving_image_path.replace(
                    origin_path.split(os.path.sep)[-1], dst_path.split(os.path.sep)[-1]
                )
            )
        )
        del transformation, template_image, moving_image
        gc.collect()
        shutil.rmtree(os.path.join(BASE_DIR, "cache"))


def calculate_entropie(image_array):
    min_val = image_array.min()
    max_val = image_array.max()
    escala = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
    normalized_array = (image_array - min_val) * 255.0 / escala
    entropie = []
    for i in range(normalized_array.shape[0]):
        slice_data = normalized_array[i, :, :].flatten()
        hist, _ = np.histogram(slice_data, bins=256, range=(0, 256))
        soma_hist = hist.sum()
        if soma_hist == 0:
            entropie.append(0)
        else:
            prob = hist / soma_hist
            entropie_slice = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
            entropie.append(entropie_slice)
    return np.array(entropie)


@pass_step
def transform_extract_relevant_slices(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Registered"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    num_slices: int = 32,
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")

    for path in tqdm(glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)):
        image = ants.image_read(
            path, reorient="ASR"
        )  # Z, Y, X - IAL = Inferior-to-superior, Anterior-to-posterior, Left-to-right
        # logger.info(path)
        image_data = image.numpy()  # -> (Z,Y,X)

        # Extract the desired slices
        # entropies = calculate_entropie(image_data)

        # selected_slices = np.argsort(entropies)[::-1][:num_slices]

        # Create a new ANTsImage from the extracted data
        extracted_image = ants.from_numpy(image_data[94:126, :, :])
        extracted_image.to_file(
            os.path.join(
                path.replace(
                    origin_path.split(os.path.sep)[-1],
                    dst_path.split(os.path.sep)[-1],
                )
            )
        )


def train_val_test_split(
    df: pd.DataFrame = None,
    domain_column: str = "Manufacturer",
    column_class: str = "Research Group",
    test_val: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: float = SEED,
    id_split: str = "Subject ID",
):
    dict_split = {}
    if domain_column:
        for domain in df[domain_column].unique():
            df_domain = df[df[domain_column] == domain]
            df_domain_filter = df_domain[[id_split, column_class]].drop_duplicates()
            train_df_sub_id, test_df_sub_id = train_test_split(
                df_domain_filter,
                test_size=test_size,
                stratify=df_domain_filter[column_class],
                shuffle=True,
                random_state=random_state,
            )
            if test_val:
                train_df_sub_id, val_df_sub_id = train_test_split(
                    train_df_sub_id,
                    test_size=val_size,
                    stratify=train_df_sub_id[column_class],
                    shuffle=True,
                    random_state=random_state,
                )
                train_df = df_domain[
                    df_domain[id_split].isin(train_df_sub_id[id_split].tolist())
                ]
                val_df = df_domain[
                    df_domain[id_split].isin(val_df_sub_id[id_split].tolist())
                ]
                test_df = df_domain[
                    df_domain[id_split].isin(test_df_sub_id[id_split].tolist())
                ]
                logger.info(
                    f"The split for the domain {domain} is: Train:{train_df.count().iloc[0]} / Val:{val_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
                )
                dict_split[domain] = {
                    "train": {
                        class_type: train_df[train_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in train_df[column_class].unique().tolist()
                    },
                    "val": {
                        class_type: val_df[val_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in val_df[column_class].unique().tolist()
                    },
                    "test": {
                        class_type: test_df[test_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in test_df[column_class].unique().tolist()
                    },
                }
            else:
                train_df = df[df[id_split].isin(train_df_sub_id[id_split].tolist())]
                test_df = df[df[id_split].isin(test_df_sub_id[id_split].tolist())]
                dict_split[domain] = {
                    "train": {
                        class_type: train_df[train_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in train_df[column_class].unique().tolist()
                    },
                    "test": {
                        class_type: test_df[test_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in test_df[column_class].unique().tolist()
                    },
                }
                logger.info(
                    f"The split for the domain {domain} is: Train:{train_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
                )

        return dict_split
    else:
        df_filter = df[[id_split, column_class]].drop_duplicates()
        train_df_sub_id, test_df_sub_id = train_test_split(
            df_filter,
            test_size=test_size,
            stratify=df[column_class],
            shuffle=True,
            random_state=random_state,
        )
        if test_val:
            train_df_sub_id, val_df_sub_id = train_test_split(
                train_df_sub_id,
                test_size=val_size,
                stratify=train_df_sub_id[column_class],
                shuffle=True,
                random_state=random_state,
            )
            train_df = df[df[id_split].isin(train_df_sub_id[id_split].tolist())]
            val_df = df[df[id_split].isin(val_df_sub_id[id_split].tolist())]
            test_df = df[df[id_split].isin(test_df_sub_id[id_split].tolist())]
            logger.info(
                f"The split for the domain All is: Train:{train_df.count().iloc[0]} / Val:{val_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
            )
            dict_split["All"] = {
                "train": {
                    class_type: train_df[train_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in train_df[column_class].unique().tolist()
                },
                "val": {
                    class_type: val_df[val_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in val_df[column_class].unique().tolist()
                },
                "test": {
                    class_type: test_df[test_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in test_df[column_class].unique().tolist()
                },
            }
        else:
            train_df = df[df[id_split].isin(train_df_sub_id[id_split].tolist())]
            test_df = df[df[id_split].isin(test_df_sub_id[id_split].tolist())]
            dict_split["All"] = {
                "train": {
                    class_type: train_df[train_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in train_df[column_class].unique().tolist()
                },
                "test": {
                    class_type: test_df[test_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in test_df[column_class].unique().tolist()
                },
            }
            logger.info(
                f"The split for the domain All is: Train:{train_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
            )
    return dict_split


def create_folder_and_move_image(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    df: pd.DataFrame = None,
    column_class: str = "Research Group",
    column_domain: str = "Manufacturer",
    split_data: dict = None,
    test_val: bool = False,
):
    if test_val:
        list_split = ["train", "val", "test"]
    else:
        list_split = ["train", "test"]
    list_class = df[column_class].unique().tolist()
    if column_domain:
        list_domain = df[column_domain].unique().tolist()
        for domain_name in list_domain:
            for class_name in list_class:
                for split_name in list_split:
                    dst_dir = os.path.join(
                        dst_path, domain_name, split_name, class_name
                    )
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    for id_mri in split_data[domain_name][split_name][class_name]:
                        shutil.copy(
                            os.path.join(origin_path, f"{id_mri}.nii.gz"),
                            os.path.join(dst_dir, f"{id_mri}.nii.gz"),
                        )
    else:
        for class_name in list_class:
            for split_name in list_split:
                dst_dir = os.path.join(dst_path, split_name, class_name)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                    for id_mri in split_data["All"][split_name][class_name]:
                        shutil.copy(
                            os.path.join(origin_path, f"{id_mri}.nii.gz"),
                            os.path.join(dst_dir, f"{id_mri}.nii.gz"),
                        )


@pass_step
def gen_class_folders(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    df: pd.DataFrame = None,
    column_class: str = "Research Group",
    domain_manufacturer: bool = False,
    domain_model: bool = False,
    test_val: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    # Filter Images
    df = df[
        df["Image ID"].isin(
            [
                id.split(os.path.sep)[-1].split(".")[0]
                for id in glob.glob(
                    os.path.join(origin_path, "*.nii.gz"), recursive=True
                )
            ]
        )
    ]
    if domain_manufacturer or domain_model:
        logger.info("Create the With Domain")
        if domain_manufacturer:
            split_data = train_val_test_split(
                df.copy(),
                "Manufacturer",
                column_class=column_class,
                test_val=test_val,
                test_size=test_size,
                val_size=val_size,
            )
            create_folder_and_move_image(
                origin_path,
                dst_path,
                df,
                column_class,
                "Manufacturer",
                split_data,
                test_val,
            )
        else:
            split_data = train_val_test_split(
                df.copy(),
                "Model",
                column_class=column_class,
                test_val=test_val,
                test_size=test_size,
                val_size=val_size,
            )
            create_folder_and_move_image(
                origin_path, dst_path, df, column_class, "Model", split_data, test_val
            )
    else:
        logger.info("Create the Without Domain")
        split_data = train_val_test_split(
            df.copy(),
            None,
            column_class=column_class,
            test_val=test_val,
            test_size=test_size,
            val_size=val_size,
        )
        create_folder_and_move_image(
            origin_path, dst_path, df, column_class, None, split_data, test_val
        )


def get_path_folder_list(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    domain: bool = False,
):
    if domain:
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[12:15])
                    for path in glob.glob(
                        os.path.join(
                            origin_path, f"*{os.path.sep}*{os.path.sep}*{os.path.sep}"
                        ),
                        recursive=True,
                    )
                ]
            )
        )
    else:
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[11:13])
                    for path in glob.glob(
                        os.path.join(origin_path, f"*{os.path.sep}*{os.path.sep}"),
                        recursive=True,
                    )
                ]
            )
        )
    return path_folder_list


def crop_nonzero(image):
    non_zero_rows = np.any(image != 0, axis=1)
    non_zero_cols = np.any(image != 0, axis=0)

    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]

    cropped_image = image[row_start : row_end + 1, col_start : col_end + 1]
    return cropped_image


def gen_3d_to_2d(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-2D"),
    path_folder_list: list = None,
):
    for class_name in path_folder_list:
        logger.info(f"Started copy the class! - {class_name}")
        class_dst_path = os.path.join(dst_path, class_name)
        if not os.path.exists(class_dst_path):
            os.makedirs(class_dst_path)
            logger.info(f"The new directory is created! - {class_dst_path}")
        for path_mri in tqdm(
            glob.glob(os.path.join(origin_path, class_name, "*.nii.gz"), recursive=True)
        ):
            name_id = path_mri.split(os.path.sep)[-1].split(".")[0]
            image = nib.load(path_mri)
            image_data = image.get_fdata()
            for i in range(image_data.shape[0]):
                dst_path_image = os.path.join(
                    class_dst_path, f"{name_id}_slice_{i}.nii.gz"
                )
                image_data_2D = image_data[i, :, :].astype(np.float32)
                image_data_2D_crop = crop_nonzero(image_data_2D)
                nib.save(
                    nib.Nifti1Image(image_data_2D_crop, affine=np.eye(4)),
                    dst_path_image,
                )


@pass_step
def transform_3d_to_2d(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-2D"),
    domain: bool = False,
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    gen_3d_to_2d(origin_path, dst_path, get_path_folder_list(origin_path, domain))


def preprocess_pipeline(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "Image", "ADNI1-Screening-Dicom"
    ),
    dataset_name: str = "ADNI1",
    df_path: str = os.path.join(
        BASE_DIR,
        "analytics",
        "ADNI1",
        "ADNI1-Screening-T1-Original-Collect.csv",
    ),
    gen_2d: bool = False,
    column_class: str = "Research Group",
    domain_manufacturer: bool = True,
    domain_model: bool = False,
    test_val: bool = False,
    num_slices: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    int_dir = os.path.join(BASE_DIR, "data", dataset_name, "Image", "Preprocess")
    df = pd.read_csv(df_path)
    df["Manufacturer"] = df["Imaging Protocol"].apply(
        lambda x: x.split(";")[0].split("=")[1]
    )
    df["Model"] = df["Imaging Protocol"].apply(lambda x: x.split(";")[1].split("=")[1])
    if "ADNI" in dataset_name:
        df = df.replace(
            {
                "GE MEDICAL SYSTEMS": "GE",
                "Philips Healthcare": "Philips",
                "Philips Medical Systems": "Philips",
                "SIEMENS|PixelMed": "Siemens",
                "SIEMENS": "Siemens",
            },
            regex=False,
        )
    df["Image ID"] = "I" + df["Image ID"].astype(str)
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)
        logger.info(f"The new directory is created! - {int_dir}")
    logger.info(f"1 Step - Transform Dicom to Nifti")
    transform_dicom2nifti(
        origin_path=origin_path,
        dst_path=os.path.join(int_dir, "1_step_dicom2nifti"),
    )

    # logger.info(f"2 Step - Apply Bias Field Correction")
    # transform_n4_bias_field_correction(
    #    origin_path=os.path.join(int_dir, "1_step_dicom2nifti"),
    #    dst_path=os.path.join(int_dir, "2_n4_bias_field_correction"),
    # )

    logger.info(f"2 Step - Apply Skull Stripping")
    transform_skull_stripping(
        origin_path=os.path.join(int_dir, "1_step_dicom2nifti"),
        dst_path=os.path.join(int_dir, "2_step_skull_stripping"),
    )
    logger.info(f"3 Step - Apply Registration")
    transform_registration(
        origin_path=os.path.join(int_dir, "2_step_skull_stripping"),
        dst_path=os.path.join(int_dir, "3_step_registration"),
        template_path=os.path.join(
            BASE_DIR, "data", "Template", "MNI152_T1_1mm_Brain.nii"
        ),
    )
    logger.info(f"4 Step - Extract Relevant Slices")
    transform_extract_relevant_slices(
        origin_path=os.path.join(int_dir, "3_step_registration"),
        dst_path=os.path.join(int_dir, "4_step_relevant_slices"),
        num_slices=num_slices,
    )
    logger.info(f"5 Step - Generate class folders")
    gen_class_folders(
        origin_path=os.path.join(int_dir, "4_step_relevant_slices"),
        dst_path=os.path.join(int_dir, "5_step_class_folders"),
        df=df.copy(),
        column_class=column_class,
        domain_manufacturer=domain_manufacturer,
        domain_model=domain_model,
        test_val=test_val,
        test_size=test_size,
        val_size=val_size,
    )
    if gen_2d:
        logger.info(f"6 Step - Transform images 3D to 2D - Optional")
        transform_3d_to_2d(
            origin_path=os.path.join(int_dir, "5_step_class_folders"),
            dst_path=os.path.join(int_dir, "6_step_nifti_2d"),
            domain=domain_manufacturer or domain_model,
        )
