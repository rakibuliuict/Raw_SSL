import os
from glob import glob
import torch
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from augmentations import get_train_transforms
from augmentations import get_test_transforms

def prepare_semi_supervised(in_dir, cache=False):
    set_determinism(seed=0)

    # Load labeled images
    path_train_volumes_t2w = sorted(glob(os.path.join(in_dir, "TrainVolumes", "t2w", "*.nii.gz")))
    path_train_volumes_adc = sorted(glob(os.path.join(in_dir, "TrainVolumes", "adc", "*.nii.gz")))
    path_train_volumes_hbv = sorted(glob(os.path.join(in_dir, "TrainVolumes", "hbv", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "Binary_TrainSegmentation", "*.nii.gz")))

    # Load unlabeled images (without segmentation)
    path_unlabeled_volumes_t2w = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "t2w", "*.nii.gz")))
    path_unlabeled_volumes_adc = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "adc", "*.nii.gz")))
    path_unlabeled_volumes_hbv = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "hbv", "*.nii.gz")))

    path_test_volumes_t2w = sorted(glob(os.path.join(in_dir, "ValidVolumes", "t2w", "*.nii.gz")))
    path_test_volumes_adc = sorted(glob(os.path.join(in_dir, "ValidVolumes", "adc", "*.nii.gz")))
    path_test_volumes_hbv = sorted(glob(os.path.join(in_dir, "ValidVolumes", "hbv", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "Binary_ValidSegmentation", "*.nii.gz")))

    def extract_patient_id(file_path):
        return os.path.basename(file_path).split("_")[0]

    train_patient_ids = [extract_patient_id(path) for path in path_train_volumes_t2w]
    unlabeled_patient_ids = [extract_patient_id(path) for path in path_unlabeled_volumes_t2w]
    test_patient_ids = [extract_patient_id(path) for path in path_test_volumes_t2w]

    # Labeled training data
    train_files = [
        {
            "patient_id": pid, "t2w": t2w_image, "adc": adc_image,
            "hbv": hbv_image, "seg": label_name
        }
        for pid, t2w_image, adc_image, hbv_image, label_name in zip(
            train_patient_ids, path_train_volumes_t2w,
            path_train_volumes_adc, path_train_volumes_hbv, path_train_segmentation
        )
    ]

    # Unlabeled training data (No segmentation)
    unlabeled_files = [
        {
            "patient_id": pid, "t2w": t2w_image, "adc": adc_image,
            "hbv": hbv_image, "seg": None  # No ground truth labels
        }
        for pid, t2w_image, adc_image, hbv_image in zip(
            unlabeled_patient_ids, path_unlabeled_volumes_t2w,
            path_unlabeled_volumes_adc, path_unlabeled_volumes_hbv
        )
    ]

    # Validation/Test data
    test_files = [
        {
            "patient_id": pid, "t2w": t2w_image, "adc": adc_image,
            "hbv": hbv_image, "seg": label_name
        }
        for pid, t2w_image, adc_image, hbv_image, label_name in zip(
            test_patient_ids, path_test_volumes_t2w,
            path_test_volumes_adc, path_test_volumes_hbv, path_test_segmentation
        )
    ]

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        unlabeled_ds = CacheDataset(data=unlabeled_files, transform=train_transforms, cache_rate=1.0)
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        unlabeled_ds = Dataset(data=unlabeled_files, transform=train_transforms)
        test_ds = Dataset(data=test_files, transform=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, unlabeled_loader, test_loader
