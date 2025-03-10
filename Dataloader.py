import os
import re
from glob import glob
from natsort import natsorted
import torch
import torchio as tio
from torch.utils.data import DataLoader, random_split
from monai.transforms import Compose, ScaleIntensityD, EnsureTypeD
from monai.data import Dataset
import torchio as tio
from monai.transforms import Compose, ScaleIntensityD, EnsureTypeD
from monai.data import DataLoader


# Define excluded IDs
excluded_ids = []  # Define your excluded IDs


#included_ids = ['AMBL-001', 'AMBL-002', 'AMBL-003', 'AMBL-004', 'AMBL-005', 'AMBL-006', 'AMBL-007', 'AMBL-008', 'AMBL-009', 'AMBL-010', 'AMBL-011', 'AMBL-012', 'AMBL-013', 'AMBL-014', 'AMBL-015', 'AMBL-016', 'AMBL-017', 'AMBL-018', 'AMBL-019', 'AMBL-020', 'AMBL-021', 'AMBL-022', 'AMBL-023', 'AMBL-024', 'AMBL-025', 'AMBL-026', 'AMBL-027', 'AMBL-028', 'AMBL-029', 'AMBL-031', 'AMBL-032', 'AMBL-033', 'AMBL-034', 'AMBL-035', 'AMBL-036', 'AMBL-037', 'AMBL-038', 'AMBL-039', 'AMBL-040', 'AMBL-041', 'AMBL-042', 'AMBL-043', 'AMBL-044', 'AMBL-045', 'AMBL-046', 'AMBL-047', 'AMBL-048', 'AMBL-049', 'AMBL-050', 'AMBL-056', 'AMBL-063', 'AMBL-068', 'AMBL-070', 'AMBL-072', 'AMBL-082', 'AMBL-085', 'AMBL-088', 'AMBL-091', 'AMBL-097', 'AMBL-496', 'AMBL-507', 'AMBL-514', 'AMBL-541', 'AMBL-555', 'AMBL-557', 'AMBL-559', 'AMBL-561', 'AMBL-562', 'AMBL-563', 'AMBL-564', 'AMBL-565', 'AMBL-566', 'AMBL-567', 'AMBL-568', 'AMBL-569', 'AMBL-570', 'AMBL-571', 'AMBL-572', 'AMBL-573', 'AMBL-574', 'AMBL-575', 'AMBL-577', 'AMBL-578', 'AMBL-579', 'AMBL-580', 'AMBL-581', 'AMBL-582', 'AMBL-583', 'AMBL-584', 'AMBL-585', 'AMBL-586', 'AMBL-587', 'AMBL-588', 'AMBL-589', 'AMBL-590', 'AMBL-591', 'AMBL-593', 'AMBL-594', 'AMBL-595', 'AMBL-596', 'AMBL-597', 'AMBL-598', 'AMBL-599', 'AMBL-600', 'AMBL-601', 'AMBL-602', 'AMBL-603', 'AMBL-604', 'AMBL-605', 'AMBL-606', 'AMBL-607', 'AMBL-608', 'AMBL-609', 'AMBL-610', 'AMBL-611', 'AMBL-612', 'AMBL-613', 'AMBL-614', 'AMBL-615', 'AMBL-616', 'AMBL-617', 'AMBL-618', 'AMBL-619', 'AMBL-620', 'AMBL-621', 'AMBL-622', 'AMBL-623', 'AMBL-624', 'AMBL-625', 'AMBL-626', 'AMBL-627', 'AMBL-628', 'AMBL-629', 'AMBL-630', 'AMBL-631', 'AMBL-632']

included_ids = ['AMBL-001', 'AMBL-003', 'AMBL-004', 'AMBL-005', 'AMBL-006', 'AMBL-007', 'AMBL-008', 'AMBL-009', 'AMBL-010', 'AMBL-011', 'AMBL-012', 'AMBL-013', 'AMBL-014', 'AMBL-015', 'AMBL-016', 'AMBL-017', 'AMBL-018', 'AMBL-019', 'AMBL-020', 'AMBL-021', 'AMBL-022', 'AMBL-023', 'AMBL-024', 'AMBL-025', 'AMBL-026', 'AMBL-027', 'AMBL-028', 'AMBL-029', 'AMBL-031', 'AMBL-032', 'AMBL-033', 'AMBL-034', 'AMBL-035', 'AMBL-036', 'AMBL-037', 'AMBL-038', 'AMBL-039', 'AMBL-040', 'AMBL-041', 'AMBL-042', 'AMBL-043', 'AMBL-044', 'AMBL-045', 'AMBL-046', 'AMBL-047', 'AMBL-048', 'AMBL-049', 'AMBL-050', 'AMBL-056', 'AMBL-063', 'AMBL-068', 'AMBL-070', 'AMBL-072', 'AMBL-082', 'AMBL-085', 'AMBL-088', 'AMBL-091', 'AMBL-097', 'AMBL-496', 'AMBL-507', 'AMBL-514', 'AMBL-541', 'AMBL-555', 'AMBL-557', 'AMBL-559', 'AMBL-561', 'AMBL-562', 'AMBL-563', 'AMBL-564', 'AMBL-565', 'AMBL-566', 'AMBL-567', 'AMBL-568', 'AMBL-569', 'AMBL-570', 'AMBL-571', 'AMBL-572', 'AMBL-573', 'AMBL-574', 'AMBL-575', 'AMBL-577', 'AMBL-578', 'AMBL-579', 'AMBL-580', 'AMBL-581', 'AMBL-582', 'AMBL-583', 'AMBL-584', 'AMBL-585', 'AMBL-586', 'AMBL-587', 'AMBL-588', 'AMBL-590', 'AMBL-591', 'AMBL-593', 'AMBL-594', 'AMBL-595', 'AMBL-596', 'AMBL-597', 'AMBL-598', 'AMBL-599', 'AMBL-600', 'AMBL-601', 'AMBL-602', 'AMBL-603', 'AMBL-604', 'AMBL-605'] # 'AMBL-606', 'AMBL-607', 'AMBL-608', 'AMBL-610', 'AMBL-612', 'AMBL-613', 'AMBL-615', 'AMBL-616', 'AMBL-617', 'AMBL-618', 'AMBL-619', 'AMBL-620', 'AMBL-621', 'AMBL-622', 'AMBL-623', 'AMBL-625', 'AMBL-626', 'AMBL-627', 'AMBL-628', 'AMBL-629', 'AMBL-631', 'AMBL-632']

"""
get_subject when soustraction between phase 3 and 1

def get_subjects(path):
    sbj_files = natsorted(glob(os.path.join(path, '*/*/5.000000-AX*')))
    subjects = []

    for sbj_file in sbj_files:
        # Extract subject ID
        tmp_sbj_name = re.split("AMBL", sbj_file)
        tmp_sbj_name = tmp_sbj_name[1]
        tmp_sbj_name = tmp_sbj_name[0:4]
        sbj_ID = 'AMBL' + tmp_sbj_name

        # Skip excluded IDs
        if sbj_ID not in included_ids:
            print(f"Skipping patient: {sbj_ID}")
            continue

        numeric_id_str = sbj_ID[-3:].lstrip('0')  # Remove leading zeros
        if not numeric_id_str.isdigit():
            print(f"Invalid numeric ID in sbj_ID: {sbj_ID}")
            continue

        numeric_id = int(numeric_id_str)

        #if numeric_id > 3:
           #continue

        print(f"Processing patient: {sbj_ID}, Numeric ID: {numeric_id}")


        # Load phase images
        files_phase_1 = natsorted(glob(os.path.join(sbj_file, '1-*')))
        files_phase_3 = natsorted(glob(os.path.join(sbj_file, '3-*')))

        #subject_data_phase_1 = tio.ScalarImage(files_phase_1)
        #subject_data_phase_3 = tio.ScalarImage(files_phase_3)

        subject_data_phase_1 = tio.Subject(
            data=tio.ScalarImage(files_phase_1),
        )
        subject_data_phase_3 = tio.Subject(
            data=tio.ScalarImage(files_phase_3),
        )

        # Subtract phase 1 from phase 3
        data_phase_1 = subject_data_phase_1.data[tio.DATA]
        data_phase_3 = subject_data_phase_3.data[tio.DATA]

        Nslices, h, w_orig, c = data_phase_1.shape

        sub_data = data_phase_3 - data_phase_1
        sub_data = torch.permute(sub_data, [3, 1, 2, 0])
        sub_data = sub_data[:, :, 200::, :]

        ROI_files = natsorted(glob(os.path.join(path, sbj_ID, '*/*ROI*')))

        if not ROI_files:
            Y = torch.zeros_like(sub_data)
        else:
            subject_seg = tio.Subject(
                data=tio.LabelMap(ROI_files),
            )
            print("Original mask shape:", subject_seg.data.shape)


            NROI = int(subject_seg.data.shape[3] / Nslices)
            Y = subject_seg.data[tio.DATA].reshape(c, h, w_orig, NROI, Nslices)
            Y = torch.sum(Y, dim=3)
            Y = torch.flip(Y, dims=(3,))
            Y = Y[:, :, 200::, :]
            # Normalize mask to binary values (0 and 1)
            Y = (Y > 0).float()
            print(f"Processed mask shape for {sbj_ID}: {Y.shape}")
            print(f"Mask unique values for {sbj_ID}: {torch.unique(Y)}")
            print(f"ROI files for {sbj_ID}: {ROI_files}")

        
        # Append data as a dictionary
        #subjects.append({
            #"image": sub_data,
            #"mask": Y,
            #"patientID": sbj_ID
        #})
        

        subjects.append({
            "image": torch.as_tensor(sub_data, dtype=torch.float32),  # Ensure tensor type
            "mask": torch.as_tensor(Y, dtype=torch.float32),  # Ensure tensor type
            "patientID": sbj_ID
        })


    return subjects

"""

def get_subjects(path):
    sbj_files = natsorted(glob(os.path.join(path, '*/*/5.000000-AX*')))
    subjects = []

    for sbj_file in sbj_files:
        tmp_sbj_name = re.split("AMBL", sbj_file)
        tmp_sbj_name = tmp_sbj_name[1]
        tmp_sbj_name = tmp_sbj_name[0:4]
        sbj_ID = 'AMBL' + tmp_sbj_name

        if sbj_ID not in included_ids:
            print(f"Skipping patient: {sbj_ID}")
            continue

        numeric_id_str = sbj_ID[-3:].lstrip('0')
        if not numeric_id_str.isdigit():
            print(f"Invalid numeric ID in sbj_ID: {sbj_ID}")
            continue

        numeric_id = int(numeric_id_str)

        print(f"Processing patient: {sbj_ID}, Numeric ID: {numeric_id}")

        # Load phase images
        files_phase_1 = natsorted(glob(os.path.join(sbj_file, '1-*')))
        files_phase_3 = natsorted(glob(os.path.join(sbj_file, '3-*')))

        subject_data_phase_1 = tio.ScalarImage(files_phase_1).data  # Shape: [1, H, W, D]
        subject_data_phase_3 = tio.ScalarImage(files_phase_3).data  # Shape: [1, H, W, D]

        # Stack phases into a multi-channel image [2, H, W, D]
        combined_data = torch.cat([subject_data_phase_1, subject_data_phase_3], dim=0)
        combined_data = combined_data[:, :, 200::, :]  # Apply same cropping

        Nslices = combined_data.shape[-1]

        ROI_files = natsorted(glob(os.path.join(path, sbj_ID, '*/*ROI*')))

        if not ROI_files:
            Y = torch.zeros((1, combined_data.shape[1], combined_data.shape[2], combined_data.shape[3]))
        else:
            subject_seg = tio.LabelMap(ROI_files)
            print("Original mask shape:", subject_seg.data.shape)

            c, h, w, total_slices = subject_seg.data.shape
            NROI = int(total_slices / Nslices)
            Y = subject_seg.data.reshape(c, h, w, NROI, Nslices)
            Y = torch.sum(Y, dim=3)
            Y = torch.flip(Y, dims=(3,))
            Y = Y[:, :, 200::, :]
            Y = (Y > 0).float()
            print(f"Processed mask shape for {sbj_ID}: {Y.shape}")
            print(f"Mask unique values for {sbj_ID}: {torch.unique(Y)}")
            print(f"ROI files for {sbj_ID}: {ROI_files}")

        subjects.append({
            "image": torch.as_tensor(combined_data, dtype=torch.float32),
            "mask": torch.as_tensor(Y, dtype=torch.float32),
            "patientID": sbj_ID
        })

    return subjects



class WholeImageDataset:
    def __init__(self, path, ratio=0.9, batch_size=1, patch_size=(128, 128, 32), samples_per_volume=32, max_length=200, num_workers=8):
        """
        Initialize WholeImageDataset with training and validation splits and queue setup.

        Args:
            path (str): Path to the dataset.
            ratio (float): Ratio of training subjects to total subjects.
            batch_size (int): Batch size for training.
            patch_size (tuple): Patch size for training and validation.
        """
        self.path = path
        self.batch_size = batch_size
        self.patch_size = patch_size

        # Get subjects and split into training and validation
        self.subjects = get_subjects(path)
        num_subjects = len(self.subjects)
        num_training_subjects = int(ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects
        self.training_subjects, self.validation_subjects = random_split(
            self.subjects, [num_training_subjects, num_validation_subjects]
        )




        # Define transformations
        self.train_transforms = Compose([
            ScaleIntensityD(keys="image"),
            EnsureTypeD(keys=["image", "mask"])
        ])
        self.val_transforms = Compose([
            ScaleIntensityD(keys="image"),
            EnsureTypeD(keys=["image", "mask"])
        ])

        # Call queuing and initialize training/validation queues
        self.patches_training_set, self.patches_validation_set = queuing(
            self.training_subjects, self.validation_subjects,
            self.patch_size, samples_per_volume, max_length, num_workers
        )




    def get_loaders(self):
        """
        Create DataLoaders for training and validation queues.

        Args:
            samples_per_volume (int): Number of samples per volume for the queue.
            max_length (int): Maximum queue length.
            num_workers (int): Number of workers for the queue.

        Returns:
            train_loader, val_loader: DataLoaders for training and validation.
        """
        """
        # Generate training and validation queues
        patches_training_set, patches_validation_set = self.queuing(
            training_subjects=self.training_subjects,
            validation_subjects=self.validation_subjects,
            patch_size=self.patch_size,
            samples_per_volume=samples_per_volume,
            max_length=max_length,
            num_workers=num_workers
        )
        """

        # Wrap TorchIO queues in PyTorch DataLoaders
        train_loader_patches = torch.utils.data.DataLoader(self.patches_training_set, batch_size=self.batch_size, drop_last=True, num_workers= 0)
        val_loader_patches = torch.utils.data.DataLoader(self.patches_validation_set, batch_size=1, drop_last=True, num_workers=0)


        print('Training set:', len(self.training_subjects), 'subjects')
        print('Validation set:', len(self.validation_subjects), 'subjects')
        print(f"Training subjects: {self.training_subjects}")
        print(f"Validation subjects: {self.validation_subjects}")

        return train_loader_patches, val_loader_patches



def queuing(training_subjects, validation_subjects, patch_size,
            samples_per_volume=4, max_length=200, num_workers=8):
    """
    Create training and validation TorchIO queues for patch-based sampling.

    Args:
        training_subjects (list): List of dictionaries, each representing a subject with keys 'image', 'mask', and 'patientID'.
        validation_subjects (list): Same structure as training_subjects, for validation data.
        patch_size (tuple): Size of patches to sample.
        samples_per_volume (int): Number of patches per volume.
        max_length (int): Maximum queue length.
        num_workers (int): Number of workers for data loading.

    Returns:
        patches_training_set, patches_validation_set: TorchIO Queues for training and validation.
    """

    # Convert dictionaries to TorchIO Subjects
    training_subjects_tio = [
        tio.Subject(
            image=tio.ScalarImage(tensor=subj["image"]),
            mask=tio.LabelMap(tensor=subj["mask"]),
            patientID=subj["patientID"]
        ) for subj in training_subjects
    ]

    validation_subjects_tio = [
        tio.Subject(
            image=tio.ScalarImage(tensor=subj["image"]),
            mask=tio.LabelMap(tensor=subj["mask"]),
            patientID=subj["patientID"]
        ) for subj in validation_subjects
    ]

    # Wrap the subjects into a TorchIO SubjectsDataset
    training_dataset = tio.SubjectsDataset(training_subjects_tio)
    validation_dataset = tio.SubjectsDataset(validation_subjects_tio)

    # Define label sampler for patch sampling
    sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name="mask",
        label_probabilities={0: 0.05, 1: 0.95}  # Adjust label probabilities for non-lesion and lesion areas
    )

    # Create TorchIO queues
    patches_training_set = tio.Queue(
        subjects_dataset=training_dataset,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_dataset,
        max_length=max_length,
        samples_per_volume=1,  # Single patch per volume for validation
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    # Debugging: Print subject IDs and patch sampling details
    print(f"Training subjects: {[subj['patientID'] for subj in training_subjects]}")
    print(f"Validation subjects: {[subj['patientID'] for subj in validation_subjects]}")

    # Verify the first few training patches
    print("\nVerifying patches from the training queue:")
    for i in range(min(5, len(patches_training_set))):  # Take a few examples to verify
        patch = patches_training_set[i]  # Get a single patch
        mask = patch["mask"][tio.DATA]  # Extract the mask tensor
        lesion_ratio = (mask > 0).float().mean().item()  # Calculate the ratio of lesion in the patch
        print(f"Patch {i + 1}: Lesion Ratio = {lesion_ratio * 100:.2f}%")

    return patches_training_set, patches_validation_set



"""
import matplotlib.pyplot as plt
import torch

def verify_dataloader(train_loader, val_loader, num_patches=3):

    print("Verifying the DataLoader...")

    # Check the first batch from the training loader
    train_batch = next(iter(train_loader))
    train_images = train_batch["image"][tio.DATA]
    train_masks = train_batch["mask"][tio.DATA]

    print("Training Batch:")
    print(f"  Image Shape: {train_images.shape}")
    print(f"  Mask Shape: {train_masks.shape}")
    print(f"  Image Type: {type(train_images)}")
    print(f"  Mask Type: {type(train_masks)}")
    print(f"  Image Tensor Min/Max: {train_images.min()}/{train_images.max()}")
    print(f"  Mask Unique Values: {torch.unique(train_masks)}")
    lesion_ratio = (train_masks > 0).float().mean().item()
    print(f"  Lesion Ratio: {lesion_ratio * 100:.2f}%")

    # Visualize patches from the training batch
    print("\nVisualizing patches from the training batch...")
    for i in range(min(num_patches, len(train_images))):
        image = train_images[i, 0].cpu().numpy()
        mask = train_masks[i, 0].cpu().numpy()
        middle_slice = image.shape[2] // 2
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Image {i + 1} (Middle Slice)")
        plt.imshow(image[:, :, middle_slice], cmap="gray")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f"Mask {i + 1} (Middle Slice)")
        plt.imshow(mask[:, :, middle_slice], cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Repeat for validation batch
    val_batch = next(iter(val_loader))
    val_images = val_batch["image"][tio.DATA]
    val_masks = val_batch["mask"][tio.DATA]

    print("\nValidation Batch:")
    print(f"  Image Shape: {val_images.shape}")
    print(f"  Mask Shape: {val_masks.shape}")
    print(f"  Image Tensor Min/Max: {val_images.min()}/{val_images.max()}")
    print(f"  Mask Unique Values: {torch.unique(val_masks)}")
    lesion_ratio = (val_masks > 0).float().mean().item()
    print(f"  Lesion Ratio: {lesion_ratio * 100:.2f}%")

    print("\nVisualizing patches from the validation batch...")
    for i in range(min(num_patches, len(val_images))):
        image = val_images[i, 0].cpu().numpy()
        mask = val_masks[i, 0].cpu().numpy()
        middle_slice = image.shape[2] // 2
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Validation Image {i + 1} (Middle Slice)")
        plt.imshow(image[:, :, middle_slice], cmap="gray")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f"Validation Mask {i + 1} (Middle Slice)")
        plt.imshow(mask[:, :, middle_slice], cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
"""

"""
# Example usage:
if __name__ == "__main__":
    # Dataset Path
    DATA_PATH = "/media/radiology/ULB-LISA/manifest-1713182663002/Advanced-MRI-Breast-Lesions"

    # Initialize the dataset and dataloaders
    dataset = WholeImageDataset(
        path=DATA_PATH,
        batch_size=2,
        patch_size=(128, 128, 16),  # Ensure divisible by 16
        samples_per_volume=16,
        max_length=200,
        num_workers=8
    )
    train_loader, val_loader = dataset.get_loaders()

    # Verify the dataloader
    verify_dataloader(train_loader, val_loader, 3)
"""