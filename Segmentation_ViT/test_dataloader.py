import os
import random
import torch
import torchio as tio
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensityD, EnsureTypeD
from glob import glob
from natsort import natsorted
import re

included_ids2 = ['AMBL-596', 'AMBL-597', 'AMBL-598', 'AMBL-599', 'AMBL-600', 'AMBL-601', 'AMBL-602', 'AMBL-603', 'AMBL-604', 'AMBL-605']
included_ids = ['AMBL-001', 'AMBL-003', 'AMBL-004', 'AMBL-005', 'AMBL-006', 'AMBL-007', 'AMBL-008', 'AMBL-009', 'AMBL-010', 'AMBL-011', 'AMBL-012', 'AMBL-013', 'AMBL-014', 'AMBL-015', 'AMBL-016', 'AMBL-017', 'AMBL-018', 'AMBL-019', 'AMBL-020', 'AMBL-021', 'AMBL-022', 'AMBL-023', 'AMBL-024', 'AMBL-025', 'AMBL-026', 'AMBL-027', 'AMBL-028', 'AMBL-029', 'AMBL-031', 'AMBL-032', 'AMBL-033', 'AMBL-034', 'AMBL-035', 'AMBL-036', 'AMBL-037', 'AMBL-038', 'AMBL-039', 'AMBL-040', 'AMBL-041', 'AMBL-042', 'AMBL-043', 'AMBL-044', 'AMBL-045', 'AMBL-046', 'AMBL-047', 'AMBL-048', 'AMBL-049', 'AMBL-050', 'AMBL-056', 'AMBL-063', 'AMBL-068', 'AMBL-070', 'AMBL-072', 'AMBL-082', 'AMBL-085', 'AMBL-088', 'AMBL-091', 'AMBL-097', 'AMBL-496', 'AMBL-507', 'AMBL-514', 'AMBL-541', 'AMBL-555', 'AMBL-557', 'AMBL-559', 'AMBL-561', 'AMBL-562', 'AMBL-563', 'AMBL-564', 'AMBL-565', 'AMBL-566', 'AMBL-567', 'AMBL-568', 'AMBL-569', 'AMBL-570', 'AMBL-571', 'AMBL-572', 'AMBL-573', 'AMBL-574', 'AMBL-575', 'AMBL-577', 'AMBL-578', 'AMBL-579', 'AMBL-580'] #, 'AMBL-581', 'AMBL-582', 'AMBL-583', 'AMBL-584', 'AMBL-585', 'AMBL-586', 'AMBL-587', 'AMBL-588', 'AMBL-590'] #, 'AMBL-591', 'AMBL-593', 'AMBL-594', 'AMBL-595'] # 'AMBL-596', 'AMBL-597', 'AMBL-598', 'AMBL-599', 'AMBL-600', 'AMBL-601', 'AMBL-602', 'AMBL-603', 'AMBL-604', 'AMBL-605' ['AMBL-606', 'AMBL-607', 'AMBL-608', 'AMBL-610', 'AMBL-612', 'AMBL-613', 'AMBL-615', 'AMBL-616', 'AMBL-617', 'AMBL-618', 'AMBL-619', 'AMBL-620', 'AMBL-621', 'AMBL-622', 'AMBL-623', 'AMBL-625', 'AMBL-626', 'AMBL-627', 'AMBL-628', 'AMBL-629', 'AMBL-631', 'AMBL-632']

"""
def get_subjects_light(path):
    subjects = []
    sbj_files = natsorted(glob(os.path.join(path, '*/*/5.000000-AX*')))

    for sbj_file in sbj_files:
        tmp_sbj_name = re.split("AMBL", sbj_file)[1][:4]
        sbj_ID = 'AMBL' + tmp_sbj_name

        if sbj_ID not in included_ids:
            continue

        files_phase_1 = natsorted(glob(os.path.join(sbj_file, '1-*')))
        files_phase_3 = natsorted(glob(os.path.join(sbj_file, '3-*')))
        if not files_phase_1 or not files_phase_3:
            continue

        img1 = tio.ScalarImage(files_phase_1).data.permute(3, 1, 2, 0)  # [D, H, W, 1] -> [1, H, W, D]
        img3 = tio.ScalarImage(files_phase_3).data.permute(3, 1, 2, 0)
        combined = torch.cat([img1, img3], dim=0)
        combined = combined[:, :, 200:, :]  # crop

        Nslices = combined.shape[-1]

        ROI_files = natsorted(glob(os.path.join(path, sbj_ID, '*/*ROI*')))
        if not ROI_files:
            Y = torch.zeros((1, combined.shape[1], combined.shape[2], combined.shape[3]))
        else:
            subject_seg = tio.LabelMap(ROI_files)
            c, h, w, total_slices = subject_seg.data.shape
            NROI = int(total_slices / Nslices)
            Y = subject_seg.data.reshape(c, h, w, NROI, Nslices)
            Y = torch.sum(Y, dim=3)
            Y = torch.flip(Y, dims=(3,))
            Y = Y[:, :, 200:, :]
            Y = (Y > 0).float()

        subjects.append({
            "image": combined,
            "mask": Y,
            "patientID": sbj_ID
        })

    print(f"✅ Total processed subjects: {len(subjects)}")
    return subjects


class WholeImageDataset:

    def __init__(self, path, batch_size=1, patch_size=(128, 128, 32), samples_per_volume=32,
                 max_length=200, num_workers=1, seed=42, train_ratio=0.8, val_ratio=0.1):

        self.path = path
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.subjects = get_subjects_light(path)
        random.seed(seed)
        all_ids = [s["patientID"] for s in self.subjects]
        random.shuffle(all_ids)

        num_total = len(all_ids)
        num_train = int(train_ratio * num_total)
        num_val = int(val_ratio * num_total)
        num_test = num_total - num_train - num_val

        train_ids = all_ids[:num_train]
        val_ids = all_ids[num_train:num_train + num_val]
        test_ids = all_ids[num_train + num_val:]

        id_to_subject = {s["patientID"]: s for s in self.subjects}
        self.train_subjects = [id_to_subject[i] for i in train_ids]
        self.val_subjects = [id_to_subject[i] for i in val_ids]
        self.test_subjects = [id_to_subject[i] for i in test_ids]

        self.train_transforms = Compose([
            ScaleIntensityD(keys="image"),
            EnsureTypeD(keys=["image", "mask"])
        ])
        self.val_transforms = Compose([
            EnsureTypeD(keys=["image", "mask"])
        ])

        def to_subject_list(subjects, transforms):
            return [
                tio.Subject(
                    image=tio.ScalarImage(tensor=s["image"].float()),
                    mask=tio.LabelMap(tensor=s["mask"].float()),
                    patientID=s["patientID"]
                )for s in subjects
            ]

        train_subjects_tio = to_subject_list(self.train_subjects, self.train_transforms)
        val_subjects_tio = to_subject_list(self.val_subjects, self.val_transforms)
        test_subjects_tio = to_subject_list(self.test_subjects, self.val_transforms)

        self.patches_training_set, self.patches_validation_set, self.patches_test_set = queuing(
            train_subjects_tio, val_subjects_tio, test_subjects_tio,
            patch_size, samples_per_volume, max_length, num_workers
        )

    def get_loaders(self):
        train_loader = DataLoader(self.patches_training_set, batch_size=self.batch_size, drop_last=True, num_workers=0)
        val_loader = DataLoader(self.patches_validation_set, batch_size=1, drop_last=True, num_workers=0)
        test_loader = DataLoader(self.patches_test_set, batch_size=1, drop_last=False, num_workers=0)

        return (
            train_loader,
            val_loader,
            test_loader,
            [s["patientID"] for s in self.train_subjects],
            [s["patientID"] for s in self.val_subjects],
            [s["patientID"] for s in self.test_subjects],
        )

def queuing(train_subjects, val_subjects, test_subjects, patch_size,
            samples_per_volume=4, max_length=200, num_workers=1):

    sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name="mask",
        label_probabilities={0: 0.05, 1: 0.95}
    )

    def make_queue(subjects_dataset, is_test=False):
        return tio.Queue(
            subjects_dataset=tio.SubjectsDataset(subjects_dataset),
            max_length=max_length,
            samples_per_volume=1 if is_test else samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=not is_test,
            shuffle_patches=not is_test,
        )

    train_queue = make_queue(train_subjects, is_test=False)
    val_queue = make_queue(val_subjects, is_test=False)
    test_queue = make_queue(test_subjects, is_test=True)

    print(f"\nSubjects: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
    return train_queue, val_queue, test_queue
"""

import os
import random
import torch
import torchio as tio
from torch.utils.data import DataLoader
from monai.transforms import Compose, ScaleIntensityD, EnsureTypeD
from glob import glob
from natsort import natsorted
import re



def get_subjects_light(path):
    subjects = []
    sbj_files = natsorted(glob(os.path.join(path, '*/*/5.000000-AX*')))

    for sbj_file in sbj_files:
        tmp_sbj_name = re.split("AMBL", sbj_file)[1][:4]
        sbj_ID = 'AMBL' + tmp_sbj_name

        if sbj_ID not in included_ids:
            continue

        files_phase_1 = natsorted(glob(os.path.join(sbj_file, '1-*')))
        files_phase_3 = natsorted(glob(os.path.join(sbj_file, '3-*')))
        if not files_phase_1 or not files_phase_3:
            continue

        ROI_files = natsorted(glob(os.path.join(path, sbj_ID, '*/*ROI*')))

        subjects.append({
            "phase1": files_phase_1,
            "phase3": files_phase_3,
            "roi": ROI_files,
            "patientID": sbj_ID
        })

    print(f"\u2705 Total processed subjects: {len(subjects)}")
    return subjects

class WholeImageDataset:

    def __init__(self, path, batch_size=1, patch_size=(128, 128, 32), samples_per_volume=32,
                 max_length=200, num_workers=1, seed=42, train_ratio=0.8, val_ratio=0.1):

        self.path = path
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.subjects = get_subjects_light(path)
        random.seed(seed)
        all_ids = [s["patientID"] for s in self.subjects]
        random.shuffle(all_ids)

        num_total = len(all_ids)
        num_train = int(train_ratio * num_total)
        num_val = int(val_ratio * num_total)
        num_test = num_total - num_train - num_val

        train_ids = all_ids[:num_train]
        val_ids = all_ids[num_train:num_train + num_val]
        test_ids = all_ids[num_train + num_val:]

        id_to_subject = {s["patientID"]: s for s in self.subjects}
        self.train_subjects = [id_to_subject[i] for i in train_ids]
        self.val_subjects = [id_to_subject[i] for i in val_ids]
        self.test_subjects = [id_to_subject[i] for i in test_ids]

        self.train_transforms = Compose([
            ScaleIntensityD(keys="image"),
            EnsureTypeD(keys=["image", "mask"])
        ])
        self.val_transforms = Compose([
            EnsureTypeD(keys=["image", "mask"])
        ])

        def to_subject_list(subjects, transforms):
            subject_list = []
            for s in subjects:
                img1 = tio.ScalarImage(s["phase1"]).data.permute(3, 1, 2, 0)  # [1, H, W, D]
                img3 = tio.ScalarImage(s["phase3"]).data.permute(3, 1, 2, 0)
                #combined = torch.cat([img1, img3], dim=0)[:, :, 200:, :]
                diff = (img3 - img1)[:, :, 200:, :]
                Nslices = diff.shape[-1]

                if not s["roi"]:
                    Y = torch.zeros((1, diff.shape[1], diff.shape[2], diff.shape[3]))
                else:
                    seg = tio.LabelMap(s["roi"])
                    c, h, w, total_slices = seg.data.shape
                    NROI = int(total_slices / Nslices)
                    Y = seg.data.reshape(c, h, w, NROI, Nslices)
                    Y = torch.sum(Y, dim=3)
                    Y = torch.flip(Y, dims=(3,))
                    Y = Y[:, :, 200:, :]
                    Y = (Y > 0).float()

                subject_list.append(tio.Subject(
                    image=tio.ScalarImage(tensor=diff.float()),
                    mask=tio.LabelMap(tensor=Y.float()),
                    patientID=s["patientID"]
                ))
            return subject_list

        train_subjects_tio = to_subject_list(self.train_subjects, self.train_transforms)
        val_subjects_tio = to_subject_list(self.val_subjects, self.val_transforms)
        test_subjects_tio = to_subject_list(self.test_subjects, self.val_transforms)

        self.patches_training_set, self.patches_validation_set, self.patches_test_set = queuing(
            train_subjects_tio, val_subjects_tio, test_subjects_tio,
            patch_size, samples_per_volume, max_length, num_workers
        )

    def get_loaders(self):
        train_loader = DataLoader(self.patches_training_set, batch_size=self.batch_size, drop_last=True, num_workers=0)
        val_loader = DataLoader(self.patches_validation_set, batch_size=1, drop_last=True, num_workers=0)
        test_loader = DataLoader(self.patches_test_set, batch_size=1, drop_last=False, num_workers=0)

        return (
            train_loader,
            val_loader,
            test_loader,
            [s["patientID"] for s in self.train_subjects],
            [s["patientID"] for s in self.val_subjects],
            [s["patientID"] for s in self.test_subjects],
        )

def queuing(train_subjects, val_subjects, test_subjects, patch_size,
            samples_per_volume=4, max_length=200, num_workers=1):

    sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name="mask",
        label_probabilities={0: 0.05, 1: 0.95}
    )

    def make_queue(subjects_dataset, is_test=False):
        return tio.Queue(
            subjects_dataset=tio.SubjectsDataset(subjects_dataset),
            max_length=max_length,
            samples_per_volume=1 if is_test else samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
            shuffle_subjects=not is_test,
            shuffle_patches=not is_test,
        )

    train_queue = make_queue(train_subjects, is_test=False)
    val_queue = make_queue(val_subjects, is_test=False)
    test_queue = make_queue(test_subjects, is_test=True)

    print(f"\nSubjects: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
    return train_queue, val_queue, test_queue
