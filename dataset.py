import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils import data

np.random.seed(0)

TRAIN_DATASET_SIZE = 393930
NUM_PTS = 971
CROP_SIZE = 128


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name="image"):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(
            sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA
        )
        sample["scale_coef"] = f

        if "landmarks" in sample:
            landmarks = sample["landmarks"].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample["landmarks"] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=CROP_SIZE, elem_name="image"):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2

        margin_h = max(margin_h, 0)
        margin_w = max(margin_w, 0)
        assert (margin_h >= 0) and (margin_w >= 0)

        sample[self.elem_name] = img[
            margin_h : margin_h + self.size, margin_w : margin_w + self.size
        ]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if "landmarks" in sample:
            landmarks = sample["landmarks"].reshape(-1, 2)
            margin = torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)
            landmarks -= margin[None, :]
            sample["landmarks"] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", data_frac=1.0):
        assert split in {"train", "val", "test"}
        super().__init__()

        landmark_file_name = (
            os.path.join(root, "landmarks.csv")
            if split != "test"
            else os.path.join(root, "test_points.csv")
        )
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []
        self.train_indices, self.val_indices = self._get_train_val_indices()

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for _ in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp)):
                if i == 0:
                    # skip header
                    continue

                if np.random.rand() > data_frac:
                    continue

                if (split == "train") and (i not in self.train_indices):
                    continue
                elif (split == "val") and (i not in self.val_indices):
                    continue

                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int16, elements[1:]))
                    assert len(landmarks) == NUM_PTS * 2
                    landmarks = np.array(landmarks, dtype=np.int16)
                    landmarks = landmarks.reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def _get_train_val_indices(self):
        with open("utils/train_indices.txt", "r") as train_file:
            train_indices = train_file.read().split(",")
            train_indices = set(map(int, train_indices))

        with open("utils/val_indices.txt", "r") as val_file:
            val_indices = val_file.read().split(",")
            val_indices = set(map(int, val_indices))

        assert not (train_indices & val_indices)
        assert sorted(train_indices | val_indices) == list(range(TRAIN_DATASET_SIZE))

        return train_indices, val_indices

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            # for train and validation
            landmarks = self.landmarks[idx]
            assert isinstance(landmarks, torch.Tensor)
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert isinstance(image, np.ndarray)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f

    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]

    return landmarks
