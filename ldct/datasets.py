import os
import csv
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class Luna16SliceDataset(Dataset):
    """Dataset that converts LUNA16 3D volumes to individual slices.

    Each sample returns a 2D slice and the corresponding bounding boxes
    of nodules in that slice (if any). The LUNA16 dataset stores scans
    in ``.mhd`` header files paired with ``.raw`` volumes. A separate
    CSV file containing nodule annotations is expected with columns
    ``seriesuid`` (study id), ``coordX``, ``coordY``, ``coordZ`` and
    ``diameter_mm``. Only nodules with diameter >3 mm are used.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_csv: str,
        transform=None,
    ) -> None:
        if sitk is None:
            raise ImportError("SimpleITK is required for reading .mhd files")

        self.root_dir = root_dir
        self.transform = transform
        self.annotations: Dict[str, List[Dict[str, float]]] = {}

        with open(annotation_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["diameter_mm"]) <= 3.0:
                    continue
                self.annotations.setdefault(row["seriesuid"], []).append(
                    {
                        "x": float(row["coordX"]),
                        "y": float(row["coordY"]),
                        "z": float(row["coordZ"]),
                        "d": float(row["diameter_mm"]),
                    }
                )

        self.series_uids = sorted(self.annotations)

        # Preload volume paths to avoid scanning disk repeatedly
        self.mhd_paths = {
            uid: os.path.join(root_dir, uid + ".mhd") for uid in self.series_uids
        }

    def __len__(self) -> int:  # pragma: no cover - simple length
        return len(self.series_uids)

    def __getitem__(self, idx: int):
        uid = self.series_uids[idx]
        path = self.mhd_paths[uid]
        image = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(image)  # shape (slices, H, W)
        spacing = image.GetSpacing()[2]  # slice spacing in mm

        boxes: List[List[float]] = []
        for ann in self.annotations[uid]:
            z_idx = int(round((ann["z"] - image.GetOrigin()[2]) / spacing))
            radius = ann["d"] / 2.0
            xmin = ann["x"] - radius
            xmax = ann["x"] + radius
            ymin = ann["y"] - radius
            ymax = ann["y"] + radius
            boxes.append([z_idx, xmin, ymin, xmax, ymax])

        sample = {"volume": volume, "boxes": boxes}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Hospital3mmDataset(Dataset):
    """Dataset for 3mm DICOM slices with annotations provided in a CSV.

    The CSV format example:

    ``DicomSet id,create time,index,type,start_x,end_x,start_y,end_y``

    Each patient has a folder named with ``DicomSet id`` that contains
    individual DICOM files. The ``index`` column identifies which slice
    the annotation belongs to (zero based). Only rows with ``type`` ==
    ``Nodule`` are used.
    """

    def __init__(self, root_dir: str, annotation_csv: str, transform=None) -> None:
        if pydicom is None:
            raise ImportError("pydicom is required for reading DICOM files")

        self.root_dir = root_dir
        self.transform = transform
        self.records: List[Tuple[str, int, Tuple[float, float, float, float]]] = []

        with open(annotation_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["type"].strip().lower() != "nodule":
                    continue
                patient = row["DicomSet id"].strip()
                slice_idx = int(row["index"])
                box = (
                    float(row["start_x"]),
                    float(row["start_y"]),
                    float(row["end_x"]),
                    float(row["end_y"]),
                )
                self.records.append((patient, slice_idx, box))

    def __len__(self) -> int:  # pragma: no cover
        return len(self.records)

    def __getitem__(self, idx: int):
        patient, slice_idx, box = self.records[idx]
        patient_dir = os.path.join(self.root_dir, patient)
        files = sorted(os.listdir(patient_dir))
        dcm_path = os.path.join(patient_dir, files[slice_idx])
        ds = pydicom.dcmread(dcm_path)
        image = ds.pixel_array.astype(np.float32)
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


class Clinical1mmDataset(Dataset):
    """Dataset for 1mm JPG images with annotation provided in an XML file."""

    def __init__(self, root_dir: str, annotation_xml: str, transform=None) -> None:
        if Image is None:
            raise ImportError("Pillow is required for reading JPG images")

        self.root_dir = root_dir
        self.transform = transform
        self.items: List[Tuple[str, List[Tuple[float, float, float, float]]]] = []

        tree = ET.parse(annotation_xml)
        root = tree.getroot()
        for image in root.iter("image"):
            name = image.attrib["name"]
            boxes: List[Tuple[float, float, float, float]] = []
            for box in image.iter("box"):
                if box.attrib.get("label", "").lower() != "nodule":
                    continue
                boxes.append(
                    (
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"]),
                    )
                )
            self.items.append((name, boxes))

    def __len__(self) -> int:  # pragma: no cover
        return len(self.items)

    def __getitem__(self, idx: int):
        name, boxes = self.items[idx]
        path = os.path.join(self.root_dir, name)
        image = np.array(Image.open(path).convert("L"))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
        }
        if self.transform:
            image, target = self.transform(image, target)
        return image, target
