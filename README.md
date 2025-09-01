# LDCT Nodule Detection

This repository provides a minimal training pipeline for detecting lung nodules
larger than 3 mm on low-dose CT (LDCT) images. The approach follows a three-stage
strategy:

1. **Pre-train** on the [LUNA16](https://luna16.grand-challenge.org/) dataset
   using ``.mhd/.raw`` volumes and corresponding CSV annotations.
2. **Fine-tune** on hospital-collected 3 mm DICOM slices with bounding boxes stored
   in a CSV file.
3. **Evaluate and adapt** to limited 1 mm clinical data comprised of JPG images
   and an ``annotations.xml`` file.

The goal is to achieve a recall above 85% on the 1 mm clinical set. The provided
scripts are intended as starting points and require further experimentation,
regularization and hyperparameter tuning to reach production-level performance.

## Repository Structure

```
ldct/
 datasets.py  # Dataset loaders for the three data sources
  train.py     # Example training pipeline
```

## Dataset Layouts

The code assumes the following directory structures:

```
LUNA16/
  *.mhd
  *.raw

3mm/
  patient_ID_1/
    *.dicom
  patient_ID_2/
    *.dicom

1mm/
  patient_ID_1/
    annotations.xml
    images/
      *.jpg
  patient_ID_2/
    annotations.xml
    images/
      *.jpg
```

## Usage

Install the required dependencies:

```bash
pip install torch torchvision SimpleITK pydicom Pillow numpy
```

Run the pipeline (adjust paths accordingly):

```bash
python -m ldct.train \
  --luna16-root /path/to/luna16/images \
  --luna16-csv /path/to/luna16/annotations.csv \
  --hospital-root /path/to/3mm/dicom_root \
  --hospital-csv /path/to/hospital_annotations.csv \
  --clinical-root /path/to/1mm/root \
  --pretrain-epochs 5 --ft-epochs 5
```

The script will report recall on the 1 mm dataset. If the recall is below 85%,
consider additional fine-tuning, data augmentation and hyperparameter tuning.
