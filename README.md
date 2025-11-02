# MRI DICOM Loader Project

Automated pipeline for processing MRI DICOM data from TCIA/NBIA for deep learning tasks.

## 📁 Project Structure

```
mri/
├── data/
│   ├── raw/                    # Raw Excel files
│   │   ├── selected_patients_3.xlsx
│   │   ├── Target-Data_2019-12-05-2.xlsx
│   │   └── TCIA-Biopsy-Data_2020-07-14.xlsx
│   ├── splitted_images/        # Image-only records (197 rows)
│   │   ├── class=1/           # PIRADS 0-2 (17 series)
│   │   ├── class=2/           # PIRADS 3 (60 series)
│   │   ├── class=3/           # PIRADS 4 (60 series)
│   │   └── class=4/           # PIRADS 5 (60 series)
│   ├── splitted_info/          # Enriched records with targets & biopsies (10,881 rows)
│   │   ├── class=1/           # PIRADS 0-2 (905 rows)
│   │   ├── class=2/           # PIRADS 3 (4,830 rows)
│   │   ├── class=3/           # PIRADS 4 (3,458 rows)
│   │   └── class=4/           # PIRADS 5 (1,688 rows)
│   ├── tcia/                   # TCIA manifest files
│   │   ├── class1.tcia
│   │   ├── class2.tcia
│   │   ├── class3.tcia
│   │   └── class4.tcia
│   ├── nbia/                   # Downloaded DICOM files (from NBIA)
│   │   ├── class1/
│   │   ├── class2/
│   │   ├── class3/
│   │   └── class4/
│   └── processed/              # Converted per-slice images
│       ├── class1/
│       ├── class2/
│       ├── class3/
│       ├── class4/
│       └── manifest_all.csv
├── tools/
│   ├── convert_xlsx2parquet.py  # Excel → Parquet converter
│   ├── merge_datasets.py        # Merge multi-source data
│   ├── tcia_generator.py        # Generate TCIA manifests
│   └── dicom_converter.py       # DICOM → PNG converter
└── requirements.txt
```

## 🚀 Complete Workflow

### Step 1: Convert Excel to Parquet (✅ Complete)

Convert raw Excel data with PIRADS scores into class-partitioned parquet files.

```bash
conda activate mri
python tools/convert_xlsx2parquet.py
```

**Output:** Class-partitioned parquet files in `data/splitted_images/class={1,2,3,4}/`

- **Class 1:** PIRADS 0, 1, 2 (combined) - Low risk
- **Class 2:** PIRADS 3 - Intermediate risk
- **Class 3:** PIRADS 4 - High risk
- **Class 4:** PIRADS 5 - Very high risk

---

### Step 1b: Merge Multi-Source Data (✅ Complete)

Enrich patient records by merging three data sources: image metadata, target lesions, and biopsy results.

```bash
conda activate mri
python tools/merge_datasets.py
```

**Input sources:**
- `data/raw/selected_patients_3.xlsx` - MRI image metadata (197 records)
- `data/raw/Target-Data_2019-12-05-2.xlsx` - Target lesion data (1,617 targets from 840 patients)
- `data/raw/TCIA-Biopsy-Data_2020-07-14.xlsx` - Biopsy core data (24,783 cores from 1,150 patients)

**Output:** Enriched dataset in `data/splitted_info/` with:
- **10,881 total rows** (55.23x multiplication from 197 original)
- **48 total columns** (17 original + 5 target + 24 biopsy + 2 source tracking)
- Multiple rows per patient due to one-to-many relationships
- All 197 patients preserved with full data coverage

**Key features:**
- Left join preserves all image records
- Handles multiple targets per patient
- Handles multiple biopsy cores per patient
- Prefixed columns (`target_*`, `biopsy_*`) to avoid conflicts

---

### Step 2: Generate TCIA Manifests (✅ Complete)

Extract MRI Series Instance UIDs and create downloadable TCIA manifest files.

```bash
conda activate mri
python tools/tcia_generator.py
```

**Note:** Uses `data/splitted_images/` as source (image-only records without data expansion).

**Output:** TCIA manifest files in `data/tcia/`
- `class1.tcia` - 17 series UIDs
- `class2.tcia` - 58 series UIDs
- `class3.tcia` - 60 series UIDs
- `class4.tcia` - 59 series UIDs

---

### Step 3: Download DICOM Files from TCIA (🔄 Manual)

Use the NBIA Data Retriever to download DICOM files:

1. Install [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
2. Open each `.tcia` manifest file from `data/tcia/`
3. Download to corresponding class directory in `data/nbia/`:
   - `class1.tcia` → `data/nbia/class1/`
   - `class2.tcia` → `data/nbia/class2/`
   - `class3.tcia` → `data/nbia/class3/`
   - `class4.tcia` → `data/nbia/class4/`

**Expected structure after download:**
```
data/nbia/class1/
  manifest-class1-.../
    Prostate-MRI-US-Biopsy/
      Prostate-MRI-US-Biopsy-0001/
        1.3.6.1.../
          1.2.840.../
            1-01.dcm
            1-02.dcm
            ...
```

---

### Step 4: Convert DICOM to Per-Slice Images (✅ Complete)

Convert NBIA-downloaded DICOM series to per-slice PNG images for deep learning.

#### Process Single Class
```bash
conda activate mri
python tools/dicom_converter.py --class 1
```

#### Process All Classes (Recommended)
```bash
conda activate mri
python tools/dicom_converter.py --all
```

#### Custom Paths
```bash
python tools/dicom_converter.py --all \
    --input data/nbia \
    --output data/processed
```

**Output:**
```
data/processed/
  class1/
    case_0001/
      {SeriesInstanceUID}/
        meta.json              # Series metadata
        images/                # Per-slice images
          0000.png
          0001.png
          ...
        labels/                # Per-slice masks (if available)
          0000.png
          0001.png
          ...
    manifest.csv              # Per-class manifest
  class2/
    ...
  manifest_all.csv            # Combined manifest for all classes
```

**Manifest CSV columns:**
- `case_id`, `series_uid`, `slice_idx`
- `image_path`, `mask_path` (empty if no labels)
- `num_labels`, `spacing_x`, `spacing_y`, `spacing_z`
- `modality`, `study_date`, `manufacturer`
- `class` (in combined manifest)

---

## 📦 Installation

### Setup Conda Environment
```bash
# Create environment
conda create -n mri python=3.12 -y
conda activate mri

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration & Options

### DICOM Converter Options

The `dicom_converter.py` can be customized via `ConverterConfig`:

```python
from pathlib import Path
from tools.dicom_converter import DicomConverter, ConverterConfig

cfg = ConverterConfig(
    root_dir=Path("data/nbia/class1"),
    out_dir=Path("data/processed/class1"),
    image_format="png",              # "png" or "jpg"
    resample_spacing=None,           # e.g., (1.0, 1.0, 1.0) for isotropic
    save_nifti=False,                # Set True to save volume.nii.gz
    window=None,                     # Manual window (lo, hi)
    auto_window=True,                # Auto percentile windowing
    percentile_window=(1.0, 99.0),   # Percentile range
)

converter = DicomConverter(cfg)
converter.convert_all()
```

---

## 📊 Data Statistics

### Current Dataset (After Step 2)

| Class | PIRADS | Series Count | Risk Level |
|-------|--------|--------------|------------|
| 1     | 0-2    | 17           | Low        |
| 2     | 3      | 58           | Intermediate |
| 3     | 4      | 60           | High       |
| 4     | 5      | 59           | Very High  |
| **Total** | - | **194**     | -          |

---

## 🔍 Features

### Label/Mask Support

The converter automatically detects and processes:
1. **DICOM SEG** objects (if `highdicom` available)
2. **NIfTI** labelmaps (`.nii`, `.nii.gz`)
3. **NRRD** labelmaps (`.nrrd`)

Masks are exported as binary PNGs (0/255) aligned to image slices.

---

### Issue: Segmentation Not Found
- DICOM SEG requires `highdicom` package
- Alternatively, export masks from 3D Slicer as NIfTI
- Place `.nii.gz` files in same directory as DICOM series

### Issue: Memory Errors on Large Datasets
Process one class at a time:
```bash
python tools/dicom_converter.py --class 1
python tools/dicom_converter.py --class 2
# etc.
```

---

### Output Format
- Images: 8-bit PNG (0-255), normalized via windowing
- Masks: Binary PNG (0=background, 255=foreground)
- Spacing preserved in manifest CSV for 3D reconstruction

---

## 📚 References

- [TCIA Prostate-MRI-US-Biopsy Collection](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/)
- [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
- [DICOM SEG Standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.51.html)

---

## 👥 Workflow Summary

```
Excel Data (3 sources)
    ↓ [convert_xlsx2parquet.py]
Image-Only Records (data/splitted_images/)
    ↓ [merge_datasets.py]
Enriched Records (data/splitted_info/) - 10,881 rows with targets & biopsies
    │
    └→ [tcia_generator.py]
       TCIA Manifests (.tcia files)
           ↓ [NBIA Data Retriever - Manual]
       DICOM Files (data/nbia/)
           ↓ [dicom_converter.py]
       Per-Slice Images (data/processed/)
           ↓
       Deep Learning Pipeline
```

**Two parallel datasets:**
1. **splitted_images/** - Compact (197 rows) for TCIA manifest generation
2. **splitted_info/** - Enriched (10,881 rows) for detailed analysis with all patient data
