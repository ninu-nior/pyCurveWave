# pyCurveWave

**Wavelet + Curvelet Augmentation Library for Python**

pyCurveWave is a lightweight Python library that provides robust frequency-domain image augmentation for research and deep learning applications.

---

## Overview

pyCurveWave offers:

* Wavelet transforms (pure Python)
* Curvelet transforms (via MATLAB Engine and CurveLab)
* Hybrid Augmentations (Curvelet+Wavelet)
* Temperature-based augmentation for controlled randomness
* Parameter-based augmentation for deterministic transforms

Ideal for:

* Computer vision research
* Deep learning augmentation pipelines
* Frequency-domain image analysis
* Medical imaging, plant disease, and texture recognition

---

## Features

### Wavelet Augmentation

* Multi-level decomposition
* High-frequency manipulation
* LL (approximation) band brightness/contrast modulation
* Dropband simulation
* Temperature-driven randomness

### Curvelet Augmentation

* Uses MATLAB Engine + CurveLab (FDCT Wrapping)
* Per-channel attenuation
* Angle-based coefficient scaling
* High-frequency texture injection
* Adjustable randomization

### Utility Functions

* Image blending, weighted combinations
* Normalization helpers

---

## Installation

```bash
# Clone the repository
git clonehttps://github.com/ninu-nior/pyCurveWave.git
cd pyCurveWave

# Install locally (editable mode)
pip install -e .


```
## MATLAB + CurveLab Setup

Curvelet augmentation requires:

1.  **MATLAB** installed
2.  **MATLAB Engine for Python**
3.  **CurveLab toolbox** added to MATLAB’s path

**1. Install MATLAB Engine for Python:**

(Adjust the path for your specific MATLAB version)

```bash
cd "C:\Program Files\MATLAB\R2022b\extern\engines\python"
python setup.py install
```

**2. Download CurveLab**

Get the toolbox from the official website: https://www.curvelet.org/

## Usage

Curvelet augmentation requires:


**Start Matlab Engine:**

(Adjust the path for your specific MATLAB version)

```bash
from pyCurveWave import core

eng = core.start_matlab_engine(
    curvelab_path="C:/path/to/CurveLab/fdct_wrapping_matlab"
)
```

**Wavelet Augmentation (Temperature Mode)**
```bash
aug, params = core.wavelet_augment(image, temperature=0.7)
```
**Wavelet Augmentation (Custom Parameter Mode)**
```bash
custom_params = {
    "wavelet_family": "db",
    "wavelet": "db2",
    "decomposition_level": 2,
    "random_factors_high_freq": [1.1, 1.0, 0.9],
}
aug, params = core.wavelet_augment(image, params=custom_params)
```
**Curvelet Augmentation (Temperature Mode)**
```bash
aug, params = core.curvelet_augment_color(
    "image.jpg", eng=eng, temperature=0.6
)
```

