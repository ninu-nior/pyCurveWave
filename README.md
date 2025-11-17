pyCurveWave
Wavelet + Curvelet Augmentation Library for Python

pyCurveWave is a lightweight Python library that provides powerful image augmentation utilities using:

Wavelet transforms (pure Python)

Curvelet transforms (via MATLAB + CurveLab)

Temperature-based augmentation for random but controlled variations

Parameter-based augmentation for deterministic transforms

This library is ideal for:

Computer vision researchers

Deep learning augmentation pipelines

Frequency-domain image analysis

Medical imaging / plant disease / texture recognition

🚀 Features
✔ Wavelet Augmentation

Multi-level wavelet decomposition

High-frequency manipulation

LL brightness/contrast modulation

Dropband simulation

Temperature-driven randomness

✔ Curvelet Augmentation

Uses MATLAB Engine + CurveLab (FDCT Wrapping)

Per-channel attenuation

Angle-based coefficient scaling

High-frequency texture injection

Adjustable randomization

✔ Utility Functions

Image blending

Weighted combinations

Normalization helpers

📦 Installation
1️⃣ Clone the repo
git clone https://github.com/<your-username>/pyCurveWave.git
cd pyCurveWave

2️⃣ Install locally
pip install -e .

3️⃣ Install required packages
pip install -r requirements.txt

⚙️ MATLAB + CurveLab Setup (For Curvelet Augmentation)

Curvelet functions require:

MATLAB installed

MATLAB Engine for Python installed

CurveLab added to MATLAB’s path

Install MATLAB engine:

cd "C:\Program Files\MATLAB\R2022b\extern\engines\python"
python setup.py install


Download CurveLab:
https://www.curvelet.org/

🧪 Usage Example
Start MATLAB engine
from pyCurveWave import core

eng = core.start_matlab_engine(
    curvelab_path="C:/path/to/CurveLab/fdct_wrapping_matlab"
)

Wavelet Augmentation (Temperature Mode)
aug, params = core.wavelet_augment(image, temperature=0.7)

Wavelet Augmentation (Custom Parameter Mode)
custom_params = {
    "wavelet_family": "db",
    "wavelet": "db2",
    "decomposition_level": 2,
    "random_factors_high_freq": [1.1, 1.0, 0.9],
}
aug, params = core.wavelet_augment(image, params=custom_params)

Curvelet Augmentation (Temperature Mode)
aug, params = core.curvelet_augment_color(
    "image.jpg", eng=eng, temperature=0.6
)