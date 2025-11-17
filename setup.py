# from setuptools import setup, find_packages

# setup(
#     name="pyCurveWave",
#     version="0.1.0",
#     author="Nehal Mantri",
#     author_email="nehalsmantri@gmail.com",
#     description="A Python library for Curvelet and Wavelet-based image augmentations and blending.",
#     packages=find_packages(),
#     install_requires=[
#         "numpy",
#         "opencv-python",
#         "matplotlib",
#         "pywavelets",
#         "scikit-image",
        
#     ],
#     python_requires=">=3.8",
# )
from setuptools import setup, find_packages
import sys

# Custom check for MATLAB Engine
def check_matlab_engine():
    try:
        import matlab.engine  # noqa
        return True
    except ImportError:
        print(
            "\n  MATLAB Engine for Python is not installed.\n"
            "   Some features of pyCurveWave (like Curvelet transforms via MATLAB) "
            "require it.\n"
            "   To install manually, run:\n"
            "   cd \"C:\\Program Files\\MATLAB\\R2023b\\extern\\engines\\python\"\n"
            "   python setup.py install\n"
        )
        return False

# Run the check (optional, won’t stop installation)
check_matlab_engine()

setup(
    name="pyCurveWave",
    version="0.1.0",
    description="A Python library for hybrid Curvelet and Wavelet image analysis and augmentation",
    author="Nehal Mantri, Aditya Mukati",
    author_email="nehalsmantri@gmail.com, adityamukati8564@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "pywavelets",
        "scikit-image"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
