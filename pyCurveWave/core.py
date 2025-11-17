import cv2
import pywt
import numpy as np
import random
import matlab.engine
import numpy as np
from skimage import io, color
import random
from .utils import blend_images_weighted, blend_images_np
def wavelet_augment(image, temperature=None, params=None, verbose=True):
    """
    Augment an image using wavelet transforms.

    Parameters
    ----------
    image : np.ndarray
        Input color image (H, W, 3).
    temperature : float or None
        If provided, controls random augmentation intensity (0 → subtle, 1 → strong).
        If None, 'params' must be provided to deterministically control augmentation.
    params : dict or None
        Optional manually specified augmentation parameters. Example:
            params = {
                'wavelet_family': 'db',
                'wavelet': 'db2',
                'decomposition_level': 2,
                'random_factors_high_freq': [1.1, 0.95, ...],
                'gaussian_noise_sigma_scaled_by_level': 3.2,
                'brightness_factor_LL': 1.03,
                'contrast_factor_LL': 0.98,
                'boolean_b': True,
                'dropped_band_index': 1,
            }
    verbose : bool
        Whether to print augmentation parameters.

    Returns
    -------
    augmented_image : np.ndarray
        The augmented image.
    params_channels : list of dict
        The parameters used for each channel.
    """

    assert image.ndim == 3 and image.shape[2] == 3, "Input must be a color image (H, W, 3)"
    if temperature is None and params is None:
        raise ValueError("Either 'temperature' or 'params' must be provided")

    augmented_channels = []
    params_channels = []

    def random_factor(base=1.0, min_scale=0.7, max_scale=1.3, expansion=2.0):
        safe_low, safe_high = min_scale, max_scale
        center = base
        low = center - (center - safe_low) * (1 + expansion * temperature)
        high = center + (safe_high - center) * (1 + expansion * temperature)
        return random.uniform(low, high)

    def random_sigma(level):
        base_sigma = 1.0 * level
        return base_sigma * (1 + 14 * temperature)

    for ch in range(3):
        channel_img = image[:, :, ch].astype(np.float32)

        if params is None:
            # TEMPERATURE-BASED RANDOM AUGMENTATION
            families = ['haar', 'db', 'sym', 'coif', 'bior', 'rbio']
            max_families = int(1 + temperature * (len(families) - 1))
            wavelet_family = random.choice(families[:max_families])
            wavelet = random.choice(pywt.wavelist(wavelet_family))

            safe_level = 1
            wild_level = 3
            max_allowed = pywt.dwt_max_level(channel_img.shape[0], wavelet)
            level = int(round(safe_level + temperature * (wild_level - safe_level)))
            level = min(level, max_allowed)

            coeffs = pywt.wavedec2(channel_img, wavelet=wavelet, level=level)
            new_coeffs = [coeffs[0]]
            random_factors = []

            for i in range(1, len(coeffs)):
                cH, cV, cD = coeffs[i]
                factor = random_factor(base=1.0, min_scale=0.9, max_scale=1.1, expansion=3)
                random_factors.append(factor)
                cH, cV, cD = cH * factor, cV * factor, cD * factor

                sigma = random_sigma(level)
                noise = np.random.normal(0, sigma, cD.shape)
                cD += noise
                new_coeffs.append((cH, cV, cD))

            brightness_factor = random_factor(base=1.0, min_scale=0.95, max_scale=1.05, expansion=4)
            contrast_factor = random_factor(base=1.0, min_scale=0.95, max_scale=1.05, expansion=4)
            LL = coeffs[0] * contrast_factor + (brightness_factor - 1) * 50 * temperature
            new_coeffs[0] = LL

            drop_prob = temperature * 0.8
            b = random.random() < drop_prob
            dropped_band_index = None
            if b and len(new_coeffs) > 1:
                drop_idx = random.randint(1, len(new_coeffs) - 1)
                dropped_band_index = drop_idx
                new_coeffs[drop_idx] = (
                    np.zeros_like(new_coeffs[drop_idx][0]),
                    np.zeros_like(new_coeffs[drop_idx][1]),
                    np.zeros_like(new_coeffs[drop_idx][2]),
                )

            used_params = {
                'wavelet_family': wavelet_family,
                'wavelet': wavelet,
                'decomposition_level': level,
                'random_factors_high_freq': random_factors,
                'gaussian_noise_sigma_scaled_by_level': sigma,
                'brightness_factor_LL': brightness_factor,
                'contrast_factor_LL': contrast_factor,
                'boolean_b': b,
                'dropped_band_index': dropped_band_index,
            }

        else:
            # PARAMETER-BASED DETERMINISTIC AUGMENTATION
            p = params
            wavelet_family = p['wavelet_family']
            wavelet = p['wavelet']
            level = p['decomposition_level']
            random_factors = p['random_factors_high_freq']
            sigma = p['gaussian_noise_sigma_scaled_by_level']
            brightness_factor = p['brightness_factor_LL']
            contrast_factor = p['contrast_factor_LL']
            b = p['boolean_b']
            dropped_band_index = p['dropped_band_index']

            coeffs = pywt.wavedec2(channel_img, wavelet=wavelet, level=level)
            new_coeffs = [coeffs[0] * contrast_factor + (brightness_factor - 1) * 50]
            for i in range(1, len(coeffs)):
                cH, cV, cD = coeffs[i]
                factor = random_factors[min(i-1, len(random_factors)-1)]
                cH, cV, cD = cH * factor, cV * factor, cD * factor
                noise = np.random.normal(0, sigma, cD.shape)
                cD += noise
                new_coeffs.append((cH, cV, cD))

            if b and dropped_band_index and 1 <= dropped_band_index < len(new_coeffs):
                new_coeffs[dropped_band_index] = (
                    np.zeros_like(new_coeffs[dropped_band_index][0]),
                    np.zeros_like(new_coeffs[dropped_band_index][1]),
                    np.zeros_like(new_coeffs[dropped_band_index][2]),
                )

            used_params = p

        augmented_channel = pywt.waverec2(new_coeffs, wavelet=wavelet)
        augmented_channel = np.clip(augmented_channel, 0, 255).astype(np.uint8)

        augmented_channels.append(augmented_channel)
        params_channels.append(used_params)

    augmented_image = np.stack(augmented_channels, axis=2)

    if verbose:
        print("Augmentation parameters by channel:")
        for i, p in enumerate(params_channels):
            print(f"Channel {i}:")
            for k, v in p.items():
                print(f"  {k}: {v}")

    return augmented_image, params_channels



def curvelet_augment_color(
    img_path: str,
    eng,
    temperature: float = None,
    params: dict = None,
    verbose: bool = True
):
    """
    Augment a color image using Curvelet transform via MATLAB CurveLab.

    Modes:
        - temperature-based random augmentation (when temperature ∈ [0,1])
        - deterministic augmentation (when params dict is provided)

    Args:
        img_path: Path to the image file
        eng: MATLAB engine instance
        temperature: Controls augmentation aggressiveness (0=conservative, 1=aggressive)
        params: Optional dict to manually specify augmentation parameters:
            {
                'attenuation_angle': 45,
                'attenuation_factor': 1.2,
                'boolean_b': False,
                'inject_texture': True,
                'noise_strength': 0.05
            }
        verbose: Whether to print used parameters
    """
    
    # Input validation
    if temperature is None and params is None:
        raise ValueError("Either 'temperature' or 'params' must be provided.")
    
    img = io.imread(img_path)
    img = img.astype(np.float32) / 255.0
    img_ycbcr = color.rgb2ycbcr(img).astype(np.float32)
    
    y_channel = (img_ycbcr[:, :, 0] - 16) / (235 - 16)
    y_channel = np.clip(y_channel, 0, 1)
    
    if params is not None:
        # ====== DETERMINISTIC MODE ======
        attenuation_angle = params.get('attenuation_angle', 0)
        attenuation_factor = params.get('attenuation_factor', 1.0)
        b = params.get('boolean_b', False)
        inject_texture = params.get('inject_texture', False)
        noise_strength = params.get('noise_strength', 0.01)
        used_params = params.copy()
    else:
        # ====== TEMPERATURE MODE ======
        temperature = max(0, min(1, temperature))
        max_angle = int(15 + (165 * temperature))
        attenuation_angle = random.randint(0, max_angle)
        
        low_factor_range = (0.9 - 0.6 * temperature, 1.1 + 0.9 * temperature)
        attenuation_factor = random.uniform(*low_factor_range)
        
        b = random.choice([True, False])
        inject_texture = random.choice([True, False])
        
        low_noise = 0.005 * (1 - temperature) + 0.0001 * temperature
        high_noise = 0.1 + 0.4 * temperature
        noise_strength = random.uniform(low_noise, high_noise)
        
        used_params = {
            'attenuation_angle': attenuation_angle,
            'attenuation_factor': attenuation_factor,
            'boolean_b': b,
            'inject_texture': inject_texture,
            'noise_strength': noise_strength
        }

    # ====== MATLAB CURVELET PROCESSING ======
    channel_mat = matlab.double(y_channel.tolist())
    coeffs = eng.fdct_wrapping(channel_mat, 0)
    
    for s in range(len(coeffs)):
        num_wedges = len(coeffs[s])
        chosen_wedge = attenuation_angle % num_wedges
        
        for w in range(num_wedges):
            if w == chosen_wedge:
                if b:  # Angular dropout
                    num_weak = 0.75 if params else random.uniform(0.5, 1.0)
                    coeffs[s][w] = eng.times(coeffs[s][w], num_weak)
                else:  # Oriented attenuation
                    coeffs[s][w] = eng.times(coeffs[s][w], float(attenuation_factor))
            else:
                jitter_low = 0.95 - 0.45 * (temperature if temperature else 0)
                jitter_high = 1.05 + 0.45 * (temperature if temperature else 0)
                coeffs[s][w] = eng.times(coeffs[s][w], float(np.random.uniform(jitter_low, jitter_high)))
            
            if inject_texture:
                scale_factor = 1.0 / (s + 1)
                ns = noise_strength * scale_factor
                coeff_array = np.array(eng.abs(coeffs[s][w]))
                noise = np.random.normal(0, ns * np.mean(coeff_array), coeff_array.shape)
                coeffs[s][w] = eng.plus(coeffs[s][w], matlab.double(noise.tolist()))
    
    # ====== RECONSTRUCTION ======
    y_augmented = np.array(eng.ifdct_wrapping(coeffs, 0)).real
    y_augmented = np.clip(y_augmented, 0, 1)
    y_augmented = y_augmented * (235 - 16) + 16
    
    img_ycbcr_augmented = img_ycbcr.copy()
    img_ycbcr_augmented[:, :, 0] = y_augmented
    aug_img = color.ycbcr2rgb(img_ycbcr_augmented)
    aug_img = np.clip(aug_img, 0, 1)
    aug_img = (aug_img * 255).astype(np.uint8)
    
    if verbose:
        print("\nCurvelet augmentation parameters:")
        for k, v in used_params.items():
            print(f"  {k}: {v}")
    
    return aug_img, used_params


def blend(p, q, method='weighted', **kwargs):
    """
    Main blending function to blend two images using a specified method.

    Parameters:
    -----------
    p, q : np.ndarray
        Input images (must have same shape).
    method : str
        Blending method: 'weighted' or 'mask'.
    kwargs : dict
        Additional parameters depending on method:
            - For 'weighted': alpha (float)
            - For 'mask': a (np.ndarray or scalar mask)

    Returns:
    --------
    out : np.ndarray
        Blended image.
    """
    if p.shape != q.shape:
        raise ValueError("Images p and q must have same shape")

    method = method.lower()

    if method == 'weighted':
        alpha = kwargs.get('alpha', 0.5)
        out = blend_images_weighted(p, q, alpha)
    elif method == 'mask':
        a = kwargs.get('a', None)
        if a is None:
            raise ValueError("Mask 'a' must be provided for 'mask' blending")
        out = blend_images_np(p, q, a)
    else:
        raise ValueError(f"Unknown blend method: {method}. Choose 'weighted' or 'mask'.")

    return out

import matlab.engine
import os

def start_matlab_engine(curvelab_path="C:/Users/Nehal/Downloads/CurveLab-2.1.3/fdct_wrapping_matlab", verbose=True):
    """
    Start a MATLAB engine session and add CurveLab to the MATLAB path.

    Parameters
    ----------
    curvelab_path : str or None
        Path to the CurveLab 'fdct_wrapping_matlab' folder.
        Example: r"C:/Users/Username/Downloads/CurveLab-2.1.3/fdct_wrapping_matlab"
        If None, the function will prompt the user to specify it later.
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    eng : matlab.engine.MatlabEngine
        An active MATLAB engine session with CurveLab added to the path.
    """
    if verbose:
        print("Starting MATLAB engine... This may take a few seconds.")

    try:
        eng = matlab.engine.start_matlab()
        if curvelab_path is not None:
            if not os.path.exists(curvelab_path):
                raise FileNotFoundError(f"CurveLab path not found: {curvelab_path}")
            eng.eval(f"addpath('{curvelab_path.replace(os.sep, '/')}');", nargout=0)
        else:
            if verbose:
                print("⚠️ No CurveLab path provided. You can manually add it using:")
                print("   eng.eval(\"addpath('<your_CurveLab_path>');\", nargout=0)")

        if verbose:
            print("MATLAB engine ready!\n")

        return eng

    except Exception as e:
        raise RuntimeError(f"Failed to start MATLAB engine: {e}")
