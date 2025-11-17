
import numpy as np
import cv2

def random_attention_mask(shape, mode="smooth", p=0.5):
    """
    Generate a random attention mask.

    Parameters:
    -----------
    shape : tuple
        Shape of the image (H, W) or (H, W, C).
    mode : str
        Type of mask to generate:
            - "binary"  : random 0/1 mask
            - "uniform" : random floats in [0,1]
            - "smooth"  : smooth random mask using Gaussian blur
            - "circle"  : random circular region mask
    p : float
        Probability for "binary" mode (fraction of ones).

    Returns:
    --------
    mask : np.ndarray
        Mask in shape (H, W) with float32 values in [0,1].
    """
    H, W = shape[:2]

    if mode == "binary":
        mask = (np.random.rand(H, W) < p).astype(np.float32)

    elif mode == "uniform":
        mask = np.random.rand(H, W).astype(np.float32)

    elif mode == "smooth":
        # random noise + Gaussian blur → soft attention
        mask = np.random.rand(H, W).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # normalize

    elif mode == "circle":
        mask = np.zeros((H, W), dtype=np.float32)
        radius = np.random.randint(min(H, W)//8, min(H, W)//3)
        center = (np.random.randint(W), np.random.randint(H))
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)

    else:
        raise ValueError("Invalid mode")

    return mask


def blend_images_weighted(p, q, alpha=0.5):
    """
    Blend two images p and q with a global weight alpha:
        out = alpha * p + (1 - alpha) * q

    Parameters:
    -----------
    p, q : np.ndarray
        Images of same shape (HxW or HxWxC), uint8 or float.
    alpha : float
        Blend weight for image p in [0,1]. (1-alpha) is weight for q.

    Returns:
    --------
    out : np.ndarray
        Blended image, same dtype as input.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")

    orig_dtype = p.dtype
    p_f = p.astype(np.float32)
    q_f = q.astype(np.float32)

    out_f = alpha * p_f + (1.0 - alpha) * q_f

    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out_f, 0, 255).round().astype(orig_dtype)
    else:
        out = out_f.astype(orig_dtype)
    return out


def blend_images_np(p, q, a):
    """
    Blend two images p and q with attention mask a:
        out = a * p + (1-a) * q

    p, q: HxW or HxWxC numpy arrays (uint8 or float)
    a:   HxW or HxWx1 or HxWxC mask. Can be float in [0,1] or boolean.
    Returns: blended image with same dtype as inputs (uint8 if inputs uint8).
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")

    # convert mask to float32 in [0,1]
    a_np = np.array(a)
    if a_np.dtype == np.bool_:
        a_f = a_np.astype(np.float32)
    else:
        a_f = a_np.astype(np.float32)
        if a_f.max() > 1.0:
            a_f = a_f / 255.0

    # ensure mask has channel dim if images are color
    if p.ndim == 3 and a_f.ndim == 2:
        a_f = a_f[..., None]

    # convert inputs to float for blending
    orig_dtype = p.dtype
    p_f = p.astype(np.float32)
    q_f = q.astype(np.float32)

    out_f = a_f * p_f + (1.0 - a_f) * q_f

    # convert back
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out_f, 0, 255).round().astype(orig_dtype)
    else:
        out = out_f.astype(orig_dtype)
    return out