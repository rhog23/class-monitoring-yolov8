"""
This code implements the Guided Bilateral Filtering (GBF) for local contrast enhancement (LCE). 
GBF is an alternative to CLAHE (Contrast Limited Adaptive Histogram Equalization). 
"""

import cv2
import numpy as np
from skimage import filters


def gbf_lce(image, guidance_image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Applies Local Contrast Enhancement using Guided Bilateral Filtering (GBF).

    Args:
        `image`: Input image (grayscale).
        `guidance_image`: Guidance image (grayscale, with enhanced local contrast).
        `diameter`: Diameter of the filtering window (default: 9).
        `sigma_color`: Standard deviation for color space similarity (default: 75).
        `sigma_space`: Standard deviation for spatial domain similarity (default: 75).

    Returns:
        The locally contrast enhanced image (grayscale).
    """

    # Apply guided bilateral filtering using the guidance image
    enhanced_image = filters.guided_filter(
        image,
        guidance_image,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    )

    return enhanced_image


# creates CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Load the image and guidance image (grayscale)
image = cv2.imread(
    "../data/Camera5_MESIN-08_MESIN-08_20240403184200_20240403191447_774495.jpg",
    cv2.IMREAD_GRAYSCALE,
)
guidance_image = clahe.apply(image)

# Apply GBF LCE
enhanced_image = gbf_lce(image, guidance_image)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Guidance Image", guidance_image)
cv2.imshow("Enhanced Image (GBF)", enhanced_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
