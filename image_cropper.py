from pathlib import Path
from PIL import Image
import numpy as np
import os

# Get the source and target directories
source_dir = 'Test_images_uncropped'
target_dir = 'Test_images'
# Iterate over images in the source directory
images = Path(source_dir).glob('*.jpg')
for image in images:
    img = Image.open(image)
    # Create a mask to set the background pixels greater than 245 (white) to 0
    mask = np.array(img)
    mask[mask > 245] = 0
    rows, cols, colors = np.where(mask)
    # Search for the maximum and minimum pixels of the silhouette
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    # Crop the image and save it to target directory
    crop = np.array(img)[y_min:y_max + 1, x_min:x_max + 1]
    cropped_img = Image.fromarray(crop)
    file_path = os.path.join(target_dir, image.name)
    cropped_img.save(file_path, 'JPEG')