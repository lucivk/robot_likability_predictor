# import required module
from pathlib import Path
from PIL import Image
import numpy as np
import os

# get the path/directory
source_dir = 'Robot_images_uncropped'
target_dir = 'Robot_images'
# iterate over files in that directory
images = Path(source_dir).glob('*.jpg')
for image in images:
    img = Image.open(image)
    mask = np.array(img)
    mask[mask > 245] = 0
    rows, cols, colors = np.where(mask)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    crop = np.array(img)[y_min:y_max + 1, x_min:x_max + 1]
    cropped_img = Image.fromarray(crop)
    file_path = os.path.join(target_dir, image.name)
    cropped_img.save(file_path, 'JPEG')