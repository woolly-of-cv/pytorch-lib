fixed_height = 800

import glob
from PIL import Image

for im in glob.glob('./images'+'/**.jp*g'):
    try:
        image = Image.open(im)
        height_percent = (fixed_height / float(image.size[1]))
        width_size = int((float(image.size[0]) * float(height_percent)))
        image = image.resize((width_size, fixed_height), Image.BILINEAR)
        image.save(im)
    except:
        print(f"Error in Processing: {im}")
