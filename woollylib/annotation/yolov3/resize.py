fixed_height = 800

import glob
from PIL import Image

print(glob.glob('Images'+'/*/**.jp*g'))

for im in glob.glob('Images'+'/*/**.jp*g'):
    image = Image.open(im)
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image = image.resize((width_size, fixed_height), Image.BILINEAR)
    image.save(im)
