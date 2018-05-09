import cv2
from PIL import Image
import os

# config - FileckrAPI APIKey
# env file does not work
key = "229ecfccbdfeccd00da6abe65546fc6a"
secret = "e1710962e2f38e68"

# config - search keyword
keywords = [
    'ストウブ',
    'シャスール',
    'バーミキュラ',
    'ル・クルーゼ'
]

# config - number of images to search
image_count = 500

# config - save image filepath
dataset_dir = "/var/local/tensorflow_billy/data/image/"

def resize_bilinear(file, size):
    name, ext = os.path.splitext(os.path.basename(file))
    img = Image.open(file, 'r')
    resize_img = img.resize((size, size))
    resize_img.save(file, ext.replace(".", ""), quality=100, optimize=True)