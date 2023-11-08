import utils
import xmltodict

from PIL import Image
import numpy as np

def gen(image_path, labels):
  for i in range(len(image_path)):
    label = labels[i]

    image = Image.open(image_path[i]).resize((448, 448))
    image = np.array(image)

    yield (image, label)
