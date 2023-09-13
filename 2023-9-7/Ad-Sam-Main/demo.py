from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

# Let's load the popular vermeer image
MY_TOKEN = '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'

image = load_image("/data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/sam-subset-11187/sa_1.jpg")
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
print(canny_image.shape)