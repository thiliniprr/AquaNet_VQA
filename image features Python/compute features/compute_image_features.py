import sys
import os
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path
from img_to_vec import Img2Vec
from PIL import Image

input_path = './test_images'

print("Getting vectors for test images...\n")
img2vec = Img2Vec()

# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}

for file in os.listdir(input_path):
    if not file.startswith('.'):
        filename = os.fsdecode(file)
        img = Image.open(os.path.join(input_path, filename)).convert('RGB')
        #img = Image.open(os.path.join(input_path, filename))
        vec = img2vec.get_vec(img)
        pics[filename] = vec
  

print('image features', vec)  # a vector from one image
print('pics', pics)           # vectors from all images in the folder

'''
# Get a vector from img2vec, returned as a torch FloatTensor
vec = img2vec.get_vec(img, tensor=True)
# Or submit a list
#vectors = img2vec.get_vec(list_of_PIL_images)

'''
