import os, random, string
from shutil import copyfile
from PIL import Image
import numpy as np
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input, decode_predictions

labels_file = open("classifier.csv", "w")
labels_file.write("path,is_dog\n")

ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


test_img_dir_path = "/cs/casmip/israelwei/Computer_Vision/Project_CV/google_data/dog_breeds"
for root, dirs, files in os.walk(test_img_dir_path, topdown = False):
	for name in files:
		img_path = os.path.join(root, name)
		cur_pred_dog = dog_detector(img_path)
		labels_file.write(img_path + "," + str(cur_pred_dog) + "\n")

labels_file.close()
