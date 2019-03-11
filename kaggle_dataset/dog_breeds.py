# Template - Kaggle and Stanford dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("input"))

# Any results you write to the current directory are saved as output.

import keras

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img

from keras.preprocessing.image import ImageDataGenerator

pd.options.mode.chained_assignment = None  # default='warn'


from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


labels = pd.read_csv("input/labels.csv")
# Get the top 120 breeds which is what we use in this notebook
top_breeds = sorted(list(labels['breed'].value_counts().head(120).index))
train_dogs = labels[labels['breed'].isin(top_breeds)]

# Get the labels of the top 120
target_labels = train_dogs['breed']

# One hot code the labels - need this for the model
one_hot = pd.get_dummies(target_labels, sparse = True)
one_hot_labels = np.asarray(one_hot)

# add the actual path name of the pics to the data set
train_folder = "input/train/"
train_dogs['image_path'] = train_dogs.apply( lambda x: (train_folder + x["id"] + ".jpg" ), axis=1)
train_dogs.head()

# Convert the images to arrays which is used for the model. Inception uses image sizes of 299 x 299
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')

# Split the data into train and validation. The stratify parm will insure  train and validation  
# will have the same proportions of class labels as the input dataset.
x_train, x_validation, y_train, y_validation = train_test_split(train_data, target_labels, test_size=0.2, stratify=np.array(target_labels), random_state=100)

# Need to convert the train and validation labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()

# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30, 
                                   # zoom_range = 0.3, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=10)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=10, seed=10)

# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with 20 classes 
#(there will be 120 classes for the final submission)
x = Dense(512, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)

# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile with Adam
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = 175,
                      validation_data = val_generator,
                      validation_steps = 44,
                      epochs = 15,
                      verbose = 2)

model_json = model.to_json()
with open("model_description.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("trained_weights.h5")

