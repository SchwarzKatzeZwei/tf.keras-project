import copy
import glob
import os
import time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras as keras

import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

input_shape = [224, 224]
input_shape_plus_ch = copy.copy(input_shape)
input_shape_plus_ch.append(3)

#######################
# Train Model Load
#######################
path = glob.glob(cfg.MODEL_PATH)[0]
model = keras.models.load_model(path)

start = time.perf_counter()

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    # preprocessing_function=keras.applications.resnet_v2.preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    "dataset3", batch_size=1, target_size=input_shape, class_mode="categorical", shuffle=False, seed=1, subset="validation"
)

train_generator = val_datagen.flow_from_directory(
    "dataset3", batch_size=1, target_size=input_shape, class_mode="categorical", shuffle=False, seed=1, subset="training"
)

model.evaluate(train_generator)
model.evaluate(val_generator)

# count = 0
# for idx, ((img, lbl), filename) in enumerate(zip(val_generator, val_generator.filenames)):
#     pred = model.predict(img)
#     true_label = np.argmax(lbl, axis=1)
#     pred_label = np.argmax(pred, axis=1)
#     print(f"T{true_label}, P{pred_label}, {true_label==pred_label} {pred[0][pred_label][0]:>7.2%} {filename}")
#     count += 1 if true_label == pred_label else 0

#     if idx == val_generator.n:
#         break

# print(f"{count} / {val_generator.n} = {count / val_generator.n:%}")
