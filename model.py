import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model


class ResNet50V2:
    def __new__(cls, shape=(64, 64, 3), classifications=2):
        cls.shape = shape
        input_tensor = Input(shape=cls.shape)
        resnet = keras.applications.resnet_v2.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=cls.shape
        )
        # resnet.trainable = False
        x = resnet.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(classifications, name="dense_logits")(x)
        predictions = Activation("softmax", dtype="float32", name="predictions")(x)
        model = Model(inputs=resnet.input, outputs=predictions)

        return model


class ResNet101V2:
    def __new__(cls, shape=(64, 64, 3)):
        cls.shape = shape
        input_tensor = Input(shape=cls.shape)
        resnet = keras.applications.ResNet101V2(include_top=False, weights="imagenet", input_tensor=input_tensor)
        x = resnet.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        model = Model(inputs=resnet.input, outputs=predictions)

        return model


class VGG16:
    def __new__(cls, shape=(64, 64, 3)):
        cls.shape = shape
        input_tensor = Input(shape=cls.shape)
        resnet = keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)
        x = resnet.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        model = Model(inputs=resnet.input, outputs=predictions)

        return model


class MLP:
    def __new__(cls, shape=(64, 64, 3)):
        cls.shape = shape
        model = Sequential()
        model.add(Dense(512, activation="relu", input_shape=cls.shape))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation="relu"))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(2, activation="softmax"))

        return model


class MobileNetV2:
    def __new__(cls, shape=(64, 64, 3)) -> tf.keras.Model:
        cls.shape = shape
        resnet = keras.applications.MobileNetV2(include_top=False, input_shape=cls.shape, pooling=None)
        h = Flatten()(resnet.output)
        model_output = Dense(2, activation="softmax")(h)
        model = Model(resnet.input, model_output)

        return model
