import copy
import os
from datetime import datetime

from tensorflow import keras as keras
from tensorflow.keras import mixed_precision as mixed_precision
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config as cfg
from keras_call_back import KerasCallBack
from model import ResNet50V2

# TensorCore 混合精度
# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_policy(policy)
# print('Compute dtype: {}'.format(policy.compute_dtype))
# print('Variable dtype: {}'.format(policy.variable_dtype))


def main():
    #######################
    # Model Building
    #######################
    input_shape = [224, 224]
    input_shape_plus_ch = copy.copy(input_shape)
    input_shape_plus_ch.append(3)  # RGB 3channel
    model = ResNet50V2(shape=input_shape_plus_ch)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(),
        metrics=[
            "accuracy",
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
        ],
    )
    # KM = KerasMetrics()
    # model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=KM.generate_metrics(2))

    #######################
    # Dataset Definition
    #######################
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=5,
        horizontal_flip=True,
        vertical_flip=True,
        # preprocessing_function=keras.applications.resnet_v2.preprocess_input,
    )
    train_generator = train_datagen.flow_from_directory(
        "dataset3",
        batch_size=cfg.BATCH_SIZE,
        target_size=input_shape,
        class_mode="categorical",
        shuffle=True,
        seed=1,
        subset="training",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        # preprocessing_function=keras.applications.resnet_v2.preprocess_input,
    )
    val_generator = val_datagen.flow_from_directory(
        "dataset3",
        batch_size=cfg.BATCH_SIZE,
        target_size=input_shape,
        class_mode="categorical",
        shuffle=True,
        seed=1,
        subset="validation",
    )

    # train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False, seed=1)
    # val_generator = val_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False, seed=1)

    print("ラベル確認(train)->{}: {}class".format(train_generator.class_indices, len(train_generator.class_indices)))

    #######################
    # Train Preparation
    #######################
    kc = KerasCallBack()
    tdatetime = datetime.now()
    tstr = tdatetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(f"tmp/weights{tstr}", exist_ok=True)
    callbacks = []
    # callbacks.append(
    #     kc.ModelCheckpointCallBack(
    #         f"tmp/weights{tstr}/"
    #         + "weights.epoch{epoch:04d}-loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}.hdf5",
    #         monitor="val_loss",
    #         save_best_only=True,
    #         save_freq="epoch",
    #         save_weights_only=False,
    #         include_optimizer=True,
    #     )
    # )
    # callbacks.append(
    #     kc.ModelCheckpointCallBack(
    #         f"tmp/weights{tstr}/"
    #         + "weights.epoch{epoch:04d}-loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}.hdf5",
    #         monitor="val_accuracy",
    #         save_best_only=True,
    #         save_freq="epoch",
    #         save_weights_only=False,
    #         include_optimizer=True,
    #     )
    # )
    from tensorflow.keras.callbacks import ModelCheckpoint

    callbacks.append(
        ModelCheckpoint(
            f"tmp/weights{tstr}/" + "weights.epoch{epoch:04d}.hdf5",
            # monitor="val_accuracy",
            save_best_only=False,
            save_freq=1,
            save_weights_only=False,
            include_optimizer=True,
        )
    )
    callbacks.append(kc.TensorBoardCallBack())

    #######################
    # Train Start
    #######################
    model.fit(
        train_generator,
        batch_size=cfg.BATCH_SIZE,
        epochs=cfg.EPOCH,
        validation_data=val_generator,
        shuffle=False,
        steps_per_epoch=train_generator.n // cfg.BATCH_SIZE,
        validation_steps=val_generator.n // cfg.BATCH_SIZE,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
