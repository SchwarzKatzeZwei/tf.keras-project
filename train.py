from dataset import DataSet
from datetime import datetime as dt
from keras_call_back import KerasCallBack
from model import MLP, VGG16, MobileNetV2, ResNet50V2, ResNet101V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
import copy
import os

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any')


def main():
    batch_size = 16
    input_shape = [32, 32]
    input_shape_plus_ch = copy.copy(input_shape)
    input_shape_plus_ch.append(3)

    model = MobileNetV2(shape=input_shape_plus_ch)
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"]
                  )

    (x_train, y_train), (x_test, y_test) = DataSet.cifar10()

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # train_generator = train_datagen.flow_from_directory("dataset/train5",
    #                                                     batch_size=batch_size,
    #                                                     target_size=input_shape,
    #                                                     class_mode="categorical"
    #                                                     )
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False, seed=1)
    val_datagen = ImageDataGenerator(rescale=1. / 255, )
    # val_generator = val_datagen.flow_from_directory("dataset/val",
    #                                                 batch_size=batch_size,
    #                                                 target_size=input_shape,
    #                                                 class_mode="categorical"
    #                                                 )
    # val_org_generator = val_datagen.flow_from_directory("dataset/val",
    #                                                     batch_size=batch_size,
    #                                                     target_size=input_shape,
    #                                                     class_mode="categorical"
    #                                                     )
    val_generator = val_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False, seed=1)

    # print("ラベル確認(train)->{}: {}class".format(train_generator.class_indices, len(train_generator.class_indices)))

    # kc = KerasCallBack()
    # tdatetime = dt.now()
    # tstr = tdatetime.strftime("%Y%m%d%H%M%S")
    # os.makedirs(f"tmp/weights{tstr}", exist_ok=True)
    # callbacks = []
    # callbacks.append(
    #     kc.ModelCheckpointCallBack(
    #         f"tmp/weights{tstr}/" +
    #         "weights.epoch{epoch:04d}-loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}.hdf5",
    #         monitor="val_loss",
    #         save_best_only=True,
    #         period=1,
    #         save_weights_only=False,
    #         include_optimizer=True))
    # callbacks.append(
    #     kc.ModelCheckpointCallBack(
    #         f"tmp/weights{tstr}/" +
    #         "weights.epoch{epoch:04d}-loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}.hdf5",
    #         monitor="val_accuracy",
    #         save_best_only=True,
    #         period=1,
    #         save_weights_only=False,
    #         include_optimizer=True))
    # model.fit(preprocessor_gen(train_generator), epochs=300, verbose=1, steps_per_epoch=train_generator.n // batch_size,
    #           validation_data=val_generator)
    # model.fit(val_org_generator, batch_size=batch_size, epochs=300, verbose=1, steps_per_epoch=val_org_generator.n // batch_size,
    #           validation_data=val_org_generator)
    # model.fit(preprocessor_gen(train_generator), batch_size=batch_size, epochs=10000, verbose=1,
    #           steps_per_epoch=train_generator.n // batch_size,
    #           validation_data=val_org_generator,
    #           validation_steps=val_org_generator.n // batch_size,
    #           callbacks=callbacks)
    model.fit(
        train_generator,
        batch_size=batch_size,
        epochs=10,
        validation_data=val_generator,
        shuffle=False,
        steps_per_epoch=train_generator.n // batch_size,
        validation_steps=val_generator.n // batch_size,
    )


if __name__ == "__main__":
    main()
