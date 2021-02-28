import tensorflow as tf


class DataSet:

    @staticmethod
    def cifar10():
        return tf.keras.datasets.cifar10.load_data()
