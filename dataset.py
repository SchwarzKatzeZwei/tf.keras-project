import tensorflow as tf


class DataSet:
    @staticmethod
    def cifar10():
        """Sample data from CIFAR10 dataset.
        (x_train, y_train), (x_test, y_test) = DataSet.cifar10()
        """
        return tf.keras.datasets.cifar10.load_data()
