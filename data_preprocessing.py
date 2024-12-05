# data_preprocessing.py

import tensorflow as tf

def load_and_preprocess_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Нормализация данных
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return (x_train, y_train), (x_test, y_test)
