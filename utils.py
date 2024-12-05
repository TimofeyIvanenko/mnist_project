# utils.py

import matplotlib.pyplot as plt

def plot_history(history):
    # Визуализация истории обучения
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на тесте')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()
