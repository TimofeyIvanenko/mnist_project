# predict.py

import numpy as np
import tensorflow as tf
from config import MODEL_PATH

class_names_numbers = [
    "Цифра 0", "Цифра 1", "Цифра 2", "Цифра 3", "Цифра 4", 
    "Цифра 5", "Цифра 6", "Цифра 7", "Цифра 8", "Цифра 9"
]

def make_predictions():
    # Загрузка модели
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Загружаем тестовые данные
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  # Нормализация

    # Прогнозирование
    pred = model.predict(x_test)
    
    # Классификация для первого изображения
    class_index = np.argmax(pred[0])
    print(f"Предсказанная цифра для первого изображения: {class_names_numbers[class_index]}")

if __name__ == "__main__":
    make_predictions()
