import tensorflow as tf
from model import build_model
from data_preprocessing import load_and_preprocess_data
from config import EPOCHS, BATCH_SIZE, MODEL_PATH
from utils import plot_history  # Импортируем функцию из utils.py

def train_model():
    # Загрузка и подготовка данных
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Построение модели
    model = build_model()

    # Компиляция модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Обучение модели
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Визуализация истории обучения
    plot_history(history)  # Вызов функции для отображения графика точности

    # Сохранение модели
    model.save(MODEL_PATH)

    return model, history

if __name__ == "__main__":
    train_model()
