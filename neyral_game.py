import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Убирает информационные и предупреждающие сообщения от TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()
import numpy as np
from random import randint


def get_model():
    if os.path.exists('guessing_model.keras'):
        model = tf.keras.models.load_model('guessing_model.keras')
        print("Модель загружена.")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, 6)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        print("Создана новая модель.")

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def get_model_with_memory():
    input_shape = (sequence_length, 2)
    if os.path.exists('memory_model.keras'):
        model = tf.keras.models.load_model('guessing_model.keras')
        print("Модель загружена.")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.LSTM(32, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        print("Создана новая модель.")

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def neural_network_guess(model):
    target = randint(1, 100)
    guess = np.array([[50]])  # Начальное предположение
    attempts = 0
    history_lenght: int = 5
    history: list = []

    # print("Компьютер загадал число от 1 до 100.")

    while True:
        prediction = model.predict(guess)[0][0]
        attempts += 1
        if abs(prediction - target) < 1:  # Учитываем округление
            print(f"Нейросеть угадала число {target} за {attempts} попыток.")
            return [attempts, target]
        else:
            if prediction < target:
                # print("Нейросеть предполагает:", int(prediction), "→ больше!")
                feedback = 1  # Меньше загаданного
            else:
                # print("Нейросеть предполагает:", int(prediction), "→ меньше!")
                feedback = -1  # Больше загаданного

            # Корректируем вес модели
            target_value = target if feedback == 1 else target - 1
            model.fit(guess, np.array([[target_value]]), epochs=1, verbose=0)

            # Подсказка и обновление
            guess = np.array([[prediction + feedback * 10]])  # Корректируем предположение


def saveModel(model):
    model.save('guessing_model.keras')


def get_repeat_count():
    count: int = 1
    while True:
        try:
            count = int(input("Введите число повторений обучения: "))
            if count < 1 and count != 0:
                # count = abs(count)
                print("Число не может быть отрицательным")
            elif count == 0:
                # count = 1
                print("Число не может быть 0")
            else:
                return count
        except:
            print("Пожалуйста, вводите целое число")


sequence_length: int = 3


def neural_network_guess_with_memory(model, target):
    # Начальные параметры
    history = []  # История попыток

    for attempt in range(1, 101):  # Максимум 100 попыток
        # Формируем входные данные из истории
        if len(history) < sequence_length:
            padded_history = [[0, 0]] * (sequence_length - len(history)) + history
        else:
            padded_history = history[-sequence_length:]

        # Преобразуем в формат np.array для модели
        input_sequence = np.array([padded_history])

        # Получаем предсказание от модели
        prediction = model.predict(input_sequence, verbose=0)[0][0]
        guess = round(prediction)
        print(f"Нейросеть предполагает: {guess}", end=" → ")

        # Проверяем результат угадывания
        if guess == target:
            print("Угадала!")
            return
        elif guess < target:
            feedback = 1
            # print("больше!")
        else:
            feedback = -1
            # print("меньше!")

        # Добавляем текущую попытку в историю
        history.append([guess, feedback])

        # Обучаем модель на текущей попытке
        target_value = target if feedback == 1 else target - 1
        model.fit(input_sequence, np.array([[target_value]]), epochs=1, verbose=0)

    print("Не удалось угадать число за 100 попыток.")


def main():
    repeat_count: int = get_repeat_count()
    my_model = get_model_with_memory()


    # attempts: list = []
    # count = 0
    # while count != repeat_count:
    #     list_attempt: int = ney_model)
    #     attempts.append(list_attempt)
    #     count += 1

    # average: float = sum(attempts) / len(attempts)
    # print(f"Среднее количество попыток: {average:.1f}")
    # print(f"Max: {max(attempts)}\nMin: {min(attempts)}")

    saveModel(my_model)
    data_frame = pd.DataFrame(attempts)
    try:
        print(data_frame)
    except:
        # notink
        print()
    data_frame.to_csv("table.csv", index=False, header=False)


main()
