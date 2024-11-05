import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()

import numpy as np
from random import randint


def get_model(name: str, path: str):
    if os.path.exists(path):
        model = tf.keras.models.load_model(name)
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


def get_model_with_memory(name: str, path: str):
    input_shape = (sequence_length, 2)
    if os.path.exists(path):
        model = tf.keras.models.load_model(name)
        print("Модель загружена.")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, activation='relu', return_sequences=False),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        print("Создана новая модель.")

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def neural_network_guess(model):
    target = randint(1, 100)
    guess = np.array([[50]])
    attempts = 0

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


def saveModel(model, name: str):
    model.save(name)


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


sequence_length: int = 30


def neural_network_guess_with_memory(model, target, log: bool):
    history = []

    for attempt in range(1, 101):
        if len(history) < sequence_length:
            padded_history = [[0, 0]] * (sequence_length - len(history)) + history
        else:
            padded_history = history[-sequence_length:]

        input_sequence = np.array([padded_history])

        prediction = model.predict(input_sequence, verbose=0)[0][0]
        guess = round(prediction)
        if log: print(f"Нейросеть предполагает: {guess}", end=" →\n")

        if guess == target:
            print("Угадала!")
            return [attempt, target]
        elif guess < target:
            feedback = 1
        else:
            feedback = -1

        history.append([guess, feedback])

        target_value = target if feedback == 1 else target - 1
        model.fit(input_sequence, np.array([[target_value]]), epochs=5, verbose=0)

    print("Не удалось угадать число за 100 попыток.")
    return [100, target]

def main():
    model_info: dict = {"name": "my_memory_model_test4.keras", "path": "my_memory_model4_test.keras"}
    repeat_count: int = get_repeat_count()
    my_model = get_model_with_memory(model_info["name"], model_info["path"])

    attempts: list = []
    count = 0
    while count != repeat_count:
        target_value: int = randint(1, 100)
        list_attempt: list = neural_network_guess_with_memory(my_model, target_value, False)
        attempts.append(list_attempt)
        count += 1

    saveModel(my_model, model_info["name"])
    # average: float = sum(attempts) / len(attempts)
    # print(f"Среднее количество попыток: {average:.1f}")
    # print(f"Max: {max(attempts)}\nMin: {min(attempts)}")

    data_frame = pd.DataFrame(attempts)
    try:
        print("Table save!")
    except:
        print("Error")
    data_frame.to_csv("table_memory6.csv", index=False, header=False)


main()