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
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation='relu'),
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

    #print("Компьютер загадал число от 1 до 100.")

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

def main():
    repeat_count: int = get_repeat_count()
    my_model = get_model()

    attempts: list = []
    count = 0
    while count != repeat_count:
        list_attempt: int = neural_network_guess(my_model)
        attempts.append(list_attempt)
        count += 1

    # average: float = sum(attempts) / len(attempts)
    # print(f"Среднее количество попыток: {average:.1f}")
    # print(f"Max: {max(attempts)}\nMin: {min(attempts)}")

    saveModel(my_model)
    data_frame = pd.DataFrame(attempts[0], attempts[1])
    try:
        print(data_frame)
    except:
        #notink
        print()
    data_frame.to_csv("table.csv", index=False)


main()