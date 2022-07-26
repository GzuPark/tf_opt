import os
import shutil

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf


class CustomModel(object):

    def __init__(self, validation: bool = False, force: bool = False) -> None:
        self.ckpt_dir = os.path.join(os.path.dirname(__file__), "../ckpt")

        if force:
            shutil.rmtree(self.ckpt_dir)
        elif not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self._model_path = os.path.join(self.ckpt_dir, "keras_model.h5")
        self.model = None
        self._validation = validation
        self._download_dataset()

    def _download_dataset(self) -> None:
        data_dir = os.path.join(os.path.dirname(__file__), "../mnist_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, "mnist.npz")
        train_dataset, test_dateset = tf.keras.datasets.mnist.load_data(path=data_path)

        self.x_train, self.y_train = train_dataset
        self.x_test, self.y_test = test_dateset

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

        self.x_val = None
        self.y_val = None

        if self._validation:
            self.x_val = self.x_train[50000:]
            self.y_val = self.y_train[50000:]

            self.x_train = self.x_train[:50000]
            self.y_train = self.y_train[:50000]

    def create_model(self) -> None:
        input_layer = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_layer)
        x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.summary()

    def train(self, again: bool = False) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        if again:
            if self._validation:
                self.model.fit(self.x_train, self.y_train, epochs=5, validation_data=(self.x_val, self.y_val))
            else:
                self.model.fit(self.x_train, self.y_train, epochs=5)

            self.model.save(self._model_path)
        else:
            self.model = tf.keras.models.load_model(self._model_path)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test)
        end_time = perf_counter()

        result = {
            "method": "keras",
            "accuracy": accuracy,
            "total_time": end_time - start_time,
            "avg_time": (end_time - start_time) / len(self.y_test),
            "model_file_size": os.path.getsize(self._model_path),
        }

        return result
