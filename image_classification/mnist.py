import os
import shutil

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf


class CustomModel(object):

    def __init__(self, root_path: str, validation: bool = False, reset: bool = False) -> None:
        self.ckpt_dir = os.path.join(root_path, "ckpt")
        self.data_dir = os.path.join(root_path, "mnist_data")
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self._model_path = os.path.join(self.ckpt_dir, "keras_model.h5")
        self._validation = validation

        self._check_dirs(reset)
        self._download_dataset()

    def _check_dirs(self, reset: bool) -> None:
        if reset:
            shutil.rmtree(self.ckpt_dir, ignore_errors=True)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _download_dataset(self) -> None:
        data_path = os.path.join(self.data_dir, "mnist.npz")
        train_dataset, test_dateset = tf.keras.datasets.mnist.load_data(path=data_path)

        self.x_train, self.y_train = train_dataset
        self.x_test, self.y_test = test_dateset

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

        if self._validation:
            self.x_val = self.x_train[50000:]
            self.y_val = self.y_train[50000:]

            self.x_train = self.x_train[:50000]
            self.y_train = self.y_train[:50000]

    def create_model(self, summary: bool = False) -> None:
        input_layer = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_layer)
        x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)

        if summary:
            self.model.summary()

    def train(self) -> None:
        kwargs = {"epochs": 5}
        if self._validation:
            kwargs["validation_data"] = (self.x_val, self.y_val)

        if os.path.exists(self._model_path):
            self.model = tf.keras.models.load_model(self._model_path)
        else:
            self.model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )

            self.model.fit(self.x_train, self.y_train, **kwargs)
            self.model.save(self._model_path)

    def evaluate(self) -> Dict[str, Any]:
        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_time = perf_counter()

        result = {
            "method": "keras",
            "accuracy": accuracy,
            "total_time": end_time - start_time,
            "avg_time": (end_time - start_time) / len(self.y_test),
            "model_file_size": os.path.getsize(self._model_path),
        }

        return result
