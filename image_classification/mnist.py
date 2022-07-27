import os

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf


class _BaseModel(object):

    def __init__(self, root_path: str, validation: bool = False, reset: bool = False) -> None:
        self.ckpt_dir = os.path.join(root_path, "ckpt")
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._data_dir = os.path.join(root_path, "data", "mnist_data")
        self._validation = validation

        self._check_dirs(reset)
        self._download_dataset()

    def _check_dirs(self, reset: bool) -> None:
        if reset:
            for model_name in os.listdir(self.ckpt_dir):
                if "mnist_" in model_name:
                    os.remove(os.path.join(self.ckpt_dir, model_name))

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

    def _download_dataset(self) -> None:
        data_path = os.path.join(self._data_dir, "mnist.npz")
        train_dataset, test_dateset = tf.keras.datasets.mnist.load_data(path=data_path)

        self.x_train, self.y_train = train_dataset
        self.x_test, self.y_test = test_dateset

        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0

    def create_model(self, summary: bool = False) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError


class BasicModel(_BaseModel):
    def __init__(self, root_path: str, validation: bool = False, reset: bool = False) -> None:
        super().__init__(root_path, validation, reset)
        self._model_path = os.path.join(self.ckpt_dir, "basic_mnist_keras.h5")

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
            kwargs["validation_split"] = 0.1

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
            "model_file_size": os.path.getsize(self._model_path),
        }

        return result
