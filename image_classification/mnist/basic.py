import logging
import os

from time import perf_counter
from typing import Any, Dict

import tensorflow as tf

from image_classification.mnist import BaseModel
from utils.dataclass import Result, TrainParams


class BasicModel(BaseModel):

    def __init__(
            self,
            root_dir: str,
            model_filename: str,
            dataset: Dict[str, Any],
            valid_split: float,
            batch_size: int,
            epochs: int,
            logger: logging.Logger,
            verbose: bool = False,
    ) -> None:
        super().__init__(root_dir, dataset, valid_split, batch_size, epochs, verbose)

        self.model = None
        self.model_path = os.path.join(self.ckpt_dir, model_filename)

        self._logger = logger
        self._logger.info("Run without optimizing")

    def create_model(self) -> None:
        input_layer = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(input_layer)
        x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10)(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)

        if self.verbose:
            self.model.summary()

    def _compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def _load_model(self) -> bool:
        success = False

        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            success = True

        return success

    def train(self) -> None:
        if self._load_model():
            return

        train_params = TrainParams(
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.valid_split,
            verbose=self.verbose,
        )

        self._compile()
        self.model.fit(self.x_train, self.y_train, **train_params.to_dict())

        tf.keras.models.save_model(self.model, self.model_path, include_optimizer=True)

    def evaluate(self) -> Result:
        if (self.model is None) and (not self._load_model()):
            self._logger.error(f"Cannot load {self.model_path}.")
            raise ValueError(f"Cannot load {self.model_path}.")

        start_time = perf_counter()
        _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        end_time = perf_counter()

        result = Result(
            method="keras",
            optimizer="none",
            accuracy=accuracy,
            total_time=end_time - start_time,
            model_file_size=os.path.getsize(self.model_path),
        )

        self._logger.info(result.to_dict())

        return result
