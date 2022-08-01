import logging
import os

from time import perf_counter
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from utils.dataclass import TFLiteModelInputs, Result


class ImageClassificationConverter(object):

    def __init__(self, inputs: TFLiteModelInputs, dataset: Dict[str, Any], logger: logging.Logger) -> None:
        self.data = dataset

        self._ckpt_dir = os.path.join(inputs.root_dir, "ckpt", inputs.dataset_name)
        _filename = f"{inputs.dataset_name}_{inputs.optimizer}_{inputs.method}.tflite"
        self._model_path = os.path.join(self._ckpt_dir, _filename)
        self._method = inputs.method if inputs.method in {"fp32", "fp16", "uint8", "dynamic", "int16x8"} else "dynamic"
        self._optimizer = inputs.optimizer
        self._converter = None
        self._interpreter = None

        self._logger = logger

    def _initialize(self) -> None:
        quant_methods = dict()

        quant_methods["fp32"] = self._get_fp32
        quant_methods["fp16"] = self._get_fp16
        quant_methods["dynamic"] = self._get_dynamic
        quant_methods["uin8"] = self._get_uint8
        quant_methods["int16x8"] = self._get_int16x8

        target_optimizer = quant_methods.get(self._method, self._get_dynamic)
        target_optimizer()

    def _get_fp32(self) -> None:
        return

    def _get_dynamic(self) -> None:
        self._converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def _get_fp16(self) -> None:
        self._converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self._converter.target_spec.supported_types = [tf.float16]

    def _get_uint8(self) -> None:
        self._converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self._converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self._converter.representative_dataset = self._representative_data_gen
        self._converter.inference_input_type = tf.uint8
        self._converter.inference_output_type = tf.uint8

    def _get_int16x8(self) -> None:
        self._converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self._converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        self._converter.representative_dataset = self._representative_data_gen

    def _representative_data_gen(self) -> Any:
        for input_value in tf.data.Dataset.from_tensor_slices(self.data["x_train"]).batch(1).take(100):
            yield [input_value]

    def convert(self, model: tf.keras.Model) -> None:
        if os.path.exists(self._model_path):
            return

        self._converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self._initialize()

        if not os.path.exists(self._model_path):
            quant_model = self._converter.convert()
            with open(self._model_path, "wb") as f:
                f.write(quant_model)

    def _get_interpreter(self) -> None:
        self._interpreter = tf.lite.Interpreter(self._model_path)
        self._interpreter.allocate_tensors()

    def evaluate(self) -> Result:
        self._get_interpreter()

        input_details = self._interpreter.get_input_details()[0]
        output_details = self._interpreter.get_output_details()[0]

        x_test_indices = range(self.data["x_test"].shape[0])
        predictions = np.zeros((len(x_test_indices),), dtype=int)

        start_time = perf_counter()
        for i, idx in enumerate(x_test_indices):
            test_image = self.data["x_test"][idx]

            if input_details["dtype"] == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point

            test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
            self._interpreter.set_tensor(input_details["index"], test_image)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(output_details["index"])[0]

            predictions[i] = output.argmax()

        end_time = perf_counter()

        result = Result(
            method=self._method,
            optimizer=self._optimizer,
            accuracy=(np.sum(self.data["y_test"] == predictions) / len(self.data["y_test"])).astype(float),
            total_time=end_time - start_time,
            model_file_size=os.path.getsize(self._model_path),
        )

        self._logger.info(result.to_dict())

        return result
