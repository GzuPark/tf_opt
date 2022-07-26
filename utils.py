import os

from typing import Any, Tuple

import tensorflow as tf


DATASET = {
    "mnist": {
        "dir": "mnist_data",
        "filename": "mnist.npz",
        "load_fn": tf.keras.datasets.mnist,
    }
}


def download_dataset(dataset_name: str) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, DATASET[dataset_name]["dir"])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_path = os.path.join(data_dir, DATASET[dataset_name]["filename"])
    (x_train, y_train), (x_test, y_test) = DATASET[dataset_name]["load_fn"].load_data(path=data_path)

    return (x_train, y_train), (x_test, y_test)
