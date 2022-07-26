import os
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 2022) -> None:
    """ref: https://hoya012.github.io/blog/reproducible_pytorch/"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
