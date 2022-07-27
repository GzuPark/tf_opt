import os.path

from image_classification import mnist, ImageClassificationConverter
from utils import set_seed


def run_mnist(path: str) -> None:
    model = mnist.BasicModel(path, validation_split=0.1)
    model.create_model()
    model.train()

    required_data = {
        "x_train": model.x_train,
        "x_test": model.x_test,
        "y_test": model.y_test,
    }

    kwargs = {
        "dataset_name": "mnist",
        "opt": "none",
        "ckpt_dir": model.ckpt_dir,
        "model": model.model,
        "data": required_data,
    }

    result = list()
    result.append(model.evaluate())

    methods = ["fp32", "fp16", "dynamic", "uint8", "int16x8"]
    for method in methods:
        converter = ImageClassificationConverter(method=method, **kwargs)
        converter.convert()
        result.append(converter.evaluate())

    pruning_model = mnist.PruningModel(path, base_model=model.model, validation_split=0.1)
    pruning_model.create_model()
    pruning_model.train()

    result.append(pruning_model.evaluate())

    kwargs["opt"] = "pruning"
    kwargs["ckpt_dir"] = pruning_model.ckpt_dir
    kwargs["model"] = pruning_model.model

    methods = ["fp32", "fp16", "dynamic", "uint8", "int16x8"]
    for method in methods:
        pruning_converter = ImageClassificationConverter(method=method, **kwargs)
        pruning_converter.convert()
        result.append(pruning_converter.evaluate())

    print(f"| {'Method':>10} | {'Model optimize':>15} | {'Accuracy':>12} | {'Total time':>15} | {'File size':>15} |")
    print(f"|{'-' * 11}:|{'-' * 16}:|{'-' * 13}:|{'-' * 16}:|{'-' * 16}:|")

    for res in result:
        _method = res['method']
        _opt = res['opt']
        _acc = res['accuracy'] * 100
        _time = res['total_time'] * 1000
        _size = res['model_file_size'] / 1024
        print(f"| {_method:>10} | {_opt:>15} | {_acc:>10.2f} % | {_time:>12.1f} ms | {_size:>12.2f} KB |")


def main() -> None:
    set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
