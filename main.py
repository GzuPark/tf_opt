import os.path

from image_classification import mnist_custom_model, ImageClassificationConverter
from utils import set_seed


def run_mnist(path: str) -> None:
    result = list()

    mnist_model = mnist_custom_model(path)
    mnist_model.create_model()
    mnist_model.train()

    result.append(mnist_model.evaluate())

    required_data = {
        "x_train": mnist_model.x_train,
        "x_test": mnist_model.x_test,
        "y_test": mnist_model.y_test,
    }

    kwargs = {
        "ckpt_dir": mnist_model.ckpt_dir,
        "model": mnist_model.model,
        "data": required_data,
    }

    methods = ["fp32", "fp16", "dynamic", "uint8", "int16x8"]
    for method in methods:
        converter = ImageClassificationConverter(method=method, **kwargs)
        converter.convert()
        result.append(converter.evaluate())

    print(f"| {'Method':>20} | {'Model optimize':>15} | {'Accuracy':>12} | {'Total time':>15} | {'File size':>15} |")
    print(f"|{'-' * 21}:|{'-' * 16}:|{'-' * 13}:|{'-' * 16}:|{'-' * 16}:|")

    for res in result:
        _method = res['method']
        _opt = "-"
        _acc = res['accuracy'] * 100
        _time = res['total_time'] * 1000
        _size = res['model_file_size'] / 1024
        print(f"| {_method:>20} | {_opt:>15} | {_acc:>10.2f} % | {_time:>12.1f} ms | {_size:>12.2f} KB |")


def main() -> None:
    set_seed()

    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
