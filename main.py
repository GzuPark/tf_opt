import os.path

from image_classification import mnist_custom_model, get_result
from utils.fix_seed import set_seed


def run_mnist(path: str) -> None:
    set_seed()

    mnist_model = mnist_custom_model(path, reset=True)
    mnist_model.create_model()
    mnist_model.train()
    res_origin = mnist_model.evaluate()

    required_data = {
        "x_train": mnist_model.x_train,
        "x_test": mnist_model.x_test,
        "y_test": mnist_model.y_test,
    }

    kwargs = {
        "dir_path": mnist_model.ckpt_dir,
        "model": mnist_model.model,
        "data": required_data,
    }

    res_fp16 = get_result(method="fp16", **kwargs)
    res_dynamic = get_result(method="dynamic", **kwargs)
    res_uint8 = get_result(method="uint8", **kwargs)

    result = [res_origin, res_fp16, res_dynamic, res_uint8]

    print("-" * 65)
    print(f"| {'Method':>10} | {'Accuracy':>12} | {'Avg. time':>15} | {'File size':>15} |")
    print("-" * 65)

    for res in result:
        _method = res['method']
        _acc = res['accuracy'] * 100
        _time = res['avg_time']
        _size = res['model_file_size'] / 1024
        print(f"| {_method:>10} | {_acc:>10.2f} % | {_time:>15.7f} | {_size:>12.2f} KB |")

    print("-" * 65)


def main() -> None:
    root_path = os.path.dirname(__file__)
    run_mnist(root_path)


if __name__ == '__main__':
    main()
