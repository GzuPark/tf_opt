from image_classification.mnist import CustomModel
from image_classification.tflite_converter import get_result


def run_mnist():
    mnist_model = CustomModel()
    mnist_model.create_model()
    mnist_model.train()
    res_origin = mnist_model.evaluate()

    required_data = {
        "x_train": mnist_model.x_train,
        "x_test": mnist_model.x_test,
        "y_test": mnist_model.y_test,
    }

    res_fp16 = get_result(mnist_model.ckpt_dir, "fp16", mnist_model.model, required_data)
    res_dynamic = get_result(mnist_model.ckpt_dir, "dynamic", mnist_model.model, required_data)
    res_uint8 = get_result(mnist_model.ckpt_dir, "uint8", mnist_model.model, required_data)

    print(f"{'Method':>10} {'Accuracy':>12} {'Avg. time':>15} {'File size':>15}")
    for res in [res_origin, res_fp16, res_dynamic, res_uint8]:
        _method = res['method']
        _acc = res['accuracy'] * 100
        _time = res['avg_time']
        _size = res['model_file_size'] / 1024
        print(f"{_method:>10} {_acc:>10.2f} % {_time:>15.7f} {_size:>12.2f} KB")


def main() -> None:
    run_mnist()


if __name__ == '__main__':
    main()
