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

    # Pruning
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

    # Quantization
    quant_model = mnist.QuantizationModel(path, base_model=model.model, validation_split=0.1)
    quant_model.create_model()
    quant_model.train()

    result.append(quant_model.evaluate())

    kwargs["opt"] = "quant"
    kwargs["ckpt_dir"] = quant_model.ckpt_dir
    kwargs["model"] = quant_model.model

    # RuntimeError: Quantization to 16x8-bit not yet supported for op: 'DEQUANTIZE'
    methods = ["fp32", "fp16", "dynamic", "uint8"]
    for method in methods:
        quant_converter = ImageClassificationConverter(method=method, **kwargs)
        quant_converter.convert()
        result.append(quant_converter.evaluate())

    # Clustering
    clustered_model = mnist.ClusteringModel(path, base_model=model.model, validation_split=0.1)
    clustered_model.create_model()
    clustered_model.train()

    result.append(clustered_model.evaluate())

    kwargs["opt"] = "clustering"
    kwargs["ckpt_dir"] = clustered_model.ckpt_dir
    kwargs["model"] = clustered_model.model

    methods = ["fp32", "fp16", "dynamic", "uint8", "int16x8"]
    for method in methods:
        clustered_model = ImageClassificationConverter(method=method, **kwargs)
        clustered_model.convert()
        result.append(clustered_model.evaluate())

    # Print out
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
