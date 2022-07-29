from time import sleep
from typing import Any, Dict, List


def print_table(
        outputs: List[Dict[str, Any]],
        time_unit: str = "ms",
        file_unit: str = "KB",
) -> None:
    time_units = dict()
    time_units["s"] = 0
    time_units["ms"] = 1
    time_units["μs"] = 2

    file_units = dict()
    file_units["b"] = 0
    file_units["kb"] = 1
    file_units["mb"] = 2

    title = f"| {'Method':>10} |"
    title += f" {'Model optimize':>20} |"
    title += f" {'Accuracy':>12} |"
    title += f" {'Total time':>15} |"
    title += f" {'File size':>15} |"

    bar = f"|{'-' * 11}:|"
    bar += f"{'-' * 21}:|"
    bar += f"{'-' * 13}:|"
    bar += f"{'-' * 16}:|"
    bar += f"{'-' * 16}:|"

    sleep(2)
    print(title)
    print(bar)

    for out in outputs:
        method = out['method']
        optimize = out['opt']
        accuracy = out['accuracy'] * 100
        total_time = out['total_time'] * (1000 ** time_units.get(time_unit.lower(), "ms"))
        file_size = out['model_file_size'] / (1024 ** file_units.get(file_unit.lower(), "KB"))

        row = f"| {method:>10} |"
        row += f" {optimize:>20} |"
        row += f" {accuracy:>10.2f} % |"
        row += f" {total_time:>12.1f} {time_unit} |"
        row += f" {file_size:>12.2f} {file_unit} |"

        print(row)
