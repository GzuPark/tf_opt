from math import pow
from time import sleep
from typing import List

from utils.dataclass import Result


def print_table(
        results: List[Result],
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

    for result in results:
        row = f"| {result.method:>10} |"
        row += f" {result.optimizer:>20} |"
        row += f" {result.accuracy:>10.2f} % |"
        row += f" {result.total_time * pow(1000, time_units.get(time_unit.lower(), 'ms')):>12.1f} {time_unit} |"
        row += f" {result.model_file_size / pow(1024, file_units.get(file_unit.lower(), 'KB')):>12.2f} {file_unit} |"

        print(row)
