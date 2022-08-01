from enum import Enum


class TFLiteQuant(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    Dynamic = "dynamic"
    UINT8 = "uint8"
    INT16x8 = "int16x8"

    def __str__(self) -> str:
        return str(self.value)
