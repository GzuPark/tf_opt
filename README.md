# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|     Method |  Model optimize |     Accuracy |      Total time |       File size |
|-----------:|----------------:|-------------:|----------------:|----------------:|
|      keras |            None |      97.50 % |        304.1 ms |       266.47 KB |
|       fp32 |            none |      97.50 % |        366.4 ms |        82.63 KB |
|       fp16 |            none |      97.50 % |        365.3 ms |        43.34 KB |
|    dynamic |            none |      97.50 % |        365.0 ms |        23.30 KB |
|      uint8 |            none |      97.54 % |        346.5 ms |        23.85 KB |
|    int16x8 |            none |      97.50 % |       1934.0 ms |        24.16 KB |
|      keras |         pruning |      97.25 % |        354.8 ms |        97.27 KB |
|       fp32 |         pruning |      97.25 % |        364.9 ms |        82.63 KB |
|       fp16 |         pruning |      97.25 % |        366.4 ms |        43.34 KB |
|    dynamic |         pruning |      97.26 % |        363.3 ms |        23.30 KB |
|      uint8 |         pruning |      97.23 % |        345.4 ms |        23.85 KB |
|    int16x8 |         pruning |      97.26 % |       1934.1 ms |        24.16 KB |
|      keras |         pruning |      97.79 % |        337.1 ms |       283.45 KB |
|       fp32 |           quant |      97.79 % |        407.4 ms |       167.73 KB |
|       fp16 |           quant |      97.79 % |        324.5 ms |        24.16 KB |
|    dynamic |           quant |      97.79 % |        324.9 ms |        24.16 KB |
|      uint8 |           quant |      97.74 % |        343.9 ms |        24.19 KB |
|      keras |      clustering |      97.24 % |        337.1 ms |        95.27 KB |
|       fp32 |      clustering |      97.24 % |        372.2 ms |        82.63 KB |
|       fp16 |      clustering |      97.24 % |        371.9 ms |        43.34 KB |
|    dynamic |      clustering |      97.26 % |        372.2 ms |        23.30 KB |
|      uint8 |      clustering |      97.32 % |        355.8 ms |        23.85 KB |
|    int16x8 |      clustering |      97.25 % |       1969.5 ms |        24.16 KB |
