# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|     Method |  Model optimize |     Accuracy |      Total time |       File size |
|-----------:|----------------:|-------------:|----------------:|----------------:|
|      keras |            None |      97.50 % |        302.0 ms |       266.47 KB |
|       fp32 |            none |      97.50 % |        372.1 ms |        82.63 KB |
|       fp16 |            none |      97.50 % |        367.9 ms |        43.34 KB |
|    dynamic |            none |      97.50 % |        365.9 ms |        23.30 KB |
|      uint8 |            none |      97.54 % |        347.0 ms |        23.85 KB |
|    int16x8 |            none |      97.50 % |       1947.2 ms |        24.16 KB |
|      keras |           prune |      97.25 % |        429.3 ms |        97.27 KB |
|       fp32 |           prune |      97.25 % |        376.8 ms |       165.09 KB |
|       fp16 |           prune |      97.25 % |        369.5 ms |        43.34 KB |
|    dynamic |           prune |      97.26 % |        378.1 ms |        23.30 KB |
|      uint8 |           prune |      97.23 % |        355.6 ms |        23.85 KB |
|    int16x8 |           prune |      97.26 % |       2002.7 ms |        24.16 KB |
|      keras |        quantize |      97.79 % |        449.6 ms |       283.45 KB |
|       fp32 |        quantize |      97.79 % |        421.0 ms |       167.73 KB |
|       fp16 |        quantize |      97.79 % |        337.2 ms |        24.16 KB |
|    dynamic |        quantize |      97.79 % |        332.6 ms |        24.16 KB |
|      uint8 |        quantize |      97.74 % |        357.7 ms |        24.19 KB |
|      keras |         cluster |      97.18 % |        455.8 ms |        95.27 KB |
|       fp32 |         cluster |      97.18 % |        404.7 ms |       402.75 KB |
|       fp16 |         cluster |      97.18 % |        405.2 ms |       363.50 KB |
|    dynamic |         cluster |      97.17 % |        398.2 ms |       343.41 KB |
|      uint8 |         cluster |      97.17 % |        382.4 ms |       343.97 KB |
|    int16x8 |         cluster |      97.16 % |       1973.6 ms |       105.44 KB |
|      keras |     cluster_qat |      97.85 % |        533.1 ms |       283.45 KB |
|       fp32 |     cluster_qat |      97.85 % |        415.6 ms |       167.76 KB |
|       fp16 |     cluster_qat |      97.85 % |        331.6 ms |        24.16 KB |
|    dynamic |     cluster_qat |      97.85 % |        328.8 ms |        24.16 KB |
|      uint8 |     cluster_qat |      97.81 % |        354.3 ms |        24.20 KB |
|      keras |    cluster_cqat |      97.66 % |        589.9 ms |       451.52 KB |
|       fp32 |    cluster_cqat |      97.66 % |        420.7 ms |       167.76 KB |
|       fp16 |    cluster_cqat |      97.66 % |        330.5 ms |        24.16 KB |
|    dynamic |    cluster_cqat |      97.66 % |        333.0 ms |        24.16 KB |
|      uint8 |    cluster_cqat |      97.65 % |        351.8 ms |        24.20 KB |
