# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|     Method |  Model optimize |     Accuracy |      Total time |       File size |
|-----------:|----------------:|-------------:|----------------:|----------------:|
|      keras |            None |      97.50 % |        295.4 ms |       266.47 KB |
|       fp32 |            none |      97.50 % |        366.0 ms |        82.63 KB |
|       fp16 |            none |      97.50 % |        366.0 ms |        43.34 KB |
|    dynamic |            none |      97.50 % |        361.6 ms |        23.30 KB |
|      uint8 |            none |      97.54 % |        346.0 ms |        23.85 KB |
|    int16x8 |            none |      97.50 % |       1940.0 ms |        24.16 KB |
| keras | pruning | 97.25 % | 350.4 ms | 97.27 KB |
| fp32 | pruning | 97.25 % | 367.9 ms | 82.63 KB |
| fp16 | pruning | 97.25 % | 365.9 ms | 43.34 KB |
| dynamic | pruning | 97.26 % | 362.5 ms | 23.30 KB |
| uint8 | pruning | 97.23 % | 345.8 ms | 23.85 KB |
| int16x8 | pruning | 97.26 % | 1933.9 ms | 24.16 KB |