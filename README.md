# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|               Method |  Model optimize |     Accuracy |      Total time |       File size |
|---------------------:|----------------:|-------------:|----------------:|----------------:|
|                keras |               - |      97.94 % |        307.5 ms |       266.47 KB |
|                 fp32 |               - |      97.94 % |        371.9 ms |        82.63 KB |
|                 fp16 |               - |      97.94 % |        374.0 ms |        43.34 KB |
|              dynamic |               - |      97.95 % |        368.7 ms |        23.30 KB |
|                uint8 |               - |      97.95 % |        344.4 ms |        23.85 KB |
|              int16x8 |               - |      97.94 % |       1952.1 ms |        24.16 KB |
