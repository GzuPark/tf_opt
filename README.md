# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

```
-----------------------------------------------------------------
|     Method |     Accuracy |       Avg. time |       File size |
-----------------------------------------------------------------
|      keras |      97.94 % |       0.0000293 |       266.47 KB |
|       fp32 |      97.94 % |       0.0000366 |        82.63 KB |
|       fp16 |      97.94 % |       0.0000369 |        43.34 KB |
|    dynamic |      97.95 % |       0.0000370 |        23.30 KB |
|      uint8 |      97.95 % |       0.0000363 |        23.85 KB |
-----------------------------------------------------------------
```
