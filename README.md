# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

```
-----------------------------------------------------------------
|     Method |     Accuracy |       Avg. time |       File size |
-----------------------------------------------------------------
|      keras |      97.94 % |       0.0000291 |       266.47 KB |
|       fp16 |      97.94 % |       0.0000371 |        43.34 KB |
|    dynamic |      97.95 % |       0.0000367 |        23.30 KB |
|      uint8 |      97.95 % |       0.0000354 |        23.85 KB |
-----------------------------------------------------------------
```
