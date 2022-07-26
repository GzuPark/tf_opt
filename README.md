# Test TensorFlow Model Optimization

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

```
    Method     Accuracy       Avg. time    File size
     keras      97.95 %       0.0000324       272864
      fp16      97.95 %       0.0000371        44380
   dynamic      97.92 %       0.0000365        23856
     uint8      98.01 %       0.0000352        24424
```
