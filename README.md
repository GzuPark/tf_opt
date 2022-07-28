# Test TensorFlow Model Optimization

Try to follow
the [model optimization reference from TensorFlow](https://www.tensorflow.org/model_optimization/guide/get_started)
& [TFLite optimized model guide](https://www.tensorflow.org/lite/performance/model_optimization).

![TFLite_quantization_decision_tree](https://www.tensorflow.org/static/lite/performance/images/quantization_decision_tree.png)
![TFMOT_collaborative_optimization](https://www.tensorflow.org/static/model_optimization/guide/combine/images/collaborative_optimization.png)

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|     Method |      Model optimize |     Accuracy |      Total time |       File size |
|-----------:|--------------------:|-------------:|----------------:|----------------:|
|      keras |                none |      97.50 % |        298.8 ms |       266.47 KB |
|       fp32 |                none |      97.50 % |        371.8 ms |        82.63 KB |
|       fp16 |                none |      97.50 % |        368.2 ms |        43.34 KB |
|    dynamic |                none |      97.50 % |        368.1 ms |        23.30 KB |
|      uint8 |                none |      97.50 % |        366.5 ms |        23.30 KB |
|    int16x8 |                none |      97.50 % |       1949.9 ms |        24.16 KB |
|      keras |               prune |      97.25 % |        314.3 ms |        97.27 KB |
|       fp32 |               prune |      97.25 % |        382.0 ms |       165.09 KB |
|       fp16 |               prune |      97.25 % |        373.1 ms |        43.34 KB |
|    dynamic |               prune |      97.26 % |        372.3 ms |        23.30 KB |
|      uint8 |               prune |      97.26 % |        372.3 ms |        23.30 KB |
|    int16x8 |               prune |      97.26 % |       1949.9 ms |        24.16 KB |
|      keras |            quantize |      97.79 % |        364.9 ms |       283.45 KB |
|       fp32 |            quantize |      97.79 % |        412.8 ms |       167.73 KB |
|       fp16 |            quantize |      97.79 % |        329.8 ms |        24.16 KB |
|    dynamic |            quantize |      97.79 % |        334.5 ms |        24.16 KB |
|      uint8 |            quantize |      97.79 % |        331.0 ms |        24.16 KB |
|      keras |             cluster |      97.18 % |        332.6 ms |        95.27 KB |
|       fp32 |             cluster |      97.18 % |        405.1 ms |       402.75 KB |
|       fp16 |             cluster |      97.18 % |        408.0 ms |       363.50 KB |
|    dynamic |             cluster |      97.17 % |        407.6 ms |       343.41 KB |
|      uint8 |             cluster |      97.17 % |        408.4 ms |       343.41 KB |
|    int16x8 |             cluster |      97.16 % |       1988.1 ms |       105.44 KB |
|      keras |         cluster_qat |      97.85 % |        339.6 ms |       283.45 KB |
|       fp32 |         cluster_qat |      97.85 % |        415.0 ms |       167.73 KB |
|       fp16 |         cluster_qat |      97.85 % |        336.9 ms |        24.16 KB |
|    dynamic |         cluster_qat |      97.85 % |        338.1 ms |        24.16 KB |
|      uint8 |         cluster_qat |      97.85 % |        334.6 ms |        24.16 KB |
|      keras |        cluster_cqat |      97.66 % |        540.4 ms |       451.52 KB |
|       fp32 |        cluster_cqat |      97.66 % |        411.6 ms |       167.73 KB |
|       fp16 |        cluster_cqat |      97.66 % |        329.3 ms |        24.16 KB |
|    dynamic |        cluster_cqat |      97.66 % |        328.3 ms |        24.16 KB |
|      uint8 |        cluster_cqat |      97.66 % |        330.0 ms |        24.16 KB |
|      keras |           prune_qat |      97.72 % |        591.4 ms |       283.45 KB |
|       fp32 |           prune_qat |      97.72 % |        413.6 ms |       167.73 KB |
|       fp16 |           prune_qat |      97.72 % |        329.2 ms |        24.16 KB |
|    dynamic |           prune_qat |      97.72 % |        331.2 ms |        24.16 KB |
|      uint8 |           prune_qat |      97.72 % |        328.7 ms |        24.16 KB |
|      keras |          prune_pqat |      97.48 % |        587.6 ms |       283.45 KB |
|       fp32 |          prune_pqat |      97.51 % |        413.5 ms |       167.73 KB |
|       fp16 |          prune_pqat |      97.48 % |        328.4 ms |        24.16 KB |
|    dynamic |          prune_pqat |      97.48 % |        330.0 ms |        24.16 KB |
|      uint8 |          prune_pqat |      97.48 % |        329.0 ms |        24.16 KB |
|      keras |   prune_cluster_qat |      97.63 % |        578.9 ms |       283.45 KB |
|       fp32 |   prune_cluster_qat |      97.63 % |        412.8 ms |       167.73 KB |
|       fp16 |   prune_cluster_qat |      97.63 % |        330.3 ms |        24.16 KB |
|    dynamic |   prune_cluster_qat |      97.63 % |        331.3 ms |        24.16 KB |
|      uint8 |   prune_cluster_qat |      97.63 % |        328.7 ms |        24.16 KB |
|      keras | prune_cluster_pcqat |      97.38 % |        585.2 ms |       451.52 KB |
|       fp32 | prune_cluster_pcqat |      97.38 % |        412.0 ms |       167.73 KB |
|       fp16 | prune_cluster_pcqat |      97.38 % |        330.5 ms |        24.16 KB |
|    dynamic | prune_cluster_pcqat |      97.38 % |        331.5 ms |        24.16 KB |
|      uint8 | prune_cluster_pcqat |      97.38 % |        328.7 ms |        24.16 KB |
