# Test TensorFlow Model Optimization

Try to follow
the [model optimization reference from TensorFlow](https://www.tensorflow.org/model_optimization/guide/get_started)
& [TFLite optimized model guide](https://www.tensorflow.org/lite/performance/model_optimization).

![TFLite_quantization_decision_tree](https://www.tensorflow.org/static/lite/performance/images/quantization_decision_tree.png)
![TFMOT_collaborative_optimization](https://www.tensorflow.org/static/model_optimization/guide/combine/images/collaborative_optimization.png)

- Test on the Apple M1 Pro & `tensorflow-macos==2.9.2`

## Image classification

### MNIST with custom model

|     Method |       Model optimize |     Accuracy |      Total time |       File size |
|-----------:|---------------------:|-------------:|----------------:|----------------:|
|      keras |                 none |      97.50 % |        308.9 ms |       266.47 KB |
|       fp32 |                 none |      97.50 % |        373.8 ms |        82.63 KB |
|       fp16 |                 none |      97.50 % |        371.8 ms |        43.34 KB |
|    dynamic |                 none |      97.50 % |        373.9 ms |        23.30 KB |
|      uint8 |                 none |      97.50 % |        364.5 ms |        23.30 KB |
|    int16x8 |                 none |      97.50 % |       1998.3 ms |        24.16 KB |
|      keras |                prune |      97.25 % |        343.5 ms |        97.27 KB |
|       fp32 |                prune |      97.25 % |        390.7 ms |       165.09 KB |
|       fp16 |                prune |      97.25 % |        382.2 ms |        43.34 KB |
|    dynamic |                prune |      97.26 % |        376.9 ms |        23.30 KB |
|      uint8 |                prune |      97.26 % |        364.7 ms |        23.30 KB |
|    int16x8 |                prune |      97.26 % |       1975.1 ms |        24.16 KB |
|      keras |                quant |      97.79 % |        346.1 ms |       283.45 KB |
|       fp32 |                quant |      97.79 % |        413.8 ms |       167.73 KB |
|       fp16 |                quant |      97.79 % |        329.6 ms |        24.16 KB |
|    dynamic |                quant |      97.79 % |        330.5 ms |        24.16 KB |
|      uint8 |                quant |      97.79 % |        331.3 ms |        24.16 KB |
|      keras |              cluster |      97.18 % |        341.7 ms |        95.27 KB |
|       fp32 |              cluster |      97.18 % |        399.7 ms |       402.75 KB |
|       fp16 |              cluster |      97.18 % |        404.7 ms |       363.50 KB |
|    dynamic |              cluster |      97.17 % |        405.2 ms |       343.41 KB |
|      uint8 |              cluster |      97.17 % |        409.9 ms |       343.41 KB |
|    int16x8 |              cluster |      97.16 % |       1985.4 ms |       105.44 KB |
|      keras |          cluster_qat |      97.85 % |        358.8 ms |       283.45 KB |
|       fp32 |          cluster_qat |      97.85 % |        411.5 ms |       167.73 KB |
|       fp16 |          cluster_qat |      97.85 % |        333.2 ms |        24.16 KB |
|    dynamic |          cluster_qat |      97.85 % |        334.0 ms |        24.16 KB |
|      uint8 |          cluster_qat |      97.85 % |        336.0 ms |        24.16 KB |
|      keras |         cluster_cqat |      97.66 % |        531.2 ms |       451.52 KB |
|       fp32 |         cluster_cqat |      97.66 % |        421.3 ms |       167.73 KB |
|       fp16 |         cluster_cqat |      97.66 % |        333.3 ms |        24.16 KB |
|    dynamic |         cluster_cqat |      97.66 % |        326.8 ms |        24.16 KB |
|      uint8 |         cluster_cqat |      97.66 % |        333.1 ms |        24.16 KB |
|      keras |            prune_qat |      97.72 % |        548.3 ms |       283.45 KB |
|       fp32 |            prune_qat |      97.72 % |        421.1 ms |       167.73 KB |
|       fp16 |            prune_qat |      97.72 % |        329.0 ms |        24.16 KB |
|    dynamic |            prune_qat |      97.72 % |        329.5 ms |        24.16 KB |
|      uint8 |            prune_qat |      97.72 % |        329.4 ms |        24.16 KB |
|      keras |           prune_pqat |      97.48 % |        571.0 ms |       283.45 KB |
|       fp32 |           prune_pqat |      97.51 % |        420.3 ms |       167.73 KB |
|       fp16 |           prune_pqat |      97.48 % |        336.2 ms |        24.16 KB |
|    dynamic |           prune_pqat |      97.48 % |        327.5 ms |        24.16 KB |
|      uint8 |           prune_pqat |      97.48 % |        328.1 ms |        24.16 KB |
|      keras |        prune_cluster |      97.23 % |        469.5 ms |        95.27 KB |
|       fp32 |        prune_cluster |      97.23 % |       6193.4 ms |       248.62 KB |
|       fp16 |        prune_cluster |      97.23 % |        365.7 ms |        43.34 KB |
|    dynamic |        prune_cluster |      97.20 % |        363.4 ms |        23.30 KB |
|      uint8 |        prune_cluster |      97.20 % |        363.3 ms |        23.30 KB |
|      keras |    prune_cluster_qat |      97.63 % |        527.0 ms |       283.45 KB |
|       fp32 |    prune_cluster_qat |      97.63 % |        409.1 ms |       167.73 KB |
|       fp16 |    prune_cluster_qat |      97.63 % |        326.4 ms |        24.16 KB |
|    dynamic |    prune_cluster_qat |      97.63 % |        328.7 ms |        24.16 KB |
|      uint8 |    prune_cluster_qat |      97.63 % |        330.4 ms |        24.16 KB |
|      keras |  prune_cluster_pcqat |      97.38 % |        580.2 ms |       451.52 KB |
|       fp32 |  prune_cluster_pcqat |      97.38 % |        416.0 ms |       167.73 KB |
|       fp16 |  prune_cluster_pcqat |      97.38 % |        330.3 ms |        24.16 KB |
|    dynamic |  prune_cluster_pcqat |      97.38 % |        333.6 ms |        24.16 KB |
|      uint8 |  prune_cluster_pcqat |      97.38 % |        332.5 ms |        24.16 KB |
