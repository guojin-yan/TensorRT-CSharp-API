# TRT10 Builder Scalar Real Readback

本记录来自本机 TensorRT 10.11.0 / CUDA 12.9 compatible host 的真实 `TensorRtExec --buildOnly`。模型是仓库内生成的 dynamic identity ONNX，报告分类保持 `build-only`。

## Host 与输入

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，driver `32.0.15.7602`
- TensorRT：`10.11.0`
- CUDA toolkit：`12.9.41`
- bridge SHA256：`d9728a0555894ee3afc32f6821dfda17fc7a21195868c3b17867881e5f4fb19c`
- normalized command SHA256：`8c82a46805ccfed333f0c02225c931ae0e168ac9c48008756d6e29a017c87157`
- report/stdout/stderr SHA256：`ef71fe868bf93bc0d819e75643c253b8e928d903e89acff079df35225506f930` / `0ae3f036af98584a11566896f6d221ee76d61f5d14b8450697082d220a36dd34` / `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Scalar 对账

| 参数 | Requested | Applied | Readback | Match | 报告分类 |
| --- | ---: | --- | ---: | --- | --- |
| `--maxNbTactics` | `32` | `True` | `32` | `True` | applied |
| `--tilingOptimizationLevel` | `Moderate` | `True` | `Moderate` | `True` | applied |
| `--l2LimitForTiling` | `268435456 B` | `False` | `3145728 B` | `False` | parse-only |
| `--quantizationFlags` | `CalibrateBeforeFusion` | `True` | `CalibrateBeforeFusion` | `True` | applied |

`--l2LimitForTiling` 没有被误报为成功：TRT10.11 setter 拒绝了请求值，实际 builder copied readback 为 `3MiB`，因此 `OptionImplementationStatus.ParseOnlyOptions` 保留该参数。

## 边界

builder snapshot 为 pointer-free copied diagnostics；本次没有 inference、output comparison、real external model 或 clean package consumer。因此 `isRuntimeExecutionProof=false`、`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`。原 deferred history 保留。
