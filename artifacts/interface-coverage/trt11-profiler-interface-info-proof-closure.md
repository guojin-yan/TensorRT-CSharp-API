# TRT11 `IProfiler::getInterfaceInfo` Proof Closure

本记录把已有的 TRT11 profiler copied metadata 路由收口为可审计的 safe-alternative proof。它不是 runtime execution proof、package-consumer-runtime proof、公开发布授权，也不删除 deferred history。

| 版本线 | `getInterfaceInfo` | `getAPILanguage` | 证据边界 |
| --- | --- | --- | --- |
| TRT8 | controlled unsupported | controlled unsupported | 当前头文件中的 profiler 不是 versioned interface |
| TRT10 | controlled unsupported | controlled unsupported | 当前头文件中的 profiler 不是 versioned interface |
| TRT11 | native copied readback | native copied readback | caller buffer + scalar out，托管侧只接收复制值 |

## 已闭环的链路

- manifest：`trt11-profiler-get-interface-info` 与 `trt11-profiler-get-api-language`。
- native：`jyppx_trt11_profiler_get_interface_info` 对 `InterfaceInfo` 做 caller-buffer/scalar copy，并通过 bridge status 返回错误。
- managed：`TensorRtProfiler.TryGetInterfaceInfo`、`TryGetApiLanguage` 和 `GetInterfaceMetadataSnapshot`。
- smoke：`ManagedProfilerCallbackSmokeRunner` 输出 `ManagedProfilerInterfaceMetadataSnapshot`，但仍按环境可用性分类。
- deferred history：`trt11-profiler-get-interface-info-deferred` 和 `twenty_third_batch_deferred.inc` 原样保留。

## 明确不证明的内容

`GetInterfaceMetadataSnapshot` 是 pointer-free copied diagnostic aggregate，`IsRuntimeProof` 固定为 `false`。它不证明真实模型、真实 enqueue、完整 package consumer、公开 NuGet 或 release close。真实 TRT11 readback 仍需要兼容 host、vendor runtime、CUDA driver 和 clean consumer 证据。
