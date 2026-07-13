# 兼容主机 Runtime 真实 Proof Gate

`runtime-compatible-host-real-proof-gate` 聚焦兼容 GPU/CUDA/TensorRT 主机上的真实 runtime proof。它只接受真实主机执行结果和日志 hash，不把 driver-blocked、DependencyProbe-only 或 build-only 结果晋级。

## 边界

- 状态：`blocked-runtime-compatible-host-real-proof-gate-owner-proof-required`
- 默认 `Passed=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push

## 准入要求

- host metadata 必须包含 OS、GPU、driver、CUDA、TensorRT 和 dotnet 信息
- runtime package key 必须与公开包和 native asset 解析路径匹配
- CUDA/TensorRT/package consumer smoke exit code 和日志 SHA256 必须完整
- TRT8/TRT10/TRT11 version guard 状态必须可追溯
