# Owner 外部真实 Proof 输入合同

`owner-external-real-proof-input-contract` 定义真实公开发布、仓库外 clean consumer、兼容主机 runtime proof、post-publish verification 和 release close approval 的 Owner 回填字段。它只是输入合同，不执行发布、不上传包、不关闭 release issue。

## 边界

- 状态：`blocked-owner-external-real-proof-input-contract-owner-input-required`
- 默认 `Passed=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push

## 必要输入

- 公开 package URL、package id/version 与公开下载 hash
- 仓库外 clean consumer 路径、公开源、restore/smoke 日志 hash
- ProjectReference、direct nupkg、local feed 计数均为 0
- 兼容主机 host id、driver、CUDA、TensorRT、runtime package key 与 version guard
- rollback review、Owner publish decision 与 Owner close decision
