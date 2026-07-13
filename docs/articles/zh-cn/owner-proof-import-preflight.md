# Owner Proof Import Preflight

`owner-proof-import-preflight` 是导入真实 owner proof 之前的预检。它扫描 placeholder、missing hash、missing path、blocked 状态和 forbidden substitutes，防止 template 或 example 被误导入为 proof。

机器可读文件：

`artifacts/final-release/owner-proof-import-preflight.json`

## 适用读者

- 准备导入真实 owner proof 的维护者。
- 检查外部 runtime / post-publish / sample-run 记录的 release owner。
- 编写 strict close record 前的审核者。

## 解决问题

owner proof 回填很容易出现“字段都在，但仍是占位值”的情况。preflight 明确扫描：

- `owner-to-fill`
- `owner-required`
- `template-only`
- `example-not-proof`
- `blocked-by-cuda-driver`
- skipped runtime 标记
- SHA256 格式问题
- log path / output JSON path / package path 占位

## 边界说明

以下内容不是 proof：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- design gate
- blocked-by-cuda-driver

preflight 只做导入前检查，固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它不能晋级 runtime proof，也不能替代 strict validator。

## 可复制命令

preflight 通过后仍要运行对应 strict validator：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

## 下一步

1. 先清理 placeholder。
2. 确认 SHA256 是真实 64 位十六进制。
3. 确认日志、output JSON、package 文件路径存在。
4. 再进入 strict validator 和 release close review。
