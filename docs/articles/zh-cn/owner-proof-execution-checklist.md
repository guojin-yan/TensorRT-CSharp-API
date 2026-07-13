# Owner Proof Execution Checklist

`owner-proof-execution-checklist` 是公开发布前的 owner 执行清单。它把 `release-proof-owner-input-dashboard` 中的三条 proof ladder 转成可填写、可验证、可审计的任务列表，但它本身仍是 checklist，不是 runtime proof。

机器可读文件：

`artifacts/final-release/owner-proof-execution-checklist.json`

## 适用读者

- release owner。
- 需要收集 package-consumer-runtime 与 post-publish verification 的发布负责人。
- 审核 sample-run-evidence 是否可晋级的维护者。

## 解决问题

发布前最关键的问题是防止三条证据线互相替代：

- `sample-run-evidence` 只能证明某个真实模型样例在 owner 环境下运行，不能替代 `package-consumer-runtime`。
- `package-consumer-runtime` 只能证明公开包 clean consumer runtime，不能替代 `post-publish verification`。
- `post-publish verification` 必须在真实公开发布之后从选定渠道重新验证。

这个 checklist 将每条 ladder 的 required fields、validator command、blocked reason 和 forbidden substitutes 放在同一张表里，方便 owner 一次性补齐输入。

## 边界说明

以下内容必须保持非 proof：

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

该 checklist 固定为 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它不会执行真实包发布，也不能关闭 release issue。

## 可复制命令

owner 回填后分三段验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

这些命令只验证 owner 已经提供的真实记录。不要把 template、report、matrix、dry-run 或 blocked 记录写成 runtime proof。

## 下一步

1. 先填写真实模型 sample-run-evidence，并保存模型、图片、labels、engine、日志、output JSON 和 SHA256。
2. 在兼容 CUDA/TensorRT 主机上采集 public package clean consumer runtime proof。
3. 真实公开发布之后，再采集 post-publish clean consumer verification。
4. 所有 strict validators 通过后，再进入 release close owner approval。
