# Release Proof Owner 输入 Dashboard

`release-proof-owner-input-dashboard` 是 release owner 的真实 proof 输入看板。它只负责把三条证据线分开，避免把样例输出、工具报告、模板或本地验证误写成发布 proof。

机器可读文件：

`artifacts/final-release/release-proof-owner-input-dashboard.json`

## 适用读者

- release owner。
- 包发布负责人。
- 需要审核 `package-consumer-runtime`、`sample-run-evidence`、`post-publish verification` 的维护者。

## 解决问题

发布前最容易混淆的地方不是缺少文件，而是把不同性质的文件放到同一个 proof 口径中。这个 dashboard 把三条 ladder 拆开：

1. `sample-run-evidence`：真实模型、图片、labels、日志、输出 JSON 和 owner review。
2. `package-consumer-runtime`：公开包来源、clean consumer、restore、native asset、runtime smoke。
3. `post-publish verification`：真实发布之后，从选定发布渠道重新验证 clean consumer。

三者不能互相替代。sample-run-evidence 不能替代 `package-consumer-runtime`，`package-consumer-runtime` 不能替代 `post-publish verification`。

## Owner 必填字段

### sample-run-evidence

- model family、task、model source URL、model license。
- model、labels、input image、generated engine、output JSON、run log 的 SHA256。
- `TensorRtExec` build command 和 `YoloVision` run command。
- stdout/stderr log path。
- owner review status。

### package-consumer-runtime

- public package source 和 package version。
- managed/runtime `.nupkg` SHA256。
- clean consumer project path。
- restore log、native asset listing、smoke log。
- host CUDA driver 与 TensorRT version。

### post-publish verification

- published package source 和 version。
- clean consumer root / project path。
- restore、native asset listing、dependency probe、smoke log。
- owner verification status。

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

这个 dashboard 的默认状态是 `blocked-owner-real-proof-required`，`performsPublish=false`，`canPublishPublicly=false`，`canCloseReleaseIssue=false`。它不会执行包发布，也不能关闭 release issue。

## 可复制命令

用于 owner 回填后验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

这些命令只用于验证 owner 已经提供的真实记录。不要在文章或 dashboard 中写入真实包发布命令，也不要把 template-only 记录写成 runtime proof。

## 下一步

1. owner 先回填 `sample-run-evidence` 的真实模型和日志。
2. 在兼容 CUDA/TensorRT 主机上采集 public package clean consumer runtime proof。
3. 真实公开发布之后，再采集 post-publish clean consumer verification。
4. 最终 release close 仍需 strict close validator 和 owner approval。
