# Owner Real Proof Field Delta Dashboard

`owner-real-proof-field-delta-dashboard` 汇总真实 proof 还缺哪些 owner 字段。它帮助 release owner 按 lane 回填 sample-run-evidence、package-consumer-runtime、post-publish verification 和 release close owner approval，但它本身不是 runtime proof。

机器可读文件：

`artifacts/final-release/owner-real-proof-field-delta-dashboard.json`

## 适用读者

- release owner。
- 需要检查真实 proof 字段缺口的维护者。
- 准备导入外部 proof 记录的审核者。

## 解决问题

真实 proof 收口时，最容易遗漏的是字段级输入：hash、log path、output JSON、package source、owner review 和 final decision。这个 dashboard 将缺失字段按 lane 列出，并给出 validator command 和 blocked reason。

覆盖 lane：

- `sample-run-evidence`
- `package-consumer-runtime`
- `post-publish-verification`
- `release-close-owner-approval`

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

该 dashboard 固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它只指出缺字段，不能替 owner 填写真实日志、真实 SHA256 或 release close approval。

## 可复制命令

字段补齐后再分 lane 验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

## 下一步

1. owner 按 lane 替换 `owner-to-fill` 和 placeholder。
2. 填写真实日志路径、output JSON 路径和 SHA256。
3. 运行 strict validators。
4. 所有 proof lane 通过后，再进入 release close owner approval。
