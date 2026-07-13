# Post-publish Clean Consumer Owner 输入指南

`post-publish-clean-consumer-owner-input.template.json` 是真实发布之后才允许填写的 owner 输入模板。它用于收集从公开发布渠道恢复、构建、探测 native asset 并执行 smoke 的 clean consumer 证据；模板本身不是 proof。

机器可读模板：

`artifacts/final-release/post-publish-clean-consumer-owner-input.template.json`

## 适用读者

- release owner。
- 负责 post-publish verification 的包发布人员。
- 审核 release close 是否具备真实公开渠道验证的维护者。

## 解决问题

`package-consumer-runtime` 可以证明公开包 clean consumer runtime，但它不能替代 `post-publish verification`。post-publish 必须发生在真实公开发布之后，并且从选定发布渠道重新 restore、检查 native assets、运行 dependency probe 和 smoke。

该模板要求 owner 回填：

- published package source。
- package version。
- clean consumer root。
- consumer project path。
- restore log path / SHA256。
- native asset listing path / SHA256。
- dependency probe log path / SHA256。
- smoke log path / SHA256。
- owner verification status。

## 边界说明

以下内容不能替代 post-publish verification：

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

post-publish verification 不能被 `package-consumer-runtime` 替代，`package-consumer-runtime` 也不能被 `sample-run-evidence` 替代。该模板保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 可复制命令

owner 回填真实记录后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

该命令只验证 post-publish 记录，不执行真实包发布。

## 下一步

1. owner 完成真实公开发布之后，再复制 clean consumer 到独立目录。
2. 从选定渠道 restore 包，而不是使用 local feed、ProjectReference 或 direct `.nupkg`。
3. 保存 restore、native asset listing、dependency probe、smoke log 和 SHA256。
4. 用 strict validator 验证 post-publish record。
5. post-publish proof 通过后，再进入 release close owner approval。
