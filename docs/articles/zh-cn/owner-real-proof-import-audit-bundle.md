# Owner Real Proof Import Audit Bundle

`owner-real-proof-import-audit-bundle` 是 owner 导入真实 proof 前的总审计包。它把 final owner blocker dashboard、field delta dashboard、import preflight、strict gate 和 freeze manifest 串到一起，但它不是 runtime proof，也不是 post-publish proof。

机器可读文件：

`artifacts/final-release/owner-real-proof-import-audit-bundle.json`

## 适用读者

- 准备回填真实 owner proof 的 release owner。
- 审核 owner 输入完整性的维护者。
- 准备进入 strict validator 的发布负责人。

## 解决问题

真实 proof 导入前最容易混淆四条 lane：`sample-run-evidence`、`package-consumer-runtime`、`post-publish-verification`、`release-close-owner-approval`。本审计包把每条 lane 的 required owner inputs 和 strict validator command 放在同一处，避免把模板、dashboard、report 或 matrix 当作 proof。

## 边界说明

以下内容不是 runtime proof、post-publish proof、publish approval 或 release close approval：

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

本审计包固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`、`importReadiness=false`。它只告诉 owner 还缺什么，不导入 proof，也不执行真实发布。

## 可复制验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~OwnerRealProofImportAuditBundle" --logger "trx;LogFileName=owner-real-proof-import-audit-bundle.trx" --results-directory .\artifacts\test-results\targeted
```

## 下一步

1. 使用 owner evidence file manifest template 填写真实文件路径。
2. 使用 strict validator command runbook 逐条验证。
3. 真实发布后再采集 post-publish verification。
4. 所有 validator 通过后再填写 release issue close owner input。
