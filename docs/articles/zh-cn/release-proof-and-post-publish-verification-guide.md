# 发布前 Proof 与 Post-Publish Verification：从 Release Hold 到最终关闭

## 适用读者

这篇文章适合 release owner、项目维护者、负责审批 NuGet 发布和 issue close 的团队成员。

## 解决问题

发布候选阶段最容易混淆两件事：发布前 readiness 和发布后 verification。前者可以通过 freeze manifest、handoff pack、dashboard、dry run、article matrix 说明“还缺什么”；后者必须在真实 public package 发布后，从公开源安装并运行 clean consumer 验证。两者不能互相替代。

## 发布前链路

发布前可以准备：

- `final-release-pre-publish-audit-matrix`
- `final-proof-owner-handoff-pack`
- `clean-consumer-runtime-proof-execution-checklist`
- `clean-consumer-proof-owner-execution-pack`
- `final-release-close-blocker-dashboard`
- `technical-article-campaign-matrix`

这些 artifact 让 owner 知道下一步做什么，也让维护者知道哪些证据还不能提升。

## 发布后链路

post-publish verification 必须等真实 public package channel 出现后执行。它需要 public package URL、install command、clean consumer logs、package hash、host metadata 和 owner confirmation。只有 post-publish proof、package-consumer-runtime proof、public package proof、strict close record 全部通过，release issue 才能进入关闭候选。

## 最终关闭门禁

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady
```

第一条验证 post-publish record；第二条是最终 close gate。它们不能在真实 proof 缺失时通过修改文本来绕过。

## 边界说明

release dashboard、dry-run、template、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics、build-only 都不是 runtime proof，也不是 post-publish verification。只有真实 public package source 上的安装和运行记录才能支撑发布后验证。

## 下一步

如果还没有真实 public package source，请继续完善文档、执行包和 owner input schema，不要关闭 release。若 public source 已存在，则按 clean consumer proof 和 post-publish proof 两条链分别回填。
