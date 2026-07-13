# ReleaseCandidate Freeze Manifest

`release-candidate-freeze-manifest` 是发布候选公开材料终检前的冻结清单。它冻结 README、docs index、toc、关键 proof boundary 文章和新增 dashboard 的本地状态，但它不是 runtime proof，也不是 post-publish proof。

机器可读文件：

`artifacts/final-release/release-candidate-freeze-manifest.json`

## 适用读者

- release owner。
- 公开材料最终审核者。
- 准备执行真实 owner proof 导入的维护者。

## 解决问题

发布前材料已经很多，单看某个 dashboard 容易误判为 release-ready。这个 freeze manifest 把当前可冻结的公开材料和仍然 blocked 的 proof lane 放在一起，明确哪些内容只是本地冻结，哪些必须等 owner 真实输入。

冻结范围包括：

- README.md。
- README.zh-CN.md。
- docs/index.md。
- docs/toc.yml。
- owner real proof field delta dashboard。
- release close strict gate dashboard。
- owner proof import preflight。
- public material final scan。
- final owner proof blocker dashboard。

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

本 manifest 固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它不能执行真实发布，也不能替代 `sample-run-evidence`、`package-consumer-runtime`、`post-publish verification` 或 release close owner approval。

## 可复制验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~ReleaseCandidateFreezeManifest" --logger "trx;LogFileName=release-candidate-freeze-manifest.trx" --results-directory .\artifacts\test-results\targeted
```

## 下一步

1. 运行 public material final scan。
2. 修复旧样例 live path 或 proof overclaim。
3. 保持所有 blocked proof lane 分离。
4. 等 owner 真实输入到位后再进入 strict validator。
