# Release Issue Close Owner Input Final Checklist

`release-issue-close-owner-input-final-checklist` 是 release issue 关闭前的 owner 最终输入清单。它记录 close record 需要的 hash、rollback plan、owner final decision、signature 和 timestamp，但当前仍保持 blocked。

机器可读文件：

`artifacts/final-release/release-issue-close-owner-input-final-checklist.json`

## 适用读者

- release owner。
- release close 审核者。
- 需要准备最终 close record 的维护者。

## 解决问题

即使 sample-run、package-consumer-runtime 和 post-publish verification 都有各自记录，release issue close 仍需要最后的 owner 输入。本清单列出 final close record 必填字段和 validator 结果要求，防止把前置 dashboard 或 runbook 当作 close approval。

## 边界说明

本清单固定 `canCloseReleaseIssue=false`、`releaseCloseReady=false`、`ownerActionRequired=true`。它不是 runtime proof、post-publish proof、publish approval 或 release close approval。

以下内容不能替代最终 owner close input：

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

## 可复制验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~ReleaseIssueCloseOwnerInputFinalChecklist" --logger "trx;LogFileName=release-issue-close-owner-input-final-checklist.trx" --results-directory .\artifacts\test-results\targeted
```

## 下一步

1. owner 回填 release evidence bundle SHA256。
2. owner 回填所有 strict validator result SHA256。
3. owner 填写 rollback plan、final decision、signature 和 timestamp。
4. 最后运行 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。
