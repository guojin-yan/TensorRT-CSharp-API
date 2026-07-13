# Owner Evidence File Manifest Template

`owner-evidence-file-manifest.template` 是 owner 真实 proof 文件清单模板。它列出 sample run、package consumer runtime、post-publish verification 和 release close approval 所需文件路径，但模板本身不是 proof。

机器可读文件：

`artifacts/final-release/owner-evidence-file-manifest.template.json`

## 适用读者

- 收集真实模型、日志、JSON 和包文件的 owner。
- 准备导入外部 proof 的维护者。
- 需要核对 SHA256 和路径存在性的审核者。

## 解决问题

没有统一文件清单时，owner 很容易只填写一部分日志或 hash，导致 validator 失败。本模板将真实文件路径按四条 lane 分开，要求路径存在、SHA256 为 64 位小写十六进制，并禁止 `owner-to-fill`、`owner-required`、`template-only`、`example-not-proof` 或 `blocked-by-cuda-driver` 作为最终值。

## 边界说明

该模板固定 `templateState=template-only-not-proof`、`ownerToFill=true`、`canBeImportedAsProof=false`、`performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

它不能替代 runtime proof、post-publish proof、publish approval 或 release close approval。TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、dry-run、local feed、ProjectReference 和 direct `.nupkg` 只能作为上下文，不能替代真实文件清单。

## 可复制验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~OwnerEvidenceFileManifestTemplate" --logger "trx;LogFileName=owner-evidence-file-manifest-template.trx" --results-directory .\artifacts\test-results\targeted
```

## 下一步

1. owner 填写真实路径。
2. owner 填写真实 64 位 SHA256。
3. 确认所有路径存在。
4. 再运行 strict validator command runbook。
