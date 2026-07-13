# 真实 Proof 导入脚本与严格 Validator 联动

`real-proof-import-validator-orchestration.json` 记录四条 release proof lane 的导入命令、严格 validator 命令、输入路径、输出校验摘要路径和缺失 evidence 时的阻塞状态。

## 当前状态

- `orchestrationState=blocked-owner-action-required`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 脚本入口

- `eng/Test-RealCaseEvidenceRecord.ps1`
- `eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1`
- `eng/Test-PackageConsumerRuntimeProofRecord.ps1`
- `eng/Test-PostPublishVerificationRecord.ps1`
- `eng/Test-ReleaseIssueCloseRecord.ps1`
- `eng/Test-ReleaseProofOwnerBackfillSummary.ps1`

新增的 summary runner 只做只读汇总，不创建 proof，不生成 hash，不执行发布，也不会把缺失日志、缺失 hash 或缺失 host metadata 的 lane 标记为 passed。

当前项目仍不能公开发布，不能关闭 release issue。
