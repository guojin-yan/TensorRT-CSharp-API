# Owner 执行结果导入与发布前复验

`owner-execution-result-import-prepublish-recheck.json` 是发布前真实 Owner 执行结果的复验总控包。它只做导入前/导入后边界核对，不执行发布，不关闭 issue，也不把模板、报告、矩阵、dry-run 或 build-only 结果提升为 proof。

## 当前结论

- `recheckState=blocked-owner-action-required`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPackageConsumerRuntimeProof=false`
- `isPostPublishVerificationProof=false`

## 四条复验线

| 顺序 | Lane | 目标记录 | 当前状态 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | `real-model-runtime` | `real-case-evidence-record.json` | 记录缺失，只有模板 | 不可提升 |
| 2 | `package-consumer-runtime` | `package-consumer-runtime-proof-record.json` | 文件存在，但仍需严格 validator 证明 | 不可提升 |
| 3 | `post-publish-verification` | `post-publish-verification-record.json` | 文件存在，但真实公开发布尚未证明 | 不可提升 |
| 4 | `release-issue-close` | `release-issue-close-record.json` | 记录缺失，只有模板 | 不可提升 |

## 复验规则

- 文件存在不等于 proof。
- JSON 格式正确不等于 proof。
- template-only、placeholder、draft、runbook、report、matrix、article 都不能放行。
- `package-consumer-runtime` 必须来自仓库外 clean consumer、公开 package source、无 local feed、无 ProjectReference、无 direct `.nupkg`。
- `post-publish-verification` 必须发生在真实公开发布之后。
- `release-issue-close` 必须等 `real-model-runtime`、`package-consumer-runtime`、`post-publish-verification` 全部严格验证通过后再执行。

## 下一步 Owner 输入

1. 补齐 `real-case-evidence-record.json`，覆盖 YoloVision `det/cls/seg/obb/pose/sem` 真实运行日志、hash、host metadata 与 owner review。
2. 重新导入并严格验证 `package-consumer-runtime-proof-record.json`，确保来自公开包源和仓库外 clean consumer。
3. 真实公开发布后，再执行 `post-publish-verification-record.json` 的 public-channel clean consumer 复验。
4. 最后创建并验证 `release-issue-close-record.json`，并固定 evidence bundle/package hash 与 rollback/deprecation plan。

当前发布候选仍保持 `canPublishPublicly=false`、`canCloseReleaseIssue=false`。
