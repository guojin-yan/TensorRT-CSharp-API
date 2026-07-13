# 公开发布前 Owner 实证输入落地包

`prepublish-owner-evidence-input-landing-pack.json` 是给 Owner 执行真实 proof 回填时使用的落地清单。它不是 proof，不执行发布，不关闭 release issue，也不允许把模板、报告、矩阵、dry-run、build-only 或本地包消费结果当作可发布证据。

## 当前状态

- `landingState=blocked-owner-action-required`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`
- `approvesPublicRelease=false`

## 四条 Owner 输入线

| 顺序 | Lane | Owner 输入文件 | 最终记录 | 当前状态 |
| --- | --- | --- | --- | --- |
| 1 | `real-model-runtime` | `real-case-evidence-record.json` | `real-case-evidence-record.json` | 缺真实输入 |
| 2 | `package-consumer-runtime` | `package-consumer-runtime-proof-owner-input.json` | `package-consumer-runtime-proof-record.json` | 文件存在但 strict proof 未证明 |
| 3 | `post-publish-verification` | `post-publish-verification-record.json` | `post-publish-verification-record.json` | 文件存在但真实公开发布未证明 |
| 4 | `release-issue-close` | `release-issue-close-record.json` | `release-issue-close-record.json` | 缺输入且依赖上游 proof |

## 关键落地要求

- `real-model-runtime` 必须覆盖 YoloVision `det/cls/seg/obb/pose/sem` 六类任务，每个任务都要有模型来源、license/source URL、ONNX hash、engine hash、输入输出 hash、stdout/stderr hash 和必要截图 hash。
- `package-consumer-runtime` 必须来自仓库外 clean consumer，并使用可追踪公开 package source；必须明确无 ProjectReference、无 local feed、无 direct `.nupkg`。
- `post-publish-verification` 必须发生在真实公开发布之后，不能用发布前 package consumer proof 替代。
- `release-issue-close` 必须最后执行，并依赖前三条 strict validator 全部通过。

## Owner 执行输出

每条 lane 都必须归档：

- 原始运行日志
- `Get-FileHash -Algorithm SHA256` 输出
- host metadata 捕获输出
- strict validator 输出
- 常见失败原因和修复动作

当前项目仍不能公开发布，不能关闭 release issue。
