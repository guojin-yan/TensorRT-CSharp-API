# ReleaseClose 真实 Proof 准入 Gate

`release-close-real-proof-readiness-gate` 汇总公开发布结果、clean consumer proof、兼容主机 runtime proof、post-publish verification、rollback approval 和 Owner close decision。全部真实 proof 通过 strict validator 前，release close 必须保持 blocked。

## 边界

- 状态：`blocked-release-close-real-proof-readiness-gate-owner-proof-required`
- 默认 `Passed=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push

## 准入要求

- 公开发布结果必须来自真实公开渠道
- 公开包下载 hash 必须与 freeze/release package hash 交叉核对
- clean consumer 和 runtime proof 必须来自外部真实执行
- rollback plan 和 Owner close decision 必须明确
- strict release issue close validator 通过前不能关闭 issue
