# 发布后 Clean Consumer 真实 Proof Gate

`post-publish-clean-consumer-real-proof-gate` 聚焦仓库外 clean consumer proof。它要求真实公开源 restore、公开包消费路径、restore/smoke 日志 hash 和项目身份完整；缺任一真实 Owner 输入时继续 blocked。

## 边界

- 状态：`blocked-post-publish-clean-consumer-real-proof-gate-owner-proof-required`
- 默认 `Passed=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push

## 准入要求

- consumer project 必须在源码仓库外
- package source 必须是公开源
- restore/smoke exit code、stdout/stderr 摘要和日志 SHA256 必须完整
- ProjectReference、direct nupkg、local feed 计数必须为 0
