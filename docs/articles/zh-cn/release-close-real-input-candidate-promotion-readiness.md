# ReleaseClose 真实输入候选晋级 Readiness

`release-close-real-input-candidate-promotion-readiness` 是发布前的候选晋级 readiness 层。它把真实 Owner 输入合同、StrictClose 校验、ReleaseClose strict record candidate、最终 ReleaseCloseRecord real validator 和 final publish proof gate 汇总成 11 条候选晋级 lane。

它的定位很窄：判断哪些候选 lane 仍然缺真实 Owner 输入，以及哪些 strict validator 必须先通过。它不会执行发布、不会下载公开包、不会运行 runtime smoke、不会晋级 proof，也不能关闭 release issue。

当前状态固定保持 `blocked-release-close-real-input-candidate-promotion-real-owner-input-required`，因为 11 条 lane 全部等待真实 Owner 输入。`failedBlockerCount=0` 只表示 readiness artifact 自身结构有效，不表示 proof ready。

## 候选 Lane

- `public-package-proof`：等待公开包 URL、包 ID、版本和下载 nupkg SHA256。
- `clean-external-consumer-runtime-proof`：等待仓库外 clean consumer 工程、运行日志、日志 SHA256 和 host metadata。
- `post-publish-clean-consumer-proof`：等待发布后的 clean consumer install/run 日志和公开包 hash。
- `hash-path-validation`：等待 stdout/stderr、日志和 nupkg hash/path 真实校验。
- `forbidden-substitute-validation`：阻断 local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate、draft、build-only 等替代物。
- `strict-close-dry-run`：等待真实输入进入 StrictClose dry-run。
- `rollback-review`：等待公开包和 post-publish proof 后的 rollback review。
- `final-close-decision`：等待 Owner 最终 close decision。
- `release-close-strict-record-candidate`：等待 strict record candidate 消除缺失 owner input、缺失 real proof 和 hash mismatch。
- `final-release-close-record-real-validator`：等待最终 ReleaseCloseRecord real validator 接受所有必填字段。
- `final-publish-proof-gate`：等待 final publish proof gate 的 action-required 归零。

## 非 Proof 边界

该 readiness 层不是 runtime proof、不是 post-publish proof、不是 publish approval、不是 release close approval、不是 package push。它也不能把 runbook、dashboard、candidate、draft、local feed、ProjectReference、direct nupkg、dry-run 或 build-only 输出提升成 proof。

Owner 真实输入导入后，下一步必须先通过 hash/path、forbidden substitute、StrictClose dry-run、final ReleaseCloseRecord real validator 和 final publish proof gate，才能进入发布前最终验收。
