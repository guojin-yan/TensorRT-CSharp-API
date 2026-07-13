# Release Close Owner Input Bridge

`release-close-owner-input-bridge` 将真实外部 proof 执行链路、post-publish、rollback/final owner decision 和 strict close validator 聚合成 release close owner gate。它用于给 Owner 看清最后关闭 release issue 前还缺哪些真实输入。

## 覆盖范围

- 聚合 7 个 close owner gate。
- 当前默认 6 个 gate blocked、1 个 gate ready。
- ready 的 classification audit gate 不能让 release close 变成 ready。
- 默认 `bridgeState=blocked-release-close-owner-input-required`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseOwnerInputBridge.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseOwnerInputBridge.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-close-owner-input-bridge.json`
- `artifacts/final-release/release-close-owner-input-bridge.md`
- `artifacts/final-release/release-close-owner-input-bridge-validation.json`
- `artifacts/final-release/release-close-owner-input-bridge-validation.md`

## 边界

该 bridge 是 blocked owner gate aggregation，不是 release close approval。gate readiness、classification audit 通过和本地 evidence 聚合都不能替代真实 runtime proof、post-publish proof、rollback approval、final owner decision、package publication 或 strict close approval。
