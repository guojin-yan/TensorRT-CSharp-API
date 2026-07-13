# Release Close Public Proof Bridge

`release-close-public-proof-bridge` 将 post-publish proof owner confirmation、release close owner bridge、close record candidate、final close decision、strict close validator 和 classification audit 聚合成最终 public proof close gate。

## 覆盖范围

- 聚合 6 个 public proof gate。
- 默认 5 个 gate blocked、1 个 classification audit gate ready。
- 默认 `bridgeState=blocked-release-close-public-proof-required`。
- strict close command 必须指向 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePublicProofBridge.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseClosePublicProofBridge.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-close-public-proof-bridge.json`
- `artifacts/final-release/release-close-public-proof-bridge.md`
- `artifacts/final-release/release-close-public-proof-bridge-validation.json`
- `artifacts/final-release/release-close-public-proof-bridge-validation.md`

## 边界

该 bridge 是 blocked public proof gate aggregation，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。classification audit ready 只能证明非 proof 边界没有漂移，不能让 release close ready。
