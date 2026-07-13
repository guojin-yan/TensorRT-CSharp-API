# Real External Proof Overlay Pack

`real-external-proof-overlay-pack` 是真实外部 proof 回填的集中 Owner 输入面。它把 package-consumer-runtime、post-publish verification、release close owner input 和 final close decision 需要的真实字段、日志、SHA256、严格校验命令集中到一个 overlay pack，方便 Owner 在最后一公里一次性回填。

它不是 proof，不会执行发布，不会上传包，不会把 template/candidate/guidance 晋级，也不会关闭 release issue。当前状态必须保持：

- `overlayState=blocked-real-owner-input-required`
- `validationState=blocked-real-owner-input-required`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofOverlayPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofOverlayPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/real-external-proof-overlay-pack.json`
- `artifacts/final-release/real-external-proof-overlay-pack.md`
- `artifacts/final-release/real-external-proof-overlay-pack-validation.json`
- `artifacts/final-release/real-external-proof-overlay-pack-validation.md`

## 覆盖范围

overlay pack 至少覆盖四条 Owner 回填线：

- `package-consumer-runtime-proof`
- `post-publish-verification`
- `release-close-owner-input`
- `release-issue-final-close-decision`

每条线必须保留 `ownerActionStatus=owner-action-required`，并继续显式拒绝 local feed、ProjectReference、direct `.nupkg`、dependency-probe-only 和 `blocked-by-cuda-driver` 等非替代 proof。

## 不能误读

- overlay pack 只是 Owner 输入清单，不是 runtime proof。
- strict validator command 只是检查入口，不代表 proof 已通过。
- 日志路径、SHA256、host metadata 和 clean consumer identity 必须来自真实外部执行。
- `failedBlockerCount=0` 只表示 overlay 形状可用，不表示 release ready。
- 只要 `failedActionRequiredCount>=1`，release gate 必须保持 blocked。
