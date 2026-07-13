# ReleaseClose 真实输入最终 Blocker 台账

`release-close-real-input-final-blocker-ledger` 是最终 blocker ledger，列出 release close 仍被真实公开发布、post-publish verification、clean consumer runtime proof、runtime compatible host proof、Owner close decision 和 strict close validator 阻塞的 proof lane。

它默认 `Passed=false`，只记录缺口，不执行发布、不关闭 release issue、不替代真实 proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseRealInputFinalBlockerLedger.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseRealInputFinalBlockerLedger.ps1 -Strict
```
