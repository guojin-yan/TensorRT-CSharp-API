# StrictClose 真实输入 Finding 报告

`strict-close-real-input-finding-report` 按公开包 hash、clean consumer、runtime host、post-publish verification、rollback review、Owner close decision 和禁止替代项分组汇总 blocker。

它只用于诊断和 Owner 下一步行动，不是 proof、不执行发布、不关闭 release issue。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-StrictCloseRealInputFindingReport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StrictCloseRealInputFindingReport.ps1 -Strict
```
