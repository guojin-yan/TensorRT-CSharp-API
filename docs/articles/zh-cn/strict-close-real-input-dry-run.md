# StrictClose 真实输入 Dry Run

`strict-close-real-input-dry-run` 汇总 Owner 输入合同、JSON 导入、hash/path validator、forbidden substitute validator 和最终真实输入准入包，输出 strict close 是否仍然 blocked。

它是 dry-run，不是 release close approval，不会关闭 issue，也不会把任何候选材料晋级为真实 proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-StrictCloseRealInputDryRun.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StrictCloseRealInputDryRun.ps1 -Strict
```
