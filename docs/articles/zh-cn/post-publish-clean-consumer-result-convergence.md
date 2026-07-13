# Post-Publish Clean Consumer Result Convergence

`post-publish-clean-consumer-result-convergence` 聚合公开发布结果导入、post-publish verification、package-consumer runtime proof、clean consumer source scan 和 final post-publish audit pack。

它的作用是暴露 clean consumer 缺口，而不是把缺口提升为 proof。只要 clean consumer log/hash/host metadata/package source 仍缺失，或仍存在 local feed、ProjectReference、direct nupkg，状态就必须保持 blocked / non-proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishCleanConsumerResultConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerResultConvergence.ps1 -Strict
```
