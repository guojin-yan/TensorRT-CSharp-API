# Public Publish Final Owner Execution Pack

`public-publish-final-owner-execution-pack` 是公开包发布前的最终 Owner 人工执行包。它把 publish handoff、publish result owner input、post-publish clean consumer convergence、strict close ready dashboard 和 release evidence classification audit 串成一组人工执行 lane，方便 Owner 在真正公开发布前逐项核对。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishFinalOwnerExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishFinalOwnerExecutionPack.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-publish-final-owner-execution-pack.json`
- `artifacts/final-release/public-publish-final-owner-execution-pack.md`
- `artifacts/final-release/public-publish-final-owner-execution-pack-validation.json`
- `artifacts/final-release/public-publish-final-owner-execution-pack-validation.md`

## 边界

该执行包默认状态为 `blocked-public-publish-final-owner-execution-required`。它只整理 Owner 手动执行顺序，不执行 `dotnet nuget push`，不上传包，不批准公开发布，不关闭 release issue。

必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 需要在真实公开渠道执行发布，并回填包 ID、版本、渠道 URL、SHA256、发布时间、命令 transcript、clean consumer 验证和最终 close decision。缺少这些真实外部输入前，本包只能作为人工执行清单，不能作为 runtime proof、post-publish proof、publish approval、release close approval 或 package push。
