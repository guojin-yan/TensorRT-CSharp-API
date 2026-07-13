# Public Publish Real Result Owner Input Contract

`public-publish-real-result-owner-input-contract` 是真实公开包发布后由 Owner 回填发布结果的输入合同。它要求 Owner 提供公开渠道 package id、version、source/channel URL、nupkg SHA256、发布时间、命令 transcript、owner reviewer 和结果摘要等字段，用来把人工发布动作转成后续 validator 可审计的输入面。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishRealResultOwnerInputContract.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishRealResultOwnerInputContract.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-publish-real-result-owner-input-contract.json`
- `artifacts/final-release/public-publish-real-result-owner-input-contract.md`
- `artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json`
- `artifacts/final-release/public-publish-real-result-owner-input-contract-validation.md`

## 边界

该合同默认状态为 `blocked-public-publish-real-result-owner-input-required`。它只定义 Owner 必须回填的真实公开发布结果字段，不执行发布、不上传包、不批准公开发布，也不关闭 release issue。

必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须在真实公开渠道完成发布后，用真实 package identity、公开下载 URL、SHA256、发布时间和命令 transcript 回填该合同。local `.nupkg`、local feed、ProjectReference、direct nupkg、dry-run、template 或 dashboard 不能替代真实公开渠道结果。
