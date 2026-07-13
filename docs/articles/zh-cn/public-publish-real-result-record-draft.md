# Public Publish Real Result Record Draft

`public-publish-real-result-record-draft` 是真实公开发布结果的 Owner 草稿填写面。它把真实公开包来源、公开下载地址、包 SHA256、发布时间、命令 transcript、reviewer 和结果摘要拆成可回填字段，供后续 import/validator 使用。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishRealResultRecordDraft.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-publish-real-result-record-draft.json`
- `artifacts/final-release/public-publish-real-result-record-draft.md`
- `artifacts/final-release/public-publish-real-result-record-draft-validation.json`
- `artifacts/final-release/public-publish-real-result-record-draft-validation.md`

## 边界

默认状态为 `blocked-public-publish-real-result-record-required`，`RequiredFields=12` 且 `Blocked=12`。它只提供 Owner 回填草稿，不执行发布、不上传包、不批准公开发布，也不关闭 release issue。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须在真实公开渠道完成发布后，回填真实 package identity、公开 URL、下载包 SHA256、发布时间、命令 transcript 和 reviewer。local `.nupkg`、local feed、ProjectReference、direct nupkg、dry-run、template 或 dashboard 不能替代真实公开发布记录。
