# Post Publish Verification Owner Input

`post-publish-verification-owner-input` 是真实发布后由 Owner 回填 post-publish verification record 的输入合同。它把公开包源、下载包 SHA256、clean consumer、restore/build/smoke 日志、stdout/stderr 摘要和主机元数据集中成一个可验证模板。

## 产物

- `artifacts/final-release/post-publish-verification-owner-input.template.json`
- `artifacts/final-release/post-publish-verification-owner-input.template.md`
- `artifacts/final-release/post-publish-verification-owner-input-validation.json`
- `artifacts/final-release/post-publish-verification-owner-input-validation.md`
- `artifacts/final-release/post-publish-verification-record.json`
- `artifacts/final-release/post-publish-verification-record.md`

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationOwnerInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordFromOwnerInput.ps1 -OwnerInputPath artifacts/final-release/post-publish-verification-owner-input.template.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

## Proof 边界

- owner input/template/record projection 都不是 proof。
- local feed、ProjectReference、direct `.nupkg`、schema-only、preflight-only 都不能替代 post-publish proof。
- 只有真实公开包源、仓库外 clean consumer、匹配的日志 SHA256、完整 host metadata 和 compatible-host smoke 通过 strict validator 后，`post-publish-verification-validation.json` 才能推动 release close。
