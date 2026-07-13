# Owner 外部真实 Proof 导入校验器

`owner-external-real-proof-import-validator` 校验 Owner 回填 JSON 的字段完整性、URL/hash/source/host metadata 和禁止替代项。它只做输入校验，不执行 `dotnet nuget push`，不下载公开包，也不把输入模板晋级为 proof。

## 边界

- 状态：`blocked-owner-external-real-proof-import-validator-owner-input-required`
- 默认 `Passed=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push

## 校验重点

- URL 必须是公开源或明确公开渠道
- SHA256 必须是 64 位十六进制值
- local feed、ProjectReference、direct nupkg、template、dry-run 和 manual handoff 不能晋级
- runtime host metadata 必须包含 CUDA、TensorRT、driver 和 runtime package key
