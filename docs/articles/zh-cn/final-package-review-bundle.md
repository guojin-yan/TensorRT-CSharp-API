# Final Package Review Bundle

`Export-FinalPackageReviewBundle.ps1` 是发布候选最终打包前的 owner review 清单。它读取本地 managed/runtime/split-runtime `.nupkg`、runtime native asset manifest、release package proof、release evidence、external runtime proof validation、post-publish validation 和 freeze summary，并输出：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPackageReviewBundle.ps1
```

输出文件：

- `artifacts/final-release/final-package-review-bundle.json`
- `artifacts/final-release/final-package-review-bundle.md`

## 记录边界

该 bundle 只回答“当前本地发布候选包有哪些、大小是多少、SHA256 是什么、对应 runtime package key 和 native asset count 是什么”。默认必须保持：

- `recordKind=final-package-review-bundle`
- `bundleState=owner-review-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canUseAsPublicPackageProof=false`
- `canCloseReleaseIssue=false`

也就是说，final package review bundle 是本地包清单和 hash 审阅物，不是 public channel proof，不会执行发布，也不能替代 compatible-host runtime proof 或 post-publish clean consumer proof。

## Package 字段

每个 package item 至少包含：

| 字段 | 含义 |
| --- | --- |
| `kind` | `managed`、`runtime` 或 `split-runtime` |
| `relativePath` | 仓库内相对路径 |
| `packageId` / `version` | 从 `.nuspec` 读取的包身份 |
| `sizeBytes` / `sizeMb` | 本地文件大小 |
| `sha256` | 当前文件重新计算的 SHA256 |
| `runtimePackageKey` | 当前审阅对应的 runtime key |
| `targetFrameworks` | 从 `lib/` 或 `ref/` 目录推断的 TFM |

SHA256 只证明本地文件内容稳定；它不能证明文件已经进入 nuget.org、GitHub Packages、GitHub Release assets 或任何 owner 认可的私有渠道。

## 与其他证据的关系

`sourceEvidence` 会引用：

- `artifacts/final-release/release-package-proof-bundle.json`
- `artifacts/final-release/release-evidence-bundle.json`
- `artifacts/final-release/post-publish-verification-validation.json`
- `artifacts/final-release/external-runtime-proof-validation.json`
- `artifacts/release/release-candidate-freeze-summary.json`
- `artifacts/runtime/<runtimePackageKey>/artifact-manifest.json`

release evidence 可以聚合 final package review 的状态，但仍必须把它当作 owner-review artifact。真实发布前还需要 owner 授权；真实发布后还需要 post-publish verification 回填 clean consumer、host metadata、`--runtime-package-key` smoke command、stdout/stderr summaries 和日志 SHA256 match。

## 不可提升项

以下内容不能被写成 public package proof 或 release issue close proof：

- template / draft / example
- runbook / collection bundle / handoff
- dependency-probe-only
- `blocked-by-cuda-driver`
- 本地 feed restore/build
- final package review bundle

只有真实 external runtime proof、真实 post-publish proof 和 owner 授权同时齐全时，才允许重新评估 `canCloseReleaseIssue=true`。
