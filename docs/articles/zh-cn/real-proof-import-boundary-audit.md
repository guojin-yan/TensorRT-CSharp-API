# Real Proof Import Boundary Audit

`real-proof-import-boundary-audit` 扫描 `artifacts/final-release`、README、docs、samples、applications 和 src，确认公开/发布面没有把 forbidden substitutes 写成真实 proof 或 Owner approval。

- 主要产物：`artifacts/final-release/real-proof-import-boundary-audit.json`
- 当前通过状态：`real-proof-import-boundary-audit-passed`
- findingCount 必须保持 `0`
- 边界：它只是 claim boundary audit，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

它重点拦截 local feed / ProjectReference / direct nupkg 被写成 package-consumer proof，build-only / dry-run / parse-only / sidecar-only 被写成 runtime proof，dashboard / runbook / candidate / draft 被写成 Owner approval，以及 `failedBlockerCount=0` 被误写成 ready。
