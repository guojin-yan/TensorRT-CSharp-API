# Final Owner Execution Checklist

`final-owner-execution-checklist` 是最终 Owner 人工执行与回填的最短路径清单。它列出 Owner authorization、仓库外 clean consumer runtime smoke、Linux runner proof、真实模型 runtime proof、post-publish verification 和 final close validation 的执行顺序。

- 主要产物：`artifacts/final-release/final-owner-execution-checklist.json`
- 校验产物：`artifacts/final-release/final-owner-execution-checklist-validation.json`
- 默认状态：`blocked-final-owner-execution-checklist-real-owner-input-required`
- 明确边界：不执行真实发布，不调用 `dotnet nuget push`，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

每条执行项都要求记录 stdout、stderr、log、hash/SHA256、exitCode 和 host identity；需要仓库外执行的步骤不能用本地 feed、ProjectReference、direct nupkg 或 build-only 输出替代。
