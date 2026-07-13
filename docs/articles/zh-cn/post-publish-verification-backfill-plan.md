# Post-Publish Verification Backfill Plan

`post-publish-verification-backfill-plan` 是真实发布之后回填 clean consumer 证据的阶段计划。它只描述发布后如何采集、回填和校验 proof，不执行 `dotnet nuget push`、GitHub Packages 上传或 GitHub Release 上传。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationBackfillPlan.ps1
```

输出：

- `artifacts/final-release/post-publish-verification-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.md`

默认状态必须保持：

- `recordKind=post-publish-verification-backfill-plan`
- `planState=blocked-real-post-publish-proof-required`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `isPostPublishVerificationProof=false`
- `canCloseReleaseIssue=false`
- `postPublishProofClassification=template-only`

## 回填顺序

计划固定把 post-publish proof 回填拆成九步：

1. 确认 owner authorization、owner decision、发布 channel、回滚计划和凭证处理。
2. 将 `post-publish-verification-record-template.json` 复制为真实 record。
3. 从真实发布 channel 下载 managed/runtime package 并记录 URL 与 SHA256。
4. 在源码仓库外创建 clean consumer，禁止 `ProjectReference`。
5. 从真实发布 channel restore/build，并记录 restore log 与 native asset listing hash。
6. 在兼容主机运行 DependencyProbe 与 runtime smoke。
7. 人工复核 stdout/stderr 摘要；stderr 为空时写 `no-stderr-emitted`。
8. 使用 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` 验证真实 record。
9. 只有 validator 晋级后刷新 release evidence、freeze summary 和 owner command plan。

## 不能误读

- backfill plan 不是 post-publish proof。
- final package review 是 local package inventory，不是 public channel proof。
- local feed、ProjectReference、template、draft、runbook、collection bundle 和 dependency-probe-only 都不能关闭 release issue。
- `blocked-by-cuda-driver` 不是 smoke passed。
- build success 不是 runtime smoke proof。
- 只有真实 `post-publish-verification-record.json` 通过 `-RequireExistingLog -FailOnNotProof`，才允许 `canCloseReleaseIssue=true`。

## 与发布链路的关系

owner command plan 会聚合本计划的 `planState` 和 step count，帮助 owner 看清发布后 clean consumer 需要补齐的证据。该聚合仍保持 `performsPublish=false`、`canMaterializeExecutableCommands=false`，不会自动执行任何发布、删除、delist 或 withdraw 操作。
