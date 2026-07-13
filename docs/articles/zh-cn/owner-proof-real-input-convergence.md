# Owner Proof Real Input Convergence

`owner-proof-real-input-convergence` 是 release close 最后一公里的 Owner 校验收敛矩阵。它把 Owner 真实输入、post-publish proof、package consumer runtime proof、final close decision、overlay candidate 和 strict close candidate 放到一个 owner-facing 视图中。

## 当前边界

- `convergenceState=blocked-owner-real-input-convergence-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

该矩阵只帮助 Owner 按顺序补齐真实输入和 proof，不执行 publish / push / upload，不关闭 release issue，不把 template、draft、candidate、hash match、local feed、ProjectReference、direct `.nupkg`、schema-only、preflight-only、dependency-probe-only 或 blocked-by-cuda-driver 当作 proof。

## 覆盖内容

- 8 个 release close real input mappings。
- post-publish verification owner input validation。
- package consumer runtime proof candidate validation。
- release issue final close decision validation。
- release issue close record candidate validation。
- release issue close record overlay candidate validation。
- owner external execution result backfill kit validation。
- release close strict record candidate validation。

## 推荐 Owner 顺序

1. 回填 public channel package source。
2. 在仓库外 clean consumer 环境执行 runtime smoke。
3. 回填 smoke log path、SHA256、exit code、host metadata 和 runtime key。
4. 回填 release issue id / url。
5. 回填 rollback owner / trigger / plan。
6. 在真实 proof 通过后回填 final close decision。
7. 刷新 overlay candidate、strict candidate、convergence matrix 和 release evidence bundle。
8. 最后运行 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 产物

- `artifacts/final-release/owner-proof-real-input-convergence.json`
- `artifacts/final-release/owner-proof-real-input-convergence.md`
- `artifacts/final-release/owner-proof-real-input-convergence-validation.json`
- `artifacts/final-release/owner-proof-real-input-convergence-validation.md`

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofRealInputConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofRealInputConvergence.ps1 -Strict
```

严格验证只证明收敛矩阵结构合法。只要真实 Owner 输入、post-publish proof、clean consumer runtime smoke log、final close decision 和 strict close validation 没有通过，`missingRealInputCount>=1`、`blockedValidatorCount>=1`、`blockedProofCount>=1` 与 `failedActionRequiredCount>=1` 就是预期状态。
