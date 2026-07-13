# Release Issue Close Record Candidate

`release-issue-close-record-candidate` 是 release close 的 owner input surface，用于把 release evidence bundle hash、release close preflight hash、stale claims audit hash、post-publish proof validation hash、rollback plan 和 owner final close decision 集中到一个候选记录中。

它不是 close proof，不执行发布，也不关闭 release issue。只有真实 post-publish proof validation、真实 rollback approval、真实 owner final close decision 和 strict close validator 全部通过后，才可能进入 close review。

## 输出

- `artifacts/final-release/release-issue-close-record-candidate.json`
- `artifacts/final-release/release-issue-close-record-candidate.md`
- `artifacts/final-release/release-issue-close-record-candidate-validation.json`
- `artifacts/final-release/release-issue-close-record-candidate-validation.md`

## 默认状态

- `recordKind=release-issue-close-record-candidate`
- `candidateState=blocked-release-close-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 必需真实输入

- release evidence bundle path/SHA256。
- release close preflight path/SHA256。
- stale claims audit path/SHA256。
- post-publish proof validation path/SHA256。
- rollback plan、rollback owner、rollback trigger。
- owner final close decision 与 timestamp。
- `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` strict validator。

## 边界

该 candidate 只描述 owner close input readiness。template、draft、repair pack、orchestrator、preflight-only、post-publish placeholder、缺失 owner decision 或缺失 rollback plan 都不能关闭 release issue，`canCloseReleaseIssue=false` 必须保持不变。
