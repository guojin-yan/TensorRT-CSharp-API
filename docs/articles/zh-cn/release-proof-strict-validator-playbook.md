# Release Proof Strict Validator Playbook

## 适用读者

本文面向发布负责人和维护者，用于执行真实 proof 输入导入、禁止替代项扫描、hash 交叉核对和 release close readiness 判断。

## 解决问题

Release close 不能依靠人工感觉。本文给出 strict validator 的执行顺序，确保 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不会被误晋级为 runtime proof、public package proof 或 post-publish proof。

## 背景与场景

项目已经有很多 runbook、collection bundle、report 和 matrix，它们都能帮助 owner 准备真实输入，但不能替代真实输入。Strict validator 的目标是在发布前自动指出 blocker，而不是为了通过测试放宽边界。

## 操作路径

1. 收集 owner 输入记录，确认 source、package、host metadata、stdout/stderr、hash 和 decision 字段完整。
2. 执行 forbidden substitute scan，拒绝 local feed、ProjectReference、direct `.nupkg`、template、dry-run 和 build-only。
3. 执行 hash cross-check，确认 package、log、record 和 report 引用一致。
4. 执行 clean consumer/runtime proof validator，确认外部 restore/build/run 真实通过。
5. 只有所有 gate 通过且 owner approval 明确时，才允许进入 release close record。

## 代码与文件入口

- `docs/articles/zh-cn/owner-external-real-proof-input-contract.md`：Owner 输入合同。
- `docs/articles/zh-cn/owner-real-input-forbidden-substitute-validator.md`：禁止替代项 validator。
- `docs/articles/zh-cn/public-package-hash-cross-check-gate.md`：hash 交叉核对。
- `docs/articles/zh-cn/release-close-real-proof-readiness-gate.md`：ReleaseClose gate。
- `tests/JYPPX.ProjectQuality.Tests`：proof/release close 相关测试。

## 图示建议

建议画 gate pipeline：input contract -> forbidden substitute scan -> hash cross-check -> runtime proof validator -> owner approval -> close record。

## 边界说明

Strict validator playbook 是执行说明，不是 proof。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 均必须保持非替代项。

## 下一步

下一轮如果 owner 提供真实输入，应直接运行 validator 并输出 blocker ledger；如果仍没有输入，应继续完善 validator 测试与 owner repair pack。
