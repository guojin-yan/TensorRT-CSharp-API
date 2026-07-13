# Release Evidence 非替代项指南

## 适用读者

本文面向发布负责人、维护者和外部验证人员，用于判断哪些材料可以进入 release evidence bundle，哪些材料只能作为辅助信息而不能替代真实 proof。

## 解决问题

项目中已经有大量 report、matrix、checklist、runbook 和 diagnostics。本文给出非替代项清单，避免把 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 误判为 runtime proof、public package proof 或 post-publish proof。

## 背景与场景

发布收口需要真实 clean consumer、兼容主机 runtime、真实 package source、hash 交叉核对和 owner decision。内部构建、样例矩阵和工具报告可以帮助定位问题，但不能证明公开包在外部消费环境中可恢复、可加载、可运行。

## 操作路径

1. 把 evidence 分为真实 proof、候选输入、辅助诊断和禁止替代项四类。
2. 对每个候选输入记录来源、机器、命令、package source、hash、stdout、stderr 和 validator。
3. 对 local feed、ProjectReference、direct `.nupkg`、template 和 dry-run 直接标记为非替代项。
4. 对 TensorRtExec report、OnnxToEngine report、YoloVision matrix 和 readonly diagnostics 只允许作为上下文引用。
5. 只有 validator 通过且 owner 字段完整的外部运行记录才能进入 close record。

## 代码与文件入口

- `docs/articles/zh-cn/release-proof-non-substitutes.md`：非替代 proof 清单。
- `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`：runtime proof 执行路径。
- `docs/articles/zh-cn/owner-external-real-proof-input-contract.md`：Owner 输入合同。
- `docs/articles/zh-cn/release-close-real-proof-readiness-gate.md`：ReleaseClose 准入 gate。
- `artifacts/interface-coverage/release-api-readiness-audit.json`：API readiness 辅助矩阵。

## 图示建议

建议用四象限图展示材料分类：真实 proof、候选输入、辅助诊断、禁止替代项。每个象限列出典型文件和是否可晋级。

## 边界说明

本文自身也是文档，不是 release approval。它不会发布包、不会关闭 issue、不会把 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report 或 readonly diagnostics 晋级为 proof。

## 下一步

下一轮应把非替代项检查继续固化到 strict validator 中，确保 release close 前所有候选 evidence 都能给出机器可读 blocker。
