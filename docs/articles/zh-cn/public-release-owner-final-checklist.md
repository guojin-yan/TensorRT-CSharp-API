# Public Release Owner Final Checklist

## 适用读者

本文面向最终执行公开发布的 owner，用一页式清单确认发布授权、真实 proof、package hash、post-publish 验证和 rollback 计划。

## 解决问题

公开发布前最危险的误判，是把准备材料当成发布授权。本文列出 owner 必须确认的最终项，并明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不能替代 runtime proof 或 post-publish proof。

## 背景与场景

TensorRtSharp 4 的发布链涉及主包、runtime package、CUDA/TensorRT/cuDNN 版本、Windows/Linux RID、clean consumer proof、真实模型运行和 release issue close。Owner final checklist 是发布执行前最后一层人工确认，不能由自动化文档代替。

## 操作路径

1. 确认 owner authorization 已记录，且发布命令、package source、API key 管理方式明确。
2. 确认 public package proof、package hash、symbols/docs 和 release notes 已准备。
3. 确认 clean consumer runtime proof 与兼容 host runtime proof 均通过 validator。
4. 确认 post-publish verification 和 rollback plan 已写入。
5. 执行发布后立即运行 post-publish clean consumer 验证，再决定是否关闭 release issue。

## 代码与文件入口

- `docs/articles/zh-cn/owner-release-execution-package.md`：owner 发布执行包。
- `docs/articles/zh-cn/public-release-owner-execution-package.md`：公开发布 owner 包。
- `docs/articles/zh-cn/release-proof-strict-validator-playbook.md`：strict validator。
- `docs/articles/zh-cn/post-publish-owner-verification-kit.md`：发布后验证采集包。
- `docs/articles/zh-cn/release-final-owner-action-sequence.md`：最终 owner 操作顺序。

## 图示建议

建议用一页 checklist：授权、package、proof、post-publish、rollback、close issue。每项附上 validator 和 blocker 字段。

## 边界说明

Checklist 不会自动授权发布，也不会执行 package push。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 仍不可替代真实 proof。

## 下一步

下一轮应等待 owner 真实输入或授权；若没有输入，继续完善 release close blocker ledger 和 repair pack，保持 can publish 与 close issue 状态为 blocked。
