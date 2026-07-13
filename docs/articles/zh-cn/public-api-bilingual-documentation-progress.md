# Public API Bilingual Documentation Progress

本文记录 Public API 双语 XML 文档 gate 的当前进度和后续维护规则。

## 当前结果

最新审计：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicApiBilingualDocumentationBacklog.ps1
```

当前结果：

- `artifacts/api-doc-audit/public-api-bilingual-documentation-audit.json`：`findingCount=0`
- `artifacts/api-doc-audit/public-api-bilingual-documentation-backlog.json`：`backlogFindingCount=0`
- release checklist 中 `Public API bilingual documentation audit` 为 ready。
- final release dry run 中 `bilingualDocumentationFindingCount=0`。

## 维护规则

- 新增 public API 时，`summary`、`param`、`returns`、`remarks` 需要同时包含英文和中文。
- callback、allocator、borrowed pointer、runtime proof 相关文档必须保留边界语义。
- 中文说明可以简短，但必须说明是否是 design gate、precheck、dry-run、runtime smoke 或真实 proof。
- 如果 finding 回归，先运行 backlog exporter 生成分批清单，再按 high-value wrapper、callback/allocator boundary、diagnostic gate 的顺序处理。

## 仍需保持的边界

双语 finding 清零只证明 public API XML 文档已满足中英文检查。它不证明：

- Linux runner 已完成真实验证。
- CUDA 13.2 runtime smoke 已通过。
- `IDebugListener::processDebugTensor` 已真实触发。
- NuGet signing、NVIDIA 再分发和 release channel 已完成审批。

这些边界继续由 final release dry run、Linux handoff 文档、package consumer evidence 和 release owner approval guide 跟踪。
