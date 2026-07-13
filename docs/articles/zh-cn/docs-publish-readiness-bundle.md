# Docs Publish Readiness Bundle

`Export-DocsPublishReadinessBundle.ps1` 用来把文档外发前的本地准备情况汇总为一个 owner-review bundle。它读取 `docs/toc.yml`、`docs/index.md`、中文文章目录、`technical-article-roadmap.md`、DocFX 本地输出、samples/smoke README、样例资产模板、核心文章覆盖和 owner action，并输出：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DocsPublishReadinessBundle.ps1
```

输出文件：

- `artifacts/final-release/docs-publish-readiness-bundle.json`
- `artifacts/final-release/docs-publish-readiness-bundle.md`

这个 bundle 的发布边界是：

- `canPublishDocsExternally=false`
- `readinessState=ready-for-owner-review` 只代表本地文档材料可供 owner 审阅
- DocFX 本地输出不是外部站点发布证明
- 文章数量和 sample-backed 统计不是营销批准
- 样例资产模板不是已下载模型，也不是 sample smoke pass

## 统计口径

脚本会统计：

| 字段 | 含义 |
| --- | --- |
| `articleCount` | `docs/articles/zh-cn/*.md` 数量 |
| `highQualityArticleCount` | 长度、命令、证据语言或样例引用满足本地启发式的文章数量 |
| `sampleBackedArticleCount` | 文中引用 `samples/`、`smoke/` 或 README 的文章数量 |
| `coreArticleCount` | 发布/外发前必须覆盖的核心文章数量 |
| `coreArticleCoveredCount` | 同时存在、进入 TOC、进入 index、满足质量启发式的核心文章数量 |
| `coreArticleMissingCount` | 仍需补齐或修正链接的核心文章数量 |
| `coreArticleCoverage` | 每篇核心文章的存在、TOC、index、质量启发式和 owner action 明细 |
| `ownerActionCount` | 文档外发前仍需 owner 确认的动作数量 |
| `hasArticleRoadmap` | 是否存在技术文章矩阵规划 |
| `docfxValidationState` | 是否存在 `docs/_site/index.html` |
| `canPublishDocsExternally` | 是否可直接外发，默认必须为 `false` |

## 核心文章覆盖

核心文章覆盖不是为了凑数量，而是为了保证外发材料至少包含项目定位、证据边界、安装消费、真实模型样例、TensorRtExec 应用和已知限制。当前脚本会检查：

- `project-overview.md`
- `release-evidence-bundle.md`
- `release-publish-execution-checklist.md`
- `external-runtime-proof-record.md`
- `post-publish-verification-record.md`
- `package-consumer-validation.md`
- `real-model-owner-backfill-checklist.md`
- `classification-model-assets.md`
- `yolovision-model-assets.md`
- `tensorrtexec-tool-getting-started.md`
- `known-limitations-4.0.0-rc.md`

每篇核心文章都需要同时满足：

- 文件存在。
- 被 `docs/toc.yml` 收录。
- 被 `docs/index.md` 链接。
- 满足本地质量启发式：内容长度足够，且包含命令、证据语言或样例引用。

## Owner Review 清单

文档外发前仍需要人工确认：

- `external-publishing-plan`：外发渠道、发布时间、平台 URL、rollback/update 方案。
- `cover-media-assets`：封面图、首图、截图、模型样例图片的版权与风格。
- `proof-claim-review`：package readiness、runtime proof、callback proof、Linux proof、post-publish proof 的表述没有越界。
- `sample-asset-claim-review`：Classification/YoloVision/YOLOX-S 仍说明模型、labels、图片由用户或 owner 提供。
- `blocked-by-cuda-driver` 没有被写成 smoke passed。

该 bundle 可以作为发布材料包的一部分，但不能单独批准文档外发。
