# README 前台入口检查清单

README 是新用户和 release owner 进入项目的第一屏。TensorRtSharp4.0 的 README 需要同时服务三类读者：想快速试用的 .NET 用户、想运行样例的模型用户、准备发布或验证 proof 的 owner。本文给出 frontpage 检查清单，确保 README、README.zh-CN、docs index、toc、samples 和 applications 的入口一致。

## 1. 当前状态必须清楚

README 第一屏应说明：

- 当前是 release candidate final evidence freeze 之后的 owner action 阶段。
- 当前状态是 `blocked-real-proof-required`。
- `performsPublish=false`。
- `canPublishPublicly=false`。
- `canCloseReleaseIssue=false`。

不要把这些状态写成公开发布已经放行。README 可以说“发布候选证据链已冻结，剩余真实 blocker 明确”，不能把自动化产物写成 owner proof。

## 2. 快速入口必须完整

README / README.zh-CN 至少应能引导到：

| 入口 | 文件 |
| --- | --- |
| 文档首页 | `docs/index.md` |
| 运行时包选择 | `docs/articles/zh-cn/runtime-package-selection.md` |
| 最终审计地图 | `docs/articles/zh-cn/release-final-audit-map.md` |
| 对外介绍素材 | `docs/articles/zh-cn/release-public-story-pack.md` |
| owner proof backlog | `docs/articles/zh-cn/release-owner-proof-backlog.md` |
| 不可替代 proof | `docs/articles/zh-cn/release-proof-non-substitutes.md` |
| 样例入口 | `samples/README.md` |
| TensorRtExec | `applications/TensorRtExec/README.md` |

如果 README 提到了 TensorRtExec，就必须同时说明：build/precheck report 不是 `package-consumer-runtime`。

## 3. 样例命名必须统一

当前 YOLO-family 样例名是 `YoloVision`。README 不应重新把旧名作为当前项目入口。

允许出现：

- `samples/YoloVision`
- `YoloVision`
- “统一 YOLO-family 样例”

不应作为 live sample 出现：

- 旧检测样例名
- 旧 csproj 名
- 旧 sample path

如果需要解释历史改名，也必须写成历史备注，而不是当前入口。

## 4. 证据边界必须保留

README 中常见的高风险边界：

| 词 | 正确语义 |
| --- | --- |
| `build-only` | 构建证据，不是 release proof |
| `parse-only` | 参数解析和报告，不是 native TensorRT 行为已执行 |
| `sidecar-only` | 模型和报告 metadata，不是 runtime proof |
| `ProjectReference` | 源码开发便利，不是 package consumer proof |
| `blocked-by-cuda-driver` | 当前主机兼容性阻塞，不是 smoke 通过 |
| `package-consumer-runtime` | 只属于 release proof record |
| `real-model-runtime` | 需要真实模型、真实输入、真实日志和 validator |
| `post-publish verification` | 只能在真实渠道发布后执行 |

README 不必解释所有细节，但必须给出链接，让读者能从 README 跳到更完整的文章。

## 5. README 与 docs index 一致性

更新 README 后，必须同步检查：

- `docs/index.md` 是否存在同一批文章入口。
- `docs/toc.yml` 是否存在同一批文章入口。
- `technical-article-roadmap.md` 是否记录新增文章编号。
- `samples/README.md` 是否使用 `YoloVision`。
- `applications/TensorRtExec/README.md` 是否保留 proof 边界。

如果 README 是“前台门面”，docs index 就是“全文目录”。两者不能一个说 release proof 仍缺，另一个暗示已经完成。

## 6. 建议 README 区块

推荐在 README 前部放置：

1. Scope / 项目范围。
2. Release candidate status / 发布候选状态。
3. Quick links / 快速入口。
4. Samples and applications / 样例与应用。
5. Runtime packages / 运行时包。
6. Release proof boundary / 发布证据边界。

这些区块能让新读者先看到项目价值，再看到边界，不会把 blocker 误读成质量失败，也不会把自动化 helper 误读成 proof。

## 7. 验证命令

README frontpage 修改后至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap"
```

如果 README 同时改动 samples 或 applications，再追加 solution build。不要直接从临时输出目录运行测试 DLL，以免测试无法上溯定位仓库根目录。
