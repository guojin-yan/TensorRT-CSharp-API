# TensorRT CSharp API v4.0 版本发布

本模块收录 TensorRT CSharp API v4.0 的正式版本说明、升级路径和面向使用者的发布解读。这里的文章只引用公开包、公开 Release、稳定文档与可复核源码，不把内部 gate、proof、Owner 记录或候选发布材料带入公开正文。

## 1. 当前文章

| ID | 标题 | 源码版本 | 状态 |
| --- | --- | --- | --- |
| `REL-001` | [TensorRT CSharp API v4.0 4.0.0 正式发布：面向 .NET 的 TensorRT 与 CUDA 全新重构](2026/2026-08-10-tensorrtsharp-4.0.0.md) | `4.0.0` | `ready` |

状态以 [`article-index.json`](../article-index.json) 为准。`ready` 表示内容、链接和仓库事实已经过技术校验，不表示文章已发布到 CSDN 或其他外部平台。

## 2. 新增规则

1. 使用 `REL-###` 作为稳定 ID，文件按正式发布日期归档到年份目录。
2. 正文必须给出准确版本、包数量、支持矩阵、安装命令、升级影响和已知限制。
3. Release、NuGet.org、GitHub Packages 等公开状态只能在实际页面可访问后写成事实。
4. 外部发布前保持 `immutable=false`；发布后写入 URL、时间、提交和正文 SHA256，并改为 `published` 与 `immutable=true`。
5. 已冻结文章需要更正时新增文章并使用 `supersedes` 建立关系，不直接改写历史正文。
