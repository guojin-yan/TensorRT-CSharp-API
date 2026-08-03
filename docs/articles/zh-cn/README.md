# 中文文章目录说明

`docs/articles/zh-cn` 是 DocFX 中文概念文档源目录，不等同于“已经可以发布到公众号或博客的技术文章库”。这里同时包含：

- 用户教程和模型案例；
- API、ABI、生命周期和兼容性设计说明；
- CI/证据 schema/发布门禁的内部工程记录；
- Owner 输入、候选 proof 和 post-publish 模板；
- `publishing` 子目录中的文章规划与发布治理材料。

2026-08-03 盘点到 485 个 Markdown 文件，其中此前没有任何文章引用执行结果图片。数量代表项目过程材料多，不代表有 485 篇完整成稿。

## 完整文章标准

技术文章只有同时满足以下条件，才可以标记为 `complete-technical-article`：

1. 有明确读者、问题、环境、可复制命令和预期输出。
2. 有真实执行结果，不把模板、build-only、local feed 或截图单独写成发布 proof。
3. 至少有一张与本次真实执行结果对应的 PNG/JPEG/WebP 配图，并说明图片来自哪个报告或运行。
4. 模型案例必须写明权重/ONNX 获取 URL、固定 revision、许可证、转换命令、ONNX 输入输出合同、SHA256 和外层 `models` 暂存路径。
5. 配图不得嵌入没有公开再分发授权的模型、测试图片或第三方素材；可以使用本项目根据真实日志生成的结果图。
6. 有证据路径、失败条件、已知限制和 proof boundary。
7. 通过 `eng/Test-TechnicalArticleCompleteness.ps1`。

## 分类

| 分类 | 含义 | 可作为完整对外文章 |
| --- | --- | --- |
| `complete-technical-article` | 正文、真实结果、配图、来源和边界齐全 | 内容完整，但公开发布仍需 Owner 授权 |
| `documentation-ready` | 适合作为项目文档，可能缺执行配图或独立叙事 | 否 |
| `internal-engineering-record` | 设计、门禁、schema、proof 或审计记录 | 否 |
| `draft-needs-runtime-or-images` | 教程骨架存在，但缺真实执行或配图 | 否 |
| `owner-input-template` | 等待外部/Owner 回填的模板 | 否 |

机器可读发布目录位于 `publication-catalog.json`。只有被显式列入 `articles` 且通过严格门禁的文件，才算内容完整；所有未列入文件默认是 `project-documentation-not-publication-ready`。

当前首篇完成门禁的文章是 [YoloVision YOLOv8n Detection 本地包消费教程](yolovision-yolov8n-det-local-package-consumer-tutorial.md)。其结果图来自真实 TensorRT、本地三包、ONNX Runtime 和 Ultralytics/PyTorch 对照，不包含未授权测试原图。

DocFX 构建成功只说明链接和站点生成正确，不等于外部文章已经发布，也不替代 public-package、post-publish、Owner acceptance 或 release proof。
