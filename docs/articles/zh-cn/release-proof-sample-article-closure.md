# 发布前真实 Proof 与高价值案例收口

## 目标

本文把 release proof、OnnxToEngine、TensorRtExec、YoloVision 和文章路线图放到同一个收口视图中。它用于指导发布前检查和后续 owner proof 回填，不是 runtime proof，也不是发布授权。

机器可读矩阵位于：

`artifacts/final-release/release-proof-sample-article-closure-matrix.json`

## 适用读者

- release owner、项目评审者和发布前质量门维护者。
- 需要区分 sample/report/design gate 与真实 runtime proof 的贡献者。
- 准备撰写 OnnxToEngine、TensorRtExec、YoloVision 或 proof boundary 宣发文章的技术作者。

## 解决问题

本阶段的问题不是缺少样例或报告，而是避免把可宣传、可采用、可审计的材料误判成 release proof。本文将工具报告、样例矩阵、文章路线图和 owner proof 输入分层说明，确保后续可以继续推广项目，同时不把 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 或 design gate 写成 runtime proof。

## 边界说明

以下内容必须保持非 proof：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- design gate
- sample-run-evidence without owner-filled logs
- blocked-by-cuda-driver

关键分层不能混用：

- sample-run-evidence 不能替代 package-consumer-runtime。
- package-consumer-runtime 不能替代 post-publish verification。
- TensorRtExec build/report 不能替代 real-model-runtime。
- YoloVision matrix 不能替代模型级输出校验。

## OnnxToEngine 收口

`samples/OnnxToEngine` 已具备 trtexec-like 转换样例的基础证据路径：

- ONNX 输入。
- engine 输出。
- min/opt/max shape profile。
- FP16 / INT8 边界。
- workspace / memory pool intent。
- timing cache intent。
- verbose diagnostics。
- report output。

这些能力适合帮助用户理解模型转换流程，但 build-only report、dry-run、shape profile matrix 和 OnnxToEngine report 仍不能晋级为 runtime proof。

## TensorRtExec 收口

`applications/TensorRtExec` 已形成 CLI + WinForms 双入口：

- console command entry。
- WinForms UI entry。
- shared normalized command model。
- build-only report。
- load-engine readonly diagnostics。
- CLI/GUI parity checklist。
- report schema。

GUI 截图、command preview、dry-run report、load-engine preflight 和 TensorRtExec report 都只能作为应用表面证据，不能替代 real-model-runtime 或 package-consumer-runtime。

## YoloVision 收口

`samples/YoloVision` 是唯一 YOLO-family 样例入口，覆盖计划包括：

- family：YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLOv26、custom。
- task：det、cls、seg、obb、pose、sem。

必须继续防止旧的 detection-only YOLO 样例名回流到 solution、docs、samples README、applications 或文章路线图。YoloVision matrix、owner asset candidate pack、sample README 和 template-only sample-run-evidence 都不是 proof。

## 文章路线图收口

`docs/articles/zh-cn/publishing/article-roadmap-30plus.json` 已覆盖不少于 30 篇文章规划，主题包括：

- 项目总览。
- 接口和 wrapper 设计。
- NuGet 安装。
- runtime packages。
- OnnxToEngine tutorial。
- TensorRtExec CLI / WinForms。
- YoloVision family tutorials。
- proof boundary 与 release evidence。
- troubleshooting。

这些文章面向微信公众号、博客和项目推广，不是 API 文档目录。文章 roadmap、draft article、截图和宣传文案都不能替代 runtime proof。

## 下一步

发布前真正可关闭的问题仍依赖 owner 回填：

1. YoloVision 真实模型、输入图片、labels、输出 JSON、stdout/stderr、SHA256 和 owner review。
2. 公开包来源的 clean consumer package-consumer-runtime proof。
3. post-publish clean consumer verification。
4. 首批完整文章正文、截图和资产 provenance。
5. Release close owner approval。
