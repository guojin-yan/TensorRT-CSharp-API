# 案例资产与证据清单

[English](README.md) | 简体中文

该目录只保存案例资产的轻量清单、许可证边界、哈希、运行证据和验证模板，不保存模型权重、ONNX、输入图片、TensorRT Engine 或包含本机隐私信息的原始日志。

## 模型目录

`demo-model-inventory.json` 是当前 10 个正式演示模型的权威清单，记录模型来源、固定版本、获取脚本、ONNX 转换方式、外层 `models` 路径、文件长度、SHA256、配套文章和运行证据。实际模型统一暂存在 Git 仓库外的：

```text
<workspace-root>/models
```

运行以下命令可只读验证模型是否齐全且哈希一致：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Sync-DemoOnnxModels.ps1 `
  -VerifyOnly
```

该验证不会上传或发布任何模型。模型、CUDA、cuDNN、TensorRT 和 NVRTC 均不得进入源码仓库、托管 NuGet 或 bridge-only 包。

## 证据边界

- `*-official-assets.json` 固定上游来源、许可证、文件长度与 SHA256。
- `*-runtime-evidence.json` 保存经过脱敏的运行结论和结果哈希。
- `*-candidate.template.json` 是待填写模板，不代表已经完成真实推理。
- `package-consumer-runtime`、公开包验证和发布证明必须由各自独立记录提供，不能由模板或源码内运行记录替代。

模型获取、转换和外层目录布局的完整说明见 `docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`。只有许可证允许、路径已脱敏且经过审核的小型 JSON、标签或结果图才可以提交。
