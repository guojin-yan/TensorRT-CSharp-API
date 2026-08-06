# 推理绑定案例

[English](README.md) | 简体中文

该案例直接在 C# 中构建一个显式批次 identity 网络，用最小模型说明 TensorRT 推理中最容易出错的资源关系：输入输出 Tensor、主机和设备内存、绑定地址、CUDA Stream、enqueue 与输出读回。

## 运行

先构建 Release，再从仓库根目录运行：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = '1'
dotnet .\samples\Inference\01.Bindings\bin\Release\net8.0\InferenceBindings.dll --tensor-rt-line 10 --batch 2
```

程序应输出网络输入输出信息、绑定状态、读回值和最终通过标记。完整代码拆解、真实 Windows Terminal 截图和运行证据见 [TensorRT 推理输入、显存绑定与 GPU 输出读回](../../../docs/articles/zh-cn/inference-bindings-tutorial.md)。

本案例使用程序内构造的 identity 网络，不需要外部 ONNX；它证明 binding 工作流，不代表某个业务模型的准确率。
