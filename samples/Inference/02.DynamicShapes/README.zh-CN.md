# 动态 Shape 案例

[English](README.md) | 简体中文

该案例构建带 `-1` 批次维度的 identity 网络，演示显式优化配置中的 min/opt/max Shape、运行时输入 Shape 设置、设备内存分配、CUDA Stream enqueue 和结果读回。

## 运行

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = '1'
dotnet .\samples\Inference\02.DynamicShapes\bin\Release\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 3
```

传入的批次必须落在案例定义的优化配置范围内。成功输出应包含 profile、实际 Shape、输入输出值和通过标记。完整步骤、真实终端截图及 Shape 错误排查见 [Dynamic Shape 推理教程](../../../docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md)。

identity 网络用于隔离动态 Shape 机制本身；业务模型仍需按实际输入名和每个动态维度建立完整 profile。
