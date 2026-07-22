# ONNX Parser Layer 输出复制元数据

## 适用场景

TensorRT 10 和 11 的 `IParser::getLayerOutputTensor` 返回由 parser 拥有的 `ITensor*`。直接把该指针包装成
公开 C# 对象会让生命周期跨过 native ABI，parser 释放后还可能留下悬空对象。本项目提供的是更窄的只读
诊断接口：在 parser owner 仍有效时复制元数据，不返回 tensor handle。

## 使用方式

```csharp
if (parser.TryGetLayerOutputTensorMetadata(
        "identity",
        0,
        out TensorRtOnnxLayerOutputTensorMetadata? metadata))
{
    Console.WriteLine(metadata.TensorName);
    Console.WriteLine(metadata.Shape);
    Console.WriteLine(metadata.DataType);
}
```

确定 layer/output 必须存在时，可以使用：

```csharp
TensorRtOnnxLayerOutputTensorMetadata metadata =
    parser.GetLayerOutputTensorMetadata("identity");
```

缺失 layer 或 output 时，`TryGet...` 返回 `false`，`Get...` 抛出 `KeyNotFoundException`。负数 output index
在进入 native 前被拒绝。

## 复制字段

快照包含：

- TensorRT adapter line、查询 layer 名称与 output index。
- tensor 名称和 64 位 shape。
- data type、location 与 allowed-format bitmask。
- dynamic dimension、shape tensor、execution tensor、network input 和 network output 标志。

`PointerFreeCopiedMetadata` 固定为 `true`，`RetainsNativeTensor` 固定为 `false`。快照的
`EvidenceKind` 是 `copied-readonly-diagnostics`，并明确不能晋级 release proof 或删除 deferred 记录。

## 版本差异

TensorRT 8 vendor header 没有该方法，因此 API 返回受控 `NotSupported`。TensorRT 10/11 header 声明的是
纯虚方法，它通过 parser vtable 调用，不对应同名 LIB/DLL export。验证应关注每条版本线独立编译、bridge
ABI/export parity 和真实 parser smoke，不能把“没有同名 DLL export”误判为方法缺失。

## 本地验证结果

TRT10.11/CUDA12.9 的内置 identity ONNX smoke 得到 `output`、shape `[-1, 4]`、`Float`、`Device`，并确认
它是 dynamic execution tensor 和 network output；`TryGet` 与 `Get` 两份复制结果一致，缺失 layer 返回
`false`，最终 enqueue/output compare 仍通过。

TRT11.0/CUDA12.9 能加载 bridge 和 vendor DLL、读取版本与 registry，但当前主机上的 vendor
`createInferBuilder`/`createInferRuntime` 返回 null，因此 metadata 路径受控跳过。这个结果只能记为
dependency/runtime probe，不能写成 TRT11 metadata runtime proof。

## 证据边界

复制元数据可以证明 public API 没有借出 parser-owned tensor，并能提供源码、ABI 和单机 runtime 诊断证据。
它不是模型准确率、外部模型、公开包 consumer、post-publish 或 release-close proof，也不授权公开发布。
