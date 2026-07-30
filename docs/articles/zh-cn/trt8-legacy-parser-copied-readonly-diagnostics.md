# TRT8 Legacy Parser 复制型只读诊断

## 适用场景

TensorRT 8.6 仍提供 UFF 与 Caffe legacy parser。`IUffParser`、`ICaffeParser` 和
`IBinaryProtoBlob` 都是 native owner 对象，`IBinaryProtoBlob::getData()` 还返回由 blob 管理的指针。
把这些对象或数据指针直接暴露给 C# 会让生命周期穿过 ABI。

本项目提供两个更窄的安全接口：

```csharp
TensorRtLegacyUffRequiredVersionSnapshot uff =
    TensorRtLegacyParserDiagnostics.GetUffRequiredVersion();

TensorRtCaffeBinaryProtoSnapshot mean =
    TensorRtLegacyParserDiagnostics.ReadCaffeBinaryProto("mnist_mean.binaryproto");
```

两个公开模型分别由 `TensorRtLegacyUffRequiredVersionSnapshot.cs` 与 `TensorRtCaffeBinaryProtoSnapshot.cs` 拥有；
不再共置于 plural snapshot 文件。文件拆分只整理源码职责，不改变下述复制和生命周期契约。

第一个接口在单次 native 调用内创建 UFF parser、读取三个版本标量并删除 parser。第二个接口创建 Caffe
parser 和 binaryproto blob，在 owner 有效期间复制 shape、data type 和所有数据字节，然后删除两个 native
对象。公开 API 不包含 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle` 或 parser/blob/data pointer。

## 复制与生命周期

binaryproto 采用两次 caller-buffer 调用：第一次查询所需长度与元数据，第二次复制数据。managed snapshot
在构造时再次复制输入数组，`Data` 属性每次还会返回新的 `byte[]`，因此调用方修改一份结果不会影响快照或
其他调用方结果。

native 边界在 C++ exception 和 Windows SEH 后都会清零输出。bridge 不调用
`shutdownProtobufLibrary`，因为它是进程级全局操作，不能由一个局部诊断请求决定。

## 版本边界

该功能只适用于 TensorRT 8 adapter。TRT10/11 在进入 native dispatch 前返回受控 `NotSupported`。
CMake 只有在 TensorRT major 为 8、两个 legacy header 存在且找到 `nvparsers` import library 时，才设置
`JYPPX_HAS_TENSORRT_LEGACY_PARSERS=1` 并链接/延迟加载 `nvparsers.dll`。

coverage 会扫描 TRT8 的 `NvCaffeParser.h` 与 `NvUffParser.h`。三个 UFF getter、
`ICaffeParser::parseBinaryProto` 以及 blob 的 data/type/dimensions getter 共 7 行通过显式 alias 记录为
`implemented-with-deferred-history`。旧 deferred manifest 不删除；Caffe/UFF network parse、plugin factory、
error recorder mutation、protobuf buffer mutation 和 process-global shutdown 继续 deferred。

## 本地验证结果

TRT8.6/CUDA12.1 的 MNIST mean binaryproto smoke 返回 UFF `0.6.9`，shape `[1,1,28,28]`、`Float`、
`3136` 个复制字节。复制 payload SHA256 为
`DF7D560B482098FAC1C6122C22BD0A54499ED9F8EC3AC6BAE8FC917D3A01774A`，独立 managed 副本与 TRT10 guard
均通过。

bindings 为 `197 manifests / 3975 records`。ABI/PE parity 为 TRT8 `993/993`、TRT10 `1087/1087`、
TRT11 `1234/1234`，missing 均为 0。三条 bridge-only 本地包 consumer 的 restore/build 均为 0 warning / 0
error，但它们只属于 `compile-surface-proof`。

## 证据边界

该结果是单机 source/runtime 诊断证据，不是公开 package consumer runtime、模型准确率、post-publish、
release-close 或发布授权。它不允许删除 deferred history，也不触发 NuGet/GitHub Packages push、GitHub
Release upload 或 GitHub Actions。
