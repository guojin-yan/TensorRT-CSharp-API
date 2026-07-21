# TensorRtExec I/O 与 Layer Precision Policies

`TensorRtExec` 和 `OnnxToEngine` 现在可以把官方 `trtexec` 的 I/O format、precision constraint 与逐层类型规则应用到已经解析的 TensorRT network。本文给出可直接运行的语法、匹配顺序、跨版本差异和报告判读方法。

这些能力解决的是 build policy，不是模型精度证明。报告中的 `ReadbackMatch=True` 说明 TensorRT 接受并回读了请求值，但不能证明调用方 buffer layout 正确、最终 tactic 符合预期、模型输出数值正确或 NuGet 包已由外部消费者验证。

## 支持的参数

| 参数 | 作用 | 应用位置 |
| --- | --- | --- |
| `--inputIOFormats` | 约束 network input 的数据类型和允许格式 | ONNX parse 成功后、optimization profile 添加前 |
| `--outputIOFormats` | 约束 network output 的数据类型和允许格式 | ONNX parse 成功后、optimization profile 添加前 |
| `--precisionConstraints` | 选择 `none`、`prefer` 或 `obey` | TRT8/10 builder flags |
| `--layerPrecisions` | 为匹配的 layer 设置计算精度 | TRT8/10 `ILayer::setPrecision` |
| `--layerOutputTypes` | 为匹配 layer 的每个 output 设置类型 | TRT8/10 `ILayer::setOutputType` |

## I/O grammar

单项格式为：

```text
type:format[+format]
```

多个 tensor specification 用逗号分隔。只有一个 specification 时会 broadcast 到全部 input 或 output；存在多个 specification 时，数量必须与 network input/output 数量完全一致。

支持的数据类型：

```text
fp32 fp16 bf16 int32 int64 int8 uint8 bool
```

支持的 format token：

```text
chw chw2 chw4 hwc8 chw16 chw32 dhwc8 cdhw32
hwc dhwc dla_linear hwc16 dla_hwc4
```

例如，所有输入使用 FP32 linear format：

```text
--inputIOFormats fp32:chw
```

两个输出分别使用 FP32 linear 和 FP16 的两种候选格式：

```text
--outputIOFormats fp32:chw,fp16:chw+chw2
```

空 token、未知 type/format、spec 数量不匹配都会在构建前失败，不会静默忽略。

## Layer rule grammar

逐层精度规则格式为：

```text
layerPattern:type[,layerPattern:type]
```

逐层 output type 格式为：

```text
layerPattern:type[+type][,layerPattern:type[+type]]
```

每个 pattern 最多包含一个 `*`。匹配顺序遵循三条规则：

1. 精确 layer 名称优先于 wildcard。
2. 同一优先级下，靠后的匹配规则覆盖靠前规则。
3. output type 只有一个时 broadcast 到该 layer 的全部 outputs；否则数量必须等于 output count。

以下规则让 `encoder` 下大部分 layer 使用 FP16，但把归一化层保持 FP32：

```text
--precisionConstraints prefer
--layerPrecisions "encoder*:fp16,encoder.norm:fp32"
--layerOutputTypes "head*:fp32"
```

`--layerPrecisions` 或 `--layerOutputTypes` 必须显式配合 `--precisionConstraints=prefer` 或 `obey`。任何 pattern 没有匹配到 layer 时会 fail closed，这能在模型重命名后及时暴露过期规则。

## 完整命令

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --tensor-rt-line 10 `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model-policy.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --inputIOFormats fp32:chw `
  --outputIOFormats fp32:chw `
  --precisionConstraints prefer `
  --layerPrecisions "encoder*:fp16,encoder.norm:fp32" `
  --layerOutputTypes "head*:fp32" `
  --buildOnly `
  --exportReport .\models\model-policy-report.json
```

首次接入真实模型时，建议先执行 `--dryRun` 检查 grammar，再执行 build-only。Layer pattern 只有在 ONNX parse 后才能对实际名称做命中检查，因此 dry-run 不能证明规则匹配。

## 跨版本行为

| TensorRT line | I/O type | Allowed formats | Precision constraints | Layer precision/output type |
| --- | --- | --- | --- | --- |
| TRT8 | set/readback；不支持本版本不存在的 BF16、INT64 | set/readback | prefer/obey raw index 按 TRT8 映射 | set/readback |
| TRT10 | set/readback | set/readback | prefer/obey set/readback | set/readback |
| TRT11 | `setType` 已移除；请求 type 必须等于 inferred type | type 匹配后 set/readback | flags 已移除，parse-only | setters 已移除，parse-only |

公开 `TensorRtBuilderFlag` 保持稳定逻辑编号，interop 边界再映射到 vendor raw index。TRT8 中 `DirectIO` 是 raw 12，`PreferPrecisionConstraints` 是 raw 11；永久 smoke 会显式检查两者隔离并输出 `BuilderFlagMapping=TRT8DirectIORaw12PreferRaw11:Isolated:True`。TRT11 已删除的 FP16、INT8、BF16、prefer/obey flags 会明确拒绝，不能把旧编号发送给新版本 API。

## 如何读报告

成功应用时，日志包含：

```text
TrtexecBuildPolicy Name=InputIOFormats Applied=True ... ReadbackMatch=True
TrtexecBuildPolicy Name=OutputIOFormats Applied=True ... ReadbackMatch=True
TrtexecBuildPolicy Name=PrecisionConstraints Applied=True ... ReadbackMatch=True
TrtexecBuildPolicy Name=LayerPrecisions Applied=True Matched=... ReadbackMatch=True
TrtexecBuildPolicy Name=LayerOutputTypes Applied=True Matched=... ReadbackMatch=True
```

这些选项随后进入 `OptionImplementationStatus.AppliedOptions`。版本不支持、TRT11 type 不匹配、dry-run、load-engine 或依赖不可用时不会产生伪 readback，对应选项保留在 `ParseOnlyOptions`。

## 常见失败

`requires one broadcast specification` 表示 I/O spec 数量既不是 1，也不等于 tensor 数量。

`did not match any network layer` 表示 pattern 拼写或模型 layer 名已变化。可以先用 `--dumpLayerInfo` / `--exportLayerInfo` 获取 engine inspector 文本，但最终规则仍针对 parsed network layer 名应用。

`tensor-set-type-removed` 表示 TRT11 的 inferred type 与请求 type 不一致。TRT11 不会通过伪 setter 改写类型；应修改模型导出、使用 strongly typed network 的类型设计，或选择与 inferred type 相同的 I/O spec。

`data-type-not-supported-on-api-line` 表示请求类型不存在于目标版本，例如 TRT8 的 BF16 或 INT64。

## 证据边界

本功能的 TRT10 identity smoke 证明 I/O、constraint、layer policy 的 set/readback、engine round-trip、enqueue 与 synthetic output match；TRT8 smoke 证明 DirectIO raw 12 与 prefer raw 11 不混淆。它们仍不是：

- 外部真实模型的 caller buffer layout 证明。
- tactic selection 或性能证明。
- FP16/BF16/INT8 数值准确率证明。
- DLA 模型执行证明。
- repository-external package consumer proof。
- NuGet/GitHub 发布批准。

真实模型文章应同时记录模型来源与许可证、ONNX/engine/input/output/log hash、具体 layer rules、运行主机、expected output 和 owner review。发布包结论继续由 clean external consumer 与 strict proof validator 决定。
