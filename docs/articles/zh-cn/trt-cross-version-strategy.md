# TRT8/TRT10/TRT11 跨版本策略

TensorRtSharp4.0 同时维护 TensorRT 8、TensorRT 10 和 TensorRT 11。跨版本目标不是把所有版本压成同一套最低公分母，而是在公共能力上提供稳定 C# 体验，在版本差异处保留明确 guard、manifest 和 runtime package key。

## 版本线为什么要分开

TensorRT 的 C++ API 会随大版本移动、替换或删除接口。简单共享一个 P/Invoke 入口会让调用方很难判断当前 DLL 是否支持某个能力，也很难把错误归因到 API 不存在、版本不匹配、native asset 缺失还是环境阻塞。

项目因此采用三层隔离：

- manifest 按 TensorRT 版本线维护。
- native bridge 按 TensorRT 版本线编译。
- runtime package key 精确写入 TensorRT/CUDA/cuDNN 组合。

常见 runtime key 形如：

```text
win-x64-trt10.11-cuda12.9-cudnn9.22
win-x64-trt11.0-cuda13.2-cudnn9.22
```

## Manifest 与 native 实现

TensorRT 相关 manifest 位于：

- `native/manifests/tensorrt/v8`
- `native/manifests/tensorrt/v10`
- `native/manifests/tensorrt/v11`

native 实现按版本线组织：

- `native/src/tensorrt/v8`
- `native/src/tensorrt/v10`
- `native/src/tensorrt/v11`
- `native/src/tensorrt/common`

新增或提升接口时，必须先确认这个接口在对应 TensorRT 版本中的真实可用性。TRT10 有的能力不一定存在于 TRT8；TRT11 的新接口也不应被包装成 TRT8 可调用。

## 托管层如何保持一致

高层 C# wrapper 应尽量给用户稳定的对象模型，例如 runtime、builder、network、engine、execution context、plugin inventory、error recorder snapshot 等。但 wrapper 内部必须尊重版本线：

- 公共 API 可用时，尽量复用同一高层类型。
- 版本专属 API 需要明确 guard 或返回能力查询结果。
- 缺失能力应返回 false、空 snapshot 或受控诊断，而不是让用户拿到裸 `IntPtr`。
- public API 不应暴露 plugin creator、allocator、debug tensor 等 borrowed pointer。

这也是项目偏向 count/copy、caller buffer、copied snapshot 的原因。跨 ABI 的对象地址不能随便进入托管世界。

## Runtime package 与版本证据

Windows 当前重点组合包括：

| Runtime key | 当前证据 |
| --- | --- |
| `win-x64-trt10.11-cuda11.8-cudnn8.9` | restore/build/native-copy/smoke 通过，探针输出 TensorRT 10.11 和 CUDA 11.8。 |
| `win-x64-trt10.11-cuda12.9-cudnn9.22` | restore/build/native-copy/smoke 通过，native asset patterns 为 `19/19`。 |
| `win-x64-trt11.0-cuda12.9-cudnn9.22` | restore/build/native-copy/smoke 通过，探针输出 TensorRT 11.0 和 CUDA 12.9。 |
| `win-x64-trt11.0-cuda13.2-cudnn9.22` | split/full package readiness clean，full package consumer smoke 被 CUDA error 35 阻塞为 `blocked-by-cuda-driver`。 |

这里的 `blocked-by-cuda-driver` 表示 packaged runtime 已启动到 CUDA runtime 边界，但当前机器驱动不满足 CUDA 13.2 runtime 执行条件。它不是 API 缺失，也不是 callback proof。

## 提升 API 时的跨版本清单

每次把 deferred row 提升为真实实现时，建议检查：

1. TRT8、TRT10、TRT11 manifest 是否都存在对应记录。
2. 该 API 是否是某个版本线独有。
3. native source 是否保持 no-throw C ABI。
4. C# interop 是否按版本路由到正确 bridge。
5. wrapper 是否能在缺失版本上给出受控结果。
6. smoke 或 quality test 是否覆盖至少一个实际 runtime 组合。
7. 文档是否避免把某个版本的能力写成所有版本通用。

## 不同状态的写法

发布材料中推荐使用这些表述：

| 状态 | 推荐写法 |
| --- | --- |
| manifest/source matched | “已纳入接口覆盖追踪” |
| non-deferred native export | “已有真实 native C ABI 实现” |
| wrapper compiled | “托管调用面已编译通过” |
| package consumer restore/build/native-copy | “消费端包布局验证通过” |
| `SmokeResult=passed` | “普通 runtime smoke 通过” |
| `blocked-by-cuda-driver` | “当前机器被驱动/runtime 兼容性阻塞” |
| callback proof false | “真实 callback runtime proof 仍为 false” |

不要把 `readiness blockers: 0` 写成“所有 runtime 和 callback 已完成”。它只说明当前 readiness 脚本没有发现包完整性和消费端证据层面的阻塞项。

## 下一步阅读

- [Runtime 包说明](runtime-packages.md)
- [Runtime Package 和 Split Package 怎么选](runtime-package-selection.md)
- [CUDA error 35 与驱动兼容排查](cuda-error-35-troubleshooting.md)
