# TensorRT CSharp API v4.0 源码编译

本模块专门介绍 TensorRT CSharp API v4.0 的本地编译路线：生成式绑定、C++ Bridge、CMake preset、托管层、NuGet 打包和构建后验证。源码编译适合需要调试 native ABI、接入新的 TensorRT/CUDA 组合、验证自有 SDK 或参与项目开发的读者；普通使用者优先阅读安装模块并使用正式包。

项目入口保持明文：

```text
项目源码：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
Native Bridge：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/native
Binding Generator：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/tools/JYPPX.BindingGenerator
CMake Presets：https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/CMakePresets.json
运行库清单：https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

## 1. 当前文章

| ID | 主题 | 状态 |
| --- | --- | --- |
| `MSC-006` | Windows/Linux C++ Bridge 编译与最小 Smoke | `ready` |
| [`BLD-001`](managed/bld-001-managed-source-build-test-and-package-validation.md) | 托管层源码编译、测试分层与本地包验证 | `review` |
| [`BLD-002`](bindings/bld-002-binding-generation-diff-audit.md) | 绑定生成、Manifest 与差异审计 | `review` |
| [`BLD-003`](native/bld-003-cmake-presets-native-debugging.md) | CMake Preset、Native Bridge 与调试 | `review` |
| [`BLD-004`](runtime/bld-004-runtime-bridge-packaging.md) | Runtime Bridge split 打包与消费者验证 | `review` |

## 2. 后续源码编译选题

| 主题 | 计划覆盖内容 |
| --- | --- |
| 新平台 | Linux、容器和自托管 runner 的可复现构建 |

计划条目只记录未来文章范围，不在已发布正文中添加下一篇跳转。每篇源码编译文章必须同时给出配置命令、构建输出、失败边界和可清理的精确目录。

## 3. 源码编译模块限制

1. 源码构建结果不能冒充 NuGet 包发布证明或其它机器的运行证明。
2. 生成文件必须由仓库 generator 产生，不能手工修改后宣称可复现。
3. TensorRT、CUDA、cuDNN 等厂商运行库的许可和二进制不属于项目开源范围。
4. 涉及 native ABI 的文章必须保存 `dumpbin` 或 `ldd` 输出，并说明目标平台和 SDK 版本。
