# CUDA Runtime Compilation（NVRTC）接入路线图

## 目标与当前边界

本项目已经交付 CUDA Runtime Compilation（NVRTC）的第一阶段 owner-safe compile API，使 .NET 用户可以提交 CUDA C++ 源码、headers、编译选项和 name expressions，并获得复制到托管内存的 PTX、CUBIN 或 LTO IR 工件。owner-safe kernel launch/readback 仍是下一阶段目标。

当前仓库已经具备 `CudaKernelLibrary.Load(byte[])`、library inventory、按名称查询和 kernel attribute 设置能力，native owner 会复制输入 code，并且不会向 public C# API 暴露 borrowed `cudaKernel_t`。这套 runtime library owner 仅在 CUDA Toolkit 12.9 及以上可用。现有 raw `cudaLaunchKernel` entry point 仍是 internal/generated 边界，不能作为 public RTC 启动方案。

NVRTC 接入不是单个 P/Invoke。它同时涉及 compiler program 生命周期、可变长日志/工件复制、编译选项与源文件可追溯性、compiled code 的 module ownership、typed kernel arguments、跨 toolkit 动态依赖和包体策略。

## 已核对的本机基线

2026-07-28 在不下载任何新文件的前提下，已核对本机安装的两个 Toolkit：

| Toolkit | Header / import library | Windows runtime |
| --- | --- | --- |
| CUDA 12.9 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `nvrtc64_120_0.dll`, `nvrtc-builtins64_129.dll` |
| CUDA 13.2 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `bin/x64/nvrtc64_130_0.dll`, `nvrtc-builtins64_132.dll` |

两个 header 均提供 version/error、program create/destroy、compile、program log、PTX、CUBIN、LTO IR、name expression 和 lowered name API。CUDA 12.9 header 仍有 deprecated NVVM output 声明；新 public API 不以该 deprecated output 为主路径。

CUDA 11.8、12.1、12.9、13.2 的 Windows header、import LIB、DLL、builtins 和 export 已由 `eng/Export-CudaRtcCapabilityMatrix.ps1` 实际审计。Linux `.so` 在本机 E 盘和 Windows Toolkit roots 中未找到，因此 Linux SONAME/symbol 仍保持未验证，不能从 Windows 结果外推。

## 2026-07-28 已实现基线

- native 新增 optional dynamic loader 与 `JYPPX_CudaRtcProgram` owner；`JYPPX_NVRTC_LIBRARY` 可指定精确 library，核心 bridge 不静态链接 NVRTC。
- ABI 覆盖 capability、dependency diagnostic、source/program name、virtual header、name expression、compile、log、PTX/CUBIN/LTO IR 与 lowered-name 的 caller-buffer/count-copy；输入有 UTF-8、embedded NUL、重复值、数量和字节上限，C++ exception 与 Windows SEH 均在边界内收敛。
- managed 新增 `CudaRtcCompiler`、`CudaRtcProgram`、`CudaRtcProgramSource`、`CudaRtcCompileOptions`、`CudaRtcCompilationResult` 与 `CudaRtcArtifact`；public surface 不暴露 `IntPtr`、`SafeHandle` 或 vendor program/kernel handle。
- `samples/CudaRuntimeCompilation` 真实覆盖 virtual header、template lowered name、成功 PTX、`sm_75` CUBIN、可用版本的 LTO IR、重复 PTX SHA256 确定性和 intentional compile failure log。
- 本机四版 compile 均成功；11.8/12.1/12.9 PTX 可由当前 CUDA 12.9 `CudaKernelLibrary` 加载，13.2 PTX 被当前 runtime/driver 以 `cudaErrorUnsupportedPtxVersion` 拒绝，因此 13.2 只记 compile proof。
- 证据位于 `artifacts/cuda-runtime-compilation/capability-matrix.json` 与 `local-smoke.json`。所有记录均保持 `kernelLaunch=false`、`gpuReadback=false`、`correctnessProof=false`。

尚未完成的 RTC 主项是 owner-bound named-kernel launch、typed argument packing、GPU readback、Linux 真机、full-runtime `cuda-rtc` 组件物化、clean package consumer 与 post-publish；这些项目未因 compile/load 成功而晋级。

## 设计不变量

1. Native program owner 使用 `JYPPX_CudaRtcProgram`，create/destroy/compile 全部 no-throw；C++ exception、Windows SEH 和 NVRTC error 都在 bridge 内转换为稳定状态与复制型诊断。
2. Native owner 复制 source、program name、header source/name、compile options 和 name expressions；不得保留 caller memory，也不得返回 `nvrtcProgram` 或 lowered-name borrowed pointer。
3. 字符串输出采用 caller-buffer size/copy，两进制输出采用 count/copy；所有长度先做上限和溢出检查。compile log 即使在编译失败时也必须可取回。
4. Public C# API 不暴露 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、`nvrtcProgram`、`cudaKernel_t`、`CUmodule`、`CUfunction` 或 kernel argument pointer array。
5. 编译失败属于可诊断结果，不应丢失 log；非法生命周期、依赖缺失和 bridge ABI 错误仍使用现有异常/状态映射。
6. NVRTC 必须是可诊断的可选能力。未使用 RTC 的 consumer 不应仅因机器缺少 NVRTC 而无法加载核心 bridge。
7. 每个输出工件均为 immutable copied bytes，并带 source/options/header/compiler/target/output hashes；不能依赖 native program owner 存活。

## 计划中的托管 API

建议以以下高层类型为目标，最终命名应通过 public API review：

- `CudaRtcCompiler`：能力探测、compiler version 和一次性 compile 入口。
- `CudaRtcProgram`：明确 `IDisposable` 生命周期的可复用 native program owner。
- `CudaRtcProgramSource`：source、virtual program name、headers 和 name expressions 的 immutable input。
- `CudaRtcCompileOptions`：目标架构、language/optimization/debug flags 和原始受控 options 的 immutable snapshot。
- `CudaRtcCompilationResult`：success/status、完整 log、lowered names、compiler/toolkit version 和 artifacts。
- `CudaRtcArtifact` / `CudaRtcArtifactKind`：PTX、CUBIN、LTO IR 的 copied payload、长度与 SHA256。

PTX、CUBIN 和 LTO IR 是 option/target dependent outputs。API 必须显式表示 unavailable，而不是返回空数组并让调用者猜测。CUBIN 需要真实 `sm_XX` target，LTO IR 需要相应编译模式；PTX hash 只证明工件内容稳定，不证明 kernel 输出正确。

## 分阶段实施

### 阶段 A：Capability 与 vendor audit

- 审计 CUDA 11.8、12.1、12.9、13.2 的 `nvrtc.h`、LIB、DLL/`.so`、builtins、symbols、版本号和 CMake target。
- 建立 machine-readable capability matrix，区分 compile log、PTX、CUBIN、LTO IR、deprecated NVVM、name expression。
- 确定 Windows delay-load/late binding 与 Linux `dlopen`/SONAME 策略，保证 NVRTC 缺失时的诊断不破坏非 RTC consumer。

### 阶段 B：Native owner 与 caller-buffer ABI

- 增加 `JYPPX_CudaRtcProgram` owner、严格参数/容量限制、幂等 destroy 和 no-throw guard。
- 实现 create、add name expression、compile、log size/copy、artifact size/copy、lowered-name size/copy、version/error diagnostics。
- 将 manifest、generated entry points、ABI declaration parity、PE export parity 和 Linux symbol parity 纳入现有生成/门禁体系。

### 阶段 C：Managed high-level API

- 增加 internal SafeHandle，但只向 public 暴露 `CudaRtcCompiler`、`CudaRtcProgram` 和 immutable DTO。
- 复制所有 log、lowered names 与 artifact bytes；验证 UTF-8、embedded NUL、重复 options、重复 name expressions、disposed owner 和超大输入。
- 增加 XML 双语文档、API surface snapshot、dependency diagnostics 和 deterministic artifact contract。

### 阶段 D：Compile-to-load-to-launch

- CUDA 12.9/13.2 首先验证 NVRTC PTX/CUBIN 能否直接进入现有 `CudaKernelLibrary.Load(byte[])`，并以 library owner + kernel name 完成启动。
- 现有 `CudaKernelLibrary` 尚无 owner-bound public launch；应增加 named-kernel launch owner，使 borrowed `cudaKernel_t` 永不越过 ABI。
- CUDA 11.8/12.1 不具备当前 runtime library API。若要维持完整支持矩阵，应设计统一的 CUDA Driver module/function owner，或明确限制；不得用裸 function pointer 补洞。
- Kernel arguments 使用 typed packing 和 bridge-owned launch storage，明确 device buffer、scalar、stream 与 module 的生命周期，并验证 grid/block/shared-memory limits。

### 阶段 E：Samples 与真实 smoke

新增 `samples/CudaRuntimeCompilation`，至少覆盖：

- vector add 或 elementwise kernel：compile、load、launch、synchronize、readback 和 expected-output comparison。
- 故意编译失败：保留完整 compiler log 和失败分类。
- C++ overloaded/template kernel：name expression 与 lowered name 闭环。
- PTX 与可用 CUBIN/LTO IR 工件的 metadata/hash 导出。

compile-only smoke、artifact hash、synthetic kernel 和 local Toolkit 都不能替代 clean package-consumer/public package proof。

### 阶段 F：Packaging 与跨平台证明

- Bridge-only NuGet 不捆绑 NVRTC；consumer 自行安装匹配 CUDA Toolkit，并获得明确 dependency diagnostics。
- GitHub full runtime 包按 runtime key 增加 `nvrtc` 与匹配的 `nvrtc-builtins`，同步 Windows/Linux manifests、split package roles、hash/size checks 和 redistribution review。
- 验证 Windows x64、Linux x64、CUDA 11.8/12.1/12.9/13.2；每个 runtime key 的声明必须与实际 native dependencies 和 package assets 一致。
- clean consumer 必须在无 ProjectReference、无开发 probing 的环境中完成 compile-to-launch；公开发布后还要重复 post-publish smoke。

## 验证矩阵与证据等级

每批至少包括 manifest 生成幂等、managed build、native build、ABI/export parity、专项测试、compile-success、compile-failure-log 和 dependency-missing smoke。进入 launch 阶段后再增加真实 GPU readback。

证据必须按以下层级记录：

| 证据 | 可以证明 | 不能证明 |
| --- | --- | --- |
| Header/LIB/DLL audit | vendor surface 存在 | API 可安全调用 |
| Compile-only | NVRTC program 与输出复制闭环 | module 可加载、kernel 正确 |
| PTX/CUBIN/LTO IR hash | 工件身份与确定性 | 数值输出正确 |
| Local compile-to-launch | 本机 Toolkit/GPU 路径可运行 | clean/public package 可运行 |
| Clean package consumer | 指定包组合可独立消费 | post-publish 来源与 hash 正确 |
| Post-publish smoke | 公开包路径可复现 | Owner 已批准最终发布/关闭 issue |

只有 compile、load、launch、readback、clean consumer、跨平台 package 和 post-publish evidence 都闭合后，CUDA RTC 才能进入 release-ready 声明。

## 与项目总收口的关系

CUDA RTC 是项目持续目标中的正式 release track，不取代 TensorRT deferred uplift、TensorRtExec/OnnxToEngine parity、YoloVision 全任务案例、双 NuGet/GitHub 包通道、技术文章和 Owner 真实 release proof。实现期间继续遵守“不以 unsafe pointer 换表面覆盖率”和“不把本地证据升级为公开发布证据”的项目边界。
