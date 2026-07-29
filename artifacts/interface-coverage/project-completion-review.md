# TensorRtSharp4.0 完成情况审查

## 2026-07-28 CUDA RTC Full-Runtime Packaging Preflight

本阶段没有直接物化或发布 `cuda-rtc` 包，而是把 runtime/split manifest 中的 planned role 升级为可执行、可复算的严格
preflight。它区分本地资产 identity、license text 存在、Owner redistribution approval、Linux proof 和 package project
物化，任一层缺失都不能进入打包。

### 实现

- 新增 `eng/Test-CudaRtcFullRuntimePackagingPreflight.ps1`，读取 public runtime manifest、split role、RTC capability
  matrix 与 ignored local root override；不下载、不复制 asset、不 pack、不 publish。
- Windows 按 CUDA 11.8/12.1/12.9/13.2 分组覆盖 6 个 runtime key，逐项核对 NVRTC 与 matching builtins 的
  relative path、expected/actual size、expected/actual SHA256；同时记录 EULA/LICENSE path 与 SHA256。
- Linux 按四版覆盖 12 个 runtime key，保持 capability matrix 的 `unverified-local-assets-not-found`、SONAME null 和
  assets unverified，不从 Windows DLL 外推 `.so`。
- runtime/split manifest 新增 preflight script/evidence、integrity source、license-text 非 approval、redistribution/platform
  proof 必需等合同字段；bridge-only 继续不引用 `cuda-rtc`。
- `Invoke-LocalSplitRuntimePackage.ps1` 对显式 `-SplitPackageRole cuda-rtc` 先执行
  `-RequireMaterializationReady` 并失败；`all` 明确警告 planned role 尚未包含，避免静默声称完整 full runtime。
- preflight 同时覆盖无 local override 的 clean-clone 路径：结构 finding 保持 0，Windows 资产转为 blocked report，
  不在 PowerShell 5.1 空数组参数绑定阶段崩溃。

### Evidence And Verification

- 结构化证据：`artifacts/cuda-runtime-compilation/full-runtime-packaging-preflight.json` / `.md`，分类为
  `local-full-runtime-packaging-preflight`。
- runtime keys：18（Windows 6、Linux 12）；Windows Toolkit version 4，asset pair integrity ready `4/4`，license text
  present `4/4`，四个唯一 pair 合计 `294150880 bytes / 280.52 MiB`；Linux asset ready `0/4`；structural findings 0。
- 四项 blocker 固定为 `redistribution-approval-pending`、`package-host-size-review-pending`、
  `linux-assets-unverified`、`cuda-rtc-role-not-materialized`；`canMaterializeFullRuntimeCudaRtcRole=false`、
  `canPublish=false`。
- `-RequireMaterializationReady` 与显式 split pack guard 均真实返回非零；无 local root 模式正常生成 blocked report；
  CUDA RTC 专项扩展后为 `16/16`。

### Proof Boundary

- Windows asset integrity 与 EULA 文件存在不等于 Owner 已批准 NVIDIA runtime 再分发，也不等于 nupkg 已生成。
- Linux 仍无真实 NVRTC/builtins `.so`、SONAME/symbol/package proof；当前不能物化跨平台 `cuda-rtc` role。
- 本批未复制 NVIDIA asset、未 pack `cuda-rtc`、未 push、未触发 Actions，未执行任何远程发布或 issue close。

## 2026-07-28 CUDA RTC Bridge Package Clean Consumer

本阶段把 bridge-only CUDA RTC 从仓库内样例推进到仓库外、本地 feed、纯 `PackageReference` consumer，验证当前
managed/bridge 本地包可以独立 restore/build，并在不使用开发目录 bridge override 的情况下完成依赖负向诊断与真实 GPU
双路径 correctness。证据分类保持 `local-feed-clean-package-consumer-candidate`。

### 实现

- 新增 `eng/Test-CudaRtcBridgePackageConsumer.ps1`，使用随机短 workspace token、独立 NuGet cache、独立
  `DOTNET_CLI_HOME` 与 `<clear />` local-only feed；consumer 无 `ProjectReference`，仓库外路径由边界检查保护。
- 脚本校验 managed 包包含当前 RTC surface，bridge 包唯一 native asset 为 `runtimes/win-x64/native/jyppxtrtbridge.dll`，
  且 managed/bridge 包中均不存在 NVRTC 或 NVRTC builtins。
- restore/build 后校验输出 bridge 与 nupkg entry SHA256 完全一致；负向运行清除 CUDA 搜索路径并指向不存在的 NVRTC，
  确认 RTC unavailable、诊断非空，同时 CUDA Driver 12090 capability 保持可用。
- 正向运行只指定用户 CUDA 12.9 Toolkit 的 NVRTC，完成成功 compile、intentional failure log、Runtime-library 与
  Driver module 两条 named typed-kernel launch/readback/correctness，并验证参与 owner 提前释放及输出 hash 一致。
- `Invoke-LocalSplitRuntimePackage.ps1` 新增 `-RunBridgeCudaRtcSmoke` 与独立 output-root 参数，可在 bridge-only pack 后
  复用该验证，并分别传递 managed/bridge version pin。
- 默认清理支持 PowerShell 5.1 超长 NuGet cache 路径：先执行常规删除，失败后仅在已验证 workspace 边界内使用
  `\\?\` 扩展路径回退；默认不保留 consumer 的清理模式已真实复验。

### Evidence And Verification

- managed nupkg SHA256：`19b43a17c29b2891356de75f00ffd210355f58f74673c2d9e5435eafa9d22d9c`；bridge nupkg
  SHA256：`79e997cd7b119c50a8faed8507b93fec5789d1e5087ffa1c8eeca19c645243a2`。
- package bridge entry 与 consumer 输出 DLL SHA256 均为
  `18493d0886b8434613c16d1dcfe3f5d134b95d60be5d463a674af65765311ba8`；两条 GPU 输出 SHA256 均为
  `33ecc0d0bc61c99a77fd5d012819fe5e366c826593739fb24b5fc86cff90b64f`。
- 结构化证据：`artifacts/cuda-runtime-compilation/bridge-package-consumer.json` / `.md`；包含包、consumer、依赖诊断、
  NVRTC/Driver 版本与加载库、runtime flags 和四份日志 hash。
- CUDA RTC 路线图专项 `15/15`；binding generator `4001 records / 201 manifests`；完整 solution Debug build 与
  双语 XML 审计均为 `0 warning / 0 error`。

### Proof Boundary

- 当前只证明本机 Windows、local-only feed、当前本地 managed/bridge 包组合，不是 public-source clean consumer、
  post-publish、Linux、CUDA 13.2 launch 或 Owner authorization proof。
- bridge-only 包不捆 NVRTC/builtins；full-runtime `cuda-rtc` role 尚未物化，不得把本地 Toolkit 依赖描述为包内依赖。
- 未 push、未触发 GitHub Actions，未执行 NuGet/GitHub Packages/Release 发布或 issue close。

## 2026-07-28 CUDA Driver Module Owner

本阶段在既有 NVRTC compile 与 CUDA 12.9+ Runtime-library launch 路径之外，完成动态 CUDA Driver module owner，
为 CUDA 11.8/12.1 等没有 Runtime library API 的版本提供统一、pointer-free 的 named-kernel typed launch 路径。

### 实现

- `driver.cpp` 动态加载 `nvcuda.dll` / `libcuda.so.1`，支持 `JYPPX_CUDA_DRIVER_LIBRARY` 精确覆盖；核心 bridge
  不静态链接 Driver import library。
- native owner retain primary context，复制 module code，以 `cuModuleLoadDataEx` / `cuModuleUnload` 管理 module；
  borrowed `CUfunction` 只在 bridge 内按名称查询，绝不穿过 C ABI。
- typed launch 接受复制型 scalar 与 owner-bound device memory，使用 context push/pop、Driver event completion owner
  和失败清理 `cuStreamSynchronize`；dynamic shared memory 在 `size_t` 转 `unsigned int` 前做原生 ABI 上限校验。
- 9 个公开 C ABI 同时具有 C++ exception containment 与 Windows SEH guard；异常报告本身也做二次 catch，
  completed-query 分支只执行一次 context pop。
- managed 新增 `CudaDriver`、`CudaDriverCapability`、`CudaDriverModule`、`CudaDriverKernelLaunch` 及 internal SafeHandle/
  interop；public surface 不暴露 `IntPtr`、`SafeHandle`、`CUmodule` 或 `CUfunction`。
- CMake 新增 `jyppx_cuda_driver_compile_probe` OBJECT target，只编译 Driver translation unit，避免 TensorRT 兼容层
  的既有错误阻断 CUDA Header-line 兼容性结论。

### 验证

- Windows CUDA 11.8/12.1/12.9/13.2 的 `cuda.h`、`cuda.lib` 与当前 `nvcuda.dll` 所需 symbols 全部审计通过；
  `cuStreamSynchronize` 已纳入 capability matrix，Linux 保持 `unverified-local-assets-not-found`。
- `driver.cpp` 分别在四版已安装 CUDA Header 下独立编译通过；CUDA 12.9 + TensorRT 11 完整 bridge 链接通过。
- 当前系统 Driver 12090 下，11.8/12.1/12.9 NVRTC PTX 的 Driver load/launch/readback/correctness/owner-retention
  全部为 true，三版 output SHA256 均为 `65dc411b0750ae9b6543bccb381f21537d69c11802db9bb45a427c2db16aa5d1`。
  CUDA 13.2 PTX 以 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 保持 load-rejected。
- bindings 为 4001 API records / 201 manifests，幂等门禁通过；Driver ABI 9/9、Runtime launch ABI 4/4、
  RTC ABI 12/12 declarations 与 PE exports 全部通过；RTC/Driver 专项测试 14/14。
- CudaSharp 全 15 个目标框架 Release build 与完整 solution Debug build 均为 0 warning / 0 error；双语 XML 审计通过。

### Proof Boundary

- 四版独立 compile-probe 证明 `driver.cpp` 对本机 Header 的编译兼容性，不证明四套完整 TensorRT bridge 都可构建。
- 11.8/12.1/12.9 运行结论是这些版本的 NVRTC PTX 在当前系统 Driver 12090 上的本机证明，不等于独立旧版
  runtime-library bridge 或 package consumer 证明。
- CUDA 13.2 launch、Linux Driver/RTC、full-runtime `cuda-rtc`、clean/public package consumer、post-publish 与
  Owner authorization 仍未闭合；未 push、未触发 Actions、未发布或关闭 issue。

## 2026-07-28 CUDA RTC Owner-Bound Launch/Readback And Managed Source Module Layout

本阶段完成 CUDA Runtime Compilation compile owner 与 CUDA 12.9+ runtime-library owner-bound named-kernel
launch/readback，并把三个托管接口项目根目录中堆叠的公开 C# 文件按职责模块化。文件整理只改变物理路径，
不改变 namespace、类型名、public API、native ABI 或对象生命周期。

### CUDA RTC 实现

- 新增 optional dynamic NVRTC loader 与 `JYPPX_CudaRtcProgram` owner；核心 bridge 不静态链接 NVRTC，
  `JYPPX_NVRTC_LIBRARY` 可指定精确依赖，缺失依赖返回可诊断 capability/diagnostic。
- 12 个 ABI 覆盖 program/source/header/name-expression、compile、log、PTX/CUBIN/LTO IR、lowered name 与幂等 destroy；
  caller-buffer/count-copy 有 UTF-8、embedded NUL、重复值、数量、容量和溢出校验，异常与 Windows SEH 不跨 ABI。
- managed 新增 `CudaRtcCompiler`、`CudaRtcProgram`、immutable source/options/result/artifact；public surface 不暴露
  `IntPtr`、`UIntPtr`、`SafeHandle` 或 vendor program/kernel/function handle。
- `samples/CudaRuntimeCompilation` 覆盖成功编译、intentional failure log、virtual header、lowered name、PTX hash
  determinism、可用的 CUBIN/LTO IR、owner-bound typed vector-add、GPU readback 和 owner 提前释放；package
  manifest 保持 bridge-only 不捆 NVRTC。
- 新增 `CudaKernelLibrary.Launch(...)`、`CudaDim3`、`CudaKernelLaunchConfiguration`、`CudaKernelArgument` 与
  `CudaKernelLaunch`；public API 只接受复制型标量和 owner-bound `CudaMemory`，borrowed `cudaKernel_t` 留在 bridge 内。
- 新增 4 个 owner-bound launch ABI；native 以 completion event 记录 stream 完成，managed launch 通过
  `DangerousAddRef` 持有 library、stream 和所有参与的 device memory，直到 synchronize/dispose。

### 托管源码模块化

- `JYPPX.CudaSharp` 的 83 个根目录文件归入 10 个模块：Core、Devices、Diagnostics、Events、Graphs、IPC、
  Kernels、Memory、RuntimeCompilation、Streams。
- `JYPPX.TensorRtSharp` 的 189 个根目录文件归入 Builder、Runtime、Engine、Execution、Inference、Network、
  Layers、Parsing、Refit、Plugins、Profiles、Serialization、ControlFlow、Diagnostics、Core 与 Callbacks；Callbacks
  再分为 Core、Debugging、MemoryAllocation、Monitoring。
- `JYPPX.TensorRtSharp.Tools` 的 22 个根目录文件归入 Artifacts、Build、Core、Runtime、Trtexec。
- 机械同步 1077 处测试、脚本、文档和结构化工件源码路径；反向扫描 294 个旧路径映射为 0 残留。
- 新增 `ManagedSourceModuleLayoutTests`，要求三个项目根目录 `.cs` 为 0，并验证所有约定模块包含源码文件。

### Verification

- Windows CUDA 11.8/12.1/12.9/13.2 NVRTC capability 与真实 compile smoke 均完成；11.8/12.1/12.9 PTX
  由当前 CUDA 12.9 runtime library owner 加载、启动并读回 257 个 float，三版 output SHA256 为
  `65dc411b0750ae9b6543bccb381f21537d69c11802db9bb45a427c2db16aa5d1`；13.2 因当前 driver/runtime 不支持其 PTX version
  只记 compile-only/load-rejected proof。
- RTC compile ABI：12/12 header declarations、12/12 PE exports；owner-bound launch ABI：4/4 declarations、4/4 PE exports，均 0 missing。
- RTC 专项测试扩展为 10/10，包含 zero dimension、null/too-many arguments、embedded NUL、disposed memory 和 scalar metadata。
- binding generator 连续两次生成 3992 API records / 200 manifests，hash 确定性和 comparison/coverage exporter 通过。
- 模块结构测试 3/3；完整 `TensorRtSharp.sln` Debug build 覆盖全部目标框架，0 warning / 0 error。
- `git diff --check` 通过；仅保留 Git 对既有 CRLF/LF 工作树规范的提示。
- build server 已关闭，仓库 `TestResults` 与相关 build/test 进程均为 0；Downloads/用户 Temp 自本批开始后
  未发现 TensorRtSharp/JYPPX/CUDA/TensorRT/NVRTC、ONNX、engine、plan、nupkg 或压缩重资产。

### Proof Boundary

- `local-smoke.json` schema 3 显式记录前三版 `kernelLaunch=true`、`gpuReadback=true`、`correctnessProof=true` 和
  `ownersDisposedBeforeSynchronize=true`；CUDA 13.2 明确记录 `loadSucceeded=false`、三个 runtime proof 字段为 false，诊断为
  `cudaErrorUnsupportedPtxVersion`。
- 本批证明是 Windows 本机 CUDA 12.9 bridge + local Toolkit/GPU 的 kernel runtime/readback；不等于 CUDA 11.8/12.1
  native runtime API、Linux NVRTC SONAME/symbol、full-runtime `cuda-rtc` 组件、clean public package consumer 或 post-publish。
- CUDA 11.8/12.1 的 unified Driver module/function owner、Linux runner、package consumer 和 post-publish 仍未闭合。
- TRT-off `win-x64-dev` 与 TRT10/CUDA12.9 native build 仍被既有非 RTC TensorRT 源码错误阻断，未记作 RTC 失败，
  也未宣称这些组合通过。
- 未 push、未触发 GitHub Actions/workflow dispatch、未执行 NuGet/GitHub 发布或 issue close。

## 2026-07-28 TensorRtExec Multi-Output Runtime Artifact Closure

本阶段在不提升危险 deferred API、不下载外部模型的前提下，完成 `TensorRtExec` / `OnnxToEngine` 的多输出
runtime artifact 闭环。此前 generic runtime 会读回全部 output，但 `--exportOutput` 与 raw dump 只保存首个 tensor，
`--dumpOutput` 也没有输出实际数值；现在三条路径统一消费同一份有序、pointer-free copied snapshot。

### 实现

- 新增 `OnnxEngineRuntimeOutputArtifact` 与 `OnnxEngineRuntimeArtifactData.OutputTensors`，复制每个 float output 的
  name、shape、element count、最多 8 个 preview value、byte length、raw bytes 和 SHA256，同时保留原有单输出属性兼容。
- `--dumpOutput` 为每个 output 写入 bounded preview、shape、长度和 SHA256；`--exportOutput` 新增有序
  `OutputTensors`，不再丢失第二个及后续 output。
- `--dumpRawBindingsToFile` 按 engine output 顺序连续写入全部 float32 bytes，并生成
  `<raw-path>.manifest.json`，记录 byte order、逐 tensor offset/length/shape/hash 和 combined hash。
- 新增 `RequestsOutputCapture`：`--dumpOutput`、`--dumpRawBindingsToFile` 或 `--exportOutput` 可在没有
  `--loadInputs` 时触发 generic bounded runtime，使用可复现的 `deterministic-generated` float input，并记录
  `InputSource`。
- 新增 `tensor-rt-exec-runtime-output-artifact-contract.json` 与 2 项 contract gate；同步 feature/parity/GUI-CLI
  matrix、gap list、应用/样例 README 和公开 parity 文章。gap summary 的陈旧 `17` 项计数同步修正为真实 `20` 项。

### Proof Boundary

- capture 与 validation 分离：未匹配 reference output 时即使 raw bytes 已写入，也必须保持
  `OutputValidated=false`、`HasTensorOutputProof=false`、`HasRawBindingProof=false` 和
  `runtime-output-captured-unverified`。
- bounded preview、raw manifest、synthetic identity output match 和 SHA256 都不是 real-model-runtime、
  package-consumer-runtime、post-publish 或 release proof；没有增加任何公共裸指针/handle surface。

### Verification

- TensorRtExec / OnnxToEngine / capability / parity / GUI-CLI / artifact 相关测试：70/70 通过；其中双输出测试验证
  JSON tensor 顺序、raw 拼接、offset、逐 tensor/combined SHA256、重复写入确定性和未验证 proof flags。
- 完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error；6 份修改的 JSON contract/matrix 均可解析；
  `git diff --check` 通过，仅有既有 README 换行规范提示。
- 本机 TRT10 identity runtime smoke 真实完成 build/serialize/deserialize/enqueue/readback：output 1 tensor、
  8 floats / 32 bytes，raw SHA256=`bcce3bba92f5737b0d780b06fcca7dff4873344d5c0e8978d44e8de6dcdcc0b7`，
  manifest offset=0 且 segment hash 与 raw hash 一致；分类保持 `synthetic-input-runtime`。
- 完整 ProjectQuality 运行 604 秒后超时，没有返回汇总，不能记作完整通过；超时前持续进入 strict release/owner
  validators。只终止本轮 13:47 启动的 dotnet/vstest/testhost/pwsh 进程树，未触碰其他任务进程，最终无本轮宿主残留。

### C 盘与发布边界

- Downloads/Temp 未发现本批新增 ONNX、engine、plan、nupkg、zip、7z 或项目下载文件。
- 可确认属于本轮的 1 个空 `MSBuildTemp` 与 6 个空 YoloVision test 目录删除命令被本机策略在执行前阻止；
  没有绕过策略。13:33 的其他空 MSBuild 目录可能与并行任务重叠，未删除。
- 未 push、未触发 GitHub Actions/workflow dispatch、未发布 NuGet/GitHub Packages/GitHub Release、未关闭 issue。

### CUDA Runtime Compilation 后续主线

- 用户新增 CUDA Runtime Compilation（NVRTC）正式需求；本阶段新增中英文
  `cuda-runtime-compilation-roadmap.md`，并接入 README、docs index/toc 与 ProjectQuality contract gate。
- 路线图基于本机既有 CUDA 12.9/13.2 header、import library、NVRTC/builtins DLL 审计，定义
  `JYPPX_CudaRtcProgram` copied owner、caller-buffer/count-copy ABI、`CudaRtcCompiler` / `CudaRtcProgram`、
  PTX/CUBIN/LTO IR artifact、compile-to-load-to-launch、typed arguments、sample 和双包通道。
- 现有 `CudaKernelLibrary.Load(byte[])` owner 仅在 CUDA 12.9+ 可用且尚无 public owner-bound launch；
  CUDA 11.8/12.1 与 Linux NVRTC/module surface 必须下一阶段真实审计，不能暴露 raw function/kernel pointer 补洞。
- compile-only、artifact hash、local Toolkit smoke、clean consumer、post-publish 和 Owner approval 保持独立证据层；
  RTC 只有在 load/launch/readback、跨平台 package 与公开包复验均闭合后才能声明 release-ready。

## 2026-07-28 Deferred Readonly Candidate Evidence Audit

本阶段把 16 个 deferred readonly candidate 的手写 evidence 记录收口为可复算的 repository linkage audit。
新增 `eng/Export-DeferredReadonlyCandidateEvidenceAudit.ps1` 和
`DeferredReadonlyCandidateEvidenceAuditTests`，并新增受版本控制的
`eng/deferred-readonly-candidate-evidence-map.json`，为已实现/安全替代候选补齐显式 `manifestSources` 与跨版本
native 聚合源。exporter 会把该 map 与被 `artifacts/` 忽略的本地 candidate list 合并，避免干净工作区丢失链接证据。

### Audit 结果

- candidate：16；implemented/design-gate status：8。
- evidence path：242 checked / 0 missing；manifest record：34 / 0 findings。
- managed public surface：94 / 0 missing；forbidden public handle：0；总 findings：0。
- plugin registry 的 TRT10/TRT11 宏生成 entry point 通过 `prefix-macro` linkage 识别；manifest `versionLine` 与路径一致。
- 首轮发现并修复 6 个问题：跨版本 native 聚合源漏列、两个 managed public surface 漏列、三个 manifest-source link miss、
  一条 ownership boundary 文案不完整。

### Proof Boundary

该 audit 只证明仓库内 candidate evidence 的路径、manifest、native symbol/linkage、托管入口和 ownership marker 彼此闭合，
不证明 vendor runtime、clean package consumer、post-publish、Linux、real model 或 owner authorization。所有
`isRuntimeExecutionProof`、`isPackageConsumerRuntimeProof`、`canPromoteRuntimeProof`、`canPromoteReleaseProof`、
`canPublishPublicly`、`canCloseReleaseIssue`、`performsPublish` 均为 `false`；没有删除 deferred history。

### Verification

- `DeferredReadonlyCandidateEvidenceAuditTests`：2/2。
- 相关只读候选/summary 专项：11/11；完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error。
- JSON/Markdown 连续导出 SHA256 稳定；audit findings：0；`git diff --check` 通过。
- stale release claims 本轮扫描超过 120 秒超时，未计作通过；上一次已记录工件仍为 1130 files / 0 findings。
- C 盘 Downloads 没有本批新增模型、engine、SDK、包或压缩重资产；Temp 没有本批重资产，只有完整 build 创建的
  15 个空 `MSBuildTemp*` 目录。精确删除命令被本机策略阻止，未绕过策略，也未删除其他进程的 fixture/log。
- 本阶段不 push、不触发 GitHub Actions、不发布 NuGet/GitHub Packages/GitHub Release、不关闭 issue。

## 2026-07-28 Release Blocker And Owner Proof Backlog

本阶段不伪造外部 runtime 或发布证据，集中把 closure ledger 的 42 条未完成 proof 投影成可执行、可复算、
可交接的单一 backlog。新增 `eng/Export-TechnicalArticleProofBacklog.ps1` 和
`TechnicalArticleProofBacklogTests`，输出 `docs/articles/zh-cn/publishing/technical-article-proof-backlog.json`
与 Markdown 台账，并将入口接入 README、docs index、toc 和 [Release Owner Handoff](../../docs/articles/zh-cn/release-owner-handoff.md)。

### Backlog Projection

- 文章 proof rows：42；proof lane 关系：89；唯一 lane：6。
- lane 分布：callback-runtime 4、linux-runner 9、owner-authorization 7、package-consumer-runtime 21、
  post-publish-verification 20、real-model-runtime 28；多 lane 文章按关系计数，不重复虚构文章。
- 每条 lane/文章保留当前 blocker、真实输入、首条 handoff 命令、严格 validator、期望工件和不可替代项；
  模板、draft、runbook、local feed、ProjectReference、direct nupkg、build-only、dependency probe、
  `Skipped=True`、blocked-by-cuda-driver 和 synthetic runtime 均不会提升 proof。
- 当前 `blockedArticleCount=42`、`readyArticleCount=0`；`canPromoteRuntimeProof=false`、
  `canPublishPublicly=false`、`canCloseReleaseIssue=false`、`performsPublish=false`。

### Verification

- backlog 专项：2/2；与 closure ledger、first/second batch audit 合并的文章专项：12/12。
- 完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error。
- backlog JSON/Markdown 连续导出 SHA256 稳定；stale release claims：1130 files scanned / 0 findings。
- `TestResults` 无残留；未触发发布、未 push、未使用 GitHub Actions，也未向 C 盘下载重资产。

## 2026-07-28 Technical Article Foundations Final Batch

本阶段一次性收口 technical article roadmap 剩余 14 条（28-32、38-45、103）。其中 28/44 和 30/45
分别共享 canonical article，因此实际扩写 12 篇唯一正文；共享关系在 Second-Batch audit 中显式核对，
没有复制正文或伪造新的 canonical mapping。正文完成只代表 source-quality closure，不替代真实 runtime、
package consumer、callback、Linux、real-model、post-publish 或 owner authorization 证明。

### 12 篇唯一正文与路线图收口

- 扩写 `blog-refit-weights-guide.md`、`trt11-modern-layers-guide.md`、
  `blog-network-layer-coverage-guide.md`、`error-recorder-diagnostics-design-gate.md`、
  `managed-logger-profiler-progress-monitor.md`、`blog-dynamic-shape-optimization-profile.md`、
  `blog-inference-bindings-identity-network.md`、`blog-onnx-parser-engine-roundtrip.md`、
  `blog-multistream-cuda-stream-event.md`、`blog-plugin-inventory-readonly-api.md`、
  `blog-cuda-memory-wrapper.md` 和 `cuda-stream-capture-to-graph-owner-safety.md`。
- 每篇补齐问题背景、Mermaid 架构图、真实仓库路径、关键代码解释、E 盘命令、输出解读、排障、
  ownership/version guard、proof boundary 和后续阅读；TRT11 Dims64/可选能力、managed callback
  invocation 边界、plugin metadata-only 复制和 CUDA stream capture owner safety 均保持原有冻结约束。
- 新增 `Export-TechnicalArticleFoundationsSecondBatchAudit.ps1`、JSON/Markdown audit 和
  `TechnicalArticleFoundationsSecondBatchTests`；审计按 14 个 roadmap entry / 12 个 unique canonical 分开计数。

### Ledger、质量门与导航

- closure ledger 重新导出为 103/103 `contentComplete`、0 `needsExpansion`；外部 proof dependency
  仍为 42 条 owner/runtime proof required，内容完成没有反向晋级 proof 状态。
- Second-Batch audit：14/14 roadmap entries、12/12 unique canonical、missing reference/link/forbidden
  全为 0；28/44、30/45 共享映射均被显式记录。closure ledger、First-Batch audit、Second-Batch audit
  连续两次运行 SHA256 均保持一致。
- 本批专项测试 7/7 通过；完整 `TensorRtSharp.sln` Debug build 0 warning / 0 error；stale release
  claims audit 扫描 1127 个文件、0 findings；`git diff --check` 通过。
- 全量 `dotnet test` 的一次复核被既有 final release evidence exporter 测试在 180 秒内拖住，未将该次
  超时计作通过；本批新增/修改文章专项测试独立通过，超时测试树已精确终止，未终止其他会话进程。

### C 盘与发布边界

- `C:\Users\guoji\Downloads` 今日无新增项目文件；用户 Temp 今日命中的是 Excel 跟踪文件和系统/第三方
  下载日志，不是本批生成物。没有发现 TensorRT、CUDA、cuDNN、ONNX、model、engine、plan、SDK、nupkg、
  zip 或 7z 等本批重资产，因此没有删除不明归属的 C 盘文件。
- 本批 exporter、build 和测试的正式输出均位于 E 盘仓库；未向 C 盘下载模型、engine、SDK、包或其他重资产。
- 未执行 push、GitHub Actions、workflow dispatch、NuGet/GitHub Packages/GitHub Release 发布或 issue close。
  `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false` 保持不变。

## 2026-07-28 Technical Article Foundations First Batch

本阶段一次性完成技术文章路线图第一组 9 条（2-6、10、15-16、19）的正文扩写，围绕 stable C ABI、
跨版本 manifest/guard、Windows 开发与 runtime package、TensorRT 对象模型、plugin serialization、
CUDA memory wrapper/range API 建立可独立发布的完整教程。正文完成只代表 source-quality closure，
不替代真实 runtime、package consumer、post-publish 或 owner authorization 证明。

### 九篇正文收口

- 9 篇文章由 22,031 字符扩为 94,705 字符，净增 72,674；当前包含 239 个标题、104 个代码块和
  17 个 Mermaid 图，2 篇达到 `complete-long-form`，7 篇达到 `complete-article`。
- native bridge 文章串联 C ABI、generated interop、safe handle、caller-buffer 与错误状态；接口清零文章
  区分 coverage、ABI presence、implemented-with-deferred-history 和真实 runtime proof。
- 跨版本与 Windows/runtime package 文章完整列明 TRT8/TRT10/TRT11、18 个 runtime key、6 个 Windows
  runtime key、version guard、local manifest、E 盘工作区和 consumer validator。
- 对象模型文章覆盖 logger、builder、runtime、engine、execution context 与 inference bindings 的所有权顺序；
  plugin serialization 文章严格区分 build-time serialization list、plugin load/register、callback 和 deferred 边界。
- CUDA 两篇文章覆盖 owner-safe memory/pinned memory/stream 生命周期、range copy/memset、offset/length 校验、
  同步/异步语义、smoke 输出和错误分层；所有 proof promotion flag 均保持 false。

### Audit、Ledger 与导航

- 新增 `eng/Export-TechnicalArticleFoundationsFirstBatchAudit.ps1` 及稳定 JSON/Markdown 输出，核对 9/9
  canonical article、必需 marker/anchor、92 个仓库引用、36 个 Markdown 链接和禁用声明。
- 新增 `TechnicalArticleFoundationsFirstBatchTests`，动态验证 exporter 确定性、正文结构、链接/引用、
  roadmap/closure ledger 投影和发布冻结字段。
- closure ledger 从 80 complete / 23 needs expansion 推进到 89 / 14；完整长文从 16 增至 18，完整文章
  从 44 增至 51，10 条操作指南和 10 条 canonical coverage 保持不变。
- 路线图、README 中英文版、docs index/toc 已同步；剩余 14 条固定为 28-32、38-45、103，下一批一次处理。

### Verification

- 首批专项：4/4 通过；文章/ledger/roadmap/Publishing 集合：86/86 通过。
- Plugin/CUDA/bridge/preflight 相关集合：33/33 通过；首批专项加精确 release docs compatibility：5/5 通过。
- 两个 exporter 连续执行两轮，四个 JSON/Markdown 输出 SHA256 均保持一致；missing marker、anchor、
  repository reference、Markdown link 和 forbidden finding 均为 0。
- 完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error；stale release claims：扫描 1124 个文件、0 findings。
- `git diff --check` 通过；提交前关闭 build server，并复核仓库相关 `vstest`/`testhost`/`pwsh` 无残留。

### C 盘与发布边界

- Downloads/用户 Temp 今日无 TensorRT、TensorRtSharp、JYPPX、ONNX 或 engine 项目目录，仓库内
  `TestResults` 为 0；唯一 100 MB 大文件是 3 月创建的 Excel `XLActiveUserTrace.etl`，明确保留。
- 本次 build 产生的随机 Temp 目录 `1xpjukus.uuc` 仅含一个 0 字节 .NET 10.0.300 workload 临时 nupkg；
  build-server shutdown 后已核对 Temp 根、绝对路径、reparse 属性和唯一文件并精确删除。
- 未执行 push、GitHub Actions、workflow dispatch、NuGet/GitHub Packages/GitHub Release 发布或 issue close。
- `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`；42 条 owner/runtime proof
  dependency 没有因文章完成而晋级。

## 2026-07-28 Technical Article Closure Ledger

本阶段以 103 条技术文章路线图为整体建立可机器复算的 closure ledger，并集中收口 63-71/79 的
重复主题、proof 教程与 callback/allocator 安全边界。ledger 将正文完成度与外部证明状态分开记录，
避免把 canonical 覆盖、长文完成或 source-quality 验证误写成 runtime/post-publish/release-close proof。

### Closure Ledger

- 新增 `eng/Export-TechnicalArticleClosureLedger.ps1`，从路线图、canonical article、声明引用和文章内容
  重新计算 JSON/Markdown；输出不写当前时间，连续执行可保持相同 SHA256。
- 主编号 103/103、补充编号 1（7.1），missing/duplicate article ID 均为 0；10 条重复主题通过显式
  canonical mapping 收口，所有 canonical article 均存在。
- 80 条正文已完成：16 条 `complete-long-form`、44 条 `complete-article`、10 条
  `complete-operational-guide`、10 条 `canonical-covered`；剩余 23 条为 `needs-expansion`。
- 56 条涉及外部资产或 proof dependency，其中 42 条仍需 owner/runtime proof；这些条目的
  `proofComplete=false` 不会反向覆盖其独立的 `contentComplete`。
- ledger 固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`，也不是
  runtime execution、post-publish 或 release-close proof。

### Canonical Long-Form Closure

- `package-consumer-runtime-proof-playbook.md` 扩为 500+ 行完整执行手册，覆盖 E 盘 clean consumer、
  runtime package key、preflight、scaffold、restore/build、native listing、真实 smoke、owner input 与 strict validator。
- `post-publish-verification-proof-playbook.md` 扩为 600+ 行完整执行手册，覆盖真实渠道重新下载、hash、
  clean scan、隔离 restore、runtime smoke、record projection、严格验证与 rollback 边界。
- `callback-allocator-safety-bridge-roadmap.md` 扩为 690+ 行安全路线，逐项审计 5 个 callback family 的
  readiness/safe-control/closure；当前 0 个 family closure-ready，14 类 owner 输入仍未到位。
- 路线图 25/26、34、63-68、71 通过 74、79、81、72、73-78、69、70、80 的 canonical article 显式收口；
  69、70、79 自身标记完整教程，未改变任何外部 proof 状态。

### Quality Gates And Navigation

- 新增 `TechnicalArticleClosureLedgerTests`，验证编号连续性、canonical mapping、文章指标、引用存在性、
  内容/proof 状态独立、发布冻结字段、目标长文长度和禁用 marker。
- README 中英文版、docs index/toc 与路线图增加 ledger 入口，用户可从前台文档直接查看剩余 23 条。
- 下一内容批次按 ledger 明确分组：先处理 2-6、10、15-16、19，再处理 28-32、38-45、103；
  每批重新导出 ledger，不以手工改计数代替正文扩写。

### Verification

- `TechnicalArticleClosureLedgerTests`：3/3 通过。
- ledger、路线图、公开文章、第三批文章、package preflight、callback readiness/closure 宽口径集合：95/95 通过。
- 完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error。
- 三篇目标长文的反引号仓库路径均可解析；目标禁用 marker 0 条；连续导出 JSON/Markdown SHA256 保持一致。
- stale release claims audit：1121 files scanned / 0 findings；`git diff --check` 通过，build server 在提交前关闭。

### C 盘与发布边界

- Downloads/Temp 今日模型、engine、plan、nupkg、压缩重资产与本项目大文件命中均为 0；一个早于本批的
  空 `jyppx-split-packages` Temp 目录已核验归属和内容后删除，仓库内 TestResults 为 0。
- 未执行 push、GitHub Actions、workflow dispatch、NuGet/GitHub Packages/GitHub Release 发布或 issue close。
- closure ledger 和三篇 source article 只是内容/边界证明；真实 package consumer、callback runtime、
  post-publish、Linux runner、real-model 与 owner authorization 状态未晋级。

## 2026-07-27 YoloVision Detection Hardening And Tutorial Closure

本阶段在 deferred 审计继续确认 immediate-safe 候选为 0 后，完成 YoloVision all-task 与 Detection
两个路线图条目的大批次收口：一方面加固 generic raw-head 数值校验，另一方面把两篇短草稿扩为
可执行长教程，并同步 Detection candidate/article/owner/generated packs。没有修改 native ABI、没有
进入 callback/calibrator/allocator/plugin/runtime ownership，也没有伪造外部模型或包消费证明。

### Detection Managed Hardening

- `YoloPostprocessOptions` 现在拒绝非有限 confidence/IoU threshold，避免 `NaN` 绕过 `[0,1]` 检查。
- generic `YoloDetectionDecoder` 对每行 box、objectness、所有 class score 与 computed score 执行
  `float.IsFinite`；负 width/height fail closed，并在异常中保留 row/class 定位。
- 为兼容既有 generic raw-head 行为，零 width/height 仍允许；YOLOv10 end-to-end 路径继续要求
  `x2>x1`、`y2>y1` 和正面积。
- 新增 threshold 与 malformed raw-head focused tests，覆盖 `NaN`、正负 Infinity 和负尺寸。

### All-Task 与 Detection 长教程

- `yolovision-all-task-overview.md` 从 56 行扩为 400+ 行，明确分离 60-row/55-supported
  `YoloCapabilityMatrix`、10-family 资产规划 matrix 与 6-task owner output contract，避免把托管
  配置能力写成所有 family/task 都有真实模型证明。
- 总览补齐 family/task 现实矩阵、E 盘资产隔离、许可证/hash、TensorRtExec build-only、YoloVision
  preflight/runtime、六任务命令、JSON/SVG、证据阶梯、validator、排障与发布清单。
- `yolovision-detection-tutorial.md` 从 89 行扩为 500+ 行，分别绑定 generic `[1,C,N]/[1,N,C]`
  raw head、YOLOv10 `[1,N,6]` end-to-end 与 YOLOX `[1,8400,85]` grid/stride 三条路径。
- Detection 教程记录 objectness 推断、score 公式、class-aware/class-agnostic/none NMS、Top-K、
  `SourceIndex`、numeric fail-closed 和 BMP/PPM 预处理，并明确 generic detection 不会自动把
  model-input boxes 逆 letterbox 到 source-image coordinates。
- 两篇命令统一使用当前 `--model`、`--layout`、`--confidence`、`--exportReport`、`.ppm + --image +
  --preprocessed-output`、`--output-json` 和 `--visualization-svg`；禁止旧参数重新出现。
- README 增加两篇入口，technical article roadmap 第 73/74 项更新为“完整教程已收口”。

### Detection Packs 与 Owner Contract

- Detection article case、candidate 与 owner pack 统一使用 `.ppm` 内置预处理并生成 JSON/SVG。
- 新增 `coordinateSpace=model-input-pixels-or-owner-confirmed`、`letterboxContract=owner-required`、
  `sourceImageInversePolicy=not-automatic-owner-transform-required`，不从 image metadata 自动推断 box inverse。
- owner exporter 与 generated pack 已同步；投影为 `projection-aligned`、0 failures。
- candidate/owner strict validators 均为 0 blockers，promotion flags 保持 false。

### Verification

- `YoloVisionManagedPipelineTests`：47/47 通过。
- 新增长教程与 pack contract 专项门禁通过。
- YoloVision、TechnicalArticleRoadmapTests、PublishingPublicArticleTests 宽口径集合：204/204 通过。
- output report strict validator：6 records、0 blockers；owner/candidate validators：0 blockers。
- 顺带修复 5 条既有 ProjectQuality nullable warning，以显式 non-empty string 断言替代 nullable
  `Assert.NotEmpty` 调用；相关 release-boundary tests 语义不变。
- 完整 `TensorRtSharp.sln` Debug build：0 warning、0 error。
- stale release claims audit：扫描 1118 个文件，`findingCount=0`。
- `git diff --check` 与 build-server shutdown 在本地提交前执行。

### C 盘与发布边界

- 本批没有向 C 盘下载或生成模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN、nupkg、zip 或 7z。
- 测试留下的 6 个空 `jyppx-yolovision-*`/`jyppx-yolox-*` Temp 父目录已核验为空并删除。
- 两个早于本批、所有权不明的 Docker 0 字节 `save.tar` 保留。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet/GitHub Packages/Release 发布或 issue close。
- 教程、matrix、preflight、build-only、JSON、SVG、synthetic tests 与 local source build 都不是
  real-model-runtime、package-consumer-runtime 或 post-publish proof；owner authorization 状态未晋级。

## 2026-07-27 YoloVision Segmentation Spatial Transform And Pose/OBB Tutorial Closure

本阶段在上一批 probability-mask 基线之上完成显式 prototype-to-source-image 空间变换，并同步收口
Pose/OBB 长教程、output schema/validator、资产包投影与负向质量门。全部行为保持纯 managed、pointer-free；
没有修改 TensorRT native ABI，也没有把 exporter-specific alignment、真实模型或 package consumer 结果
写成已证明能力。

### Segmentation Spatial Transform

- 新增 `YoloSegmentationSpatialTransform`、显式 `model-input|normalized` coordinate space、bilinear
  prototype sampling、source-image resize-back 与可选 detection-box crop。
- CLI 新增 `--mask-spatial-transform`、`--mask-coordinate-space`、`--mask-crop-to-box`；请求 transform
  时必须使用 `--task seg` 和 `--image`，外部 tensor 不允许提供推断式 inverse metadata。
- crop 采用 `[left,right) x [top,bottom)` 半开栅格边界，避免 right/bottom 多覆盖一行或一列。
- effective scale 从取整后的 `ResizedWidth/SourceWidth`、`ResizedHeight/SourceHeight` 推导，和真实 resize
  pixel-center 栅格一致，不依赖可能因 letterbox round 产生轻微偏差的理想等比 scale。
- report/visualization 公共 overload 对缺少 image metadata 或非 segmentation result 的 spatial 请求
  fail closed；旧 overload 与默认 prototype-grid 行为保持兼容。
- output JSON 新增可选 `spatialTransform`：applied/coordinate/crop/interpolation、source/target/resized
  shape、pad/effective scale、final mask shape/count/threshold/scope、source box 和固定 owner boundary。
- source-image SVG 使用最终 probability mask 的有界 48x48 采样，保留 `data-spatial-mask-cell=true`；
  默认 prototype preview 仍保持 24x24 与原字段语义。

### Schema、Validator 与资产包

- `yolovision-output.schema.json` 增加完整 `segmentationSpatialTransform` definition。
- `Test-YoloVisionOutputReport.ps1` 增加 applied、coordinate、bilinear、shape product、active<=total、
  threshold、source scope/boundary 和 image/letterbox 来源链 blocker。
- segmentation example 升级为 1280x720 source image、640x640 letterbox、720x1280 final mask 的自洽
  spatial 正例；负向测试覆盖伪造 coordinate/interpolation/count/threshold/boundary 和不可信 input metadata。
- article case、owner backfill、generated projection 与 segmentation candidate 同步 `.ppm + --image +
  --preprocessed-output` 命令及 spatial metadata；exporter 能从命令解析 image/tensor/output 路径。
- owner projection 保持 `projection-aligned`，strict owner-pack validator 通过；所有 promotion flag 仍为 false。

### Pose 与 OBB 长教程

- `yolovision-pose-tutorial.md` 从 52 行扩为 218 行，绑定独立 keypoint tensor、`SourceIndex`、
  `[1,N,K*stride]`/`[1,K*stride,N]`、stride/score、坐标与 skeleton owner boundary、E 盘命令和证据链。
- `yolovision-obb-tutorial.md` 从 50 行扩为 238 行，绑定 angle tensor、degree/radian normalization、
  `SourceIndex`、angle range/axis/width-height owner contract，并明确当前是 axis-aligned NMS 后附加角度，
  没有实现 rotated-IoU NMS。
- 两篇命令统一使用真实 `--exportReport`，路线图第 76/77 项更新为“完整教程已收口”；新增长度、
  代码路径、命令、proof boundary 和禁止过度声明的专项门禁。

### Verification

- spatial/schema/validator/tutorial 定向集合：58/58 通过。
- YoloVision、TechnicalArticleRoadmapTests、PublishingPublicArticleTests 宽口径集合：201/201 通过。
- output examples strict validator：6 records、0 blockers；两类 segmentation 负向报告均被拒绝。
- owner backfill exporter：`projection-aligned`、0 failures；strict owner-pack validator 通过。
- 完整 `TensorRtSharp.sln` Debug build：0 warning、0 error。
- stale release claims audit：扫描 1118 个文件，`findingCount=0`。

### C 盘与发布边界

- 本批没有下载或生成模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN、nupkg、zip 或 7z 到 C 盘。
- 测试更新的 5 个空 `jyppx-yolovision-*` Temp 目录已确认归属并删除；没有本批 NuGet package cache 目录。
- 两个 0 字节 Docker Temp `save.tar` 早于本批、所有权不明，未删除。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet/GitHub Packages/Release 发布或 issue close。
- spatial report/SVG 是 owner review diagnostic；exporter-specific alignment、real-model-runtime、
  package-consumer-runtime、post-publish verification 和 owner authorization 状态均未晋级。

## 2026-07-27 Segmentation Probability Mask And TensorRtExec Option Tutorial Closure

本阶段按“每批做更多”要求同时推进可用能力与宣传文章：补齐 YoloVision segmentation probability
mask/threshold/report/preview 路径，并将 Segmentation 与 TensorRtExec 参数分层两篇概要扩展为完整教程。
该批只增加 managed pointer-free 行为和结构化证据，不改变 TensorRT native ABI 或 deferred ownership
边界。

### Segmentation 实现

- 保留现有 `ComposeLinearMask` API 和 `RawLogits` 语义，新增 `ComposeProbabilityMask` 与数值稳定
  sigmoid；runtime multi-output decode 改为 probability mask。
- `YoloMultiOutputMetadata` 以兼容构造函数/重载新增 `MaskThreshold`；CLI 增加
  `--mask-threshold`，并进入 preflight、runtime decode、report 和 SVG。
- `YoloSegmentationMask` 新增 value kind、threshold、probability readback 和 active pixel count；不暴露
  native pointer 或不透明 ownership。
- output report 将 `maskPixelCount` 明确为 active prototype-grid pixels，并新增
  `maskTotalPixelCount`、`maskValueKind`、
  `maskPixelCountScope=prototype-grid-before-crop-resize`。
- SVG 从整框填色提升为真实 probability 值驱动、最多 24x24 的有界网格预览；仍明确不是
  model-specific crop/resize-back final overlay。
- strict output validator 新增 shape product、active<=total、threshold range、value kind/scope 检查，并增加
  篡改 total pixel count 必须被拒绝的 E 盘负向测试。
- article case pack、owner backfill pack、candidate template 和 exporter 同步显式
  `--mask-threshold 0.5` 与新 metadata；生成投影保持 aligned。

### 两篇长文

- `yolovision-segmentation-tutorial.md` 扩展为模型/许可证、E 盘资产、role/coefficient/prototype、
  SourceIndex、sigmoid/threshold、build/preflight/runtime、JSON/SVG、validator、排障和发布检查完整教程。
- `tensorrtexec-option-layering-deep-dive.md` 绑定 31 项 capability JSON、85 项 GUI/CLI field map、
  17 项 gap list，以及 dry-run/build-only/readonly/bounded-runtime 四层命令和 report validator。
- 修正旧 segmentation 文章中 4 个不存在的代码文件名和过度实现描述；crop/resize-back 回到明确的
  owner adapter 边界。
- README 增加两篇入口，technical article roadmap 第 72、75 项更新为“完整教程已收口”。
- 新增两组长文交叉门禁，验证代码、schema、example、validator、pack、README、roadmap 和
  machine-readable counts。

### Verification

- 本批定向集合：124/124 通过。
- segmentation output examples：6 records、0 blockers；篡改 pixel total 的负向样例被 strict 拒绝。
- owner backfill exporter：`projection-aligned`、0 failures；strict pack validator 通过。
- `TensorRtExec --help-json`：31 entries、26 implemented/bounded、4 parse/diagnostic、1 blocked。
- 完整 `TensorRtSharp.sln` Debug build：0 warning、0 error。
- stale release claims audit：扫描 1117 个文件，`findingCount=0`。

### C 盘与发布边界

- 未下载或生成模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN、nupkg、zip 或 7z 到 C 盘。
- C 盘 Temp 有两个不属于本批的 0 字节 Docker `save.tar`；本批没有调用 Docker，未删除无法确认
  所有权的文件。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet/GitHub Packages/Release 发布或 issue close。
- `blocked-real-proof-required`、`canPublishPublicly=false`、`canCloseReleaseIssue=false` 不变；
  通用 prototype-grid preview 不是 owner final overlay 或 real-model/package-consumer proof。

## 2026-07-27 YoloVision Classification/Semantic Combined Tutorial Closure

本阶段继续案例和文章收尾，把原先只有概要的
`yolovision-classification-semantic-tutorial.md` 扩展为 classification 与 semantic
segmentation 的对照式完整教程。该批绑定现有 pointer-free managed decode、CLI、output report、
SVG、case pack 和 validator，不新增 native ownership 风险，也不把示例或 build-only 写成真实运行证明。

### 实现

- 教程新增 cls/sem 判断矩阵、Mermaid 全链路、E 盘资产工作区、许可证/hash 清单、预处理边界、
  两套 TensorRtExec build-only/preflight/runtime 命令和输出报告检查。
- 明确 classification 支持 `[C]`、`[1,C]`、`[C,1]`，执行 threshold/order/Top-K，当前 decoder
  不自动 softmax；`--classification-output` 绑定 tensor name/role。
- 明确 semantic 支持 `[C,H,W]`、`[1,C,H,W]`、受 class count 约束的 `[1,H,W,C]`，归一化为
  class-major float map；SVG 的逐像素 argmax/32x24 预览不是完整 map proof。
- 绑定两个 example JSON、task output contract、article/owner case pack、output validator、owner proof
  validator、sample-run evidence 和发布前检查清单。
- `samples/YoloVision/README.md` 增加组合教程入口；technical article roadmap 第 78 项更新为
  “完整教程已收口”。
- 新增 `ClassificationSemanticTutorialBindsCodeCommandsReportsAndProofBoundaries` 专项质量门。

### Verification

- `YoloVisionDocumentationMatrixTests`：5/5 通过。
- YoloVision 文档/技术文章组合：56/56 通过。
- `Test-YoloVisionOutputReport.ps1 -Strict`：6 records，0 blockers。
- 完整 `TensorRtSharp.sln` Debug build：0 warning、0 error。
- stale release claims audit：`findingCount=0`；`git diff --check` 通过。

### C 盘与发布边界

- 本批未下载或生成模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN、nupkg、zip 或 7z 到 C 盘。
- C 盘审计发现一个不属于本任务、0 字节的 Docker Temp `save.tar`；本批没有运行 Docker，未删除
  无法确认所有权的临时文件。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet/GitHub Packages/Release 发布或 issue close。
- 当前仍为 `blocked-real-proof-required`、`canPublishPublicly=false`、
  `canCloseReleaseIssue=false`；expected real-log success marker 仍需 owner 真实资产回填。

## 2026-07-27 YOLO Family Multi-Task Tutorial Closure

本阶段在 TensorRT/CUDA deferred 审计确认没有 immediate-safe 候选后，转向用户明确要求的 YOLO
宣传与案例收口。扩展 `yolo-family-profile-and-postprocess-guide.md`，把全系列配置底座变成
可执行、可审计的多任务接入教程，不把规划矩阵写成 runtime proof。

### 实现

- 新增模型来源/许可证、E 盘 case workspace、ONNX/labels/input SHA256 和 derived artifact
  目录约定，避免教程指导用户把大资产散落到 C 盘。
- 新增 Mermaid 全链路：来源 -> hash -> TensorRtExec build-only -> YoloVision task/profile
  -> decode/NMS -> JSON/SVG/log -> sample-run evidence -> owner review。
- 新增 det/cls/seg/obb/pose/sem 六任务接入矩阵、profile/output metadata 合同和每任务命令骨架。
- 新增 binding metadata 严格验证、任务专属检查、证据归档顺序、排障表和发布前 checklist。
- 新增 `YoloFamilyProfileGuideIsPublishableLongFormAndBindsEachTaskToEvidence` 专项测试。

### Verification

- `YoloVisionDocumentationMatrixTests`：4/4 通过。
- 教程继续明确 synthetic/build-only/sidecar/local feed/ProjectReference/direct nupkg
  不是 real-model-runtime 或 package-consumer-runtime proof。
- 未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN 或 NuGet 到 C 盘；未执行 Actions、push
  或发布。

## 2026-07-27 Source Build/C++ Bridge Guide And Stale Claims Closure

本阶段完成源码构建与 C++ bridge 公开教程的长文收尾，并清理 release-facing 文档中的 stale claim
命中。该批仍是 source-quality/documentation closure，不是 runtime、package-consumer 或发布证明。

### 实现

- 扩展 `docs/articles/zh-cn/tensorrtsharp-source-build-cpp-guide.md`：构建全景图、E 盘
  workspace/assets 布局、环境快照、runtime key 决策表、生成物、CMake/build 日志、
  `dumpbin /dependents`、聚焦测试、本地 package validation、NuGet 小核心/bridge 与 GitHub
  full-runtime 路线、故障树和配图建议。
- 新增 `CppBridgeMasterGuideIsPublishableLongFormAndKeepsProofBoundaries` 专项门禁。
- 修正 5 篇 release-facing 文章中的 marker/retired sample 文案上下文，重新生成
  `artifacts/final-release/stale-release-claims-audit.json/.md`，`findingCount=0`。
- 刷新 `release-candidate-final-evidence-freeze.json/.md`；状态仍为
  `blocked-real-proof-required`，`performsPublish=false`、`canPublishPublicly=false`、
  `canCloseReleaseIssue=false`。

### Verification

- SourceBuildCmakeWindowsGuideTests：4/4 通过。
- TechnicalArticleRoadmapTests、PublishingPublicArticleTests、SourceBuildCmakeWindowsGuideTests：
  81/81 通过。
- stale release claims audit：0 findings。
- 未执行 GitHub Actions、push、NuGet/GitHub Packages/Release 发布或 issue close。
- 未下载模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN 或 NuGet 重资产到 C 盘。

## 2026-07-27 Tool Capability JSON And YoloVision Offline Contract Self-Test

本阶段进入工具与案例收尾：不触发 GitHub Actions，不发布包，不下载模型或依赖到 C 盘，优先补可离线验证、低 ownership 风险的机器可读能力说明和 self-test。

### 实现

- 新增 `src/JYPPX.TensorRtSharp.Tools/Trtexec/TrtexecLikeOptionCapabilities.cs`，输出 `trtexec-like-option-capabilities.v1` JSON，覆盖 31 个 trtexec-like option capability rows。
- `applications/TensorRtExec` 和 `samples/OnnxToEngine` 增加 `--help-json` / `--capabilities-json`，离线输出 option group、alias、implementation class、parse/report-only 或 blocked 状态、`releaseFrozen=true` 和 `canPromoteRuntimeProof=false`。
- `samples/YoloVision` 增加 `--self-test-capabilities`，离线验证 60 个 family/task rows、55 个 supported rows、5 个 YOLOX unsupported rows 和 `IsRuntimeProof=False` 边界。
- 更新 `applications/TensorRtExec/README.md`、`samples/OnnxToEngine/README.md`、`samples/YoloVision/README.md` 与外层 plan/diary/prompt。
- 新增 `ToolCapabilityJsonSurfaceTests`，守住 JSON schema、release freeze、CLI switch 暴露和 YoloVision capability self-test 输出。

### Verification

- `ToolCapabilityJsonSurfaceTests`：3/3 通过。
- 相关集合 `ToolCapabilityJsonSurfaceTests|OnnxToEngineTrtexecLikeTests|TrtexecBuildPolicyTests|YoloVisionManagedPipelineTests`：93/93 通过。
- `TensorRtExec`、`OnnxToEngine`、`YoloVision` Debug build：0 warning、0 error。
- `OnnxToEngine --help-json`、`TensorRtExec --help-json`、`YoloVision --self-test-capabilities` 均可离线执行。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批是 source-quality capability surface 和 offline matrix contract self-test，不是 runtime proof、real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-23 Native Bridge Build Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/native-bridge-build-public-article.md`，将 native bridge
构建文章从短说明扩展为面向外部用户的源码构建与 ABI 审计文章。文章覆盖环境前提、
manifest/generated interop、CMake preset、ABI 设计规则、质量门、两条 package 路线和
proof boundary。

### 实现

- 文章补齐 `native/manifests/tensorrt/v8`、`v10`、`v11`、`native/manifests/cuda`、
  `native/generated/bridge_api_catalog.g.h`、`bridge_entrypoints.g.h` 与 generated C#
  interop 的证据路径。
- 公开说明 `Generate-Bindings.ps1`、`Test-BindingGeneratorOutputs.ps1`、
  `Export-InterfaceCoverageMatrix.ps1`、`Export-NativeMethodsComparison.ps1`、
  `Export-WrapperLiftCandidates.ps1` 和 `Export-GeneratedApiCoverage.ps1` 的验证用途。
- 写入 Windows/Linux TRT8/TRT10/TRT11 CMake preset 与
  `JYPPX_ENABLE_TENSORRT_BINDINGS`、`JYPPX_ENABLE_CUDA_BINDINGS`、
  `JYPPX_TENSORRT_LINE`、`JYPPX_CUDA_LINE`、`JYPPX_CUDA_VERSION`、
  `JYPPX_TENSORRT_CUDA_VERSION`、`JYPPX_CUDNN_MAJOR` 的版本线边界。
- 明确 ABI 规则：C ABI entrypoint 稳定、跨 ABI 不抛 C++ exception、native failure
  转换为 `JYPPX_StatusCode`、字符串/数组走 count/copy 或 caller buffer、
  public wrapper 不泄露裸 `IntPtr`。
- 连接 `TensorRtNativeAbiSurfaceParityTests`、`PublicApiHandleExposureAuditTests`、
  `NativeBridgePathResolverTests`、`NativeVendorBoundaryGuardTests` 与源码构建/roadmap
  质量门。
- 明确 native bridge build、generated interop、readonly diagnostics、local feed、
  ProjectReference、direct `.nupkg` install 与 GitHub Actions dry-run 都不是
  package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `NativeBridgeBuildPublicArticleCoversAbiGenerationPresetsPackagesAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `12/12` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Parser Refitter Copied Diagnostics Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/onnx-parser-parserrefitter-诊断-copied-diagnostics-release-gate.md`，
将 ONNX Parser / ParserRefitter copied diagnostics 从短说明扩展为 release gate
边界文章。文章说明 parser/refitter snapshot、summary、TensorRtExec report projection、
package consumer surface gate 与 proof boundary 的关系。

### 实现

- 文章补齐 `TensorRtOnnxParser.cs`、`TensorRtOnnxParserDiagnosticSnapshot.cs`、
  `TensorRtOnnxParserDiagnostic.cs`、`TensorRtOnnxParser.ModelSupport.cs`、
  `TensorRtOnnxParserRefitter.cs`、`TensorRtOnnxParserRefitterDiagnosticSnapshot.cs`、
  `NativeBridgeApi.ParserRefitterDiagnostics.cs`、`OnnxEngineBuildDiagnostics.cs`、
  `OnnxEngineParserPreflightSnapshot.cs` 和 `OnnxEngineBuildService.cs` 的证据路径。
- 明确 parser/refitter snapshot 复制 `Line`、`ErrorCount`、`Diagnostics`、
  `DiagnosticSummary`、`UsedVCPluginLibraries`、`IdentityOperatorSupported`、
  `CopiedDiagnosticCount`、`DiagnosticSummaryLength`、`RuntimeEvidenceKind` 与
  pointer-free proof boundary。
- 写入 TensorRtExec report 中的 `ParserPreflightSnapshot`、`DiagnosticsState`、
  `ModelSupportState`、`CopiedSubgraphCount`、`ParserDiagnosticsEvidenceKind`、
  `ParserRefitterDiagnosticsEvidenceKind`、`CopiedDiagnosticsBoundary`、
  `ForbiddenSubstitutes` 与 `CanPromoteCopiedDiagnosticsToRuntimeProof = False`。
- 明确 `onnx-parser-diagnostic-readiness` 和
  `onnx-parser-refitter-diagnostic-readiness` 都是 `compile-surface-proof` /
  `proof=false`，不是 runtime/package/post-publish/release-close proof。
- 为 `PublishingPublicArticleTests` 增加
  `OnnxParserParserRefitterCopiedDiagnosticsArticleCoversSnapshotsReportsAndReleaseBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `11/11` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 OnnxToEngine Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/onnx-to-engine-public-article.md`，将 ONNX 到
TensorRT engine 的转换文章从短流程扩展为完整可发布教程。文章围绕
`samples/OnnxToEngine`、TensorRtExec 共享 trtexec-like 参数、report/evidence sidecar、
MNIST real-model 边界和 YoloVision 关系，明确转换链路的证据价值和 proof 边界。

### 实现

- 文章补齐 `samples/OnnxToEngine/Program.cs`、`TrtexecLikeParser.cs`、
  `TrtexecLikeOptions.cs`、`OnnxEngineBuildOptions.cs`、`OnnxEngineBuildService.cs`、
  `OnnxEngineBuildResult.cs`、`OnnxEngineBuildDiagnostics.cs`、
  `OnnxEngineBuildReportWriter.cs`、`OnnxEngineBuildEvidenceSidecar.cs`、
  `TensorRtExecCommand.cs`、`MainForm.cs` 与 TensorRtExec parity/gap matrix 的证据路径。
- 公开说明 `--onnx`、`--saveEngine`、`--loadEngine`、shape profile、precision、
  memory pool、timing cache、layer info、refit、weight streaming、runtime output、
  report 和 evidence sidecar 等 trtexec-like 参数面。
- 写入 `OnnxEngineBuildResult` / diagnostics 核心字段，包括 `ProofClassification`、
  `BuildEvidenceOnly`、`IsRuntimeExecutionProof`、`IsRealModelRuntimeProof`、
  `IsPackageConsumerRuntimeProof`、`NormalizedCommandSha256`、`PreflightMetadata`、
  `LoadedEngineDiagnostics`、`BuilderConfigDeploymentSnapshot`、`ParserPreflightSnapshot`
  与 `.engine-readback.json`。
- 明确 `MnistOnnxRuntimeService` 可作为 real-model-runtime 方向的样例，但仍不是
  package-consumer-runtime proof。
- 连接 `samples/YoloVision`、`yolovision-article-case-pack.json` 和
  `yolovision-family-task-real-asset-roadmap.json`，要求公开材料使用 YOLO 系列和
  det/cls/seg/OBB/pose/semantic segmentation 的广义口径，不退回 `samples/YoloDet`。
- 为 `PublishingPublicArticleTests` 增加
  `OnnxToEnginePublicArticleCoversSharedParserReportsYoloVisionAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `10/10` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Plugin Inventory Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/plugin-inventory-public-article.md`，将 plugin inventory
从短说明扩展为可发布的 readonly diagnostics 文章。文章解释 builder/global/
builder-capability/runtime registry source、creator metadata、field metadata、diagnostics
和 smoke runner 输出，并明确它不能替代 plugin lifecycle、真实 inference 或包消费验证。

### 实现

- 文章补齐 `TensorRtPluginRegistryInventory.cs`、
  `TensorRtBuilder.PluginRegistryInventory.cs`、`TensorRtRuntime.PluginRegistryInventory.cs`、
  `TensorRtEnvironmentProbe.PluginRegistryInventory.cs`、`plugin_registry_inventory.inc`、
  managed interop、`PluginRegistryInventorySmokeRunner` 与 interface coverage matrix 的证据路径。
- 明确 `TensorRtPluginRegistrySource.Builder`、`Global`、`BuilderCapability`、`Runtime`
  四类 source，以及 `HasErrorRecorder`、`ParentSearchEnabled`、`CreatorCount`、
  `RecursiveCreatorCount`、`FindCreator`、`TryFindCreator`、`GetCreatorSummaries`、
  `GetFieldSummaries` 与 `GetDiagnostics` 的 copied/read-only 边界。
- 写入 creator/field metadata 字段，包括 `InterfaceKind`、`InterfaceMajor`、
  `InterfaceMinor`、`ApiLanguage`、`TensorRtVersion`、`FieldType`、`Length` 与
  `HasData`，并说明 `HasData` 不暴露 field data pointer。
- 明确 native bridge 的 `getAllCreators` / `getAllCreatorsRecursive`、SEH guard、
  vendor mismatch/missing 与 probe 异常分类边界。
- 明确 `createPlugin`、`clone`、`serialize`、`deserializePlugin`、`attachToContext`、
  `enqueue`、`registerCreator`、`deregisterCreator`、`loadLibrary`、callback trampoline、
  resource acquire/release 和 borrowed pointer API 仍不能伪装成低风险完成。
- 为 `PublishingPublicArticleTests` 增加
  `PluginInventoryPublicArticleCoversSourcesCopiedMetadataSmokeAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `9/9` 通过。
- 本批未改实现代码，仅更新 public article、ProjectQuality 专项门禁和 review 记录。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Engine Inspector Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/engine-inspector-public-article.md`，将 Engine Inspector
从短说明扩展为可发布的 readonly diagnostics 文章。文章解释 engine 名称、IO tensor、
layer count、profile count、device memory、auxiliary streams、profiling verbosity、
inspector text 与 readback hash 的用途，并明确这些证据不能替代真实 inference 或包消费验证。

### 实现

- 文章补齐 `TensorRtEngineInspector.Trt11Diagnostics.cs`、`OnnxEngineBuildResult.cs`、
  `OnnxEngineRuntimeArtifactWriter.cs`、`OnnxEngineBuildDiagnostics.cs`、
  `TensorRtExecReport.cs`、`TensorRtExecCommand.cs`、`MainForm.cs` 与
  `tensor-rt-exec-trtexec-parity-matrix.json` 的证据路径。
- 明确 `GetLayerInformation`、`HasExecutionContext`、`TryGetErrorRecorderSnapshot`、
  `OnnxLoadedEngineDiagnostics`、`ReadbackFingerprint`、`ReadbackSha256` 与
  `EvidenceBoundary` 属于 copied/read-only 诊断面。
- 写入 TensorRtExec build 后 `--dumpLayerInfo` / `--exportLayerInfo` 与 `--loadEngine`
  readonly diagnostics 示例，并覆盖 `trtexec-like-engine-readback` 与
  `trtexec-like-engine-readback-skipped` artifact 边界。
- 明确 Engine Inspector does not create execution bindings，不做 enqueue inference，
  不 validate outputs，不伪造 per-layer timing；它不是 real-model-runtime proof，也不是
  package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `EngineInspectorPublicArticleCoversReadbackArtifactsAndReadonlyProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `8/8` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Source Build Public Article Expansion

本批继续补齐二次矫正中的源码 C++ 编译教程与宣传文章要求，将
`docs/articles/zh-cn/publishing/source-build-windows-public-article.md` 从简短入口扩展为可发布的 Windows
源码构建文章。文章现在覆盖 C# / C++ bridge / NVIDIA runtime 三层构建路线图、环境清单、推荐目录、
`CUDA_PATH`、TensorRT/cuDNN 检查、6 个 Windows CMake release preset、生成绑定、managed build、native bridge
构建后验证、CMake/CUDA/TensorRT/DLL/CUDA error 35 排障，以及 GitHub full runtime 包与 NuGet 小包双路线边界。

同步新增 `PublishingPublicArticleTests.SourceBuildPublicArticleCoversCppBridgeEnvironmentPresetsPackagesAndTroubleshooting`，
锁定 C++ bridge、CUDA/TensorRT/cuDNN、preset、`Generate-Bindings.ps1`、ABI/public handle quality gate、
`dumpbin /dependents`、双包路线和 deep-dive article 链接，防止 public article 退化成短清单。

### Verification 与边界

- 定向测试：
  `PublishingPublicArticleTests|SourceBuildCmakeWindowsGuideTests` 共 `6/6` 通过；编译阶段仅出现 5 条既有
  ProjectQuality nullable warning。
- 本批没有执行 native build、没有下载 CUDA/TensorRT/cuDNN/ONNX/engine/nupkg；只更新文章与质量门。
- 源码构建文章仍明确 build-only 边界，不把 CMake/dotnet build、local feed 或 `Skipped=True` 宣称为
  package-consumer-runtime、post-publish、publish approval 或 release-close proof。
- 本批未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages/GitHub Release 发布。

## 2026-07-23 YOLOv10 Public Article Evidence Backfill

本批转向宣传文章序列的真实内容质量，把 `YoloVision YOLOv10 End-to-End 输出接入：从官方模型到 TensorRT 结果`
从通用教程增强为带本仓库 source-tree runtime closure 数值的完整技术文章。新增章节
“本仓库已验证的 YOLOv10n v1.1 闭环”，明确记录官方 YOLOv10 v1.1 revision、AGPL-3.0-only 边界、
ONNX/engine/output/run-log SHA256、TensorRT/CUDA 版本、GPU/driver、`output0:[1,300,6]` 和
`dog=0.91683036` 运行结果。

同步更新 `YoloVisionDocumentationMatrixTests`，要求文章持续包含关键 hash、shape、top prediction、
许可证和 `not package-consumer-runtime` 边界，防止后续宣传材料退化成没有证据的泛泛介绍。

### Verification 与边界

- 定向文档/路线图质量门：
  `YoloVisionDocumentationMatrixTests|ArticleRoadmap30PlusTests` 共 `5/5` 通过；编译阶段仅出现 5 条既有
  ProjectQuality nullable warning。
- 本批没有下载模型、ONNX、engine 或图片；只引用已存在的 hash-pinned closure 与 acquisition manifest。
- 本批不触发 GitHub Actions、不 push、不执行 NuGet/GitHub Packages/GitHub Release 发布。
- 文章中的 YOLOv10n 记录仍是 `source-tree-real-model-runtime`；不是 package-consumer-runtime、post-publish、
  release-close proof，也不是 AGPL 资产公开再分发批准。

## 2026-07-23 Runtime Deserialization Deferred Boundary Audit

本批回到 deferred 主线，针对 `runtime-deserialization-boundary` 设计组做集中审计，而不是尝试把
`IRuntime::deserializeCudaEngineV2` 或 `IRuntime::loadRuntime` 伪装为低风险晋级。审计固定 5 条
medium-risk deferred-only 行：TRT10/11 `deserializeCudaEngineV2` 与 TRT8/10/11 `loadRuntime`。

新增 `runtime-deserialization-deferred-boundary-audit.json/.md`，记录每条候选的 manifest id、entrypoint、
version guard、vendor vtable/外部 runtime 边界、ownership blocker、现有 safe alternative 和
`keep-deferred` 决策。现有 `TensorRtRuntimeDeserializationBoundaryPrecheck` 继续证明 direct
`IRuntime::deserializeCudaEngine` 已由 scoped-buffer bridge 覆盖；`TensorRtRuntimeDeserializationDependencyDiagnostics`
继续把 dependency-probe-only、CUDA driver/runtime blocker、full package consumer runtime proof 缺失和
`loadRuntime` ownership blocker 分类为 non-proof；`TensorRtStreamIoInterfaceInfoDesignGate` 继续锁住
stream reader/writer callback 与 owner handle 前置条件。

### Verification 与边界

- 新增 `RuntimeDeserializationDeferredBoundaryAuditTests`，验证 5 条 runtime boundary rows 仍为
  `deferred-only`，deferred history 保留，未尝试 native promotion，GitHub Actions 与发布副作用均为 false。
- 聚焦测试：
  `RuntimeDeserializationDeferredBoundaryAuditTests|RuntimeDeserializationBoundaryPrecheckTests|StreamIoInterfaceInfoDesignGateTests`
  共 `12/12` 通过；仅出现 5 条既有 ProjectQuality nullable warning。
- 本批不修改 native ABI surface、不新增 entrypoint、不删除 deferred manifest、不运行 GitHub Actions、不 push、
  不发布 NuGet/GitHub Packages/GitHub Release。
- 这些审计与 design gate 是 source-quality / non-proof evidence，不是 runtime execution proof、package-consumer-runtime
  proof、external lean runtime proof 或 release-close proof。

## 2026-07-23 TRT8 Legacy Parser Copied Readonly Diagnostics

本批完成 TRT8 legacy parser 的 owner-scoped safe alternative：UFF required version 三个标量 getter、
`ICaffeParser::parseBinaryProto` 以及 `IBinaryProtoBlob` 的 data/type/dimensions 共 7 行。native 在单次调用
内创建并删除 parser/blob；binaryproto 使用 caller-buffer 复制 shape、data type 和字节，不暴露 parser、blob
或 data pointer，也不调用进程级 `shutdownProtobufLibrary`。Windows SEH 与 C++ exception 路径均清零输出。

coverage 扫描器现在仅对 TRT8 纳入 `NvCaffeParser.h` 与 `NvUffParser.h`，每套 TRT8 package 为 `880` 行，
`760 implemented / 120 deferred-only`；7 行均为 `implemented-with-deferred-history`，相邻的 parser/destroy
仍为 deferred-only。旧 deferred manifests 保留，长期 deferred 边界未缩小。

### Runtime 与证据

- bindings：`197 manifests / 3975 records`，连续生成幂等，输出校验通过。
- TRT8/CUDA12.1 smoke：UFF `0.6.9`；MNIST binaryproto `[1,1,28,28] / Float / 3136 bytes`；复制 payload
  SHA256 为 `DF7D560B482098FAC1C6122C22BD0A54499ED9F8EC3AC6BAE8FC917D3A01774A`；独立 managed copies 与
  TRT10/11 非目标 guard 通过。
- native ABI/PE：TRT8 `993/993`、TRT10 `1087/1087`、TRT11 `1234/1234`，missing 均为 0。
- solution Debug：0 warning / 0 error；Release：5 条既有 nullable warning / 0 error。
- 公共 API 文档、双语文档通过；DocFX `930 models / 0 warning / 0 error`。
- 受影响专项测试 `52/52`；legacy evidence strict validator `16/16`；classification finding `0`；strict
  release required failure `0`。
- TRT8/10/11 bridge-only package consumer 均 restore/build 通过，0 warning / 0 error，证据分类仍为
  `compile-surface-proof`，不提升为 package-consumer runtime proof。

### 发布与 C 盘边界

本批没有 push、workflow dispatch、NuGet/GitHub Packages publish、GitHub Release upload 或 issue close。
Owner final gate 仍 blocked，因真实外部 owner evidence 尚未提供。C 盘未发现本批 TensorRT/CUDA/cuDNN/ONNX/
模型/engine/nupkg 下载；本轮 workload 日志与三个空临时目录的精确删除命令在执行前被工具策略拦截，未发生
部分删除。Downloads 中的既有用户资产未触碰。

## 2026-07-22 ONNX Parser Layer Output Copied Metadata

本批重新导出并审计 TensorRT deferred inventory：598 条 deferred rows 中 low risk 为 0、medium 为 112、
high 为 486，CUDA immediate-safe candidate 为 0。最终没有直接包装 parser-owned `ITensor*`，而是选择
`IParser::getLayerOutputTensor` 的 owner-scoped copied metadata safe alternative。TRT8 vendor header 没有该
方法，因此保持无 native entrypoint 和 managed `NotSupported` guard；TRT10/11 通过 parser vtable 调用纯虚
方法，同名 import-library/DLL symbol 为 0 不构成 ABI 缺失。

native C ABI 在 parser owner 有效期间复制 tensor name、64-bit shape、data type、location、allowed formats、
dynamic/shape/execution/input/output flags，字符串使用 caller-buffer，所有 output 在失败前重置，并隔离 C++
exception 与 Windows SEH。公开 C# surface 为 `TryGetLayerOutputTensorMetadata`、
`GetLayerOutputTensorMetadata` 和 pointer-free `TensorRtOnnxLayerOutputTensorMetadata`，不暴露 `IntPtr`、
`nint`、`UIntPtr`、`SafeHandle` 或 tensor handle。TRT11 旧 deferred manifest 保留，coverage 通过显式 alias
记录 implemented-with-deferred-history；没有删除历史记录来改变统计。

### Runtime 与证据

- bindings 为 `196 manifests / 3973 records`，连续生成幂等且输出验证通过；coverage 为 TRT10
  `761 implemented / 118 deferred-only`、TRT11 `814 / 87`，TRT8 保持受控版本边界。
- TRT8/CUDA12.1 runtime 与 builder creation 成功，但该 bridge 构建时没有 ONNX parser dependency；parser
  construction 现在以 exit `0` 和 `Skipped=True Reason=ParserConstruction:...` 受控收敛，metadata 未查询。
- TRT10.11/CUDA12.9 identity smoke 实际得到 tensor `output`、shape `[-1, 4]`、`Float / Device`；dynamic、
  execution、network-output 为 true，`TryGet`/`Get` 一致，缺失 layer 返回 false，enqueue/output match 通过。
- TRT11.0/CUDA12.9 bridge/vendor DLL/version/registry probe 成功，但 vendor builder/runtime creation 返回 null；
  metadata 未查询，继续分类为 `dependency-runtime-probe-only`。
- compact evidence 与 validation 位于 `onnx-parser-layer-output-metadata-runtime-evidence*`，strict validator
  `15/15`。TRT8/10/11 三个无 ProjectReference package consumer restore/build 均成功，但只属于
  `compile-surface-proof`。

### Verification 与边界

- TRT8/CUDA12、TRT10/CUDA12、TRT11/CUDA12 native build 通过；ABI/PE parity 分别为 TRT8 `991/991`、
  TRT10 `1087/1087`、TRT11 `1234/1234`，missing 均为 0。
- solution Debug/Release 均为 0 error；Release 保留 5 条既有 nullable test warning。Public API documentation
  warning 0、bilingual finding 0，DocFX `926 models / 0 warning / 0 error`。
- 最终三类受影响测试为 `50/50`，此前 broader affected set 为 `74/74`；一次包含整个
  `ReleaseCandidateReadinessTests` 的组合在 600 秒超时，不能声明完整 suite pass。
- classification/public-proof finding 均为 0，strict release required failure 为 0。Owner convergence 保持
  structural `9/9`、accepted `0/9`、gates `2/3`，final owner gate blocked `5`。

C 盘定向审计未发现 TensorRT、CUDA、cuDNN、ONNX、模型、engine 或 nupkg 下载；Downloads、Desktop、
Documents 任务命名匹配为 0，全局 NuGet 中的 JYPPX 项均为旧缓存。本轮 `dotnet build-server shutdown` 已
关闭 MSBuild/VB/C# servers，并清除 15 个空 `MSBuildTemp*` 目录。仍有 230 个当日 workload 小日志
（441,942 bytes）以及 `C:\jyppx-pkgcache`、`%TEMP%\jyppx-split-packages`、`%TEMP%\MSBuildTemp` 三个
空目录；标准精确 `Remove-Item` 在执行前被工具策略拦截，未发生部分删除，也未改用绕过方式。

本批不构成公开 package consumer、post-publish、模型准确率或 release-close proof，不允许删除 deferred
history，也不授权 NuGet/GitHub Packages push、GitHub Release upload 或 issue close。

## 2026-07-22 TensorRtExec Engine Packaging、Refit 与 Weight Streaming

本批延续 deferred inventory 的安全审计结论：598 条 TensorRT deferred rows 中仍无 low-risk 候选，
因此没有删除 deferred history 或引入 pointer-shaped public API。工作重心转向现有 typed owner surface，
把 `--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights`、`--refit`、
`--allowWeightStreaming` 和 `--weightStreamingBudget` 从 parse/report-only 推进到真实 builder、runtime
和 engine set/readback。

Weight streaming budget 现支持 `-2`、`-1`、`0..100%`、bytes/K/M/G/KiB/MiB/GiB；无后缀数字继续按
MiB 解释以保留已有 managed contract。构建新 engine 时，weight streaming 要求 strongly typed，budget
要求 builder weight-streaming flag；load-engine 路径允许对已有 weight-streaming plan 单独设置 budget。
预算在 execution context 创建前解析、设置和读回，并同时记录 streamable weights、automatic budget 和
scratch bytes。version-compatible plan 在反序列化前设置并读回 `EngineHostCodeAllowed`。

TRT10/11 使用 `StripPlan`；默认配合 `RefitIdentical`，显式 `--refit` 时使用 `Refit`。TRT8 可真实应用
version-compatible、exclude-lean 与单独 refit，但 strip/weight streaming 保持版本 guard；实测 TRT8 的
version-compatible + refit vendor readback 冲突已固化为显式 guard。exclude-lean 和 strip weights 要求
build-only/skip-inference，避免把尚未实现的 external lean runtime 或 stripped-plan refit lifecycle 伪装
成可推理路径。另修复外部 ONNX build 后 readonly diagnostics 错用空 `LoadEnginePath` 的问题，改为使用
实际 preflight engine path。

### Runtime 与证据

- TRT10 version-compatible + refit：config flags、runtime host-code 与 engine refittable readback 全部匹配；
  engine round-trip、enqueue 和 identity output match 成功。
- TRT10 strip weights：`StripPlan=True`，默认 `RefitMode=RefitIdentical`，两项 readback match。
- 官方 YOLOX-S：ONNX 35,858,002 bytes，streamable weights 35,829,504 bytes；`50%` 解析和 readback
  均为 17,914,752 bytes，scratch 为 5,901,824 bytes；`images [1,3,640,640]` 到
  `output [1,8400,85]` enqueue 成功。通用 runner 未做 decode/NMS，输出严格保持
  `captured-unverified`。
- TRT10 load-engine `-1`：readonly diagnostics 得到 2 个 I/O、206 层和 1 个 profile；automatic/readback
  均为 35,829,504 bytes，随后 bounded enqueue 成功。
- TRT8 保持 `dependency-probe-only`（当前 bridge 无 ONNX parser build support）；TRT11 runtime creation
  仍遇到已知 structured exception `3228369022`，同样未提升为 applied/runtime proof。
- compact evidence 为 `trtexec-engine-packaging-runtime-evidence.{json,md}` 与对应 validation；strict
  验证 `20/20`，evidence JSON SHA256
  `AC9C9368C7DE55D43F6A1FEB54B4F68FD3BBE4A50240C36468A8A7A1782A101D`。

### Verification

- bindings 连续生成与验证稳定：`194 manifests / 3971 records`，工作树幂等。
- solution Debug `0 warning / 0 error`；Release 为 `5` 条既有 ProjectQuality nullable warning、
  `0 error`。Public API documentation warning `0`，bilingual finding `0`。
- TRT8/CUDA12、TRT10/CUDA12、TRT11/CUDA12 native Release build 成功；ABI/PE 分别为 TRT8
  `991/991`、TRT10 `1086/1086`、TRT11 `1233/1233`，missing declaration/export 均为 `0`。
- 本批 focused tests `74/74`，release workflow contract `22/22`，bounded N-S/T-Z `74/74`，无失败或
  超时。GUI/CLI checklist strict 为 `19/19`，runtime-proof item `0`。
- DocFX 应用 `921 models`，`0 warning / 0 error`。managed package 14,805,776 bytes，SHA256
  `DFB8C921C26159F1B864467D0D8830809590005298AEE716B827FC8329C54029`；TRT10 bridge-only package
  351,092 bytes，SHA256 `BD5E4B221B2A9FBFA651CF43F3700C54633F74C574640858E4F496C473177A89`。
  无 ProjectReference consumer restore/build 为 `0 warning / 0 error`，证据仍为 compile-surface-proof。
- strict classification/public-proof finding 均为 `0`，strict release required failure 为 `0`。owner
  convergence 保持 structural `9/9`、accepted `0/9`、gates `2/3`，final owner gate blocked `5`。
- staging audit 补充 `artifacts/user-acceptance` 文档 bucket，避免有意更新的验收文档被误报为
  `manual-review`；复验 `reviewCandidate=0`。

### C 盘与边界

本批未把 TensorRT、CUDA、ONNX、YOLOX、engine、源码或 nupkg 下载到 C 盘；Downloads、Desktop、
Documents、Temp 顶层和全局 NuGet package root 的任务相关新增均为 `0`。consumer 清空了自己的 cache
子目录；当前仅剩 54 个 dotnet workload 小日志（71,194 bytes）及 `C:\jyppx-pkgcache`、
`%TEMP%\jyppx-split-packages`、`%TEMP%\MSBuildTemp` 三个空目录。标准删除命令被工具安全策略在执行前
拒绝，未换壳绕过。E 盘 ignored evidence 目录中的 42,007,516-byte YOLOX plan 同样因删除策略拒绝而
仍待人工清理。

本批没有执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。builder/engine
readback、真实权重 enqueue、本地 nupkg 与 bridge-only consumer 都不能替代跨版本 lean runtime、完整 refit、
模型准确率、公开 package consumer 或 post-publish proof。

## 2026-07-21 TensorRtExec I/O And Layer Precision Policies

本批先审计 598 条 TensorRT deferred rows 与 50 个去重候选：low-risk 为 0，剩余项仍落在 callback、
borrowed `IDimensionExpr`、error-recorder refcount、plugin ownership、runtime deserialization ownership 或
TRT8 consistency-checker 缺 symbol 边界，因此没有为了数量删除 deferred history 或暴露 native pointer。

随后修复稳定 public `TensorRtBuilderFlag` 与 vendor raw index 混用的问题。公开 enum 数值保持不变，
`SetFlag/GetFlag/ClearFlag/SetFlags/GetFlags` 在 managed interop 边界按 TRT8/10/11 双向映射，unsupported
flag fail closed，vendor-only raw bit 不会误报成别的逻辑 flag。TRT8 `DirectIO` raw 12 与
`PreferPrecisionConstraints` raw 11 已加入永久 NetworkBuilder smoke 并实跑隔离成功。

应用层实现 `--inputIOFormats`、`--outputIOFormats`、`--precisionConstraints`、`--layerPrecisions` 与
`--layerOutputTypes`：parser 校验 type/format grammar、IO broadcast/count、pattern 单 wildcard、layer exact
优先 wildcard、后规则覆盖前规则、output type broadcast/count；ONNX parse 后对 network tensor/layer 使用
typed set/readback。TRT8/10 支持完整策略；TRT11 仅在 requested type 等于 inferred type 时设置 allowed
formats，已移除的 tensor type、constraint flags 与 layer setters 明确保持 version guard。

TRT10.11/CUDA12.9 identity smoke 中五项策略全部 `Applied=True`、`ReadbackMatch=True`，进入
`AppliedOptions`，engine round-trip、两次 measurement enqueue 与 output match 成功。TRT8.6/CUDA12.1
NetworkBuilder build/enqueue/output match 成功并输出 raw flag isolation true。TRT11/CUDA12.9 在 network
creation 前遇到已知 vendor structured exception `3228369022`，报告保持 dependency-probe-only，五项策略
全部留在 parse-only，没有伪造 applied。

### Verification

- binding generator/output validation 连续生成两次一致：`194 manifests / 3971 records`。
- 完整 solution Debug/Release 均为 `0 error`；Debug 保留 5 条既有 nullable warning，Release 为
  `0 warning`。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9 native Release build 成功；ABI declaration 为
  TRT8 `991/991`、TRT10 `1086/1086`、TRT11 `1233/1233`，三份 PE export parity missing 均为 `0`。
- 受影响 ProjectQuality focused 与 bounded shard 均为 `139/139`；GUI/CLI checklist strict 为
  `18/18`，runtime-proof item 为 `0`。
- managed package 为 `14,805,725` bytes，SHA256
  `B3A8BA1C90DFE4C381716FD3399E6775C591CF76E5492BF68B478A527039C0DA`；TRT10 bridge-only package
  为 `351,087` bytes，SHA256 `31EDA06CFF2B87F18202D77D758D8E95117F6646F27BEFAEB73786BDF0EF40BB`。
  无 `ProjectReference` consumer restore/build 为 `0 warning / 0 error`，分类仍是 compile-surface-proof。
- Public API documentation/bilingual 均通过；DocFX `920 models`、`0 warning / 0 error`。strict
  classification/public-proof finding 均为 `0`，strict release quality required failure 为 `0`。
- owner convergence 保持 structural `9/9`、accepted `0/9`、gates `2/3`；final owner gate blocked `5`，
  没有把本地包、synthetic identity 或 dependency probe 提升为公开 proof。

### C 盘审计

本批没有向 `Downloads`、全局 NuGet package root 或 C 盘其他位置下载 TensorRT、CUDA、cuDNN、模型、
engine 或 nupkg。bridge consumer 已自动删除本批 `C:\jyppx-pkgcache` 子 cache 与
`%TEMP%\jyppx-split-packages` 子目录；最终审计发现 81 个 dotnet workload 小日志（合计 105,684 bytes）
以及 `C:\jyppx-pkgcache`、`%TEMP%\jyppx-split-packages`、`%TEMP%\MSBuildTemp` 三个空目录。
两次标准 PowerShell `Remove-Item` 都在执行前被工具安全策略拒绝，未换壳绕过；用户 Downloads、NuGet、
Codex、CUDA 与系统缓存未触碰。

compact evidence 为 `trt-deferred-safe-uplift-candidate-audit.{json,md}` 与
`trtexec-io-layer-precision-runtime-evidence.{json,md}`。这些结果是本地 ProjectReference、builder policy
与 synthetic identity 证据，不是 caller binding layout、tactic、数值准确率、真实模型、外部 package
consumer 或发布证明。未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-21 TensorRtExec Deployment Controls

本批把 `--device`、`--useDLACore`、`--allowGPUFallback`、`--tacticSources`、`--directIO`、
`--sparsity` 与 `--stronglyTyped` 从 parser/report 对齐推进到有版本 guard 的真实 build 控制。
指定 device 时，完整 build/load/bounded-runtime 在专用 host thread 内执行 CUDA device set/readback，
避免污染调用线程；DLA core 在写 config 前按 `builder.DlaCoreCount` fail closed；fallback、tactic mask、
DirectIO 与 SparseWeights 均走 typed set/get 并记录 requested/readback/match。

`--sparsity=enable|disable` 已真实应用，`force` 因官方语义还包含权重重写而继续 parse-only。
strongly typed 的 vendor 位语义按版本拆开：TRT10 使用 raw `1u << 1`；TRT11 不复用该 raw bit，
而是依赖 vendor 的 always-strongly-typed network 契约；TRT8 继续 parse-only。GUI/CLI field map、
feature/parity matrix、release gap、用户验收说明与中文部署控制文章已同步，DLA config readback 不会被
写成 DLA layer/model 执行证明。

### Runtime 与验证

- TRT10.11/CUDA12.9 identity smoke 成功：device `0/0`、GPU fallback、tactic sources、DirectIO、
  SparseWeights 和 TRT10 strongly typed raw bit 均应用/readback match；engine round-trip、enqueue 与
  output match 为 true。`sparsity=force` 独立 build 成功，但 `--sparsity` 保持 parse-only，builder
  snapshot 不含 SparseWeights。
- 无 DLA 的 RTX 主机报告 0 cores；请求 core 0 实际非零退出 `-532462766`，没有静默 GPU fallback
  或 DLA engine。TRT8 device applied、strongly typed parse-only，因该 bridge 无 ONNX parser 保持
  dependency-probe-only。TRT11 在 network creation 前遇到已知 vendor structured exception
  `3228369022`，因此也只记录 dependency-probe-only，不宣称 strongly typed applied。
- focused TensorRtExec/OnnxToEngine tests `72/72`；bounded N-S/T-Z 两分片、9 个相关 class 为
  `62/62`，无失败或超时。GUI/CLI checklist strict 为 `16/16`。
- bindings 为 `194 manifests / 3971 records`，脚本内连续生成两次幂等。TRT10/CUDA12.9 与
  TRT11/CUDA12.9 native Release 增量构建成功；TRT8/10/11 ABI declarations 为
  `991/991`、`1086/1086`、`1233/1233`，TRT10/TRT11 PE exports 为 `1086/1086`、`1233/1233`，
  missing 均为 0。
- solution Debug/Release 均为 0 error；最终 Release 为 0 warning，Debug/单独 test build 保留 5 条
  既有 nullable test warning。Public API documentation/bilingual 均 0 finding；DocFX 应用
  919 models，`0 warning / 0 error`。
- managed nupkg 为 14,794,656 bytes，SHA256
  `11BE575C8DC8CCC9580835C56CFF3A895A1FADD36190867A2F9865351954740F`；TRT10.11/CUDA12.9
  bridge-only nupkg 为 351,089 bytes，SHA256
  `889F1FF4A9B19C847E311C022B5FBF4AD2C1CEAE49B70E8585C0672FAD1CF700`。无 ProjectReference
  consumer restore/build 为 0 warning / 0 error，分类严格保持 `compile-surface-proof`。
- strict classification/public-proof finding 均为 0，strict release quality
  `RequiredFailureCount=0`。owner convergence 保持 structural `9/9`、accepted `0/9`、gates `2/3`；
  final owner gate 仍有 5 个真实外部输入阻塞。

### C 盘与证据边界

本批没有把 TensorRT、CUDA、ONNX、模型、engine、源码或项目 nupkg 下载到 C 盘；
Downloads/Documents/Desktop 无相关命中，共享 NuGet package 根也无本轮新增 package 目录。本地 pack
的 NuGet cache 显式使用 E 盘；consumer 临时 `C:\jyppx-pkgcache` 与 Temp split 子目录在脚本内清空。
精确审计仍发现旧 `eb0bl5in.vut` workload metadata 33,710 bytes、本轮 23 个 workload logs、15 个空
MSBuildTemp 目录及两个空 task cache 根。标准 `Remove-Item` 在执行前被工具安全策略整体拒绝，未换壳
或绕过，因此这些目标仍待策略允许后清理。

compact evidence 为 `trtexec-deployment-controls-runtime-evidence.{json,md}`，继续保持
`isDlaModelExecutionProof=false`、`isSparseTacticSelectionProof=false`、
`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`、
`canPublishPublicly=false`。未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或
issue close。

## 2026-07-21 TensorRtExec Runtime Controls

本批按 TensorRT 10.11 官方 `trtexec --help` 与 v10.11 `sampleOptions.cpp` / `sampleInference.cpp`
语义，把上一批仍为 parse-only 的 `--threads`、`--useSpinWait`、`--useCudaGraph` 和
`--noDataTransfers` 接入真实 bounded runtime。`--threads` 作为布尔开关，每个 effective stream
使用独立 host driver thread，并在子线程恢复 worker 创建时的 CUDA device；spin wait 通过
`CudaEvent.IsReady()` 主动查询；graph 持有独立 graph/graph-exec owner，完成 capture、instantiate
和 launch，捕获失败时安全结束 capture、释放全部 graph owner、统一回退 direct enqueue并记录
reason；no-transfer 只 allocate/bind device buffer，跳过 input H2D 与 output D2H/readback。

报告新增 `UseSpinWaitApplied`、`UseCudaGraphRequested/Applied/FallbackReason` 与
`MeasurementRoundsPerContext`。只有真实执行后才将这些 controls 标 applied；graph fallback 的 run
仍保留 parse-only。no-transfer run 是成功的 scheduler/enqueue 行为证据，但保持
`OutputMatch=false`，output/raw export options 仍 parse-only，artifact 明确
`HasBenchmarkExecutionEvidence=true`、`OutputElementCount=0`、`HasRawBindingProof=false`。
`--sleepTime` 继续 parse-only，因为仓库没有忠实的 device-side launch-to-compute delay 原语，禁止用
普通 `Thread.Sleep` 冒充。

### 最终验证

- focused TensorRtExec/OnnxToEngine/schema/application tests 为 `49/49`；按仓库 bounded runner 覆盖
  9 个相关 class、3 个批次，`55/55`，无失败或超时，且未触发递归执行完整门禁的
  `ReleaseCandidateReadinessTests`。
- TRT10.11/CUDA12.9 threads+spin+graph smoke：2 contexts / 2 threads，per-context rounds `[3,3]`，
  6 raw timing samples / 3 averaged samples，spin wait 与 graph 均 applied，fallback reason 为空，
  output match true。no-transfer smoke 完成 2 次真实 enqueue，H2D/D2H 均为 0，output match false、
  output count 0、raw proof false；output/raw options 正确保持 parse-only，times artifact 明确
  `HasBenchmarkExecutionEvidence=true`。
- solution Debug/Release 均为 0 error；Debug 通过，Release 保留 5 条既有 test nullable warning。
  bindings 为 `194 manifests / 3971 records`，连续生成两次幂等。TRT10/CUDA12.9 与
  TRT11/CUDA12.9 native Release 增量构建成功；TRT8/10/11 ABI declarations 分别为
  `991/991`、`1086/1086`、`1233/1233`，missing 0；TRT10/TRT11 PE exports 分别为
  `1086/1086`、`1233/1233`，missing 0。
- managed package `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` 为 14,786,378 bytes，SHA256
  `6A0DC0B568A76CC52D6451689BD8D7DE4E0495C570F164402F7E427E80B3BA32`；TRT10.11/CUDA12.9
  bridge package 为 351,093 bytes，SHA256
  `9C32942A72385574CA6DF652EA0221758C4A7A5C72DD74C321481B2B95187646`。无 ProjectReference
  consumer restore/build 均成功，分类严格保持 `compile-surface-proof`。
- Public API documentation 与 bilingual audit 均通过，三个公共项目均 0 warning / 0 error；
  DocFX 应用 918 models，`0 warning / 0 error`。strict classification 与 public-proof finding
  均为 0，strict release quality `RequiredFailureCount=0`。owner convergence 为 structural
  `9/9`、accepted `0/9`、gates `2/3`；final owner gate 仍明确 blocked，未把本地证据晋级。

### C 盘与证据边界

本批没有把 TensorRT/CUDA/ONNX、模型、源码或项目 nupkg 下载到 C 盘；split package restore cache
显式位于 `build-out/runtime-controls-pack/nuget-cache`（E 盘）。Downloads、Documents、Desktop
今日没有本任务相关文件。Temp 仅命中一个由本轮 `dotnet` 调用生成的 33,710-byte workload metadata
包：`C:\Users\guoji\AppData\Local\Temp\eb0bl5in.vut\microsoft.net.workloads.10.0.300.msi.x64\10.302.0\microsoft.net.workloads.10.0.300.msi.x64.10.302.0.nupkg`；
目录无占用进程，但标准 `Remove-Item` 被当前工具安全策略拒绝，未换壳或绕过，因此仍待环境策略允许
后清理。NuGet、Codex、CUDA 用户缓存以及 Downloads 中无关的 Cockpit 安装包、Typora 更新包均未
触碰。

compact evidence 为 `trtexec-runtime-controls-runtime-evidence.{json,md}`，继续保持
`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`、
`canPublishPublicly=false`。未执行 NuGet push、GitHub Packages publish、GitHub Release upload
或 issue close。

## 2026-07-21 TensorRtExec/OnnxToEngine Bounded Benchmark Scheduler

本批在 CUDA deferred candidate safety audit 后没有发现可安全提升的 immediate-safe 函数：311 条
deferred rows、64 个唯一函数全部仍属于 callback/user object、external/graphics、generic graph
params、raw symbol/entry point 或 resource/context ownership。没有为了凑接口数量而改变 deferred
边界，审计记录与旧 deferred history 均保留。

应用层转向 trtexec-like scheduler 完整度：compatible float engine 现在为每个 effective
`--infStreams`（优先）或 `--streams` 创建独立 execution context、bindings、CUDA stream 与
start/stop event；`--iterations` 与 `--duration` 是同时满足的双下限，`--warmUp` 以实际 GPU
enqueue 达到最低时长，`--idleTime` 在 measurement rounds 间生效，`--avgRuns` 产出连续平均窗口，
`--percentile` 明确基于 raw GPU timing samples。worker 构造失败会回收已创建 owner，dispose 前
drain stream。`--sleepTime`、`--useSpinWait`、`--threads`、`--useCudaGraph` 和
`--noDataTransfers` 继续 parse-only；应用状态只在实际执行后标记相应 applied option。

### 验证与证据

- 真实 TRT10.11/CUDA12.9、RTX 3060 Laptop、driver 576.02 bounded smoke：2 contexts、6 raw
  samples、3 average windows、p90 `0.319488 ms`；duration run 6440 rounds、1000.1398 ms，
  两次 output match 均为 true。证据为
  `trtexec-bounded-benchmark-scheduler-runtime-evidence.{json,md}`，严格保持 synthetic/local
  project-reference boundary：`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`、
  `canPublishPublicly=false`。
- CUDA candidate audit 专项 1/1；bounded ProjectQuality 为 3/3 shard、54/54 tests；focused 与
  相邻 TensorRtExec/OnnxToEngine 测试保持通过。
- solution Debug 为 0 warning / 0 error；Release 为 0 error，保留 5 条既有 test nullable
  warnings。TRT10/CUDA12.9 与 TRT11/CUDA12.9 existing native Release build 成功；本批无 native
  source/ABI contract change。TRT8/10/11 source ABI declarations 为 `MissingDeclarations=0`，
  TRT10/TRT11 PE export parity 为 `MissingExports=0`。
- 无 ProjectReference TRT10 bridge consumer restore/build 为 0 warning / 0 error，分类保持
  `compile-surface-proof`，不提升为 package-consumer runtime proof。Public API bilingual audit、
  public-proof boundary audit、strict release quality gate 均为 0 findings / 0 required failures；
  DocFX 为 917 models、0 warning / 0 error。

### C 盘与发布边界

本批没有将 ONNX、engine、模型、nupkg、源码副本或 benchmark 资产下载到 C 盘。审计发现的本轮
临时项仅为今天生成的 88 个 .NET workload logs（109,635 bytes）、3 个空 Temp 目录
（`MSBuildTemp`、`MSBuildTempinxxb011.4kd`、`jyppx-split-packages`）与空
`C:\jyppx-pkgcache`；Downloads、Documents、Desktop 今日相关命名命中均为 0，全局 NuGet、Codex、
CUDA 与既有用户文件不触碰。标准 `Remove-Item` 清理命令被当前执行策略拦截，未绕过策略，因此这些
临时痕迹在本地仍需用户/环境策略允许后清理。未执行 NuGet push、GitHub Packages publish、GitHub
Release upload 或 issue close。

## 2026-07-21 CUDA IPC Import Owner-Safe Uplift

本批把 `cudaIpcOpenEventHandle`、`cudaIpcOpenMemHandle` 与 `cudaIpcCloseMemHandle` 从
`deferred-only` 提升为真实 owner-safe 路径，与上一批 export token 组成完整跨进程生命周期。
event import 返回由 `cudaEventDestroy` 释放的 owner wrapper；memory import 返回带
`MemoryReleaseMode::IpcClose` 的 `CudaMemory`，只允许 `cudaIpcCloseMemHandle`，普通
`cudaFree` 与 `cudaFreeAsync` 均 fail closed。public surface 只接收不可变 64-byte token 或
token+精确 allocation size descriptor，不暴露 `IntPtr`、`UIntPtr`、`SafeHandle` 或 device
pointer。

### Vendor、coverage 与 runtime

- CUDA 11.6、11.8、12.1、12.3、12.9、13.2 的 header 与 `cudart.lib` 均包含三条 symbol；
  已安装的 11.6 至 12.9 runtime DLL export 均存在。CUDA 13.2 本机没有独立 runtime DLL，仍只
  记录 header/import-library/native-build proof。
- bindings 为 `194 manifests / 3971 API records`，两次生成幂等。六套 CUDA coverage 中三条
  function 均为 `implemented-with-deferred-history`；旧 deferred manifest 保留。
- TRT10/CUDA12.9 与 TRT11/CUDA13.2 native Release build 通过；两份 PE 的新增导出均为 `3/3`。
  TRT8/10/11 ABI source parity 以及 TRT10/TRT11 PE gate 均为
  `MissingDeclarations=0 MissingExports=0`。
- TRT10/CUDA12.9 在 RTX 3060 Laptop、driver 576.02、WDDM 上完成真实父子进程 smoke：先建立
  memory/event share handle，再提交 exporter 写入并 record event；child import、event
  synchronize、64-byte 读回、反向写入、`FreeAsync` 拒绝与专用 close 均通过，父进程保持源 owner
  存活并验证 child 写入。token 经 redirected stdin 传输，不进入命令行或日志。

### Managed、package consumer 与质量门禁

- `CudaIpcExportToken.FromBytes` 精确校验 64-byte 并复制输入；
  `CudaIpcMemoryExportDescriptor` 把 token 与 allocation size 作为一个 transport unit。
  `CudaEvent` / `CudaMemory` 提供 Import/TryImport 与 `IsIpcImported`，新增
  `CudaDeviceAttribute.IpcEventSupport`。
- 最终 managed nupkg 为 14,786,429 bytes，SHA256
  `AA74FBF8AA06EBFA18A4D41782233676EAC6632891BFE2AAA325F107A3D6B2B3`；TRT10/CUDA12.9
  bridge-only nupkg 为 351,092 bytes，SHA256
  `A9B03F40DA2E9350D832352780EA14D1548EA0373256031C15B1644D6291ED8A`。无
  ProjectReference consumer restore/build 为 `0 warning / 0 error`，分类保持
  `compile-surface-proof`。
- 专项 `8/8`；CUDA bounded shard 覆盖 32 个 class、`123/123`。完整 solution Debug/Release
  均为 `0 error`，保留 5 条既有 test nullable warning。Public API documentation 与 bilingual
  finding 均为 0；DocFX `917 model(s)`，`0 warning / 0 error`。
- strict classification 与 public-proof finding 均为 0；strict release quality
  `RequiredFailureCount=0`。owner convergence 结构验证 `FailedBlockers=0`，5 个真实外部 owner
  input surface 仍保持 blocked。
- TRT10 PluginRegistryInventory、NetworkBuilder 与 InferenceBindings net8 smoke 均通过。legacy
  net48 PowerShell CUDA smoke 因当前进程 DLL 搜索路径报 `0x8007007E`，未计为通过；本批 net8
  cross-process smoke 已真实覆盖普通 allocation/free 与 imported close 两条释放路由。

### 证据边界

候选与 runtime 证据为
`cuda-ipc-import-owner-safe-candidate-audit.{json,md}` 和
`cuda-ipc-import-cross-process-runtime-evidence.{json,md}`。runtime evidence 冻结 bridge、runner
与 managed CUDA assembly hash，不保存 token/device pointer，并明确
`isPackageConsumerRuntimeProof=false`、`canPromoteRuntimeProof=false`、
`canPublishPublicly=false`。未执行 NuGet/GitHub Packages push、GitHub Release upload 或 issue
close。

## 2026-07-20 Public API Documentation Zero-Finding Closure

本批把 managed public API documentation 从“已有审计、仍有历史缺口”推进为可持续的
zero-finding contract。16 个 CUDA/TensorRT source 文件补齐或改为中英双语 XML 注释，未修改
public signature、枚举值、控制流、native manifest 或 ABI。

### Closure 与持续门禁

- compiler-reported `CS1591` 从 139 条降为 0；三个公共项目以 Release/net8.0 强制重建，均为
  `0 warning / 0 error`。
- 非中英双语 XML 元素从 24 条降为 0；重新导出的 bilingual backlog 为
  `inputFindingCount=0`、`backlogFindingCount=0`。
- `release-quality-gate.yml` 在 solution build 后运行不带 `-SkipBuild` 的
  `Test-PublicApiBilingualDocumentation.ps1`；该脚本先执行 compiler documentation audit，再检查
  XML 中的 `summary`、`param`、`returns` 与 `remarks`。documentation build 任一非零退出也会
  fail closed，不能借旧 XML 误通过。
- `Test-ReleaseQualityGate.ps1` 新增 required check
  `workflow-public-api-documentation`；专项 `PublicApiDocumentationClosureTests` 同时锁定
  `139 -> 0`、`24 -> 0`、workflow 契约与 proof boundary。

### Verification

- binding generator/output validation：`193 manifests / 3968 API records`，重复生成幂等。
- focused closure/workflow tests：`8/8`；按 CI 顺序完成 Debug build 后，N-S bounded shard
  `13/13`。
- 完整 solution Release build：`0 warning / 0 error`；Debug build：`0 error`，保留 5 条既有
  test nullable warning。
- DocFX：`916 model(s)`，`0 warning / 0 error`。
- strict classification audit finding `0`；public-proof claim boundary finding `0`；strict release
  quality `RequiredFailureCount=0`。
- 闭环证据：`public-api-documentation-closure.{json,md}`，明确
  `isRuntimeExecutionProof=false`、`isPackageConsumerRuntimeProof=false`、
  `canPublishPublicly=false`。

### C 盘与发布边界

- 未发现本批模型、包、源码副本或项目资产下载到 C 盘。
- 累计清理了本轮 `dotnet/MSBuild` 留下的 33 个 workload 日志和 30 个空 `MSBuildTemp*` 目录；
  清理后同一时间窗残留为 0。
- 保留 NuGet、Codex、CUDA、Downloads、既有 `.jyppx` 配置与 action runner；未执行 NuGet
  push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-20 CUDA Graph Memory Allocation Owner-Safe Uplift

本批将 `cudaGraphAddMemAllocNode` 与 `cudaGraphAddMemFreeNode` 从 `deferred-only` 提升为
真实 owner-safe 路径。`CudaGraphMemoryAllocation` 只保存 graph-bound bridge metadata，
device address、allocation node token、`IntPtr`、`UIntPtr` 和 `SafeHandle` 都不进入 public
surface；memset、device-to-pinned-host copy 与 free 由所属 `CudaGraph` 代为组合，并自动依赖
allocation node。跨 graph 使用、二次 free、allocation wrapper 活跃时释放 graph 都 fail
closed，generic node removal 也拒绝 memory-allocation/free node，防止绕过 wrapper。

### Vendor 与版本证据

- CUDA 11.8、12.1、12.9 的 `cuda_runtime_api.h`、`cudart.lib` 与 runtime DLL 均含
  `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode`。
- CUDA 13.2 header/import archive 含两条 symbol；本机未安装独立 cudart DLL，当前驱动也不
  支持 CUDA 13.2 runtime，因此只记录 TRT11/CUDA13.2 native build，不冒充 runtime pass。
- native 使用独立 `CUDART_VERSION >= 11040` guard；旧 deferred manifest 未删除，六套
  CUDA 11.6/11.8/12.1/12.3/12.9/13.2 coverage 行全部为
  `implemented-with-deferred-history`。

### Verification

- binding generator/output validation：`193 manifests / 3968 API records`，重复生成幂等。
- TRT10/CUDA12.9 与 TRT11/CUDA13.2 native Release build 通过；两份 bridge 的 5 条新增
  export 均为 `5/5`。TRT10/TRT11 既有 ABI surface gate 均为
  `MissingDeclarations=0 MissingExports=0`。
- 完整 solution Release build 为 `0 errors`，保留 5 条既有 test nullable warning；新增专项
  `7/7`，相邻 CUDA graph 集合 `50/50`，handle/coverage/release source 集合 `64/64`，
  bounded shard `17/17`。
- TRT10/CUDA12.9 本机真实 smoke 完成 64-byte allocation、`0x6B` memset、D2H copy、free、
  instantiate/launch/synchronize，64 个字节全部匹配；同时输出
  `GraphDisposeRejected=True CrossGraphRejected=True SecondFreeRejected=True`。
- 最终 managed nupkg 与 TRT11/CUDA13 bridge-only nupkg 重打后，无 ProjectReference consumer
  restore/build 为 `0 warning / 0 error`；probe 未请求，分类保持 `compile-surface-proof`。
- DocFX build 为 `0 warning / 0 error`；strict classification/public-proof audit finding 均为
  `0`，strict release quality required failure 为 `0`。
- 全量 public API documentation audit 仍报告 139 条既有 CUDA CS1591 warning；本批新增
  `CudaGraphMemoryAllocation` 与 graph methods 不在 finding 中。该历史审计未记为通过。
- owner input convergence 结构验证 `FailedBlockers=0`，但 5 个真实 owner input surface
  仍缺外部公开发布证据，状态保持
  `blocked-owner-input-contract-convergence-real-owner-input-required`。

### 证据与清理边界

候选审计和本地 runtime evidence 位于
`cuda-graph-memory-allocation-{candidate-audit,local-runtime-evidence}.{json,md}`。证据明确为
ProjectReference local runtime：`isPackageConsumerRuntimeProof=false`、
`canPromoteRuntimeProof=false`、`canPublishPublicly=false`，且不记录 device pointer。

本轮没有下载文件到 C 盘，Downloads 时间窗新增为 `0`。清理 66 个本轮 .NET workload
小日志、16 个空 MSBuild 临时目录、空 `C:\jyppx-pkgcache` 与空
`Temp\jyppx-split-packages`；复查均不存在。未触碰 NuGet、Codex、CUDA、Downloads 或系统
缓存；未执行 NuGet/GitHub Packages push、GitHub Release upload 或 issue close。

## 2026-07-20 YoloVision TRT8/TRT10/TRT11 本地包消费者矩阵与公开交接

本批把单一 TRT10 consumer 扩展为由 `RuntimePackageKey` 驱动的三版本矩阵。单行脚本从
`split-runtime-packages.manifest.json` 推导 bridge package ID 与 TensorRT line，显式拒绝
调用参数和 manifest 不一致；consumer 在运行前输出 bridge 的实际 TensorRT/CUDA build
identity，脚本要求 build major、请求 line 和 YoloVision output JSON 三者一致。SDK/build root
与 runtime DLL root 已分离，避免把只有 headers/import libs 的 TensorRT 目录误判为完整运行时。

`Test-YoloVisionLocalPackageConsumerMatrix.ps1` 对 TRT8、TRT10、TRT11 分别创建短路径 E 盘
workspace 和隔离 NuGet cache，逐行执行无 ProjectReference restore/build、官方 YOLOX-S engine
build、enqueue、grid/stride decode、NMS、JSON/SVG 与清理。最终矩阵为 2 pass / 1 blocker：

- TRT10/CUDA12.9：bridge `10.11.0/12.9`，5 detections，`14.360 ms`。
- TRT11/CUDA12.9：bridge `11.0.0/12.9`，使用已由 Release digest 校验的 E 盘 assembled
  runtime，5 detections，`11.679 ms`。
- TRT8/CUDA12.1：bridge `8.6.1/12.1` 已加载，但本机 cuDNN 8 developer root 中没有
  `cudnn64_8.dll`；CMake 安全门因此在 bridge build 时关闭 ONNX parser。该行只记录为
  `runtime-attempt-blocked`，prediction count 为 0，不冒充 runtime pass。

新 `Test-YoloVisionPackageSurface.ps1` 对实际 nupkg 内 DLL/XML 与 Release build hash 做一致性
检查，并反射 39 个 exported types / 368 个 public declared members。`IntPtr`、`nint`、
`UIntPtr`、`SafeHandle`、pointer 和 `JYPPX.SampleSupport.OnnxSampleOptions` 泄漏 finding 均为 0；
`YoloVisionCommand.Run` 的双语 XML 契约已补齐。YoloVision 包为 77,963 bytes，SHA256
`6823e236086dcaaee84f830a6272989a1601b96d1eae8c5e0a1d11370f0d47e0`。

路径无关 compact proof 位于
`artifacts/interface-coverage/yolox-multi-version-local-package-consumer-runtime-proof-closure.{json,md}`。
raw matrix/log/report 继续位于 ignored `artifacts/yolovision/yolox-local-package-consumer-matrix`。
proof 同时记录 5 个本地包 hash、逐行 stdout/stderr hash、bridge identity、runtime root 来源、
TRT8 blocker、surface audit 与 C 盘审计；workspace 全部删除，C 盘测试目录/命名资产匹配均为
0，已知 `C:\jyppx-pkgcache` / `C:\jyppx-split-packages` 不存在。

公开发布交接新增：

- `yolovision-public-package-owner-handoff.{json,md}`：5 个精确 package ID/version/hash、预期
  nuget.org URL、GitHub Packages source、TRT10/TRT11 clean command 与 TRT8 rebuild blocker。
- `Test-YoloVisionPublicPackageConsumer.ps1`：未来发布后使用单一公开 NuGet source、E 盘隔离
  cache、NuGet `.nupkg.metadata` source、下载 nupkg hash 和真实 YOLOX runtime 生成 proof。
- `Test-YoloVisionPublicPackageProof.ps1`：要求下载文件 hash 同时匹配 proof 与冻结 handoff，
  local feed、ProjectReference、缺字段、无 runtime marker 或 workspace 未清理均 fail closed。
  使用 local handoff 冒充公开 proof 的自测得到 48 个 failure、退出码 1。

最终验证：完整 `TensorRtSharp.sln` Release build 为 0 warning / 0 error；YoloVision consumer、
pipeline、output/schema、asset 与文档核心集合 49/49，文章/发布材料 readiness 6/6。新增 consumer
专项单独为 9/9。宽泛 `FullyQualifiedName~YoloVision` 聚合集合曾在 300 秒命令上限超时，未将
其写成通过。binding 两次生成保持 191 manifests / 3961 records 且幂等；sample asset manifest
finding 0，strict classification finding 0，strict release quality required failure 0。

本批仍是 `local-package-consumer-runtime-matrix`。成功行不是公开
`package-consumer-runtime`，TRT8 blocker 不是 runtime proof；所有
`packagesDownloadedFromPublicFeed`、owner approval、post-publish、publish/close 标志继续为
false。未执行 NuGet/GitHub Packages push、GitHub Release upload 或 issue close。

## 2026-07-20 YoloVision YOLOX 本地 PackageReference 消费者闭环

本批把已有官方 YOLOX-S 源码树真实运行推进到无 ProjectReference 的本地 NuGet consumer。
`YoloVision.csproj` 现在可打包为 `JYPPX.TensorRT.CSharp.API.YoloVision`，并公开
pointer-free 的 `YoloVisionCommand.Run(string[] args)`；原 CLI 只做薄转发，因此 package
consumer 与源码样例复用完全相同的 profile、BGR/top-left 预处理、YOLOX raw grid/stride
decoder、NMS、JSON 和 SVG 路径。YoloVision 包只依赖主 managed API，不再直接声明冗余的
CUDA ProjectReference。

仓库新增 `samples/YoloVision.PackageConsumer` 模板和
`eng/Test-YoloVisionLocalPackageConsumer.ps1`。脚本把临时工程、隔离 NuGet cache、tensor 和
运行输出放在外层 E 盘，只启用三个本地 file feed，并检查 `project.assets.json` 中 project
library 为 0。bridge-only 包通过 `runtimes/win-x64/native` 自动复制
`jyppxtrtbridge.dll`；TensorRT 10.11/CUDA 12.9 继续由系统安装提供。

最终 clean consumer restore/build 为 0 warning / 0 error，真实 YOLOX run 输出
`YoloVision Passed=True`、5 个检测、`bicycle=0.954854`、`dog=0.913407`，enqueue elapsed
为 `14.238 ms`。E 盘 restore cache 共 109 个文件、105,810,697 bytes，运行结束后整个
workspace 已删除。可提交 proof closure 位于
`artifacts/interface-coverage/yolox-local-package-consumer-runtime-proof-closure.{json,md}`；
含本机路径的 stdout/stderr/report 保持 ignored。

C 盘 Temp、Downloads、Documents、Desktop 文件名审计未发现 YOLOX/consumer 的 ONNX、
engine、tensor、PPM、labels 或 nupkg；最终删除 6 个测试重新创建的空目录，
`C:\jyppx-pkgcache` 与 `jyppx-split-packages` 均不存在。系统 CUDA、全局 NuGet、Codex 和
用户文件未删除。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed API 4.0.0 | 14,615,786 | `4bfe3c013ea2de89630b7d91cf471c082f68d1a9fe9c7e1015235a759e21b111` |
| YoloVision 4.0.0 | 77,621 | `bb13fc1574b121ab4a304bb0be3abb5efb5c543ac94ebaf791fbbf04a5d36507` |
| TRT10 bridge-only 4.0.0 | 347,493 | `13e82c8080a09754979b80108bcbdbadaf47d6f10161f71d59438477e8a13336` |

完整 solution Release build 为 0 error，保留 5 个既有 nullable warning；YoloVision/new
consumer/sample layout 专项 47/47，post-publish article/sample readiness 3/3。文章规划从
42 增到 43，并新增完整中文 PackageReference consumer 教程。bindings 两次生成均保持
191 manifests / 3961 records 且工作树状态不变；sample manifest audit 为 9 份、finding 0；
strict classification finding 0，strict release quality required failure 0。

证据分类严格保持 `local-package-consumer-runtime`：本批确实完成真实模型 runtime 和纯
PackageReference consumer，但包来自本地 file feed，不是公开 URL 下载。因此
`isPackageConsumerRuntimeProof=false`、`packagesDownloadedFromPublicFeed=false`、
`publicRedistributionOwnerApproval=false`、`canPublishPublicly=false`。未执行 NuGet push、
GitHub Packages/Release upload 或 issue close。

## 2026-07-19 TensorRtExec Builder Scalar 对齐

本轮将 `--maxNbTactics`、`--tilingOptimizationLevel`、`--l2LimitForTiling` 和 `--quantizationFlags` 接入共享 trtexec-like parser、TensorRtExec CLI/WinForms、ONNX build service、JSON/schema 与 OptionImplementationStatus。TRT10/11 的 max tactics、tiling level、L2 tiling limit 使用现有安全 `TensorRtBuilderConfig` wrapper；TRT8 输出 controlled unsupported。quantization flags 在 TRT8/10 应用并 copied readback，TRT11 输出 removed-by-vendor diagnostics。

`AppliedOptions` 只接受日志中明确的 `TrtexecBuilderScalar ... Applied=True`，因此 dry-run、setter 拒绝和版本不支持不会被误报为已应用。定向测试 34/34 通过，Tools/TensorRtExec build 为 0 warning / 0 error，schema 与 `git diff --check` 通过。dry-run artifact `artifacts/interface-coverage/trtexec-builder-scalar-precheck-proof.{json,md}` 的分类为 `precheck`，四个参数均为 parsed + parse-only，未创建 builder config、未执行 runtime，也不能晋级 real-model-runtime、package-consumer-runtime 或 release proof。C 盘本轮未发现可归因的 TensorRT 临时文件；共享 NuGet、CUDA、Codex 和系统缓存保留。

TRT10.11.0/CUDA12.9 compatible host 的真实 build-only 证据位于 `artifacts/real-case/trtexec-builder-scalar-trt10-cuda12/`。max tactics、tiling level、quantization flags 均完成 `Applied=True/ReadbackMatch=True`；L2 请求被 vendor setter 拒绝并读回 `3145728` bytes，因此保留为 parse-only。该记录包含 host metadata、report/stdout/stderr SHA256 和 copied builder snapshot，但 `InferenceRan=False`，不能替代 real-model-runtime、package-consumer-runtime 或 release proof。

跨版本补充矩阵位于 `artifacts/real-case/trtexec-builder-scalar-multi-version/`。TRT10/11 对 L2 `3MiB` 的有效值均完成 applied/readback match，同时保留 `256MiB` 受控拒绝证据；TRT8 实际创建 runtime/builder/config 并完成 quantization flags readback，现代 tactics/tiling/L2 API 按版本 guard 保持 unsupported。TRT11 quantization flags 明确记录 `RemovedByTensorRT11`。TRT8 因缺少 `cudnn64_8.dll` 只到 dependency-probe-only，TRT10/11 是 build-only；所有 report strict validator failed blocker 为 0，仍不构成 inference 或 package-consumer proof。

## 2026-07-19 BuilderConfig Deployment Readback 收口

本批将已有 `TensorRtBuilderConfig.GetDeploymentSnapshot()` 接入 `OnnxEngineBuildService`、`OnnxEngineBuildResult`、TensorRtExec CLI/WinForms 与报告 schema。`DeploymentOptions` 记录请求值，`BuilderConfigDeploymentSnapshot` 记录 builder config 成功创建后复制读回的 timing、workspace、device/DLA、flags、tactic、plugin path 和 diagnostics。该字段保持 pointer-free，不新增 native ownership API，也不删除 deferred history。

本机 dry-run 保持 `precheck`；TRT8/CUDA12.1 bridge 加载后在 vendor runtime creation 触发 structured exception `3228369022`，因此本批没有伪造真实 readback。证据分类仍为 `build-only`/`dependency-probe-only`，`isRuntimeExecutionProof=false`、`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`。详见 `artifacts/interface-coverage/builder-config-deployment-readback-proof.{json,md}`。

生成日期：2026-07-18

## 审查范围

本次审查基于项目开发总方案、本地源码 manifest、已生成的 interop 文件，以及以下本地 NVIDIA 头文件目录：

- `third_party/nvidia`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`

详细的机器可读检查清单位于：

- `artifacts/interface-coverage/tensorrt-interface-coverage.csv`
- `artifacts/interface-coverage/cuda-runtime-interface-coverage.csv`
- `artifacts/interface-coverage/interface-coverage-summary.md`
- `artifacts/api-inventory/tensorrt-api-inventory.md`
- `artifacts/interop-comparison/generated-api-coverage.md`
- `artifacts/real-case/multi-version-onnx-runtime/multi-version-runtime-evidence-matrix.json`
- `artifacts/test-analysis/project-quality-test-inventory.json`

## 2026-07-18 TRT8 Consistency Checker Vendor Symbol 审查

本轮针对 TRT8 官方 `NvInferConsistency.h` 中的 `Global::createConsistencyChecker_INTERNAL` 与 `IConsistencyChecker::validate` 做了安全提升设计审查。设计要求是 native 复制完整 engine blob、checker handle 拥有 blob、logger 在 checker 生命周期内保持 owner attachment，并以 `implemented-with-deferred-history` 记录旧 deferred manifest。

### 结论：继续 deferred

在实现进入 native build 后，TRT8 CUDA 11.8 与 CUDA 12.1 的实际 `nvinfer.lib` 均无法解析 `createConsistencyChecker_INTERNAL`。进一步使用 Visual Studio `dumpbin /symbols` 检查两套 TensorRT 8.6.1.6 import library，并使用 `dumpbin /exports` 检查对应 TensorRT DLL，均没有 `createConsistencyChecker_INTERNAL`、`Consistency` 或 `consistency` 导出。链接错误为：

```text
unresolved external symbol createConsistencyChecker_INTERNAL
```

因此本轮没有保留一个只能编译 manifest、却无法链接或无法证明真实 vendor 调用的假实现；本轮临时 manifest、native payload、托管 wrapper、smoke 与 consumer marker 均已撤回。原有 `trt8-global-create-consistency-checker-internal-deferred` 与 `trt8-consistency-checker-validate-deferred` manifest 未删除，coverage 统计保持 TRT8 `751 implemented / 96 deferred-only`，TRT10 `759 / 120`，TRT11 `813 / 88`。

### 本轮收尾验证

- Generate-Bindings 恢复为 `181 manifests / 3919 records`，生成器幂等验证通过。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA11.8、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native rebuild 全部通过；version guard 未被临时 candidate 改变。
- 完整 `TensorRtSharp.sln` Debug build 通过，0 warning / 0 error；`Start-Process -Wait` 捕获的退出码为 0。
- 恢复后的 TRT8 capability/BuilderConfig/RNNv2/plugin 专项与 coverage alias 防回归分片为 `80/80` 通过；完整 ProjectQuality 分片 run `vendor-symbol-final-20260718` 运行 `382 classes / 1284 tests`，G-M `72/72`、T-Z `167/167` 通过，A-F 与 N-S 因既有 owner/evidence artifact 生成链超过 900 秒而超时，summary 记录 `239` 条已执行且通过、`2` 个 timed-out 分片，不冒充完整 suite pass。
- coverage 显式 alias 已验证 `Global::getBuilderPluginRegistry` 与 `IPluginRegistry::getBuilderSafePluginRegistry` 为 `implemented-with-deferred-history`；TRT8/TRT10/TRT11 native ABI source declaration parity 分别为 `983/983`、`1078/1078`、`1226/1226`，三份主 bridge PE export parity missing 均为 0。
- Plugin Registry、NetworkBuilder、InferenceBindings smoke：TRT8/TRT10 全部通过；TRT11 Plugin Registry 通过、NetworkBuilder 按 vendor structured exception 受控跳过，InferenceBindings 同一 vendor runtime creation exception 退出，未被标记为通过。
- 三个 bridge-only `PackageReference` consumer restore/build 均为 0 warning / 0 error，`ProjectReference=False`，证据分类保持 `compile-surface-proof`，不提升为 runtime proof。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-22 TensorRtExec Refitted Plan Local Package Consumer Runtime

本阶段把上一批 persisted full-weight plan 从源码树运行推进到本地 NuGet package consumer。新增
`samples/RefittedPlan.PackageConsumer` 和 `eng/Test-TrtexecRefittedPlanPackageConsumer.ps1`：consumer
只引用 managed API 与 TRT10 bridge-only 两个本地包，不使用 `ProjectReference`，不从源码目录手工加载
managed assembly，也不启用 development probing。

### Package、workspace 与 owner 边界

- `NuGet.config` 只有两个声明的本地 source，nuget.org disabled；consumer 使用仓库外 E 盘工作区和独立
  `RestorePackagesPath`。`project.assets.json` 验证两个目标包均进入隔离 cache。
- persisted plan 与 float input 先复制到 consumer 工作区，执行命令只读取副本；managed assembly 与
  `jyppxtrtbridge.dll` 都必须位于 consumer output，且不位于源码树。
- consumer 通过 public `TensorRtLogger -> TensorRtRuntime -> TensorRtEngine -> TensorRtExecutionContext ->
  TensorRtInferenceBindings -> CudaStream` surface 完成 deserialize、binding、enqueue 和 readback，没有
  `IntPtr/nint/UIntPtr/SafeHandle` 或 borrowed pointer 暴露。只有所有 `using` owner 离开作用域后才输出
  `OwnerScopeExited=True`。
- 运行完成后删除整个 E 盘 consumer workspace；compact evidence 只保留 path-free hash/metadata，raw
  命令、路径和日志保留在 ignored `artifacts/package-consumer/trtexec-refitted-plan/`。

### 真实 package consumer runtime

- managed nupkg：`14,805,837` bytes，SHA256
  `9afd8486da2094027fc73bc222e61df277e406d295559c136637b5824b6125da`；TRT10 bridge-only nupkg：
  `351,091` bytes，SHA256 `24dfbdb294de81b418f9e72aaa1c265968a2f8d2b2706cd7bd39b623ffb6829d`。
- copied plan：`408,876` bytes / `5594817d...99441eb`；copied input：`3,136` bytes /
  `81f2cd77...187564`；restore/build/runtime exit code 均为 `0`。
- reload engine `IsRefittable=False`，I/O/layers/profiles 为 `2/5/1`；输入 `Input3 [1,1,28,28] / 784`
  floats，输出 `Plus214_Output_0 [1,10] / 10` floats。bindings readiness、enqueue、owner scope 和
  workspace cleanup 全部通过。
- raw output 为 `40` bytes，SHA256
  `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`，与 same-process、第二
  TensorRtExec 进程和 full-weight baseline 三路完全一致，argmax 为 `7`。
- compact evidence/validator 为
  `trtexec-refitted-plan-package-consumer-evidence/validation.{json,md}`，strict `46/46`。分类是
  `local-package-consumer-refitted-plan-runtime`；`isPackageConsumerRuntimeProof=false` 继续表示没有公开
  feed/post-publish proof。

### 质量门与 package inventory 事实

- bindings 保持 `194 manifests / 3971 records`，连续生成幂等；solution Debug `0 warning / 0 error`，
  Release 保留 5 个既有 nullable test warning / `0 error`。
- native TRT8/10/11 CUDA12 configure/build 通过；ABI declaration/PE export 分别为 `991/991`、
  `1086/1086`、`1233/1233`，missing declaration/export `0`。
- 新增测试 `4/4`，refit/persistence/package consumer 相邻集合 `45/45`；正式 shard runner 为 `6/6`，
  inventory `1460 tests / 402 classes / unassigned 0`；workflow/source-quality 集合 `22/22`。
- Public API bilingual finding `0`；DocFX `924 models / 0 warning / 0 error`。classification 与 public-proof
  finding 均为 `0`；标准 strict release gate required failure `0`。
- Owner convergence 保持 structural `9/9`、accepted `0/9`、gates `2/3`；final owner gate blocked `5`。
- 额外 `RequirePackageInventory` 强门禁未通过：TRT10 package set 只有 managed/full/bridge，缺
  `split-cuda-cudnn`、`split-tensorrt`、`split-meta`。补打时确认 CUDA12 cuDNN 9.22 资产只有 headers/
  import libs，没有所需 runtime DLL；CUDA13 目录中的 DLL 未被替换使用。因此标准 release gate 通过，
  但不得声明完整 TRT10 split package set ready。

### C/E 盘与发布边界

- C 盘 Downloads/Desktop/Documents 本轮命名匹配为 `0`；E 盘 consumer workspace 与失败 split staging
  均已删除。没有把 plan、input、output 或 package cache 写入 C 盘 consumer workspace。
- 通用 bridge consumer 清理了 `C:\jyppx-pkgcache` 子目录；该根仍为空。失败的 full split 尝试在
  `%TEMP%\jyppx-split-packages` 留下一个 0-byte 空 runtime-key 子目录；标准精确删除在执行前被工具
  策略拒绝，没有部分删除，也未换壳绕过。`%TEMP%\jybr` 为 7 月 18 日历史空目录，本轮未删除。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-22 TensorRtExec Refitted Plan Persistence 与独立 Reload

本阶段为 managed extension `--saveRefittedEngine <path>` 完成从 ONNX stripped-plan refit commit 到
持久化 full-weight plan、原 owner 释放、新 owner reload、same-process enqueue 和第二进程
`--loadEngine` 的闭环。实现复用既有 TRT10/11 owner-safe serialization config/host memory/runtime surface，
没有新增 native ABI 或 pointer-bearing public API。

### 真实问题与修复

- 首次直接调用默认 `TensorRtEngine.Serialize()` 时，TensorRT 10 生成的 plan 虽与 stripped plan 不同、
  能独立反序列化且 metadata 正常，但真实 MNIST 输出 10 个 float 全为零。这证明 artifact/hash/metadata
  gate 本身不足以证明 refitted weights 已持久化。
- 根因是 stripped engine 的 serialization config 保留 `ExcludeWeights`。最终路径创建
  `TensorRtSerializationConfig`，显式 `ClearFlag(ExcludeWeights)` 并回读 flags，再调用
  `Serialize(config)`；真实 flags 为 `3 -> 2`，`RefittableWeightsIncludedInSerialization=true`。
- 完整权重 plan 在 TRT10 reload 后 `IsRefittable=false`，但 I/O `2`、layers `5`、profiles `1`，可创建
  context 并正确 enqueue。因此 `ReloadEngineRefittable` 保留为事实诊断字段，不再作为 full-weight
  reload gate；runtime 也不重复套用 build 阶段的 refittable-state 检查。

### Runtime 与跨版本证据

- stripped plan：`459,764` bytes，SHA256
  `6f939a9a033b1b2b834362b2d94a80e5ad121cfeb96da6203605bed572d3bb5a`。
- persisted full-weight plan：`408,876` bytes，SHA256
  `5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb`；原 refitted engine 在 reload
  前释放，reload metadata/context gate 通过。
- same-process reload、独立第二 TensorRtExec `--loadEngine` 进程和固定 full-weight baseline 三路输出均为
  40 bytes，SHA256 `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`。
- TRT8 dry-run 保持 `--saveRefittedEngine` parse-only 且不写文件；non-dry exit code `2`，在 native 调用前
  命中 parser-refitter version guard。TRT11 为 `dependency-probe-only`，该选项未进入 AppliedOptions。
- compact evidence strict validator 为 `35/35`；GUI/CLI checklist 为 `21/21`。本地 persistence/enqueue
  证据仍不是 model accuracy、package-consumer runtime、public package 或 release-close proof。

### 质量门

- bindings：`194 manifests / 3971 records`，连续生成与幂等验证通过。
- solution Debug：`0 warning / 0 error`；Release：5 个既有 nullable test warning / `0 error`。
- refit/persistence focused `44/44`；application/schema/parity `29/29`；文档更新后相关集合 `47/47`。
- native TRT8/10/11 CUDA12 build 通过；ABI/PE 分别为 `991/991`、`1086/1086`、`1233/1233`，
  missing declaration/export 为 `0`。
- Public API warning `0`、bilingual finding `0`；DocFX `923 models / 0 warning / 0 error`。
- managed nupkg `14,805,622` bytes，SHA256
  `51A3680B4CF4719F0333B64B4A123849029F2311A3AC01E93E18A0D6746A5C13`；TRT10 bridge-only nupkg
  `351,088` bytes，SHA256 `09DDF0C225774EB7FAF53BFF32D9D3F3130F5AF01A8C76BACF5A27194C635EDC`。
- 无 ProjectReference consumer restore/build/probe 通过，分类保持 compile-surface-proof；classification finding
  `0`、public-proof failed blocker `0`、strict release required failure `0`。
- owner convergence 保持 structural `9/9`、accepted `0/9`、gates `2/3`，不可公开发布、不可关闭 issue。

### C/E 盘与发布边界

- C 盘 Downloads/Desktop/Documents 在本轮时间窗未发现 MNIST、TensorRT、trtexec、plan、engine 或 nupkg
  新文件；未触碰 NuGet、Codex、CUDA 或系统缓存。
- 可明确归因的 C 盘残留仅为空的 `C:\jyppx-pkgcache` 与 `%TEMP%\jyppx-split-packages`。E 盘本轮
  real-case 目录保留 3 个 ignored plan，共 `1,274,636` bytes，以及可再生 probe/readback JSON。
- 对上述精确目标执行的标准清理在执行前被工具安全策略整体拒绝；没有部分删除，也没有换壳绕过。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-22 TensorRtExec ONNX Stripped-Plan Refit Lifecycle

本阶段补齐 TensorRtExec managed extension `--refitFromOnnx <path>`，复用已有 owner-safe
`TensorRtEngine`、`TensorRtRefitter` 与 `TensorRtOnnxParserRefitter` surface，把 stripped plan
从构建、反序列化、ONNX 权重装载、engine refit commit 到 context gate 串成单一生命周期。

### Lifecycle 与安全边界

- 参数必须同时具备 `--onnx --stripWeights --refit`；当前拒绝 `--loadEngine` 组合，避免隐式猜测
  build/refit source。TRT8 dry-run 可解析为 parse-only，non-dry 在 native 调用前拒绝；TRT10/11
  才进入 parser-refitter lifecycle。
- 实际顺序固定为 stripped build、deserialize、`IsRefittable`、copied missing/all inventory、
  `RefitFromFile`、copied parser diagnostics、`RefitCudaEngine()` commit、再次检查 missing/error/
  refittable，完整成功后才设置 `ContextCreationAllowed=True` 并允许创建 execution context。
- snapshot 独立记录 `ParserRefitReturned` 与 `EngineRefitReturned`。实测证明仅 parser load 而不执行
  engine commit 会令 MNIST 输出全零，因此 `RefitCudaEngine()` 是不可省略的正确性边界。
- public surface 没有新增 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、borrowed pointer 或 native
  ownership；`OutputSha256` 与最多 64 个 float 的 comparison sample 只用于对照，不提升 generic
  output 的 proof classification。

### Runtime evidence

- TRT10.11/CUDA12.9 使用仓库已有 TensorRT MNIST ONNX，模型 SHA256
  `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`，没有下载新模型。
- refit 前后 engine 均 `IsRefittable=True`；parser load 与 engine commit 均返回 true；missing
  `0 -> 0`、all inventory `6 -> 6`、parser error `0`、copied diagnostic `0`；context gate 通过并
  完成 `[1,10]` bounded enqueue。
- refit 输出 10 floats / 40 bytes，SHA256
  `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`，与独立 full-weight
  baseline 完全一致。compact evidence strict 为 `24/24`。
- TRT8 证据为 `dry-run-precheck`；TRT11 因 `createInferRuntime` 返回 null 保持
  `dependency-probe-only`。本证据不证明 refitted plan 持久化、模型准确率、package-consumer
  runtime、public package 或 release readiness。

### Verification

- bindings 保持 `194 manifests / 3971 records`，连续生成和输出验证通过；本批没有 native manifest
  或 ABI entry 变更。
- solution Debug 为 `0 warning / 0 error`；Release 为 `0 error`，保留 5 个既有 nullable test
  warning。先前 focused/application/schema/parity/release 集合为 `42/42` 与 `71/71`；最终相关
  宽集合再次通过 `56/56`。
- native TRT8/CUDA12、TRT10/CUDA12、TRT11/CUDA12 增量构建成功；ABI declaration/PE export
  parity 为 TRT8 `991/991`、TRT10 `1086/1086`、TRT11 `1233/1233`，missing 均为 `0`。
- GUI/CLI checklist 为 `20/20`；Public API documentation warning `0`、bilingual finding `0`；
  DocFX `922 models / 0 warning / 0 error`，新文章已生成 HTML。
- managed nupkg 为 `14,805,698` bytes，SHA256
  `362FB60971660695659F8D391165A28113D5F6AE191D438973FB12C4819525CA`；TRT10 bridge-only
  nupkg 为 `351,093` bytes，SHA256
  `FF029C62AADD96798368EDEF83404895D8FA751F063805F42407EEA858998329`。无 ProjectReference
  consumer restore/build `0 warning / 0 error`，dependency probe ready，但仍为 compile-surface-proof。
- strict classification/public-proof finding 均为 `0`，strict release required failure `0`。owner
  convergence 为 structural `9/9`、accepted `0/9`、gates `2/3`；final owner gate blocked `5`。

### C/E 盘与发布边界

- C 盘 Downloads/Desktop/Documents/Temp 定向审计未发现本批 ONNX、plan、engine、nupkg、TensorRT
  或 JYPPX 下载资产。consumer 脚本已自动删除本次 restore cache 与 split-package 子目录。
- C 盘仍有本批 34 个 `.NET workload` 日志，共 `41,141` bytes、15 个空 MSBuild 临时目录，另有
  空 `C:\jyppx-pkgcache` 与 `%TEMP%\jyppx-split-packages` 根目录。标准删除命令在执行前被工具
  安全策略拒绝，未发生部分删除，未换壳绕过；NuGet、Codex、CUDA 和系统缓存未触碰。
- E 盘 real-case 目录仍有 3 个可再生 plan，共 `1,262,292` bytes；精确路径标准删除同样在执行前
  被策略拒绝。plan 不提交，保留小型 JSON evidence。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-20 CUDA IPC Export-Only Copied Token Uplift

本批将 `cudaIpcGetEventHandle` 与 `cudaIpcGetMemHandle` 提升为 export-only copied token
安全路径。native 通过 caller-owned 64-byte buffer 复制 opaque token；managed 只返回不可变
`CudaIpcExportToken` 副本，不公开 event handle、device pointer、`IntPtr`、`UIntPtr` 或
`SafeHandle`。`cudaIpcOpenEventHandle`、`cudaIpcOpenMemHandle` 与
`cudaIpcCloseMemHandle` 继续 deferred。

### Owner 与 fail-closed 边界

- event 必须由 `Interprocess | DisableTiming` 创建，否则 native 返回结构化失败。
- memory 必须是同步 `cudaMalloc` 基址；managed、async 与 pool allocation 都拒绝导出。
- token 不拥有源资源；其他进程使用 token 期间，源 `CudaEvent` 或 `CudaMemory` 必须保持存活。
- `ToArray()` 每次返回新副本，`ToString()` 只输出 kind/length；证据文件不记录 token 内容。
- import/close 涉及跨进程 owner、device affinity、peer access 和恢复策略，不在本批实现。

### Verification

- binding generator/output：`192 manifests / 3963 API records`，两次生成幂等通过。
- TRT10/CUDA12.9 与 TRT11/CUDA13.2 native build 通过；两份 DLL 均导出 event/memory
  token entrypoint。ABI declaration/export 检查均为 missing `0`。
- solution Release build：`0 warning / 0 error`；专项 ProjectQuality 与 shard runner：`7/7`。
- coverage 在 CUDA 11.6、11.8、12.1、12.3、12.9、13.2 共 6 组头文件中将两个 get
  接口标为 `implemented-with-deferred-history`；三条 open/close 行保持 `deferred-only`。
- CUDA 12.9 本机 smoke：event/memory token 均为 64 bytes，默认 event 与 managed memory
  负路径均被拒绝，`ExportOnly=True`。
- 重打本地 managed nupkg 后，TRT11/CUDA13 bridge-only package consumer restore/build
  `0 warning / 0 error`；该结果仅为 `compile-surface-proof`，probe 未请求。
- strict classification 与 public-proof boundary audit finding 均为 `0`；strict release
  quality gate required failure 为 `0`。

### 证据与发布边界

- `cuda-ipc-export-token-candidate-audit.{json,md}` 记录候选与 owner 边界；
  `cuda-ipc-export-token-local-runtime-evidence.{json,md}` 记录本机 smoke、bridge hash 和
  明确的非 proof 分类。
- 本机 smoke 是 ProjectReference local runtime evidence：
  `isCrossProcessRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`、
  `canPublishPublicly=false`，不能替代第二进程 import/lifetime proof。
- 本轮没有下载文件到 C 盘；Downloads 时间窗新增为 `0`。package consumer 创建的空
  `C:\jyppx-pkgcache` 根和 15 个已核实为空的 MSBuild 临时目录已删除。
- 未触碰 NuGet、Codex、CUDA、Downloads 或系统缓存；未执行 NuGet/GitHub Packages push、
  GitHub Release upload 或 issue close。

### 下一阶段入口

下一阶段只应从实际存在于目标 TensorRT import library 的 deferred entry 中选择候选。任何新的 consistency checker 设计必须先在所有目标 TRT8 vendor package 上完成 PE/import-library symbol proof，再进入 manifest/native/wrapper；在 symbol proof 通过前不得重新提升该接口。

## 2026-07-18 TRT11 / CUDA 12.9 Compatible-Host Runtime Proof 复审

本阶段基于起始提交 `6fb1833e21e2d62f96f45467b23316b3334f718a`，使用既有 GitHub Release `v4.0.6156` 的 TRT11/CUDA12.9 vendor 组件和当前源码构建的 bridge，完成 TRT11 真实 builder、parser、engine round-trip、enqueue 与 package runtime consumer 证明。Release 资产只读下载和校验；没有执行 package push、Release upload 或 issue close。

### Plugin Registry 与 ONNX 兼容处理

- `TensorRtEnvironmentProbe` 的 global/capability registry inventory 与 creator lookup 新增 `includeCreatorFields` overload；旧 overload 固定转发 `true`，保持既有完整 inventory 行为。
- TRT11 内置 V3 creator 的 field hook 仅供 parser 使用，直接枚举会输出 vendor `Unexpected Internal Error`。TRT11 Plugin Registry smoke 现在传入 `false`，只复制 creator identity、interface 与 API language；显式字段 API、TRT8/TRT10 完整字段路径继续保留。
- `OnnxToEngineSmokeRunner` 对 TRT11 已移除的 `PlatformHasFastFp16`、`PlatformHasFastInt8`、`PlatformHasTf32` 查询单独捕获 `BridgeStatusCode.NotSupported`，输出 `Unavailable:NotSupported` 后继续 parse/build，而不是把版本差异误判为 runtime 失败。
- public surface 仍只返回 copied managed snapshot，不公开 `IntPtr`、`nint`、`SafeHandle`、device/plugin/tensor pointer；plugin create/register/deregister/load、callback、allocator/resource 与 borrowed pointer 继续 deferred。

### Release 资产与真实运行

三个既有 Release 包的本地 SHA256 与 GitHub Release digest 完全一致：

| 角色 | 大小 | SHA256 |
| --- | ---: | --- |
| base | 3,923 bytes | `F1E1E896B5066472DD900CBD830950781E2215D967BA47BC97B3D103377FD0F3` |
| TensorRtRuntime | 258,004,802 bytes | `93E8CA4FD0B95CFB49C3E2CDC6BB94AFA8126853BE66D0B5013D60875C325A2C` |
| TensorRtBuilder.Sm75Sm86 | 449,336,913 bytes | `FE26D320160AF0B8CCD76F044429CE106DF8D89FE89B62859693FBC80FCD79CB` |

- `PluginRegistryInventorySmokeRunner`：通过，`CreatorFieldCollection Included=False`，日志不含 `Unexpected Internal Error`。
- `NetworkBuilderSmokeRunner`：通过，identity engine build/serialize、`Enqueue=True`、`OutputMatch=True`。
- `OnnxToEngineSmokeRunner`：通过，parser/config owner lease、engine 文件/流 round-trip、refitter diagnostics、enqueue 与 output compare 均成功。
- `InferenceBindingsSmokeRunner`：`ExecuteV2` 与 `EnqueueV3` 均 `OutputMatch=True`。
- 仓库外纯 `PackageReference` consumer 使用本轮 managed + TRT11 bridge-only 包：`createInferRuntime` 返回非空、engine 2,580 bytes、enqueue 完成、identity output match。bridge DLL 为 777,728 bytes，SHA256 `7B8C4311A68CF8A4D2C81F4F9EE8A7381D8F4178F29DD71FB3064A1FF5EFD4EE`。

机器可读证明由 `eng/Export-Trt11Cuda129CompatibleHostProof.ps1` 生成：

- `artifacts/real-case/trt11-cuda12-compatible-host-proof/trt11-cuda12.9-compatible-host-proof.json`
- `artifacts/real-case/trt11-cuda12-compatible-host-proof/trt11-cuda12.9-compatible-host-proof.md`
- `artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.json`

状态为 `passed-compatible-host-engineering-proof` / `compatible-host-bridge-package-runtime`，`isRuntimeExecutionProof=true`。由于 managed/bridge 均来自本地 feed，`isPackageConsumerRuntimeProof=false`、`canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`；该证明不能替代 public clean consumer 或 post-publish proof。

### Coverage、构建、测试与包

- bindings 生成与幂等验证保持 181 manifests / 3919 records；coverage 为 TRT8 `751/96`、TRT10 `759/120`、TRT11 `813/88`，CUDA 六个版本行为 `211/57`、`216/57`、`219/58`、`229/63`、`241/66`、`257/73`。
- 完整 `TensorRtSharp.sln` Release build 为 0 warning / 0 error。TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 五套 native 和 TensorRT ABI parity 6/6 全部通过。
- Plugin Registry、BuilderConfig、RNNv2、ONNX、consumer、ABI 受影响分片 83/83；ProjectQuality inventory 为 1284 tests / 382 classes / unassigned 0，新增类 4/4，累计 hash-verified coverage 为 382/382、185 份有效 TRX、missing 0、invalid evidence 0。既有 one-shot 已知超过命令上限，本阶段不冒充 one-shot pass。
- managed 包和 TRT8/TRT10/TRT11 bridge-only 包均重新打包；三个临时 consumer 只使用 `PackageReference`、无 `ProjectReference`，restore/build 为 0 warning / 0 error，并编译所有 `includeCreatorFields` overload。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,697,081 bytes | `86165D8EA577BD6BC662246591540B3FC19255EB67668BDF8060F0A51FE79755` |
| TRT8/CUDA12.1 bridge-only | 313,595 bytes | `704A23DF72455E2FC24C0656918E39150E8C9622ACD7BC2D1E8DA24081B8500F` |
| TRT10/CUDA12.9 bridge-only | 339,349 bytes | `54B5377746582B1EEF591CF5735D30DCA9060DD7C984D681D1399DDB462C83BE` |
| TRT11/CUDA12.9 bridge-only | 268,769 bytes | `EC5371BDB0CD47CE938E232F0139975C13848A66F5D99CE5C4ECB1AE44A24B26` |

### Gate 与发布边界

strict classification 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；要求 classification 的 strict release gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3、validation blocker 0；所有 publish/upload/close 标志继续为 false。

## 2026-07-18 CUDA Managed-Memory Location V2 复审

本阶段基于起始提交 `ca0d5907e7e864433888e756ee00645faf735a53`，提升 CUDA 12.3/12.9 的 `cudaMemAdvise_v2` 与 `cudaMemPrefetchAsync_v2`。CUDA 13.2 已将相同 location 语义迁移到非 `_v2` 名称，因此 bridge 保持统一 owner-safe ABI，在 native 内按 Toolkit 版本独立选择 vendor 调用。旧 deferred manifest 全部保留。

### Owner、Location 与版本边界

- 新增 `CudaMemoryLocationKind` 与不可变 `CudaMemoryLocation`，只允许 `Device`、`Host`、`HostNuma`、`CurrentHostNuma` 四类规范位置。device/NUMA id 必须非负，忽略 id 的位置固定为 0；默认无效 struct 会在调用前被拒绝。
- 两个公开 overload 只位于 `CudaManagedMemory`，接受现有 managed-memory owner、offset/count 与 `CudaStream` owner，不公开 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr` 或 device pointer。异步 prefetch 文档要求 memory/stream 保持存活到同步完成。
- native 会校验 handle kind、`MemoryObject::is_managed`、range、location、advice 与 accessed-by/location 组合；flags 固定为 `0U`。C++ exception 和 Windows SEH 均转换为 bridge status。
- CUDA 12.3-12.9 调用 `cudaMemAdvise_v2` / `cudaMemPrefetchAsync_v2`；CUDA 13.x 调用 location 形态的 `cudaMemAdvise` / `cudaMemPrefetchAsync`；CUDA 11.8/12.1 明确返回 `NotSupported`。callback、allocator/resource、plugin mutation、裸/borrowed pointer 继续 deferred。

### Coverage、构建与测试

generator 为 181 manifests / 3919 records，bindings 生成和幂等验证通过。两条接口在 CUDA 12.3 与 12.9 的四个版本行均为 `implemented-with-deferred-history`，匹配 real safe entry 与原 deferred entry：

| CUDA Toolkit | 扫描接口 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: |
| 11.6 | 268 | 211 | 57 |
| 11.8 | 273 | 216 | 57 |
| 12.1 | 277 | 219 | 58 |
| 12.3 | 292 | 229 | 63 |
| 12.9 | 307 | 241 | 66 |
| 13.2 | 330 | 257 | 73 |

- `JYPPX.CudaSharp` 全目标框架 Release build 与完整 `TensorRtSharp.sln` Release build 均为 0 warning / 0 error。
- 新专项、相邻 memory-range/batch 与 package-consumer 分片 16/16。正式 ProjectQuality inventory 为 1280 tests / 381 classes / unassigned 0；本轮 bounded shard 16/16，累计 hash-verified coverage 为 381/381、184 份有效 TRX、missing 0、invalid evidence 0。
- one-shot 完整 ProjectQuality Tests 在 604 秒命令上限内未返回最终结果，因此不记为通过；可审计结论使用 bounded shard 与累计类覆盖。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 五套 native 均成功。TensorRT 静态声明与三条主 DLL 的 PE export parity 继续为 missing declaration 0、missing export 0。

### Runtime、NuGet 与 Consumer

- CUDA 12.9 真实 smoke 加载新 entry 并完成 `cudaMemAdvise_v2`。本机 RTX 3060 Laptop GPU 为 `ConcurrentManagedAccess=False`，location prefetch 已进入 NVIDIA runtime 后受控返回 `cudaErrorInvalidDevice(101)`；输出明确记录 `Advise=True`、`PrefetchAttempted=True`、能力约束与 sticky error 清零，不冒充 prefetch success。
- TRT10/CUDA12.9 Plugin Registry inventory/lookup/parent-search、NetworkBuilder 和 InferenceBindings 真实通过；identity engine 的 ExecuteV2、EnqueueV3 与 output compare 匹配。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 包已重打。三个无 ProjectReference、纯 PackageReference consumer restore/build/validation 全部通过，新 location 类型与 overload 已进入 compile surface。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,403,087 bytes | `BB3712F0D961BAECCB9CE859D6CFB806F49AB0E35BEFAD749B533EC88000E285` |
| TRT8/CUDA12.1 bridge-only | 313,595 bytes | `1E4F73C3C5EB41C3EEDA2D8323CB8DA9A930B96311BA265FEF9C2DA09D19CE93` |
| TRT10/CUDA12.9 bridge-only | 339,345 bytes | `BA41765A70196BB87FE8288EA1F936147C094E065D7B6D39936C9C8DA45EB37C` |
| TRT11/CUDA13.2 bridge-only | 276,838 bytes | `2AADE3A2822BABD84761CB152968164B787A954F12E1956EF831426157008D48` |

### Gate 与发布边界

strict classification 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；strict release gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3、validation blocker 0；`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-18 TensorRT Native ABI Export Parity 复审

本阶段基于起始提交 `b991f6419aca394159292c3bfc39b960bd88abbf`，收敛 manifest、公共 C 头声明与 Windows PE export 三者的一致性。修复范围只覆盖已有安全实现和 managed wrapper、但公共头缺少 `JYPPX_C_API` 声明的入口；没有扩展 callback、plugin mutation、allocator/resource acquire/release 或 pointer-bearing public API。

### ABI 与实现收敛

- TRT8、TRT10、TRT11 公共头分别补齐 17、31、14 个声明，共 62 个。新增 `eng/Test-TensorRtNativeAbiSurface.ps1`，默认验证三版本全部 manifest entry；可选 `-BridgePath` 后通过 `dumpbin /exports` 逐项校验真实 DLL export。
- 门禁使用严格 token 匹配，并兼容既有 `*_DECL(entry)` 声明宏，避免把 `lookup` 与 `lookup_get_*` 等子串误判为同一入口。GitHub workflow 已接入静态 ABI validation 并上传 `artifacts/native-abi/**`。
- PE 检查暴露 TRT11 refitter 两个 copied diagnostics entry 虽有 manifest/wrapper、却未进入 source 编译。现已补齐 `get_error_recorder_snapshot_info` 与 `get_error_recorder_error`：校验 refitter owner，复制 count/interface/error code/description，校验 index，并包含 C++ exception 与 Windows SEH containment；不暴露 recorder pointer。
- TRT11 Plugin Registry runner 只把 `BridgeProbeException` 的 `RuntimeError + "returned a null TensorRT object."` 明确签名视为 compatible-host skip。`EntryPointNotFoundException` 等 ABI 回归仍会失败，防回归测试已锁定该边界。

### Coverage、构建与测试

generator 保持 180 manifests / 3917 records，bindings 生成和幂等检查通过。coverage 分类未被 ABI 声明修复错误改变：TRT8 为 751/96、TRT10 为 759/120、TRT11 为 813/88；TRT8 两条 builder registry alias 仍为 `implemented-with-deferred-history`。

- 完整 `TensorRtSharp.sln` Release build 为 0 warning / 0 error。
- ABI、Plugin Registry、BuilderConfig、RNNv2、Refitter、workflow、public handle exposure 等受影响合并分片 68/68 通过。
- 正式 ProjectQuality inventory 为 1275 tests / 380 classes / unassigned 0；11 个受影响类逐类 bounded run 全部通过，累计 hash-verified coverage 为 380/380、183 份有效 TRX、missing 0、invalid evidence 0。既有 one-shot 会被共享 evidence 长尾与文件竞争影响，本阶段不冒充 one-shot pass。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 五套 native 均构建成功；最终三条主 DLL 的 manifest export parity 为 TRT8 983/983、TRT10 1078/1078、TRT11 1226/1226。旧 TRT10 DLL 的 1047/1078 与 31 项缺口证据保留在 `artifacts/native-abi/trt10-before-rebuild.json`。

### Runtime、NuGet 与 Consumer

- TRT8/CUDA12.1 与 TRT10/CUDA12.9 的 Plugin Registry、NetworkBuilder、InferenceBindings 已真实通过；TRT10 OnnxToEngine 原 `jyppx_trt10_builder_plugin_registry_exists` 缺失导出已消失，parse/build/serialize/deserialize/refitter diagnostics/enqueue/output compare 完成。
- TRT11/CUDA13.2 可读取 global/capability plugin inventory；runtime 与 builder 创建因本机 CUDA error 35 返回 null，runner 分别输出精确 `Skipped=True`。NetworkBuilder 记录受控失败，OnnxToEngine 记录受控 skip；不晋级 TRT11 runtime proof。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 包已重打；三个纯 `PackageReference` consumer 无 `ProjectReference`，restore/build/validation 全部通过，分类保持 `compile-surface-proof`。
- TRT10 bridge package runtime consumer 真实完成 CUDA preflight、8652-byte identity engine、enqueue 与 output compare，`IdentityOutputMatch=True`。该本地 compatible-host runtime evidence 仍不能替代公开包或 post-publish proof。
- TRT11/CUDA13.2 split bridge/CudaCudnn/TensorRt/meta 四角色齐全，package inventory 为 `packageSetReady=true`、`missingSplitRoles=[]`、`sha256Ready=true`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,376,358 bytes | `4BEE8ADB329CC9AF502BECA143D994ADED4FA8461980D34E225EA6EF2C631C30` |
| TRT8/CUDA12.1 bridge-only | 312,356 bytes | `D5190A5E3F8FC937BDE22FD83E99A6B20EDE72662DF46B893D67E03B95BAF307` |
| TRT10/CUDA12.9 bridge-only | 338,407 bytes | `0CA750BA128946BAFE4222CEC25594E4BDC02301FF2CBCBEA06C97A34B489886` |
| TRT11/CUDA13.2 bridge-only | 276,098 bytes | `CBAC36D30FC47082478E31EA6FF4684831D0ECEDC4915E7EA14CD734304043C9` |

### Gate 与发布边界

strict classification 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；要求 package inventory/classification 的 strict release gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3；`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-18 TRT11 ONNX Parser Builder-Config Attachment 复审

本阶段基于起始提交 `4531b04a49a8a9e11171aefb82b6f0c350229465`，提升 TRT11 `IParser::setBuilderConfig`，并补齐官方 `REPORT_CAPABILITY_DLA=2`、`ENABLE_PLUGIN_OVERRIDE=3`、`ADJUST_FOR_DLA=4` parser flag。旧 `parser-set-builder-config-deferred` manifest 保留，coverage 通过显式 real/deferred-history alias 归并，不以删除历史 deferred 改变统计。

### Owner、ABI 与部署路径

- public `TensorRtOnnxParser.SetBuilderConfig(TensorRtBuilderConfig)` 只接受现有 owner。调用前创建 `SafeTensorRtObjectHandleLease`；vendor 返回 `true` 后才替换旧 lease，返回 `false` 或抛错时释放新 lease 并保留旧关联。
- parser 释放顺序为 native parser、builder-config lease、initializer pins；同一把锁串行化 attachment 与 dispose。parser/config 版本线不一致时 fail closed，TRT8/TRT10 明确 `NotSupported`。
- native 同时校验 parser/config handle kind 与 TRT11 line；vendor 调用被 C++ exception 和 Windows SEH containment 包围。public API 不暴露 `IntPtr`、`nint`、`SafeHandle`、device/plugin/tensor pointer。
- `OnnxEngineBuildService` 的 TRT11 DLA 路径先设置 default device/DLA core/GPU fallback，再把 config 关联到 parser，并启用 capability report 与 DLA adjustment；ONNX smoke 和纯 package consumer 同步覆盖该强类型 surface。
- callback trampoline、plugin create/register/deregister/load、allocator/resource acquire/release、裸 pointer、borrowed plugin/tensor 与 ownership 不明确的 handle 继续 deferred。

### Coverage、构建与测试

generator 最终为 180 manifests / 3917 records。`IParser::setBuilderConfig` 在 TRT11/CUDA12.9 与 TRT11/CUDA13.2 两行均为 `implemented-with-deferred-history`：

| TensorRT line | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 | 847 | 847 | 751 | 96 |
| TRT10 | 879 | 879 | 759 | 120 |
| TRT11 | 901 | 901 | 813 | 88 |

- bindings 生成与幂等检查通过；完整 `TensorRtSharp.sln` Release build 为 0 warning / 0 error。
- 新专项 7/7，通过 ONNX/parser/coverage/tool/package-consumer 相关分片 97/97。正式 ProjectQuality inventory 为 1268 tests / 379 classes；受影响 bounded shard 7/7，累计 hash-verified coverage 为 379/379、missing 0、invalid evidence 0。
- one-shot 完整 ProjectQuality Tests 运行 3600 秒后超时，没有最终计数；残留 testhost/dotnet 已清理。本审查只声明可审计的 bounded shard 全类覆盖，不把 one-shot 尝试写成通过。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 五套 native 均基于最终生成物构建成功，version guard 独立。

### Runtime、NuGet 与 Consumer

- TRT8 与 TRT10 Plugin Registry、NetworkBuilder、InferenceBindings 真实运行通过；identity engine 的 `ExecuteV2`、`EnqueueV3` 和 output compare 匹配，TRT8 额外完成 `EnqueueV2`。
- TRT11 Plugin Registry runner 安全完成，但 vendor SEH `3228369022` 使 inventory 受控 skip；NetworkBuilder 与 OnnxToEngine 在相同 runtime/builder preflight 阻断。因此本机没有把新 attachment 晋级为 TRT11 runtime proof。
- TRT10 OnnxToEngine 被既有缺失导出 `jyppx_trt10_builder_plugin_registry_exists` 阻断；其他 TRT10 registry/network/inference smoke 已通过，该问题不属于本批 attachment。
- managed 4.0.0 与 TRT8/TRT10/TRT11 bridge 包已重打。三个纯 `PackageReference` consumer 无 `ProjectReference`，restore/build 为 0 warning / 0 error，编译 attachment 与 parser flags；分类保持 `compile-surface-proof`、`Runtime execution proof=False`。
- TRT11/CUDA13.2 bridge、CudaCudnn、TensorRt、meta 四角色已恢复，package inventory 为 `packageSetReady=true`、`missingSplitRoles=[]`、`sha256Ready=true`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,376,700 bytes | `30BE9A82AE44D9619FA1999DB1577F56AD22134C658FEA16040909DB84606B18` |
| TRT8/CUDA12.1 bridge-only | 306,080 bytes | `FB27BD71C1ABCB96475CB2A689AC2ED7736D30DDE5B91BCA82711F3C3932901E` |
| TRT10/CUDA12.9 bridge-only | 330,494 bytes | `373F428BF2299D091AA816392C958BB49EAAE2660DEB61243A60ADAB6B21091D` |
| TRT11/CUDA13.2 bridge-only | 274,146 bytes | `3C5E806E9FFC73D3E05D5949914665F14D23184D41823D0F1E9164DAACFB9046` |

### Gate 与发布边界

strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；要求 package inventory/classification 的 strict release gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3、validation blocker 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-18 CUDA Kernel Library Symbol 与 Attribute Owner-Safe 复审

本阶段基于提交 `1261c080b8a6af34e141b4dfea923519f89acd4c`，提升 CUDA 12.9/13.2 的 `cudaLibraryGetGlobal`、`cudaLibraryGetManaged`、`cudaLibraryGetUnifiedFunction` 与 `cudaKernelSetAttributeForDevice`。实现复用现有 bridge-owned `CudaKernelLibrary`，只公开 copied size、存在性与 owner-bound setter；旧 deferred manifest 全部保留。

### Pointer、错误与版本边界

- `TryGetGlobalSymbolSize` 与 `TryGetManagedSymbolSize` 让 native 将 vendor `dptr` 参数设为 null，只复制 `size_t`；公开面只返回 `bool` 与 `ulong`。
- `ContainsUnifiedFunction` 仅在 native 调用栈内检查 function pointer 是否非空，不保存、不调用、不跨 ABI。CUDA 12.9 对 raw PTX library 的 missing unified function 实际返回 `cudaErrorInvalidValue`；高层保留异常，smoke 受控消费 sticky error 后确认 `AfterClear=0`，不伪造 `false`。
- `SetAttributeForDevice` 接受 library owner、kernel name、7 个允许修改的 `CudaKernelAttribute`、value 与 device ordinal。native 临时调用 `cudaLibraryGetKernel`，setter 返回后立即丢弃 borrowed kernel。
- public API 不暴露 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device/plugin/tensor/function pointer；字符串使用调用期 UTF-8 buffer。C++ exception 与 Windows SEH 均在 C ABI 内转换。
- 真实 vendor 调用只在 `CUDART_VERSION >= 12090` 编译；CUDA 11.8 smoke 明确得到 `CudaKernelLibrary Skipped=True VersionGuard=NotSupported Runtime=11080`。callback、resource acquire/release、raw pointer 与 ownership 不明确的 handle 继续 deferred。

### Coverage、构建与测试

generator 最终为 179 manifests / 3916 records。四个目标函数在 CUDA 12.9/13.2 的 8 行全部为 `implemented-with-deferred-history`：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 211 | 57 |
| 11.8 | 273 | 273 | 216 | 57 |
| 12.1 | 277 | 277 | 219 | 58 |
| 12.3 | 292 | 292 | 227 | 65 |
| 12.9 | 307 | 307 | 239 | 68 |
| 13.2 | 330 | 330 | 257 | 73 |

- bindings 连续生成与幂等检查通过；完整 `TensorRtSharp.sln` Release build 为 0 warning / 0 error。
- API inventory 中四个新 safe entry 的 manifest/source 双向缺口均为 0。全仓仍有 224 条历史 deferred/声明型 manifest 无 source，未删除或误记为本批回归。
- 受影响 ProjectQuality 最终分片 65/65 通过。正式 inventory 为 1261 tests / 378 classes；累计 hash-verified 类覆盖 378/378、missing 0。上一阶段 one-shot 已证明共享 final-release evidence 长尾会互相干扰，本轮不把该不稳定路径写成完整套件通过。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9 与 TRT11/CUDA13.2 五套 native 均基于最终生成物构建成功。

### Runtime、NuGet 与 Consumer

- CUDA 12.9 raw PTX/data 与 file library 均真实查询 global size `4`；missing global/managed 为 false，attribute setter 成功，最终 last error 为 0。
- TRT8/CUDA12.1 与 TRT10/CUDA12.9 的 Plugin Registry、NetworkBuilder、InferenceBindings 均通过；identity engine 的 build/serialize/deserialize、`ExecuteV2`、`EnqueueV2`/`EnqueueV3` 与 output compare 匹配。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 包已重打。三个纯 `PackageReference` consumer 无 `ProjectReference`，restore/build 为 0 warning / 0 error，强类型编译四个新方法与 7 个 enum 值；分类保持 `compile-surface-proof`、`Runtime execution proof=False`。
- TRT11/CUDA13.2 的 CUDA/cuDNN、TensorRT 与 meta 三角色从保留的 full-runtime nupkg 恢复，bridge 角色保留本阶段当前 build-out 产物；package inventory 为 `packageSetReady=true`、`missingSplitRoles=[]`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,356,996 bytes | `1BBD555C4D7336DA81CA4C3A77A66A4344BBF37645B19674928EC1B581DDDB6D` |
| TRT8/CUDA12.1 bridge-only | 306,080 bytes | `4AD06C17AA7F23FF00E745B1714EBB02A6A7DC2DA96C1203A980EBF2A3108489` |
| TRT10/CUDA12.9 bridge-only | 330,492 bytes | `3441B0BF4A96B5EB36396088D945C752C3B17428AC78F6E971BB0A98E70DDE68` |
| TRT11/CUDA13.2 bridge-only | 273,892 bytes | `56EDA4CD1DCD645651A27CECBBE2CB355663E1B1F0A718DA99A1F4EACA880885` |

### Gate 与发布边界

strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；带 package inventory/classification 要求的 strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3、validation blocker 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Managed-Memory Batch Owner-Safe 复审

本阶段基于起始提交 `e97a29f57c1e7c20539ce46b28ed97f51d6d119f`，提升 CUDA 13 新增的 `cudaMemPrefetchBatchAsync`、`cudaMemDiscardBatchAsync` 与 `cudaMemDiscardAndPrefetchBatchAsync`。三条接口复用现有 `CudaManagedMemory` 和 `CudaStream` owner，不暴露 device pointer，并保留全部旧 deferred manifest。

### Owner、ABI 与版本边界

- native 新增 `JYPPX_CudaManagedMemoryBatchRange` caller array。bridge 在调用栈内校验 owner、managed allocation 类型、offset/count、目标设备和 stream，再临时构造 pointer/size/location/location-index 数组；flags 固定为 `0ULL`。
- `MemoryObject::is_managed` 区分 `cudaMallocManaged` 与普通、async、memory-pool device allocation，三条 batch entry 会拒绝非 managed owner。
- managed 新增 `CudaManagedMemoryRange`、`CudaManagedMemoryPrefetchRange` 与 `CudaManagedMemoryBatch`。internal interop 对每个 `SafeCudaMemoryHandle` 建立调用期 lease，固定 descriptor array，并在 finally 中逆序释放；异步提交后由公开文档要求调用方保持 memory owner 与 stream 存活到同步完成。
- public API 不含 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device/plugin/tensor pointer。native 将 `std::bad_alloc`、其他 C++ exception 与 Windows SEH 转换为 bridge status，不允许异常跨 C ABI。
- 真实 vendor 调用仅在 `CUDART_VERSION >= 13000` 编译；CUDA 11/12 明确返回 `NotSupported`。callback、external/resource pointer、allocator acquire/release、generic tagged union 与 ownership 不明确的 handle 继续 deferred。

### Coverage、构建与测试

generator 最终为 178 manifests / 3912 records。三条 CUDA 13.2 行均为 `implemented-with-deferred-history`，旧 CUDA 版本不生成伪匹配：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 211 | 57 |
| 11.8 | 273 | 273 | 216 | 57 |
| 12.1 | 277 | 277 | 219 | 58 |
| 12.3 | 292 | 292 | 227 | 65 |
| 12.9 | 307 | 307 | 235 | 72 |
| 13.2 | 330 | 330 | 253 | 77 |

- bindings 生成与幂等通过；完整 solution Release build 成功，0 error，保留 5 条仓库既有 nullable warning。
- API inventory 的三条新 entry 均为 manifest/source 双向缺口 0。全仓 inventory 仍报告 224 条历史 deferred/声明型 manifest 无 source，属于既有口径，未将其误记为本批回归。
- 受影响 ProjectQuality 分片 40/40 通过，覆盖新 batch、memory range、Plugin Registry、BuilderConfig、TRT8 setter 与 RNNv2。正式 inventory 为 1251 tests / 376 classes；累计 hash-verified 类覆盖 376/376，`-MissingOnly` 四分片均为空集。
- one-shot 完整 ProjectQuality 运行约 55 分钟后出现共享 `artifacts/final-release` 的长尾竞态并被终止，已观察的 9 个失败均位于 owner/final-release evidence 竞态或真实 owner blocker；本轮不宣称 one-shot 完整套件通过。
- native 的 TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 与额外 TRT11/CUDA12.9 均构建成功，CUDA 13 batch symbol 未泄漏到旧 header/version guard。

### Runtime、NuGet 与 Consumer

- CUDA 12.9 smoke 实际调用 typed batch surface并得到 `VersionGuard=NotSupported Runtime=12090`。CUDA 13.2 bridge 在当前仅支持 CUDA 12.9 的驱动主机上于 `cudaRuntimeGetVersion` 返回 error 35，保持 compatible-host blocked。
- TRT8/CUDA12.1 与 TRT10/CUDA12.9 的 Plugin Registry、NetworkBuilder、InferenceBindings 均通过，identity build/serialize/deserialize/enqueue/output compare 匹配。TRT11/CUDA13 runtime 创建返回空对象；TRT11/CUDA12.9 受控记录 Windows SEH `3228369022`，均不伪造 runtime proof。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 本地包已重打。三个纯 `PackageReference` consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error，编译两个 typed range 与三个 batch 方法；proof 分类保持 `compile-surface-proof`、`Runtime execution proof=False`。
- 为完整 strict package inventory 恢复了 TRT11/CUDA13.2 四角色本地 split set；最终从当前源码重新构建 TRT11 bridge 后覆盖 bridge 角色，避免旧 full-runtime nupkg 中的 bridge 资产回流。`packageSetReady=true`、`missingSplitRoles=[]`、`sha256Ready=true`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,317,043 bytes | `85F1C0DD0638523380B6B23FE0318C6B125AB093A2CFEDD32BE1FC664216AD3E` |
| TRT8/CUDA12.1 bridge-only | 305,544 bytes | `D128F0B2DBA8ED321A12C89848955FBCE57CA19C279CE3427556B7C8D4335500` |
| TRT10/CUDA12.9 bridge-only | 329,492 bytes | `BCA75FE283819D76E7B5CA4BF56CF1E453CF76BCED6C1C4AE3501FDCA04C919A` |
| TRT11/CUDA13.2 bridge-only | 272,826 bytes | `CBA3848FCF0260A8199FDCE4874A2FC736CC35AFFD9C81EB17D8B509B9B22209` |

### Gate 与发布边界

strict release quality gate 在 `-RequirePackageInventory -RequireClassificationAudit` 下为 `release-quality-gate-passed`、required failure 0；classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0。Owner convergence 保持 structural 9/9、accepted 0/9、gates 2/3、validation blocker 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Primary Execution Context Owner-Safe 复审

本阶段基于起始提交 `bfed406d246610046896ed41385f61c043965929`，从 CUDA 13.2 的 execution context、device resource、graph 与 kernel-library 候选中审查 21 条边界，只提升 7 个能够复用稳定 owner 的 primary execution context API：`cudaDeviceGetExecutionCtx`、`cudaExecutionCtxGetDevice`、`cudaExecutionCtxGetId`、`cudaExecutionCtxSynchronize`、`cudaExecutionCtxStreamCreate`、`cudaExecutionCtxRecordEvent` 与 `cudaExecutionCtxWaitEvent`。

### Owner 与 ABI 边界

- native 新增 bridge-owned `ExecutionContextObject`。该对象只包装 `cudaDeviceGetExecutionCtx` 返回的设备主上下文；释放 entry 只删除 bridge wrapper，绝不调用 `cudaExecutionCtxDestroy`。官方头文件明确指出，对该 primary context 调用 destroy 属于未定义行为。
- managed 新增 internal `SafeCudaExecutionContextHandle` 与 public `CudaPrimaryExecutionContext`。公开面只提供 `IsPrimary`、copied device/id、同步、创建 bridge-owned stream、record/wait 现有 event；不暴露 `cudaExecutionContext_t`、`IntPtr`、`nint`、`SafeHandle`、device pointer 或 plugin/tensor pointer。
- stream 复用现有 `CudaStream` owner；event 复用现有 SafeHandle 且只在 native 调用栈内借用。native entry 保持 C++ exception containment 与 Windows SEH guard。
- CUDA 13 使用独立 `CUDART_VERSION >= 13000` guard；CUDA 11/12 明确返回 `NotSupported`。`cudaExecutionCtxDestroy`、green context、device resource、graph generic tagged union、library global/managed/unified pointer 等旧 deferred 全部保留。

### Coverage、构建与测试

generator 最终为 177 manifests / 3909 records。7 条 CUDA 13.2 目标行均为 `implemented-with-deferred-history`：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 211 | 57 |
| 11.8 | 273 | 273 | 216 | 57 |
| 12.1 | 277 | 277 | 219 | 58 |
| 12.3 | 292 | 292 | 227 | 65 |
| 12.9 | 307 | 307 | 235 | 72 |
| 13.2 | 330 | 330 | 250 | 80 |

- bindings 生成与幂等通过；完整 solution Release build 为 0 warning / 0 error。
- 受影响专项 87/87 通过，覆盖 owner-safe 专项、coverage alias、public handle exposure、bridge consumer、Plugin Registry、BuilderConfig、RNNv2 与 CUDA 13 version guard。
- ProjectQuality Debug inventory 为 1251 tests / 376 classes。新增类独立补跑 5/5 通过；累计 hash-verified bounded shard 覆盖为 376/376、169 份有效 TRX、missing 0、invalid evidence 0。该结果是仓库正式 bounded shard 全类覆盖，不冒充会并发改写共享 evidence 且耗时数小时的 one-shot 单进程通过。
- native 的 TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 四套预设均重新 configure/build 成功，version guard 彼此独立；保留仓库既有 native warning 边界。

### Runtime、NuGet 与 Consumer

- CUDA 12.9 `CudaSmokeRunner` 真实运行通过，并输出 `CudaPrimaryExecutionContext Skipped=True VersionGuard=NotSupported Runtime=12090`。CUDA 13.2 在当前 CUDA 12.9 driver 主机最早的 `cudaRuntimeGetVersion` 返回 error 35，保持 compatible-host blocked，不伪装为 execution-context runtime proof。
- TRT8/TRT10 Plugin Registry inventory、NetworkBuilder、InferenceBindings 均完成真实运行。两版本 identity engine build/serialize/deserialize、ExecuteV2/EnqueueV3 与 output compare 成功；TRT8 额外完成 EnqueueV2。TRT11 可读取 global/capability inventory，但 runtime 创建仍因 CUDA 13 compatible-host 条件返回空对象。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 本地包已重打。三个临时 consumer 只使用 `PackageReference`、无 `ProjectReference`，restore/build 均为 0 warning / 0 error，并编译 `CudaPrimaryExecutionContext` 的完整 public surface；`Runtime execution proof=False`。
- strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 保持 accepted 0/9、gates 2/3、validation blocker 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。
- 本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Kernel Library Metadata 与 ProjectQuality Bounded Shard 复审

本阶段基于起始提交 `5d9aeec2e3369682f3235ac7335857c76dfe595a`，新增 6 个 CUDA Kernel Library owner-safe entry：`cudaLibraryLoadData`、`cudaLibraryLoadFromFile`、`cudaLibraryUnload`、`cudaLibraryGetKernelCount`、`cudaLibraryEnumerateKernels` 与 `cudaLibraryGetKernel`。同时将 ProjectQuality 从依赖数小时 one-shot 套件的状态，收敛为可续跑、可审计的类级 bounded shard 证据。

### 实现与生命周期边界

- native 新增 bridge-owned `JYPPX_CudaKernelLibrary`。内存加载会在 owner 中保留 code 副本，文件加载和 unload 均由 SafeHandle 生命周期控制；`cudaKernel_t` 只在 native 调用栈内用于存在性查询，不逃逸到 managed。
- managed 新增 `SafeCudaKernelLibraryHandle`、`CudaKernelLibrary.Load(byte[])`、`LoadFromFile(string)`、`KernelCount`、copied `CudaKernelLibraryInventorySnapshot` 与 `ContainsKernel(string)`。
- public API 不暴露 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device/function/plugin/tensor pointer。inventory 使用 count/copy，文件名使用 UTF-8 caller-owned string 输入；native entry 保持 C++ exception 与 Windows SEH containment。
- CUDA 12.9/13.2 调用真实 vendor library API；CUDA 11.6/11.8/12.1/12.3 明确返回 `NotSupported`。missing kernel 被转换为 `false` 后调用 `cudaGetLastError()` 清除已消费的 error 500，避免污染后续 last-error。
- library global、managed/unified function pointer、kernel mutation/raw symbol、execution-context/resource/green-context ownership 链继续 deferred。旧 deferred manifest 全部保留。

### Coverage 与 ProjectQuality

generator 最终为 176 manifests / 3901 records。6 个目标函数在 CUDA 12.9 与 13.2 的 12 个可用版本行全部为 `implemented-with-deferred-history`：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 211 | 57 |
| 11.8 | 273 | 273 | 216 | 57 |
| 12.1 | 277 | 277 | 219 | 58 |
| 12.3 | 292 | 292 | 227 | 65 |
| 12.9 | 307 | 307 | 235 | 72 |
| 13.2 | 330 | 330 | 243 | 87 |

- shard runner 新增 `-MissingOnly` 与 coverage/inventory SHA256 校验；缺口模式默认每类独立 testhost，保留 timeout、process-tree cleanup、TRX counters、声明类覆盖和 TRX SHA256 的 fail-closed 审计。
- coverage exporter 仅接受 `state=passed`、hash 匹配、passed>0 且 failed/error/timeout/aborted 全为 0 的 TRX，并输出单类耗时排名与历史 failed/timed-out 执行单元。
- 正式 Release inventory 为 1246 tests / 375 classes；累计类级证据最终为 375/375、168 份有效 TRX、missing 0、invalid evidence 0。该结论是跨 bounded runs 的 hash-verified class coverage，不冒充一次性 one-shot 全套通过。
- `ProjectQualityShardRunnerTests` 使用隔离的两类/TRX 夹具验证 `2/2 complete`，随后篡改一份 TRX 必须降为 `1/2 incomplete` 且产生 SHA256 mismatch；3/3 通过。
- CUDA/TRT8 alias/Plugin Registry/BuilderConfig/RNNv2/coverage 专项最终 84 项中先暴露 1 个配置 inventory 错配；按 Debug 配置重建后 shard runner 3/3，通过项未发现实现回归。Release evidence 的剩余 6 类独立续跑为 8/8。

### 构建、Smoke、NuGet 与 Consumer

- bindings 生成与幂等通过：176 manifests / 3901 records。
- 完整 solution Release build 成功，0 error；保留 5 条既有 nullable warning，位于两个 release-proof 测试文件，本阶段未扩大。
- native：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部构建成功，CUDA/TensorRT version guard 独立。
- CUDA 12.9 真实 kernel library smoke：内存与文件加载均为 count 1、inventory complete、named lookup true、missing lookup false、`LastError=0`；CUDA 11.8 明确 `VersionGuard=NotSupported`。
- TRT8 与 TRT10 的 Plugin Registry、NetworkBuilder、ExecuteV2/EnqueueV2/EnqueueV3 与 output compare 全部成功。TRT11 Plugin Registry/NetworkBuilder 安全记录 vendor structured exception `3228369022`；Inference runner同样在 runtime 创建处受 compatible-host 条件阻塞，不晋级 runtime proof。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 本地包已重打。三个纯 `PackageReference` consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error，并编译 `CudaKernelLibrary` 的 load/count/inventory/contains/dispose surface；`IsRuntimeExecutionProof=false`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,253,134 bytes | `86A40EE80E785E8791BA31E3B65A4328619DA97A6D7BA23BF6E5117D487F051B` |
| TRT8/CUDA12.1 bridge-only | 303,108 bytes | `55A59C0E8EA874E83EDC4B9F8A477E9B5B300EFCBAB709CF6C0948A90E648FAB` |
| TRT10/CUDA12.9 bridge-only | 326,898 bytes | `49635F52A7E555FEE0883013477C6CC9192B5BDA2EACFF11939BE2D147D4C3C2` |
| TRT11/CUDA13.2 bridge-only | 268,946 bytes | `B2CB0C8069351DDD8CA78843280870017C527D1E9375F45DA76FCFCF353FBF4B` |

### Gate 与发布边界

strict release quality gate 为 `release-quality-gate-passed`、required failure 0；classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Texture/Surface Owner 与 ProjectQuality 隔离复审

本阶段基于起始提交 `f4e882494d62aa53123a48766a9bd3b4c9b76be4`，新增 10 个 CUDA texture/surface 安全 entry，覆盖 `cudaCreate/DestroySurfaceObject`、`cudaGetSurfaceObjectResourceDesc`、`cudaCreate/DestroyTextureObject`、CUDA 11.8 `_v2` create/get descriptor，以及 texture resource、texture descriptor 和 resource-view copied query。旧 deferred manifest 全部保留，coverage 通过显式 real alias 优先并合并 deferred history。

### 实现与生命周期边界

- `CudaTextureObject` 与 `CudaSurfaceObject` 只接受 `CudaArray` owner；创建前对 `SafeCudaArrayHandle` 建立 `DangerousAddRef` lease，对象销毁后才 `DangerousRelease`。即使调用方先显式 `Dispose` array，底层 CUDA array 仍保持有效直到 texture/surface 销毁。
- native 只保存 bridge-owned `cudaTextureObject_t` / `cudaSurfaceObject_t` 值，不向 managed public API 返回 array、device pointer 或 vendor object pointer；资源查询只复制 enum、尺寸、pitch、size 和 pointer-presence bool。
- `cudaCreateTextureObject_v2` 与 `cudaGetTextureObjectTextureDesc_v2` 仅在 CUDA 11.8 descriptor ABI 路径调用真实 `_v2` vendor API，CUDA 12/13 返回带诊断的 `NotSupported`，不跨版本误绑定。
- 真实 CUDA 11.8 smoke 发现：创建时传入 null resource-view 后，vendor query 返回 success 但输出字段未定义。bridge 现记录创建时是否提供 view，并把该情况归一化为 `CudaTextureResourceViewSnapshot.IsSpecified=false` 的零快照；不再把垃圾字段包装成有效 descriptor。
- native entry 使用 `noexcept` exception containment 与 Windows SEH guard。public API 未暴露 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device/plugin/tensor pointer。
- linear/pitch2D texture、mipmapped texture、custom resource view、external memory/semaphore、graphics interop、callback、library/kernel handle 与 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 最终为 175 manifests / 3895 records。10 个函数在可用版本上的 50 行全部为 `implemented-with-deferred-history`：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 211 | 57 |
| 11.8 | 273 | 273 | 216 | 57 |
| 12.1 | 277 | 277 | 218 | 59 |
| 12.3 | 292 | 292 | 226 | 66 |
| 12.9 | 307 | 307 | 228 | 79 |
| 13.2 | 330 | 330 | 236 | 94 |

- bindings 生成与两次幂等校验通过，175 manifests / 3895 records。
- 完整 solution Release build 通过，0 warning / 0 error。
- native：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部成功。
- texture/surface、coverage alias、public API、consumer 与既有 owner-scoped 受影响分片最终 70/70；`RealProofCandidatePromotionGuardTests` 独立 2/2。
- ProjectQuality 新增程序集级串行边界，完整套件三小时内未重现 canonical artifact 文件锁，但因串行累计耗时超过三小时被 bounded timeout 终止，未生成最终 TRX，因此不记为完整通过。超时时正在执行 promotion guard pipeline；该类独立复跑通过，没有证据表明单项死锁。

### Runtime Smoke、NuGet 与 Consumer

- CUDA 11.8 与 CUDA 12.1 smoke 均真实得到 `OwnerDisposedBeforeQuery=True`、`Lease=True`、Array resource、`HasDevicePointer=False` 和 `IsSpecified=False` 零 view；CUDA 11.8 `_v2` create/query 成功，CUDA 12.1 `_v2` 明确 `NotSupported`。CUDA 13.2 在当前 driver 12.9 主机以 error 35 明确跳过。
- TRT8/CUDA12.1 与 TRT10/CUDA12.9 的 Plugin Registry、NetworkBuilder、InferenceBindings 全部通过；TRT8 `ExecuteV2/EnqueueV2/EnqueueV3`、TRT10 `ExecuteV2/EnqueueV3` 输出匹配。
- TRT11/CUDA13.2 可读取 global/capability copied inventory，但 runtime 创建受当前 driver 12.9 阻塞；不将 native build、inventory 或 compile consumer 冒充 runtime proof。
- managed 4.0.0 与三个 Bridge-only 4.0.0 本地包已重打并校验。三个纯 `PackageReference` consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error，`RuntimeExecutionProof=False`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,221,437 bytes | `9EE86496E5147586A7CF31718A89129894C916910B02D2E03588AA3A3714B2A2` |
| TRT8/CUDA12.1 bridge-only | 301,869 bytes | `46E1EF97ED95CFF265E3CF4A7C6D00A2B6E21CF1ECA1BF76B61151C54800C24A` |
| TRT10/CUDA12.9 bridge-only | 323,800 bytes | `6C5BC50388A7B445565E342A3AEA300AD821BB93E0045852517060DC6D562DE4` |
| TRT11/CUDA13.2 bridge-only | 266,128 bytes | `6A6DBD72477865D0907DDFAF2FEE0EAB76EA705AB2765027FD34E212E7B30BD5` |

### Gate 与发布边界

strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Owner-Scoped Graph Diagnostics 安全提升复审

本阶段基于起始提交 `b9bc959af6939bb003cf1f7aeda2f5305f645540`，从 CUDA graph/stream deferred-only 清单中提升 12 个安全 entry，覆盖 11 个官方函数：memset node 默认/after 创建、graph exec memset 参数更新、owner-scoped node 删除、kernel/host/memalloc/memfree 参数 copied snapshot、external semaphore signal/wait copied snapshot、stream capture copied summary 与 capture dependency token array 更新。旧 deferred manifest 全部保留，coverage 通过显式真实 alias 优先并合并 deferred history，不以删除历史记录改变统计。

### 实现与安全边界

- memset node 创建与 exec 更新均要求托管 `CudaMemory` owner，native 校验 destination owner 和 graph/exec 生命周期；node 返回 graph-owned token，不暴露 `cudaGraphNode_t`。
- `CudaGraph.RemoveNode` 在 native 删除前核对 node 属于指定 graph；成功后同步使 owner-bound token 失效，避免跨 graph 删除与悬空复用。
- kernel/host/memalloc/memfree 与 external semaphore 查询只复制 dimensions、count、scalar 和 pointer-presence bool，不返回 function、callback、user data、device pointer、semaphore 或参数数组。
- capture summary 按 CUDA ABI 分线：CUDA 13+ 使用无后缀七参数 ABI，CUDA 12.3-12.x 使用 `_v3`，CUDA 11.3-12.2 使用 `_v2`，更旧版本使用基础三参数；各版本 guard 独立。
- capture dependency 更新只接受同一 active capture graph 的 owner-bound node token 数组，在调用栈内完成验证与复制。CUDA 13 路径传空 edge-data，保持旧 API 语义；`_v2` edge-data 的公开建模继续 deferred。
- native entry 捕获 C++ exception 与 Windows SEH；public API 未暴露 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device pointer、callback pointer、external resource handle 或 borrowed vendor object。
- `cudaStreamBeginCaptureToGraph`、kernel/host/memalloc/memfree 创建、external resource mutation、callback/user object、library/kernel/resource handle 与 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 最终为 174 manifests / 3885 records。最终 CUDA coverage：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 203 | 65 |
| 11.8 | 273 | 273 | 206 | 67 |
| 12.1 | 277 | 277 | 210 | 67 |
| 12.3 | 292 | 292 | 218 | 74 |
| 12.9 | 307 | 307 | 220 | 87 |
| 13.2 | 330 | 330 | 228 | 102 |

目标 62 个版本行全部为 `implemented-with-deferred-history`。`Find-ExplicitCudaManifestApis` 为 11 个函数建立真实 alias 和 deferred-history alias；专项测试同时约束 matcher 优先顺序与旧 deferred 保留。

- bindings 生成与幂等：通过，174 manifests / 3885 records。
- 完整 solution Debug build：0 warning / 0 error。
- native：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部成功，CUDA 11.8 编译实际验证了 capture info ABI 分线。
- 新专项：6/6；CUDA/coverage/public API/consumer 受影响分片：106/107。唯一失败是 canonical artifact 并发文件锁，独立串行复跑通过；两个过时的 owner/release readiness 断言清理后为 2/2。
- 完整 ProjectQuality 运行超过 100 分钟，多个测试并行改写 `artifacts/final-release` canonical 文件，产生文件锁和状态串线，未形成可信总结果，不记为完整通过；未发现本批 CUDA 实现相关失败。

### Runtime Smoke、NuGet 与 Consumer

- CUDA graph smoke 真实验证 active capture graph、空 dependency replace、memset output、owner-scoped remove，以及 memalloc/memfree copied snapshot；全部通过。
- TRT8/TRT10 Plugin Registry、NetworkBuilder 与 InferenceBindings 全部通过，ExecuteV2/EnqueueV2/EnqueueV3 输出匹配；TRT11 保持当前 compatible-host/runtime 边界，不将 build 或 dependency probe 冒充 enqueue proof。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 4.0.0 本地包已重打。三个纯 `PackageReference` consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error，`RuntimeExecutionProof=False`。
- TRT11 bridge-only 重打后，已从 `artifacts/pack-current/trt11-cuda13-preserved-full` 恢复 CudaCudnn、TensorRt 与 meta 包，同时保留最新 Bridge 包，release candidate inventory 为 3/3。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,121,698 bytes | `A06558A4F729DBC180F24AE84B7382CDED79D21494F80CDFE9118CA7C54B93DE` |
| TRT8/CUDA12.1 bridge-only | 298,341 bytes | `A9E401652C552059B59C5D75238322AF6AE914AE48AA9450F2A46391CFC7E38C` |
| TRT10/CUDA12.9 bridge-only | 320,394 bytes | `D4B75BBA305124C8D3C283B00E724F8EBA8EB93924714000D0000715564D8256` |
| TRT11/CUDA13.2 bridge-only | 262,689 bytes | `5CDFCF6E43E34EB36EAA0073D2E7C13FF32EE07026AF5FE7B1A292FC4000B70C` |

### Gate 与发布边界

strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。`git diff --check` 通过，仅有既有 CRLF/LF 转换提示。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 CUDA Child Graph、Exec Update 与 Runtime Logs 安全提升复审

本阶段基于起始提交 `2dfab034c9df29567782ddb0fe0500b53f672f28`，新增 14 个 CUDA 安全 entry，覆盖 child graph 创建与 copied topology、graph exec child 参数更新、exec update copied failure metadata、parameterized instantiate、kernel attribute copy 和 CUDA 13.2 runtime logs。旧 deferred manifest 全部保留，coverage 先匹配显式真实 alias，再合并 deferred history，不以删除历史记录改变统计。

### 实现与安全边界

- `CudaGraph.AddChildGraphNode/After` 返回 graph-owned node token；embedded child graph 只在 native 调用栈内读取并复制为 `CudaGraphChildSnapshot`，不会向 public API 逃逸 borrowed graph handle。
- `CudaGraphExec.Update` 在 `cudaErrorGraphExecUpdateFailure` 下保留 result、error node type 与 error-from node type，返回 `CudaGraphExecUpdateSnapshot`，不丢弃 vendor 失败元数据，也不暴露 node pointer。
- `CudaGraph.InstantiateWithParameters` 支持默认与 upload stream overload；成功后返回 bridge-owned `CudaGraphExec`，失败路径不泄漏 executable graph。
- `CopyKernelNodeAttributes` 已按 CUDA ABI 的 source、destination 顺序调用，并以 destination、source 的托管签名保持调用意图清晰。
- `CudaRuntimeLogs` 使用 typed `CudaLogCursor`、caller-buffer memory dump 与 UTF-8 file path；buffer 上限为 25,600 bytes，返回 `CudaLogSnapshot`，不公开原生 iterator pointer。
- `get_cuda_child_graph` 整体位于 `JYPPX_HAS_CUDA_TOOLKIT` guard；parameterized instantiate 与 logs 分别使用 CUDA 12.0、13.2 version guard，TRT8/TRT10/TRT11 构建树保持独立。
- public API 未暴露 `IntPtr`、`nint`、`SafeHandle`、`UIntPtr`、device pointer、plugin pointer、tensor pointer 或 borrowed graph object。callback 注册、device/resource acquire/release 与 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 最终为 173 manifests / 3873 records。最终 CUDA coverage：

| CUDA Toolkit | 官方函数 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 193 | 75 |
| 11.8 | 273 | 273 | 196 | 77 |
| 12.1 | 277 | 277 | 200 | 77 |
| 12.3 | 292 | 292 | 207 | 85 |
| 12.9 | 307 | 307 | 209 | 98 |
| 13.2 | 330 | 330 | 218 | 112 |

`cudaGraphAddChildGraphNode`、`cudaGraphChildGraphNodeGetGraph`、`cudaGraphExecChildGraphNodeSetParams`、`cudaGraphExecUpdate`、`cudaGraphInstantiateWithParams`、`cudaGraphKernelNodeCopyAttributes`、`cudaGraphNodeGetContainingGraph`、`cudaLogsCurrent`、`cudaLogsDumpToMemory` 与 `cudaLogsDumpToFile` 均为 `implemented-with-deferred-history`。`Find-ExplicitCudaManifestApis` 的 matcher 顺序与旧 deferred 保留均有防回归断言。

- bindings 生成与幂等：通过，173 manifests / 3873 records。
- 完整 solution Debug build：0 warning / 0 error。
- native：TRT8/CUDA11、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部成功。额外 `win-x64-dev` preset 只在既有 TensorRT stub `validate_profile_selector`、`get_payload`、`create_handle_with_payload` 失败，与本批 CUDA 文件无关，未扩散修改。
- 新增专项：11/11；受影响 ProjectQuality 分片：67/67；release candidate package inventory 独立串行复核：3/3。
- 完整 ProjectQuality 曾启动，但多个既有测试并行写同一 `artifacts/final-release` canonical 文件，引发 `File.Replace`/`Get-Content` 文件锁和状态串线；进程已终止，该运行不记为完整通过。受影响分片、package inventory 与 strict gate 均已串行通过。

### Runtime Smoke、NuGet 与 Consumer

- TRT10 CUDA graph smoke：child snapshot 为 `Nodes=2 / Roots=1 / Edges=1`，exec update 为 `Success`，parameterized instantiate 默认与 upload stream overload 均成功。
- CUDA smoke：常规路径成功；CUDA 12.9 logs 按版本 guard 正确记录 `Skipped`，不冒充 CUDA 13.2 runtime log proof。
- TRT8/TRT10 Plugin Registry 与 NetworkBuilder 全部通过；NetworkBuilder 均为 `Enqueue=True / OutputMatch=True`。
- TRT8 InferenceBindings 的 ExecuteV2/EnqueueV2/EnqueueV3、TRT10 ExecuteV2/EnqueueV3 均为 `OutputMatch=True`。
- managed 4.0.0 与 TRT8/TRT10/TRT11 三个 bridge-only 4.0.0 本地包已重打；三个纯 `PackageReference` consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error，`RuntimeExecutionProof=False`。
- 为恢复 release inventory 的完整 split 集合，TRT11/CUDA13.2 额外顺序重打 full runtime 与 Bridge/CudaCudnn/TensorRt/meta 四角色 split 包；这仍是本地 package inventory，不是公开渠道 proof。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,323,665 bytes | `6197F92ACFE427A7E7147F2E1D30CD3F10C7046A2433A76ECED9275E455644FB` |
| TRT8/CUDA12.1 bridge-only | 293,806 bytes | `482779E341F6D081051DCB712A864C93C097F28125037E53E7E12EF1482B70EB` |
| TRT10/CUDA12.9 bridge-only | 316,932 bytes | `5F009E88D3EEDA6139950C01E930F310F312D5CDA1612567E173E4D6660DF940` |
| TRT11/CUDA13.2 bridge-only | 259,084 bytes | `CFEA46FD0BB22A52C99D63D66464A0D2F394E946C8979CEFDFC5C76924539C4B` |

### Gate 与发布边界

strict classification audit 为 `FindingCount=0`；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。`git diff --check` 通过，仅有既有 CRLF/LF 转换提示。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 ONNX Config 生命周期与 Coverage 收敛复审

本阶段基于提交 `ec4a2976d6c017353034140f514c45f0db9819c2`，完成三版本 `Global::createONNXConfig` 真实实现优先收敛，并将 TRT8 `IOnnxConfig::destroy` 映射到通用 bridge-owned object destroy。旧 deferred manifest 全部保留，coverage 通过显式 alias 优先选择真实 entry，再合并 deferred history，不以删除历史记录改变统计。

### 实现与安全边界

- `JYPPX_HAS_TENSORRT_ONNX_CONFIG` 与 `JYPPX_HAS_TENSORRT_ONNXPARSER` 已拆分。ONNX Config 是 bridge-owned、header-only 对象，不再错误依赖 parser runtime；TRT8 本机可保持 `ONNXPARSER=0 / ONNX_CONFIG=1`。
- ONNX Config 创建后先由 `std::unique_ptr` 接管，bridge handle 分配成功后才释放所有权，修复 handle 分配失败时的原生对象泄漏。
- TRT8/TRT10/TRT11 include site 各自声明 expected-major guard；配置对象创建、标量和 caller-buffer 字符串控制继续位于 C++ exception 与 Windows SEH 边界内。
- public API 只暴露 `TensorRtOnnxConfig`、copied snapshot/summary 与 `Dispose`，不暴露 `IntPtr`、`nint`、`SafeHandle` 或 parser/plugin/tensor/device pointer。
- RNNv2 gate weights/bias setter、callback trampoline、plugin create/register/deregister/load library、allocator/resource acquire/release 与 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 为 172 manifests / 3859 records。最终 coverage：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 751 | 96 |
| TRT10 10.11.0.33 | 879 | 879 | 759 | 120 |
| TRT11 11.0.0.114 | 901 | 901 | 812 | 89 |

六个包变体的 `Global::createONNXConfig` 均为 `implemented-with-deferred-history`；两个 TRT8 包变体的 `IOnnxConfig::destroy` 同样为该状态。TRT8 `Global::getBuilderPluginRegistry` 与 `IPluginRegistry::getBuilderSafePluginRegistry` 继续正确命中真实 capability/safe registry entry 并合并旧 deferred history。

- bindings 生成与幂等通过：172 manifests / 3859 records。
- 最终源码状态下完整 solution Release build：0 warning / 0 error。
- native Release：TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部成功；TRT8 已验证 `ONNXPARSER=0 / ONNX_CONFIG=1`。
- ONNX Config、coverage alias、Plugin Registry、BuilderConfig、RNNv2、consumer 与 version guard 受影响分片：92/92；最终 public handle/ONNX Config 复核另为 11/11。
- DocFX：0 warning / 0 error。
- 完整 ProjectQuality 使用串行 xUnit 配置尝试执行，在 4m46s 和 6m09s 复现两个既有 owner canonical artifact/fixture 失败：`FinalOwnerExecutionBlockerLedgerTests` 仍要求历史 `blockerCount >= 90`，当前 fail-closed 产物为 2；`FinalOwnerExecutionCloseReadinessFromRealInputTests` 的 owner template validator 保持 2 个 blocker。该尝试未记为完整通过，也未弱化门禁。

### Runtime Smoke 与 Package Consumer

- ONNX Config runtime smoke 在 TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 三版本全部通过；该路径不需要 GPU enqueue。
- TRT8/TRT10 Plugin Registry、NetworkBuilder 与 InferenceBindings 已通过；identity build/serialize/deserialize/enqueue/output compare 均成功。
- TRT11 GPU enqueue 仍受本机 driver 576.02 / CUDA error 35 限制，不记为 runtime proof。
- managed NuGet 与 TRT8/TRT10/TRT11 三个 bridge-only 4.0.0 本地包已重打。三个 baseline consumer 均为无 `ProjectReference` 的纯 `PackageReference` restore/build，新增 ONNX Config owned lifecycle、snapshot、summary 与 `Dispose` 编译可见。
- TRT8/TRT10 仓库外 bridge package runtime consumer 均为 `smokeStatus=passed`、`EnqueueCompleted=True`、`IdentityOutputMatch=True`；本地 feed 证据不冒充 public clean package-consumer proof。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 14,239,095 bytes | `9C6B739DB5A326F7A08DCC8B64CC4036B6A0FB00B89DC0E9F2F15A63399D2C04` |
| TRT8/CUDA12.1 bridge-only | 290,765 bytes | `D85EC4E11BC064F5926782E865E23F45E9B8574C5238E70F2B6A8BFC7152AE7C` |
| TRT10/CUDA12.9 bridge-only | 315,019 bytes | `3065C5BADACA580724822BE2049537C79C49828199C521D750E4C3139F8EA326` |
| TRT11/CUDA13.2 bridge-only | 256,270 bytes | `D9860BCCDD4D92A7A2391FB37EA612C06B44988E7E1FF536B14B65BBCF79E237` |

### Gate 与发布边界

strict classification、public proof claim boundary、real-proof import boundary 与 stale release claims finding 均为 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。`git diff --check` 通过，仅有既有 CRLF/LF 转换提示。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-17 Synchronous Inference 与 Global Plugin Registry Control 复审

本阶段基于提交 `e4ac4edca685ef6cbbcd6225a886f4743eccf2b1`，新增 18 个安全 bridge entry：TRT8 的 `execute`、`executeV2`、`enqueueV2`，TRT10/TRT11 各一个 `executeV2`，TRT8 global Plugin Registry copied inventory/control 11 个，以及 TRT10/TRT11 global registry parent-search setter 各 1 个。旧 deferred manifest 全部保留，coverage 通过显式 alias 优先匹配真实实现并合并 deferred history，不以删除历史记录改变统计。

### ABI、生命周期与降级边界

- 同步推理 native 实现只收集 execution context 已绑定的 tensor/binding address，在调用栈内构造临时数组；public API 不暴露 `IntPtr`、`nint`、`SafeHandle` 或 device/tensor pointer。
- `TensorRtInferenceBindings.ExecuteV2` 与 TRT8 `ExecuteLegacy` 为同步调用；`EnqueueV2AndSynchronize` 在返回前强制 `CudaStream.Synchronize()`，不会把异步 device work 生命周期转嫁给调用方。
- TRT8 global creator inventory 只复制 name/version/namespace/field metadata。`getFieldNames()`、集合指针、count 与 storage 校验位于同一 Windows SEH guard；单个 vendor creator 返回 `RuntimeError`、`InvalidState`、`NotSupported` 或 `NotImplemented` 时只将该 creator 的字段列表降级为空，creator 身份仍保留。
- TRT8 global registry 没有独立 recursive creator count，因此 managed `RecursiveCreatorCount` 为 `null`，不伪造为 0。真实 smoke 中 inventory 为 2 creators、`Recursive=n/a`、诊断一致。
- global parent-search setter 返回前读回校验；Plugin Registry smoke 在 `finally` 恢复原值。TRT8 creator lookup 基于 copied inventory，不返回 creator pointer。
- 字符串继续使用 caller-buffer，数组继续使用 count/copy；C++ exception 和 Windows SEH 不跨 ABI。TRT8/TRT10/TRT11 include site 各自声明 expected major，不匹配 translation unit 只生成 vendor-missing stub。
- RNNv2 `setWeightsForGate` / `setBiasForGate`、callback trampoline、plugin create/register/deregister/load library、allocator/resource acquire/release 及 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 最终为 172 manifests / 3859 records。2026-07-17 重导 coverage：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 749 | 98 |
| TRT10 10.11.0.33 | 879 | 879 | 758 | 121 |
| TRT11 11.0.0.114 | 901 | 901 | 811 | 90 |

TRT8 `Global::getBuilderPluginRegistry`、`IPluginRegistry::getBuilderSafePluginRegistry`、`Global::getPluginRegistry`、`IPluginRegistry::setParentSearchEnabled`、`IExecutionContext::execute`、`executeV2` 与 `enqueueV2` 均核对为真实实现或 `implemented-with-deferred-history`。两条最初错误显示 deferred-only 的 builder registry 行已通过显式 alias 优先规则修复，并有 matcher 顺序防回归断言。

- bindings 生成与幂等：通过，172 manifests / 3859 records。
- 完整 solution Release：0 warning / 0 error。
- native Release：TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 全部成功；TRT10 跨版本 translation unit 保留既有 MSVC warning，但无编译或链接错误。
- 新专项以及 Plugin Registry、BuilderConfig、RNNv2、coverage alias、consumer 与 vendor guard 相关测试：85/85；仓库 shard runner 的 A-F/G-M/N-S/T-Z 受影响分片同样为 85/85。
- DocFX：0 warning / 0 error。
- 完整 ProjectQuality 本阶段已尝试，但多个测试并行改写相同 canonical `artifacts/final-release` 文件，出现 Windows 文件占用与状态串线，testhost 随后长时间无 CPU/子进程，未形成可靠全绿结果。直接相关类已全部串行通过；既有累计 shard 缺口不包含本阶段专项类。门禁未被弱化。

### Runtime Smoke 与 Package Consumer

- TRT8/CUDA12.1：Plugin Registry、NetworkBuilder、InferenceBindings、TensorRtSmoke、Lifecycle 全部通过。`executeV2`、enqueueV3、同步 `enqueueV2` 均输出一致；global parent-search 修改与恢复成功。
- TRT10/CUDA12.9：global/capability registry inventory、parent-search 往返、NetworkBuilder、InferenceBindings、TensorRtSmoke、Lifecycle 全部通过；`executeV2` 与 enqueueV3 输出一致。runtime-local/builder-owned inventory 在当前本地 bridge 缺旧 entry 时按既有可跳过策略记录，不冒充覆盖。
- TRT11/CUDA13.2：bridge、global/capability inventory 与 parent-search 往返可执行；CUDA runtime 报 error 35，`createInferRuntime` 返回 null，NetworkBuilder/InferenceBindings/runtime smoke 均不记为通过。
- managed NuGet 与三个 bridge-only 包已重打。三个 baseline consumer 都是仓库外纯 `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error，并编译新增 global registry 与同步推理高层 API。
- TRT8/TRT10 bridge package runtime consumer 均完成 identity engine build/serialize/deserialize/enqueue/output compare，`RuntimeSmoke=Passed`、`IdentityOutputMatch=True`。TRT11 package compile 通过，runtime 证据严格记录为 `compatible-host-bridge-package-runtime-failed`、CUDA error 35、`isRuntimeExecutionProof=false`。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,959,148 bytes | `6E6D264EBA2316177ADA2978C034713B68E4DE13678BC9472D043F08F65B71AB` |
| TRT8/CUDA12.1 bridge-only | 286,583 bytes | `6FA307EC55DFBAE60ECA8149F723715FA5BF5AC07F9522B15D3B1536184FEA8E` |
| TRT10/CUDA12.9 bridge-only | 314,904 bytes | `0F3637728A77DA4B06F8A42AC9FE0CCAACF53A766E4C4AD67931ADAE9909051F` |
| TRT11/CUDA13.2 bridge-only | 256,054 bytes | `AB6DEADCB8B564043DFB80B5817235F3886F40EBE2B3E2EF58706E818120A715` |

### Gate 与发布边界

strict classification、public proof claim boundary 与 real-proof import boundary finding 均为 0；stale release claims finding 为 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 继续为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。`git diff --check` 通过，仅有既有 CRLF/LF 转换提示。

本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-16 Execution Context Auxiliary Stream 三版本生命周期复审

本阶段基于提交 `747b44ac8b1dbfe8b1060a78c0f7068c9315450f`，将 `IExecutionContext::setAuxStreams` 从仅 TRT11 可用扩展为 TRT8/TRT10/TRT11 三版本安全能力。TRT8 与 TRT10 各新增 set/clear 两个正式 entry；TRT11 既有 entry 迁移到共享 native 实现。TRT8/TRT10 旧 deferred manifest 全部保留，coverage 通过显式 alias 优先合并 deferred history，不以删除历史记录改变统计。

### 生命周期与 ABI 边界

- native set 拒绝负数、超过 1,000,000 的 count、正 count 配 null 数组、default/invalid stream 和重复 stream；vendor `setAuxStreams` 同时受 C++ exception 与 Windows SEH guard 保护。
- managed wrapper 对每个调用方拥有的 `CudaStream` 建立持久 `DangerousAddRef` lease。native set 成功后才替换旧 lease，native clear 成功后才释放旧 lease；context dispose 时先尝试 native clear，再 teardown context，最后 `DangerousRelease`。
- wrapper 不 dispose 调用方 stream。`TensorRtAuxiliaryStreamAssignmentSnapshot` 只暴露版本线、数量、clear/lease 状态和文本诊断，`NativeStreamPointerExposed=False`、`BorrowedHandleEscaped=False`。
- public API 未新增裸 `IntPtr`、`nint`、`SafeHandle` 或 device/plugin/tensor pointer。字符串继续 caller-buffer，数组继续 count/copy；TRT8/TRT10/TRT11 version guard 保持独立。
- RNNv2 `setWeightsForGate` / `setBiasForGate`、callback trampoline、plugin mutation、allocator/resource acquire/release 与 ownership 不明确的 pointer 继续 deferred。

### Coverage、构建与测试

generator 为 168 manifests / 3841 records。最终 coverage：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 743 | 104 |
| TRT10 10.11.0.33 | 879 | 879 | 757 | 122 |
| TRT11 11.0.0.114 | 901 | 901 | 810 | 91 |

TRT8/TRT10 `IExecutionContext::setAuxStreams` 均为 `implemented-with-deferred-history`。bindings 生成与幂等、managed `net8.0`、完整 solution Release 均通过，solution 为 0 warning / 0 error。TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 全部成功。专项测试 7/7、受影响分片 51/51、release source-only 22/22、CI bounded shard 11/11；ProjectQuality inventory 为 1212 tests / 370 classes。

本轮未重复运行完整 ProjectQuality。上一阶段已完整运行 2 h 14 m，结果为 1165 passed / 40 failed / 0 skipped / 1205 total；40 个失败集中在既有 canonical artifact 顺序、owner template/fixture 和历史 snapshot 断言。该结果未记为完整通过，本轮以新增专项、受影响分片、source-only 与 bounded CI 分片覆盖本次 blast radius。

### Runtime Smoke、NuGet 与 Consumer

- TRT8/CUDA12.1 与 TRT10/CUDA12.9 的 Plugin Registry、NetworkBuilder、InferenceBindings、TensorRtSmoke 均通过真实 build/serialize/deserialize/enqueue/output compare；NetworkBuilder 输出 pointer-free auxiliary-stream clear snapshot。
- TRT10 bridge package runtime consumer 使用仓库外纯 `PackageReference` 项目，识别 TensorRT 10.11.0 / CUDA 12.9 / NVIDIA GeForce RTX 3060 Laptop GPU，`EnqueueCompleted=True`、`IdentityOutputMatch=True`、`RuntimeSmoke=Passed`。
- TRT11/CUDA12.9 Plugin Registry 与 NetworkBuilder 正确诊断/skip；vendor runtime creation 的 SEH 被 guard 捕获，code `3228369022`。InferenceBindings 非零退出，TensorRtSmoke 输出 blocked 诊断，因此不记为 TRT11 runtime proof。TRT11/CUDA13.2 本轮只完成 native build，不冒充 runtime proof。
- 三版本 bridge-only consumer 均无 `ProjectReference`，restore/build 为 0 warning / 0 error。docfx 在修复既有 `Chinese.items` TOC 缩进后为 0 warning / 0 error。

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,906,253 bytes | `27A965C1225094F1A4979D83E1351E0C6E76BD28F0030BACBE4F2BD2E1411D33` |
| TRT8/CUDA12.1 bridge-only | 283,736 bytes | `7BF1567E551C9498D9368522025F61ACD576C47748FFF8E66AAAFD1654075FD9` |
| TRT10/CUDA12.9 bridge-only | 313,069 bytes | `8F401C82DC194B2198EBF09A5D51921E79C0C3C87930A6D4BB9326A1A5A81435` |
| TRT11/CUDA12.9 bridge-only | 248,044 bytes | `9FB182C1629CF67F378FE84C00538F279422A2639913697140BBEE5C068D109C` |

### Gate 与发布边界

strict classification、public proof claim boundary 与 real-proof import boundary finding 均为 0；stale release claims finding 为 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。final blocker convergence 继续为 `blocked-owner-action-required`，4 lanes / 21 missing inputs。real Owner proof convergence 为 accepted 0/9、gates 2/3、validation failed blockers 0，`canPublishPublicly=false`、`canCloseReleaseIssue=false`。`git diff --check` 通过，仅有既有 CRLF/LF 转换提示。本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-16 Runtime Alias、Direct Deserialize/Enqueue 与 CUDA 13 Nested Bin 复审

本阶段基于提交 `ebe760f3ffe5dede512f51fdb2318cdbabd68444` 收口 coverage alias 与 TRT11/CUDA13 runtime 诊断。旧 deferred manifest 全部保留；coverage 通过显式 alias 优先级选择真实实现，不以删除历史记录改变统计。公开 runtime deserialize 路径继续使用调用期间 pinned managed buffer，返回 bridge-owned `TensorRtEngine`，不暴露裸 `IntPtr`、`nint`、`SafeHandle` 或 device/plugin/tensor pointer。

### Coverage 与安全边界

generator 为 166 manifests / 3837 records。最终 coverage：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 742 | 105 |
| TRT10 10.11.0.33 | 879 | 879 | 756 | 123 |
| TRT11 11.0.0.114 | 901 | 901 | 810 | 91 |

TRT8 `Global::getBuilderPluginRegistry` 与 `IPluginRegistry::getBuilderSafePluginRegistry` 均继续为 `implemented-with-deferred-history`，显式 alias 先匹配 capability/safe exists 实现，再合并旧 deferred history。三版本 `IRuntime::deserializeCudaEngine` 和 `IExecutionContext::enqueueV3` 现在匹配真实 direct/scoped-buffer 与 async enqueue entry；TRT10/TRT11 `IRuntime::deserializeCudaEngineV2`、TRT8 `IExecutionContext::enqueueV2` 继续为 `deferred-only`。

deserialization precheck 现在明确记录 `DirectDeserializeCudaEngineRowsDeferred=False`、`DirectDeserializeCudaEngineRowsImplemented=True`、`DirectDeserializeCudaEngineV2RowsDeferred=True` 与 `LoadRuntimeDeferred=True`。这只纠正真实实现状态，不解除 V2/loadRuntime ownership 边界，也不把 precheck 晋级为 runtime execution proof。

### CUDA 13 Runtime 诊断

Windows CUDA 13.2 runtime DLL 位于 `CUDA\v13.2\bin\x64`。bridge runtime consumer 与 lifecycle smoke 的搜索顺序已加入并优先该目录，native asset evidence 同时扫描 TensorRT `bin/lib` 和 CUDA `bin\x64/bin`。最终 TRT11/CUDA13.2 package runtime consumer 证实解析到 `cudart64_13.dll`，但本机 NVIDIA driver 576.02 在 `cudaDriverGetVersion` 返回 CUDA error 35。

root-cause 被精确分类为 `trt11-create-runtime-null-cuda-runtime-error / cuda-driver-insufficient-for-runtime`；`cudaPreflightDriverInsufficient=True`。DLL resolution 报告区分 `presentAtCapture` 与 `currentPathExists`，consumer 清理临时输出后仍保留 capture-time bridge SHA256 证据。TRT11 runtime/enqueue proof 继续为 false，没有把允许失败的 smoke 写成成功。

### 构建、测试与 Smoke

- bindings 生成与幂等通过：166 manifests / 3837 records。
- `TensorRtSharp.sln` Release 完整构建通过：0 warning / 0 error。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 全部成功。
- runtime alias/CUDA13 诊断专项 14/14；Plugin Registry、BuilderConfig、RNNv2、runtime serialization、coverage alias 与 consumer 核心分片 88/88，合计受影响专项 102/102。
- 完整 ProjectQuality 在显式关闭 collection parallel 后完整运行 2 h 14 m：1165 passed / 40 failed / 0 skipped / 1205 total。失败集中于共享 canonical `artifacts/final-release` 的顺序状态、既有 owner template/fixture 与 release snapshot 断言；该结果不记为完整通过。测试后已按正确顺序重建 canonical package consumer、runtime readiness、release evidence 与 owner convergence。
- TRT8/CUDA12.1 Plugin Registry capability inventory、NetworkBuilder 与 InferenceBindings 已通过；TRT10/CUDA12.9 global/capability registry 为 7 creators，NetworkBuilder、InferenceBindings 与 TensorRtSmoke 均完成 build/serialize/deserialize/enqueue/output compare。
- TRT10 bridge package runtime consumer 使用仓库外 `PackageReference` consumer 完成 identity engine build/serialize/deserialize/enqueue，`IdentityOutputMatch=True`。
- TRT11/CUDA13.2 full package consumer restore/build 为 0 warning / 0 error、native assets 19/19，runtime 到达 packaged CUDA 后按 CUDA error 35 分类为 `blocked-by-cuda-driver`。

### NuGet、Consumer 与发布边界

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,859,440 bytes | `0E41FE445BE61D5A1E48E3F1C4F9D05FD63DAB912BE21D93E4AC23CD1D2A5E36` |
| TRT8/CUDA12.1 bridge-only | 282,272 bytes | `A14111C91AF27609D3553E7FE5017ABCF74C9D6051FB51D708DB370468F57729` |
| TRT10/CUDA12.9 bridge-only | 311,591 bytes | `A7894D3D9825E031E1A3C4B5F53CA75D8BE556019C2B2981E95E48B5E9B19D32` |
| TRT11/CUDA12.9 bridge-only | 247,626 bytes | `147BA8202433A13E9996372730DD36898E6A619AA44CB1E4D2F302577EC6DD48` |
| TRT11/CUDA13.2 bridge-only | 254,189 bytes | `AF6F757D9A3CF25B40D441164CCA5B206928EE85AAF09C1E50D594314EE4801A` |

TRT8/TRT10/TRT11 三个 bridge-only consumer 均仅使用 managed/bridge `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error；`-SkipProbe` 结果严格保持 compile/package-layout proof、`isRuntimeExecutionProof=false`。

最终 strict classification、public proof claim boundary 与 real-proof import boundary finding 均为 0；stale release claims finding 为 0；strict release quality gate 为 `release-quality-gate-passed`、required failure 0。Owner convergence 为 accepted 0/9、gates 2/3、validation failed blockers 0、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。本阶段未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

本阶段实现提交 `747b44ac8b1dbfe8b1060a78c0f7068c9315450f` 已推送；GitHub Actions run `29507133319` 为 `completed/success`（<https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29507133319>）。source-quality job `87650695840` 的 bindings、coverage、solution build、source-only tests、bounded ProjectQuality shard 与 summary upload 全部成功；`release-artifact-audit`、`split-package-all`、`package-managed-dry-run` 按 push 条件 skipped，没有公开发布副作用。

## 2026-07-16 Safe Lifecycle、Shape、Serialization 与 Error Metadata 复审

本阶段在基线提交 `67034f9b5266e5a5acdc687d4c520a159b3fe6f7` 上完成一组跨版本安全生命周期与部署元数据提升。TRT8 新增 direct engine build、plugin serialization copied path、legacy shape binding setter 与 error-code upper-bound；TRT10 新增 direct engine build 与 error-code upper-bound；TRT11 新增 refitter logger owner-scoped presence 与 error-code upper-bound。旧 deferred manifest 全部保留，coverage 通过显式优先 alias 合并历史，不以删除 deferred 记录改变统计。

### 真实实现与安全边界

| 能力 | TRT8 | TRT10 | TRT11 | 公开边界 |
| --- | ---: | ---: | ---: | --- |
| `IBuilder::buildEngineWithConfig` | 1 | 1 | 已有 | 返回 bridge-owned `TensorRtEngine`，失败销毁 vendor engine |
| plugin serialization path set/get | 2 | 已有 | 已有 | 输入复制字符串数组；输出 caller-buffer/count-copy snapshot |
| `IExecutionContext::setInputShapeBinding` | 1 | n/a | n/a | managed 数组复制，校验 shape binding、元素数和维度乘积 |
| `IErrorRecorder::EnumMax` metadata | 1 | 1 | 1 | 只公开 exclusive upper bound |
| `IRefitter::getLogger` presence | 已有 | 已有 | 1 | owner 调用栈内读取，只公开 bool |

direct engine、plugin path、shape binding、logger presence 与 error metadata 的 vendor 调用均有 C++ exception 和 Windows SEH 边界。公开 API 未新增 `IntPtr`、`nint`、`SafeHandle`、device/plugin/tensor pointer。字符串采用 caller-buffer，数组采用 count/copy；plugin path 与 shape value count 均有 1,000,000 上限。RNNv2 gate weights/bias setter、callback trampoline、plugin create/register/deregister/load、allocator/resource acquire/release 和 ownership 不明确的 pointer 继续 deferred。

`NetworkBuilderSmokeRunner` 同时保留 serialized build/deserialization proof，并新增 direct build proof。非 refittable identity engine 的 refitter logger 诊断现在记录受控 `Skipped:BridgeProbeException`，不会阻断后续 enqueue；这不把 refitter 创建失败写成 logger presence proof。

### Coverage 与 Alias

generator 当前为 166 manifests / 3837 records。2026-07-16 15:19 重导结果：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 742 | 105 |
| TRT10 10.11.0.33 | 879 | 879 | 753 | 126 |
| TRT11 11.0.0.114 | 901 | 901 | 808 | 93 |

以下目标行已核对为 `implemented-with-deferred-history`：TRT8/TRT10 `IBuilder::buildEngineWithConfig`，TRT8 `IBuilderConfig::getPluginToSerialize` / `setPluginsToSerialize`、`IExecutionContext::setInputShapeBinding`，三版本 `IErrorRecorder::EnumMax`，TRT11 `IRefitter::getLogger`，以及 TRT8/TRT10 两条 `IPluginV2DynamicExt` broadcast 查询。TRT8 `Global::getBuilderPluginRegistry` 与 `IPluginRegistry::getBuilderSafePluginRegistry` 继续优先匹配 capability/safe exists 实现并合并 deferred history。

### 构建、测试与 Runtime Smoke

- bindings 生成与幂等通过：166 manifests / 3837 records。
- `TensorRtSharp.sln` Release 完整构建通过：0 error；仅 5 条既有 ProjectQuality nullable warning。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release configure/build 全部成功；保留既有 MSVC warning 和 CUDA13 delay-load warning。
- SafeLifecycle、Plugin Registry、BuilderConfig、RNNv2、plugin serialization、coverage alias 与 readonly evidence 扩大分片最终为 87/87；deferred safety triage 更新后的独立测试为 1/1。
- 完整 ProjectQuality 并行尝试在约 1 分 17 秒内出现共享 `artifacts/final-release/*.json` 的文件占用/原子替换竞态，以及既有 artifact 快照/脚本结构断言失败；随后精确终止进程树并确认无残留。该尝试不记为完整通过，也不改变本批 88 项独立受影响测试全绿的结论。

真实 smoke：

- TRT8/CUDA12.1 plugin serialization set/get/snapshot/clear 通过；capability registry 为 2 creators，inventory/lookup/snapshot 一致；NetworkBuilder direct build、serialized build、deserialize、enqueue、output compare 通过；TensorRtSmoke 与 InferenceBindings identity enqueue/output compare 通过。
- TRT10/CUDA12.9 plugin serialization set/get/snapshot/clear 通过；global/capability registry 为 7 creators，inventory/lookup/snapshot 一致；NetworkBuilder direct build、TensorRtSmoke high-level chain 与 InferenceBindings enqueue/output compare 通过。
- TRT11/CUDA12.9 plugin serialization、global/capability registry、runtime 与 builder 创建继续触发既有 SEH `3228369022`。bridge 将其转换为稳定诊断；PluginSerialization、PluginRegistry、NetworkBuilder、TensorRtSmoke 安全 skip。InferenceBindings runner 在同一 runtime 创建点退出，因此没有 TRT11 enqueue proof。

### NuGet、Consumer 与发布边界

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,858,414 bytes | `27E737C202A39627269F6BC63BD10E756CA963B3DCA0F9799ED33A2122CC4613` |
| TRT8/CUDA12.1 bridge-only | 282,270 bytes | `8303E7138065C0467ADAF21A7833C4964D3CF2A6AC3A8393D0C4AFD8C9314FEC` |
| TRT10/CUDA12.9 bridge-only | 311,594 bytes | `4CFA250EAB1EC692F18470823568CC5A15430D2709B46E4355C01A6D93E03118` |
| TRT11/CUDA12.9 bridge-only | 247,627 bytes | `E3B93D1CA011B9F0DAE4A82D03E3998FF4145B41437BC4FC3EEC3D5678C64A34` |

三个独立 consumer 均只使用 managed/bridge `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error；报告分别位于 `artifacts/package-consumer/trt8`、`trt10`、`trt11`。`-SkipProbe` 结果严格分类为 compile-surface/package-layout proof、`isRuntimeExecutionProof=false`，不是 public clean package proof。

本阶段最终门禁结果：strict release quality gate 为 `release-quality-gate-passed`、required failure 0；strict classification audit 为 `classification-audit-passed-non-proof-boundaries-intact`、finding 0；stale release claims finding 0；Owner convergence 为 accepted 0/9、gates 2/3、validation failed blockers 0、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

本阶段不执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。Owner convergence 未获得真实输入前继续 fail-closed。

## 2026-07-16 Plugin Layer Owner-Scoped Format 与 Serialization 查询复审

本阶段在提交 `513b6c90fd3e9fa654c91ae155afcaf7097612cc` 上继续完成 20 个 network-owned Plugin 查询 entry：TRT8 2 个，TRT10 9 个，TRT11 9 个。实现只在 `TensorRtLayer` 的 network owner lease 内访问 borrowed plugin/capability/tensor，并在单次调用结束前复制出格式支持、IO 类型、alias 索引与 serialization field metadata。

### 真实 API 增量与安全边界

| 能力 | TRT8 | TRT10 | TRT11 | 公开结果 |
| --- | ---: | ---: | ---: | --- |
| PluginV2 DynamicExt 当前格式组合 copy | 1 | 1 | 1 | `TensorRtPluginFormatSupportSnapshot` |
| PluginV2 IOExt 当前格式组合 copy | 1 | 1 | 1 | `TensorRtPluginFormatSupportSnapshot` |
| PluginV3 build IO count | 0 | 1 | 1 | `TensorRtPluginV3BuildIoSnapshot` |
| PluginV3 output data types count/copy | 0 | 1 | 1 | copied `TensorRtDataType` list |
| PluginV3 aliased input index count/copy | 0 | 1 | 1 | copied bounded index list，缺少 V2 capability 时为 `-1` |
| PluginV3 当前格式组合 count/copy | 0 | 1 | 1 | `TensorRtPluginFormatSupportSnapshot` |
| PluginV3 runtime serialization field count/name/metadata | 0 | 3 | 3 | `TensorRtPluginV3SerializationFieldInventory` |

字符串继续使用 caller-buffer/required-size，数组继续使用 count/copy。`PluginField.data` 只复制非空标志，不解引用、不返回、不缓存。managed 对 Plugin IO 总元素和 serialization field count 均设置 1,000,000 上限，避免异常 vendor count 触发无界分配。公开 API 不暴露 `IntPtr`、`nint`、`SafeHandle`、device/plugin/tensor pointer；C++ exception 与 Windows SEH 均转换为 bridge status。

共享 native `.inc` 使用适配器级 `JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3`：TRT8 为 0，TRT10/TRT11 为 1。V3 导出由适配器决定，vendor 类型调用再检查实际 TensorRT major，因此 TRT10 preset 编译 TRT8 source 时不会误引用 V3 helper，较低 vendor 版本编译现代适配器时仍保留明确的 vendor-missing 路径。

`getOutputShapes`、workspace、valid tactics、configure、enqueue、attachToContext、RNNv2 gate weights/bias setter、callback trampoline、plugin mutation 与 allocator/resource acquire/release 继续 deferred；旧 deferred manifest 未删除。

### Coverage 与 Alias

为以下接口增加显式优先 alias，并通过独立 history alias 合并旧 deferred：

- `IPluginV2DynamicExt::supportsFormatCombination`
- `IPluginV2IOExt::supportsFormatCombination`
- `IPluginV3OneBuild::getAliasedInput`
- `IPluginV3OneBuild::getOutputDataTypes`
- `IPluginV3OneBuild::supportsFormatCombination`
- `IPluginV3OneRuntime::getFieldsToSerialize`

generator 当前为 163 manifests / 3828 records。2026-07-16 重导后的矩阵如下：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 731 | 116 |
| TRT10 10.11.0.33 | 879 | 879 | 749 | 130 |
| TRT11 11.0.0.114 | 901 | 901 | 806 | 95 |

六条目标接口均为 `implemented-with-deferred-history`。上一阶段的 `Global::getBuilderPluginRegistry` 与 `IPluginRegistry::getBuilderSafePluginRegistry` 在 TRT8 也继续保持 `implemented-with-deferred-history`，没有通过删除历史记录改变统计。

### 构建、测试与 Runtime Smoke

- bindings 生成与幂等通过：163 manifests / 3828 records。
- `TensorRtSharp.sln` Release 完整构建通过；最终重建仅有 5 个既有测试 nullable warning，0 error。所有 `JYPPX.TensorRtSharp` target framework 单独构建为 0 warning / 0 error。
- Plugin、BuilderConfig、RNNv2、coverage alias、public handle 与 package consumer 受影响分片最终为 113/113。
- public API documentation audit warning 0；bilingual documentation finding 0。
- 完整 ProjectQuality 1195 项单进程尝试 30 分钟后工具超时，期间无失败输出且无残留 testhost；该结果不记为完整套件通过。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 均成功；保留既有 MSVC warning 与 CUDA13 delay-load warning。

真实 smoke：

- TRT8/CUDA12.1 Plugin Registry capability inventory 为 2 creators，lookup/snapshot 成功；NetworkBuilder、NetworkLayers、InferenceBindings 均完成 build/deserialize/enqueue，输出匹配。
- TRT10/CUDA12.9 global/capability inventory 为 7 creators，lookup/snapshot 成功；NetworkBuilder、NetworkLayers、InferenceBindings 均完成真实 enqueue，输出匹配。
- TRT8/TRT10 的新 owner-scoped 查询在非 Plugin layer 上返回预期受控诊断，没有 pointer 逃逸或异常越过 ABI。
- TRT11/CUDA12.9 global/capability registry、runtime、builder 创建继续触发既有 SEH `3228369022`，被 bridge 防护并稳定 skip；没有写成 TRT11 runtime/enqueue proof。

### NuGet、Consumer 与 Gate

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,829,429 bytes | `1AAF34DF764DBF573FB53AC5DC99488BAF5D9DDE420A75BB9372EEBF9F273BC0` |
| TRT8/CUDA12.1 bridge-only | 280,267 bytes | `558DD0282081014FBF7A97B79FEE6439CAB6F2F6329A40AB0D20F33AD3470A10` |
| TRT10/CUDA12.9 bridge-only | 310,355 bytes | `0C38C272626CAE1E15E23E4AEE9D3C99BE0826306F51A64483070A833F0C96D3` |
| TRT11/CUDA12.9 bridge-only | 247,426 bytes | `673C543E04D4A448F401980D1003EC23023A215408DC00972C2305E17FCADF01` |

三个 consumer 均只使用 managed/bridge `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error；独立报告保存在 `artifacts/package-consumer/bridge-only/<runtime-key>`。`-SkipProbe` 仍严格分类为 compile-surface-proof、`isRuntimeExecutionProof=false`。

- strict release quality gate：`release-quality-gate-passed`，required failure 0。
- strict classification audit：`classification-audit-passed-non-proof-boundaries-intact`，finding 0。
- real Owner proof convergence：accepted 0/9、gates 2/3、validation failed blockers 0。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 Release Issue close。
- 本阶段实现提交 `67034f9b5266e5a5acdc687d4c520a159b3fe6f7` 已推送；GitHub Actions run `29476096667` 为 `completed/success`（<https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29476096667>）。source-quality job `87549284359` 的 bindings、coverage、solution build、source-only tests、bounded ProjectQuality shard 与 quality summary upload 全部成功；三个 opt-in 发布/大包 job 按 push 条件 skipped。

## 2026-07-15 Owner-Scoped Versioned Interface Metadata 收口复审

本阶段在提交 `d56078fa7841b046343a2e19c263d96d1719f954` 上继续完成 TRT10/TRT11 的 owner-scoped versioned-interface metadata 查询，共新增 2 个 manifest、22 个真实 bridge entry。TRT8 不声明也不编译这些入口；上一阶段的 TRT8 capability registry、legacy setter 与 RNNv2 owner-bound setter 实现保持不变。

### 真实 API 增量与安全边界

| 能力 | TRT10 | TRT11 | Entry 总数 | 高层边界 |
| --- | ---: | ---: | ---: | --- |
| owner-attached error recorder versioned metadata | 7 | 7 | 14 | runtime、refitter、engine、execution context、builder、network、engine inspector 只复制 interface kind/version/language |
| execution-context callback API language | 3 | 3 | 6 | output allocator、temporary-storage allocator、debug listener 只在 context owner 调用栈中读取 borrowed interface |
| builder-config progress-monitor versioned metadata | 1 | 1 | 2 | 只在 config owner 调用栈中复制 progress monitor metadata |

字符串使用 caller-buffer/required-size；所有 owner、tensor name、buffer、size 和输出参数均显式校验。borrowed native interface 不返回、不缓存、不跨调用存活，公开 API 只返回 `TensorRtVersionedInterfaceMetadata`，不暴露 `IntPtr`、`nint`、`SafeHandle`、device/plugin/tensor pointer。所有 TensorRT 虚调用均被 C++ exception 与 Windows SEH 边界包围。

高层 API 统一采用 `TryGet*VersionedMetadata(out metadata, out diagnostic)`，TRT8 或未附加 callback/error recorder 时返回安全诊断而不是 native pointer。RNNv2 weights/bias setter、callback trampoline、plugin mutation、resource acquire/release 与 ownership 不明确的 pointer 路径继续 deferred。

### Coverage 与 Alias

除保留上一阶段 `Global::getBuilderPluginRegistry` / `IPluginRegistry::getBuilderSafePluginRegistry` 的显式优先规则外，本阶段为以下官方接口增加显式优先 alias，并用独立 history alias 保留旧 deferred：

- `IPluginV2Ext::getTensorRTVersion`
- `IPluginV2IOExt::getTensorRTVersion`
- `IVersionedInterface::getAPILanguage`
- `IVersionedInterface::getInterfaceInfo`

最终 generator 为 160 manifests / 3808 records。coverage 矩阵已于 2026-07-15 18:19 重导：

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only |
| --- | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 727 | 120 |
| TRT10 10.11.0.33 | 879 | 879 | 741 | 138 |
| TRT11 11.0.0.114 | 901 | 901 | 798 | 103 |

TRT8 的两条 builder registry 行以及四条新增 alias 目标行均保留旧 deferred history；没有删除 deferred manifest 来改变统计结果。

### 构建、测试与 Smoke

- `Generate-Bindings.ps1` 与 `Test-BindingGeneratorOutputs.ps1` 通过：160 manifests / 3808 records，生成幂等。
- public API documentation 与 bilingual audit 通过；补齐 27 个缺失 XML 注释，并修复 30 个只有英文占位的双语元素。
- `TensorRtSharp.sln` Release 完整构建通过：0 warning / 0 error。
- owner-scoped metadata、TRT8 alias、Plugin Registry、BuilderConfig、RNNv2、native guard 等专项共 83/83 通过。
- 完整 ProjectQuality 单进程执行持续推进 30 分钟后被工具超时终止，期间未输出失败；残留的本次 `dotnet/vstest/testhost` 进程树已按 PID 精确清理。该结果不得写成完整套件通过。
- ProjectQuality inventory 当前为 1189 tests / 366 classes / unassigned 0。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 全部成功；保留既有 MSVC constant-condition/conversion 与 CUDA13 delay-load warning。

真实 smoke 结果：

- TRT8/CUDA12.1 PluginRegistryInventory：capability registry exists，2 creators，inventory consistent，lookup/snapshot 成功；NetworkBuilder legacy scalar round-trip、enqueue、output compare 通过。
- TRT10/CUDA12.9 PluginRegistryInventory：global/capability inventory 与 lookup 成功；NetworkBuilder、TensorRtSmokeRunner、InferenceBindings 的 build/serialize/deserialize/enqueue/output compare 通过。
- TRT10/CUDA12.9 ManagedProgressMonitor：`OwnerScopedMetadata Available=True`，`IProgressMonitor 1.0 / Cpp`，attach/clear 通过。
- TRT11/CUDA12.9 CallbackAllocatorSafeControls：设计门禁与 allocator dry-run 成功，但真实 runtime 创建仍被既有 SEH `3228369022` 防护并安全 skip，未冒充 TRT11 runtime proof。
- CallbackAllocatorSafeControls 的 TRT10 调用暴露其既有辅助输出硬编码 TRT11 allocator entry 的限制；本阶段用正确的 TRT11 bridge 验证该 runner，不把 TRT10 的 `EntryPointNotFoundException` 记作新 ABI 失败。

### NuGet 与 Package Consumer

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,685,806 bytes | `258389EBBE7996A92931037FA4BCD9AD6D1E878BC11A90CDAD0C73A89F8135F5` |
| TRT8/CUDA12.1 bridge-only | 278,748 bytes | `1A7E05443E52D2269B5C35B83046B777CA2E53E7471118D9A0D0A034CA002735` |
| TRT10/CUDA12.9 bridge-only | 305,588 bytes | `0698439DAAD2209D0E7C2A8F9C6D7B23AE9659D60225ECBA63C99C281E3DAD13` |
| TRT11/CUDA12.9 bridge-only | 242,628 bytes | `5CCE72943528D64F126B57D36E961E0B7445D3C34FB1886BFB4D76FF882D06B2` |

managed package content audit 通过。三个 bridge consumer 均只包含 managed/bridge `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error，dependency probe 为 `environment-probe-succeeded`。该证据严格分类为 compile-surface/package-layout 证明，不是 clean public package runtime proof。

### Gate、Owner 与发布边界

- strict release quality gate：`release-quality-gate-passed`，required failure 0。
- strict classification audit：`classification-audit-passed-non-proof-boundaries-intact`，finding 0。
- owner input contract validation：failed blockers 0；real Owner proof convergence 仍为 accepted 0/9、gates 2/3、validation failed blockers 0。
- `git diff --check` 通过，仅有既有 CRLF/LF 转换提示。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 Release Issue close。
- 本阶段提交 `513b6c90fd3e9fa654c91ae155afcaf7097612cc` 已推送至 `origin/TensorRtSharp4.0`；GitHub Actions run `29408378232` 已 `completed/success`（<https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29408378232>），source-quality job `87329282271` 的 bindings、coverage、solution build、source-only tests 与 bounded shard 全部成功。

## 2026-07-15 TRT8 Builder Capability Registry 与 Legacy Setter 收口复审

本阶段在上一提交 `8775b4c476fcbe4919fa68367c14dcaf5887d624` 的基础上，为 TRT8 新增 2 个 manifest、27 个真实 bridge entry：builder capability plugin registry copied inventory 17 个、builder/config legacy scalar setter 4 个、RNNv2 owner-bound setter 6 个。旧 deferred manifest 全部保留，coverage 通过显式 alias 优先匹配安全实现，并同时记录 deferred history。

### 真实 API 增量与安全边界

| 能力 | Entry | 高层 C# | 边界 |
| --- | ---: | --- | --- |
| builder capability/safe plugin registry existence | 2 | `TensorRtEnvironmentProbe` Get/Try API | 只返回 bool，不暴露 registry pointer |
| capability registry creator inventory/lookup | 15 | copied `TensorRtPluginRegistryInventory` / creator / field metadata | 字符串 caller-buffer，字段按 count/index 复制，TRT8 creator version 复制为整数 |
| legacy builder/config scalar setter | 4 | max batch、workspace、min timing、flags | TRT8-only compatibility API；TRT10/11 明确 unsupported |
| RNNv2 enum/tensor setter | 6 | operation、direction、input mode、cell/hidden/sequence tensor | 公开 API 接受 owner 包装对象，不暴露 `IntPtr`、`nint`、`SafeHandle` 或 tensor pointer |

native 实现对 capability、enum、索引、输出指针和 handle kind 做显式校验；字符串采用 caller-buffer/required-size，数组与字段采用 count/index copy。所有 TensorRT 虚调用均有 C++ exception 和 Windows SEH 防护。`setWeightsForGate` / `setBiasForGate` 因权重内存生命周期未解决继续 deferred；plugin create/register/deregister/load、resource acquire/release、callback trampoline 与 device/borrowed pointer 也未提升。

### Coverage 修复

`eng/Export-InterfaceCoverageMatrix.ps1` 现在对以下接口先执行显式 alias，再进入通用启发式：

- `Global::getBuilderPluginRegistry`
- `IPluginRegistry::getBuilderSafePluginRegistry`

主 alias 严格指向 `id:*builder-capability-plugin-registry-exists` 和既有 safe/capability exists 组合；TRT8 旧 deferred 通过独立 history alias 合并，因此两行状态为 `implemented-with-deferred-history`，而不是 `deferred-only` 或丢失历史后的纯 `implemented`。新增专项测试同时锁定 alias 文本、前置优先块顺序和最终 CSV 状态。

### 当前覆盖率

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only | 本阶段变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 725 | 122 | +13 / -13 |
| TRT10 10.11.0.33 | 879 | 879 | 737 | 142 | 不变 |
| TRT11 11.0.0.114 | 901 | 901 | 794 | 107 | 不变 |

generator 从 156 manifests / 3759 records 增至 158 / 3786。27 个 entry 映射为 13 条 TRT8 官方接口行提升；CUDA coverage 未变化。

### 构建、测试与运行

- bindings 生成与幂等校验通过：3786 records / 158 manifests。
- `TensorRtSharp.sln` Debug 整解构建通过，0 warning / 0 error；PluginRegistryInventorySmokeRunner 最终分流修改后项目单独构建也为 0/0。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 全部成功；TRT10 保留既有 constant-condition、conversion 和 unreachable-code warning。
- 专项及 Plugin Registry、BuilderConfig、RNNv2、coverage alias、public handle、consumer 相关测试 79/79；smoke 分流更新后的专项 6/6、Plugin Registry 合并 15/15。
- ProjectQuality inventory：1183 tests / 365 classes / unassigned 0；受影响分片最终 run `local-trt8-capability-legacy-setters-20260715-rerun` 为 85/85。
- 单进程完整 ProjectQuality 尝试运行约 20 分钟仍无汇总，随后终止并清理测试子进程；不得写成完整测试通过。

TRT8/CUDA12.1 runtime 正向结果：builder capability registry exists，copied inventory 2 creators、diagnostics consistent、creator lookup/snapshot 成功、TRT version 为 8601；safe registry exists=false。NetworkBuilder 验证 max batch/workspace/min timing/flags setter，engine build/deserialize/enqueue/output compare 通过；InferenceBindings identity enqueue/output compare 通过。

TRT10/CUDA12.9 capability registry inventory/lookup 正向通过，NetworkBuilder 与 InferenceBindings enqueue/output compare 通过。TRT11/CUDA12.9 的 global/capability registry、runtime 和 builder 创建均被既有 SEH `3228369022` 防护；NetworkBuilder 稳定 skip，InferenceBindings 在 runtime 创建处终止，未形成 TRT11 enqueue proof。

### NuGet 与 Package Consumer

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| managed `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,588,134 bytes | `8879DCD5F016D6AF14AF5211F7650CC6973B2DBABBCB2C38B2AF83BF6A077F0D` |
| TRT8/CUDA12.1 bridge-only | 278,049 bytes | `61E99C2AE40790D56E0C3862DD2280CD197CE00A465E6072DC4549BD359E8250` |
| TRT10/CUDA12.9 bridge-only | 302,509 bytes | `B58A618AC5BC436A094BC005E5218405F6463DFBB9D473C27C6FB865FB6845AA` |
| TRT11/CUDA12.9 bridge-only | 239,662 bytes | `930796BE7653845D5D8993879D0F671E7E9B43445486172073718FEE4E3611CE` |

三个 consumer 均仅使用 managed/bridge `PackageReference`，无 `ProjectReference`，restore/build 为 0 warning / 0 error。`-SkipProbe` 结果严格分类为 `compile-surface-proof`、`isRuntimeExecutionProof=false`；bridge-only 包未被描述为完整 runtime 包或 public package proof。

### Gate、Owner 与发布边界

- strict release quality gate：`release-quality-gate-passed`，required failure 0。
- strict classification audit：`classification-audit-passed-non-proof-boundaries-intact`，finding 0。
- real Owner proof convergence：accepted 0/9、gates 2/3，继续 blocked。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 Release Issue close。
- 本阶段提交 `d56078fa7841b046343a2e19c263d96d1719f954` 已推送至 `origin/TensorRtSharp4.0`；GitHub Actions run `29397394518` 已 `completed/success`（<https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29397394518>），source-quality job `87293891351` 完成 158 manifests / 3786 records、source tests 22/22、bounded shard 11/11。

### 下一批技术方向

优先从 TRT8 剩余 122 条 deferred-only 中继续筛选已有 owner、可 copy-out、无 device pointer 的接口，同时评估 TRT10/TRT11 的 stream reader/writer stable state、execution-context copied diagnostics 与 versioned-interface metadata。RNNv2 gate weights/bias setters、callback trampoline、plugin mutation、allocator/resource acquire/release 和 ownership 不明确的 borrowed pointer 继续 deferred。Owner 仍为 0/9 时只允许本地代码、包和 CI 验证，不执行公开发布副作用。

## 2026-07-15 PluginV2 Layer Capability Owner-Scoped 查询最新复审

本轮继续执行“deferred 边界提升”，新增 3 个 manifest、22 个真实 bridge entry：TRT8/TRT10 各 8 个，TRT11 6 个。实现从 network-owned `TensorRtLayer` 进入，borrowed `IPluginV2*`、输入 tensor 和临时数组只在单次 native 调用内存在；原 deferred 历史全部保留。

### 本轮真实 API 增量

| 能力 | TRT8 | TRT10 | TRT11 | 高层 C# 与边界 |
| --- | --- | --- | --- | --- |
| plugin output count | 已实现 | 已实现 | 已实现 | `TensorRtPluginV2LayerMetadata.OutputCount` |
| Ext/IOExt/DynamicExt presence | 已实现 | 已实现 | 已实现 | 三个 bool capability 属性，不暴露 capability pointer |
| legacy output dimensions | 已实现 | 已实现 | 已实现 | `GetPluginV2LegacyOutputDimensions`，从 owner-held layer 输入复制维度 |
| legacy workspace size | 已实现 | 已实现 | 已实现 | `GetPluginV2LegacyWorkspaceSize`，只返回字节数 |
| legacy format support | 已实现 | 已实现 | 已实现 | `SupportsPluginV2LegacyFormat`，只返回 bool |
| Ext output data type | 已实现 | 已实现 | 已实现 | `GetPluginV2OutputDataType`，从 layer 输入复制类型 |
| implicit-batch broadcast | 已实现 | 已实现 | 官方已移除 | TRT11 managed 明确返回 unsupported，不伪造一致性 |

native 使用 `dynamic_cast` 仅做 capability presence/Ext 调用，scratch 数组采用 non-throwing allocation；索引、count、0/1 flag 均校验。所有 plugin 虚调用均有 C++ exception 与 Windows SEH 防护，公开 API 未新增 `IntPtr`、`nint`、plugin pointer、tensor pointer 或 device pointer。

### 当前覆盖率

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only | 本轮非 deferred 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 712 | 135 | +7 |
| TRT10 10.11.0.33 | 879 | 879 | 737 | 142 | +7 |
| TRT11 11.0.0.114 | 901 | 901 | 794 | 107 | +5 |

当前 generator 共处理 156 个 manifest、3759 条 API 记录。22 个 bridge entry 映射为 19 条官方接口行提升；TRT11 少两条是因为官方已删除两个 broadcast 方法。相关行状态为 `implemented-with-deferred-history`，不是删除历史后的纯 implemented。CUDA 覆盖未变化。

### 验证、运行与包消费

- bindings 生成与幂等校验通过：3759 API records / 156 manifests；coverage 导出通过。
- `TensorRtSharp.sln` Debug 整解构建通过，0 warning / 0 error。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 全部成功；保留既有 constant-condition、deprecated 和 delay-load warning。
- TRT8/CUDA12.1 与 TRT10/CUDA12.9 `NetworkLayersSmokeRunner` 完整通过；非 PluginV2 layer 的 capability query 均稳定拒绝，后续 engine build/deserialize/enqueue/output compare 通过。
- TRT11/CUDA12.9 在 runtime 创建阶段触发已防护 SEH `3228369022`；TRT11/CUDA13.2 的 `createInferRuntime` 返回 null。两者均未到达 layer 查询，不能写成 TRT11 broadcast runtime proof。
- 新增 PluginV2 capability 专项测试 5/5；PluginV2/V3 相关测试合并 16/16；受影响 ProjectQuality 分片 33/33。inventory 为 1177 tests / 364 classes。
- managed NuGet：`JYPPX.TensorRT.CSharp.API.4.0.0.nupkg`，13,855,210 bytes，SHA256 `D2AB3C76492835316644883C643620F52DFF3537C4F6C0653AA401324386D74E`。
- TRT10 bridge-only NuGet 已从本轮 Release DLL 重打：302,022 bytes，SHA256 `B20FB9E77A74E3A98A739096BF765BB387ABEC5D283CE5BD4A7B676CFD03DADC`。独立 consumer 仅通过 PackageReference restore/build 新 API，0 warning / 0 error，严格分类为 compile-surface-proof。
- TRT10 完整 runtime/split 重打在输入校验阶段被缺失的十个 cuDNN 9.22 DLL 阻塞；没有绕过校验或把 bridge-only 包冒充完整 runtime 包。
- strict release quality gate 通过，required failure 0；strict classification audit 通过，finding 0；`git diff --check` 通过，仅有既有换行提示。
- 功能提交 `8775b4c476fcbe4919fa68367c14dcaf5887d624` 已推送；GitHub `release-quality-gate` run `29389207806` completed/success。`source-quality` 的 bindings 3759/156、solution build、source-only 22/22、bounded shard 11/11 和 artifact upload 全部成功；三个 opt-in 大包/审计 job skipped。

### 发布与 Owner Proof 边界

- real Owner proof convergence 仍为 accepted 0/9、gates 2/3，状态继续 blocked。
- 本轮没有执行 `dotnet nuget push`、GitHub Packages publish、GitHub Release upload 或 Release Issue close。
- 当前正向 runtime 仅证明 TRT8/TRT10 非 plugin 拒绝路径和常规 network 执行；仓库仍没有可复用的正向 PluginV2 layer 创建路径。

### 下一批技术方向

下一轮先收口本轮 commit/CI 与包 SHA，再继续从 135/142/107 条 deferred-only 中批量选择 owner-scoped、可 copy-out 的只读接口。优先评估 versioned-interface metadata、stream reader/writer 状态、execution-context copied diagnostics 和 allocator/resource presence；descriptor、workspace/device pointer、register/create/clone/enqueue、callback trampoline 继续 deferred。若 Owner lane 出现真实输入，则按现有九 lane contract 导入，不新增同义 dashboard，也不自动公开发布。

## 2026-07-15 PluginV3 Layer Owner-Scoped 元数据最新复审

本轮继续执行“deferred 边界提升”，在 TRT10/TRT11 各新增 13 个真实 bridge API，共 26 个 manifest/native entry。原始 `IPluginV3Layer::getPlugin`、`IPluginV3::getCapabilityInterface`、PluginV3 callback、descriptor/context 依赖方法和 `addPluginV3` deferred 历史均保留，没有通过删除 deferred 记录制造完成度。

### 本轮真实 API 增量

| 能力 | TRT8 | TRT10 | TRT11 | 高层 C# 与边界 |
| --- | --- | --- | --- | --- |
| PluginV3 wrapper interface info/API language | 不支持 | 已实现 | 已实现 | `TensorRtPluginV3LayerMetadata.PluginInterface`，复制 kind/version/language |
| core/build/runtime capability presence | 不支持 | 已实现 | 已实现 | build/runtime 使用 nullable 子快照，不公开 capability pointer |
| core name/version/namespace/interface | 不支持 | 已实现 | 已实现 | `TensorRtPluginV3CoreMetadata`，字符串 caller-buffer copy-out |
| build interface、output/tactic count、format limit、timing cache ID、metadata string | 不支持 | 已实现 | 已实现 | `TensorRtPluginV3BuildMetadata?`，只复制标量和字符串 |
| runtime interface info/API language | 不支持 | 已实现 | 已实现 | `TensorRtPluginV3RuntimeMetadata?` |
| layer 高层入口 | 明确返回 unsupported | 已实现 | 已实现 | `TensorRtLayer.GetPluginV3Metadata()` / `TryGetPluginV3Metadata(...)`，强制 network owner lease |

native 内可以在一次 owner-scoped 调用中短暂访问 `IPluginV3*` 和 capability interface，但返回前只复制字符串、标量、枚举和 interface version。公开 API 没有新增 `IntPtr`、`nint`、plugin pointer 或 capability pointer；vendor C++ exception 和 Windows SEH 均转换为 status/last error，不跨 C ABI 抛出。

### 当前覆盖率

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only | 本轮非 deferred 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 705 | 142 | 0 |
| TRT10 10.11.0.33 | 879 | 879 | 730 | 149 | +15 |
| TRT11 11.0.0.114 | 901 | 901 | 789 | 112 | +15 |

当前 generator 共处理 153 个 manifest、3737 条 API 记录。26 个新 bridge entry 使每条 TRT10/TRT11 版本线各有 15 条官方接口行进入 `implemented-with-deferred-history`；差异来自一个安全 owner-scoped entry 可以覆盖多个官方只读语义。CUDA 覆盖未变化。

### 验证、运行与包消费

- bindings 生成和幂等校验通过：3737 API records / 153 manifests。
- TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2 native Release build 成功；保留既有 deprecated/constant-condition warning。
- TRT8/CUDA12.1 `NetworkLayersSmokeRunner` 通过，PluginV3 查询稳定返回“仅支持 TensorRT 10 和 11”。
- TRT10/CUDA12.9 `NetworkLayersSmokeRunner` 通过，非 PluginV3 layer 被稳定拒绝，诊断为 `PluginV3 layer plugin interface info query requires a TensorRT PluginV3 layer.`，后续 network build、engine deserialize、enqueue 和 output compare 均通过。
- TRT11/CUDA13.2 在创建 network 前由 vendor `createInferRuntime` 返回 null；既有根因报告将其归类为 driver 576.02 只支持 CUDA 12.9、CUDA 13.2 preflight error 35。TRT11/CUDA12.9 也在 runtime 创建阶段触发已防护的结构化异常 `3228369022`。两者均未触及本阶段 PluginV3 API，不能形成 TRT11 runtime proof。
- 仓库没有非 deferred `addPluginV3` 或可复用 PluginV3 模型，因此本轮没有正向 PluginV3 layer runtime proof；不得用 native 编译、负向 smoke 或静态测试替代。
- managed NuGet 已重打：`JYPPX.TensorRT.CSharp.API.4.0.0.nupkg`，13,800,724 bytes，SHA256 `709496E6239605D3B7DEC7393C595563681C77F6F449A1F063BCD4DB73AADF55`。
- TRT10 bridge package consumer 从本地 nupkg restore/build 成功，新 PluginV3 Get/Try、子模型与公开属性通过 freshness、delegate 和 `nameof` 编译；使用 `-SkipProbe`，严格分类为 compile-surface evidence，不是 runtime/public/post-publish proof。
- PluginV3 专项测试 5/5 通过；目标 ProjectQuality 分片 30/30 通过；GitHub workflow 同款 source-only 测试 22/22、critical shard 11/11 通过；inventory 为 1172 tests / 363 classes。单进程全量 ProjectQuality 在 7 分钟上限超时且没有失败输出，不能记录为全量通过。
- strict release quality gate 通过，required failure 0；strict classification audit 通过，finding 0；`git diff --check` 通过。
- 功能提交 `7487a315a4c119e968f4ccac3365462cf3a4135b` 已推送到 `TensorRtSharp4.0`；GitHub `release-quality-gate` run `29382425918` completed/success，`source-quality` 的 bindings、coverage、solution build、source-only tests、bounded shard 和 artifact upload 全部成功。

### 发布与 Owner Proof 边界

- real Owner proof convergence 仍为 accepted 0/9、gates 2/3，状态为 `blocked-real-owner-proof-convergence-real-owner-input-required`。
- 本轮没有执行 `dotnet nuget push`、GitHub Packages publish、GitHub Release upload 或 Release Issue close。
- 本地 source-tree smoke、native build 和 package compile 均不能替代 compatible-host TRT11 proof、公共包 clean consumer、post-publish 或 owner authorization。

### 下一批技术方向

下一轮先从最新 CSV 对剩余 149/112 条 TRT10/TRT11 deferred-only 行做一次集中安全分流，优先一次完成 20-40 个可由既有 owner 获取且能 copy-out 的只读 API；重点评估 PluginV2 capability scalar/metadata、通用 versioned-interface metadata、stream reader/writer 只读状态和 execution-context copied diagnostics。需要 descriptor、workspace、device pointer、plugin instance ownership 或 callback trampoline 的路径继续 deferred。若九条 Owner lane 出现真实输入，则使用现有 contract/preflight/convergence 导入并执行 owner 授权的发布验证，但仍禁止模型自行发布。

## 2026-07-14 PluginV2 Layer 元数据与 TRT8 Creator 版本最新复审

本轮继续执行“deferred 边界提升”，没有把 manifest/source 100% 匹配解释为 100% 可用，也没有删除任何 deferred 历史记录。新增 19 个真实 bridge API 条目，并使 19 条官方接口版本行从 `deferred-only` 移动到 `implemented-with-deferred-history`。

### 本轮真实 API 增量

| 能力 | TRT8 | TRT10 | TRT11 | 高层 C# | 生命周期边界 |
| --- | --- | --- | --- | --- | --- |
| PluginV2 layer plugin type/version/namespace | 已实现 | 已实现 | 已实现 | `TensorRtLayer.GetPluginV2Metadata()` | network owner lease 内读取并复制字符串 |
| PluginV2 serialization size | 已实现 | 已实现 | 已实现 | `TensorRtPluginV2LayerMetadata.SerializationSize` | 只复制标量 |
| PluginV2 packed TensorRT version | 已实现 | 已实现 | 已实现 | packed/tag/major/minor/patch 解码属性 | 只复制标量 |
| Plugin creator compile-time TensorRT version | builder/runtime inventory 与 lookup 已实现 | 官方当前 creator surface 不提供，返回 `null` | 官方当前 creator surface 不提供，返回 `null` | `TensorRtPluginCreatorInfo.TensorRtVersion`、`TensorRtPluginCreatorSummary.TensorRtVersion` | registry owner scope 内复制标量 |
| 原始 `IPluginV2Layer::getPlugin` | deferred 历史保留 | deferred 历史保留 | deferred 历史保留 | 不公开裸指针 | borrowed pointer 不跨 ABI |

公开 API 没有新增 `IntPtr`、`nint`、`IPluginV2*` 或 plugin ownership。字符串使用 caller-buffer/count-copy 模式；native 调用包含 C++ exception 与 Windows SEH 防护，异常不跨 C ABI。

### 当前覆盖率

| TensorRT 版本线 | 官方接口 | manifest/source 已匹配 | 非 deferred 实现 | deferred-only | 本轮非 deferred 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRT8 8.6.1.6 | 847 | 847 | 705 | 142 | +7 |
| TRT10 10.11.0.33 | 879 | 879 | 715 | 164 | +6 |
| TRT11 11.0.0.114 | 901 | 901 | 774 | 127 | +6 |

当前 generator 共处理 151 个 manifest、3711 条 API 记录。CUDA 覆盖本轮未变化：CUDA 11.6/11.8/12.1/12.3/12.9/13.2 的非 deferred 数量分别为 188/191/194/200/202/207。

### 验证与包消费

- TRT8、TRT10、TRT11 native Release configure/build 均成功；保留既有 MSVC constant-condition、deprecated API 和 delay-load warning，不能写成 native 0 warning。
- TRT10 `NetworkLayersSmokeRunner` 真实运行通过；非 PluginV2 layer 被稳定拒绝，诊断为 `PluginV2 layer plugin type query requires a TensorRT PluginV2 layer.`，没有 `EntryPointNotFoundException` 或 access violation。
- TRT8 `PluginRegistryInventorySmokeRunner` 通过，runtime/builder registry 均存在；当前进程未加载标准 plugin library，creator count 为 0，因此本机未形成 `TensorRtVersion > 0` 的正向 creator 运行证明。
- managed NuGet 已重新打包：`JYPPX.TensorRT.CSharp.API.4.0.0.nupkg`，13,695,324 bytes，SHA256 `C2B8FD94F0A0A245A24553ADA3740DB87354F29402CE03411EB3F1C4A9754579`。
- TRT10 bridge package consumer 仅通过本地 nupkg restore/build，新 PluginV2 snapshot 与 creator version compile surface 通过，consumer build 为 0 warning / 0 error；使用本地 feed 和 `-SkipProbe`，因此只是 package compile-surface evidence，不是公开包或 runtime proof。
- 新增专项测试 6/6 通过；关键 ProjectQuality 分片 33/33 通过；当前 inventory 为 1167 tests / 362 classes。
- 单进程全量 ProjectQuality 在 15 分钟上限超时，不能记录为通过；仓库既定分片 runner 已清理超时进程并用于本轮受影响类验证。
- `Test-ReleaseQualityGate.ps1 -Strict` 通过，required failure 0；classification audit finding 0；`git diff --check` 通过，仅有既有 CRLF/LF 提示。
- GitHub 首次 run `29329170033` 暴露 TRT8 manifest 显式 `IntPtr` override 违反 source-only contract；修复提交 `be9d6ea0b7afd12f6e18ab62dd8f48c51a6061f6` 移除多余 override，最终 `release-quality-gate` run `29379794020` completed/success，`source-quality` 的 bindings、coverage、build、source-only tests、bounded shard 和 artifact upload 全部通过。

### 发布与 Owner Proof 边界

- 九条 real Owner proof lane 仍为 accepted 0/9，convergence 为 `blocked-real-owner-proof-convergence-real-owner-input-required`。
- 本轮没有执行 `dotnet nuget push`、GitHub Packages 发布、GitHub Release 上传或 Release Issue 关闭。
- 本地 package consumer 成功不能替代公开源 clean consumer、post-publish、真实 YoloVision 模型或 compatible-host TRT11/CUDA13 proof。

### 下一批技术方向

下一批优先处理 TRT10/TRT11 owner-scoped PluginV3 layer metadata：在 network owner lease 内读取 `IPluginV3`/capability interface，只复制 core name/version/namespace、interface info、API language 和安全 build/runtime 标量或字符串，不公开 `IPluginV3*`、capability pointer 或 plugin ownership。registry register/deregister/load、plugin resource acquire/release、create/clone/enqueue 和 callback trampoline 继续保留 deferred。

## 最新总体状态

项目主线已经从“missing 接口清零”转为“deferred 边界提升、完整包消费验证和真实运行证据收口”。`100% manifest/source 匹配` 只表示官方接口已经进入追踪范围，不等于 `100% 可用`。真实完成度继续以非 deferred native/source 实现、C# 高层 wrapper、明确的生命周期边界、clean package consumer 和 compatible-host runtime proof 为准。

截至 2026-07-10，当前机器可确认：

- B-tier alias/safe-alternative 工作项 `btier-001` 至 `btier-045` 已稳定收口，45/45 均保留 deferred history，不通过删除历史记录制造完成度。
- `win-x64-trt11.0-cuda13.2-cudnn9.22` 的四角色 split package set 已真实生成，`missingSplitRoles=[]`、`splitRuntimePackagesReady=true`、`packageSetReady=true`、`sha256Ready=true`。
- managed package 已重新打包，`JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` 为 13,196,765 bytes，SHA256 为 `A19F22391F84F03FC95E3E724E76EF29B15FFF1236FF1F77BD3F243C135677A8`。
- bridge-only clean consumer 已完成 Release restore/build、最新高层 API compile surface 和 native dependency probe，`NativeDependencyStatus=ready`、`ProbeResult=environment-probe-succeeded`；完整 consumer 复制 native assets 19/19，并真实到达 packaged CUDA 13.2 runtime，但被本机 driver `576.02` / CUDA capability `12.9` 以 CUDA error 35 阻塞。
- `win-x64-trt10.11-cuda12.9-cudnn9.22` bridge-only NuGet 已在仓库外 `PackageReference` consumer 中完成真实 identity runtime：engine build/serialize/deserialize、execution context、enqueue 和 output compare 均成功；该记录分类为 `compatible-host-bridge-package-runtime`，可晋级当前兼容主机运行证明，但不是公共源 clean package-consumer proof，也不解除 TRT11/CUDA13.2 blocker。
- runtime package readiness 为 `ready`，release candidate readiness 只剩 1 条 compatible-host runtime proof blocker；final dry-run 为 `ready-needs-manual-approval`、自动 blocker 0、manual approval 11。最终公开发布仍保持 `canPublishPublicly=false`、`canCloseReleaseIssue=false`。
- TensorRT 8 RNNv2 五个只读 getter `getCellState`、`getHiddenState`、`getSequenceLengths`、`getWeightsForGate`、`getBiasForGate` 已全部由 deferred-only 提升为 owner-bound/copy-out 高层 API；C-tier triage 从 370 降至 360，RNNv2 getter 的 remaining deferred triage 为 0。
- TRT10/CUDA12.9 source-tree 已完成 MNIST 0–9 十张官方 PGM 的真实 inference/output validation，全部为 `real-model-runtime`，预测数字全部匹配，最低置信度为数字 4 的约 `0.991077`；embedded identity 继续严格分类为 `synthetic-input-runtime`。
- TRT10/CUDA11.8 已完成 embedded identity runtime 和 MNIST digit 7 真实模型 runtime：identity `OutputMatch=true`、约 `1.558528 ms`；MNIST 预测 7、置信度 `0.99999285`、约 `1.612 ms`。
- TRT8/CUDA11.8 与 TRT8/CUDA12.1 identity runtime 均通过；TRT8 MNIST 因 `cudnn64_8.dll` 缺失保持 `blocked-by-cudnn8-runtime-missing`。TRT11/CUDA12.9 因 vendor root 缺少 `nvinfer_11.dll`、`nvinfer_plugin_11.dll`、`nvonnxparser_11.dll` 保持 `blocked-by-runtime-assets-missing`。
- 新增跨版本机器可读矩阵，共 19 个案例：15 passed、4 blocked、4 个 `synthetic-input-runtime`、11 个 `real-model-runtime`、0 个 `package-consumer-runtime`。所有 source-tree runtime 证据都不能替代 clean package-consumer runtime proof。
- ProjectQuality 已从 inventory-only 推进到稳定分片累计覆盖：当前 inventory 为 1034 个测试、308 个测试类，A-F/G-M/N-S/T-Z 四个分片的 308/308 类均存在至少一个通过且 TRX SHA256 匹配的完整执行单元；累计接受 59 个严格 TRX，缺失类 0、坏哈希 0。该结论是累计类级覆盖，不冒充一次性单进程全量运行。

## 2026-07-10 最新阶段复审

### 完整 Split Package Set

当前 runtime key：

```text
win-x64-trt11.0-cuda13.2-cudnn9.22
```

| 角色 | 大小 | SHA256 |
| --- | ---: | --- |
| meta | 10,411 bytes | `EAE695C780EAF7539BD56971DC5CF09B5B4E0610E792DFA7E38CEBA1F77EC666` |
| bridge | 231,930 bytes | `39C5ECEE0FAEE35A2012DF0CB6F2609A13C315C872D0904A9E8BC1355206AFAE` |
| cuda-cudnn | 398,300,715 bytes | `C1009FD92ED8EE1D27C96BF230FDAC9AF7C06C1AF0D0106095558579E7465CE9` |
| tensorrt | 1,931,405,943 bytes | `6A736CC39511EA1B953439B87C75862E581D06EC579D38D5B2A3B853378E6001` |

Release managed package：

| 包 | 大小 | SHA256 |
| --- | ---: | --- |
| `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg` | 13,196,765 bytes | `A19F22391F84F03FC95E3E724E76EF29B15FFF1236FF1F77BD3F243C135677A8` |

完整 full runtime nupkg 仍保留在 `artifacts/runtime-nupkg`。split package ready 只证明包集合、依赖关系、文件布局和哈希已准备好，不等于公开包 proof 或真实 runtime proof。

### Clean Consumer

| 路径 | Restore | Build | Native assets | Runtime smoke | Runtime proof |
| --- | --- | --- | ---: | --- | --- |
| split-meta/full consumer | 成功 | 成功，0 warning / 0 error | 19/19 | `blocked-by-cuda-driver`，已到达 packaged runtime，CUDA error 35 | false |
| bridge-only consumer | 成功 | 成功，0 warning / 0 error | bridge compile/copy surface | `environment-probe-succeeded` | false |
| TRT10 bridge runtime consumer | 成功 | 成功 | 9 个资产均记录 SHA256 | identity build/serialize/deserialize/enqueue/output compare 全部通过 | `compatible-host-bridge-package-runtime` |

两条 canonical 报告分别位于：

- `artifacts/package-consumer/package-consumer-validation-summary.json`
- `artifacts/package-consumer/bridge-package-consumer-validation-summary.json`

### TRT10 Bridge Package Runtime Proof

当前兼容主机运行证明位于：

- `artifacts/package-consumer/bridge-runtime/win-x64-trt10.11-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.json`
- `artifacts/package-consumer/bridge-runtime/win-x64-trt10.11-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.md`

| 项目 | 结果 |
| --- | --- |
| Runtime key | `win-x64-trt10.11-cuda12.9-cudnn9.22` |
| Managed NuGet | 13,196,765 bytes / `A19F22391F84F03FC95E3E724E76EF29B15FFF1236FF1F77BD3F243C135677A8` |
| Bridge NuGet | 288,677 bytes / `15EAAC7DA0B2D0BF4DF81FCDF2FF08D82E03D91D5A9C91A977A444AFB0550099` |
| `jyppxtrtbridge.dll` | 897,024 bytes / `3BC489017FB537093A3E6A5C8A561F864D04A60762DB2C4B6FA4E667CDCAADD0` |
| Host | NVIDIA GeForce RTX 3060 Laptop GPU / driver 576.02 |
| Runtime | TensorRT 10.11.0 / CUDA 12.9 |
| Consumer | 仓库外、仅 `PackageReference`、无 `ProjectReference` |
| Engine / enqueue / compare | 2604 bytes / 成功 / 匹配 |
| Combined log SHA256 | `2FB7D9D077898FBD4C683541B814A0894C3DA756B9A1F1CCF1BED1D483A8590D` |
| Proof boundary | `isRuntimeExecutionProof=true`、`isPackageConsumerRuntimeProof=false`、`canPromoteCompatibleHostRuntimeProof=true` |

发布证据 inventory 当前包含 8 个包、1 个 compatible bridge package，且 `compatibleBridgeRuntimeProofReady=true`。本结果证明本地 managed NuGet 与 bridge NuGet 可在匹配的系统 TensorRT/CUDA 依赖上真实执行，但本地 package source 不等于公共源 clean package proof；`canPublishPublicly=false`、`canCloseReleaseIssue=false` 继续保持。

### TensorRtExec / OnnxToEngine Load-Engine Bounded Runtime

新增共享工具层 bounded runtime 路径后，`samples/OnnxToEngine` 与 `applications/TensorRtExec` 均可复用同一套 `OnnxEngineBuildService`：

- `--loadEngine` 先执行 readonly deserialize/readback，记录 `PreflightMetadata`、`LoadedEngineDiagnostics`、engine SHA256、I/O tensor、layer/profile/device memory 和 readback SHA256。
- 当 engine 满足 one-float-input、float-output、runtime shape 可估算时，继续创建 execution context、绑定输入/输出、enqueue、读回输出。
- identity output 与输入匹配时仍分类为 `synthetic-input-runtime`；其他 bounded output 只写 `runtime-output-captured-unverified`，不晋级 real-model 或 package-consumer proof。
- raw binding 二进制只允许 embedded identity 且 output matched 的源树 synthetic runtime 写出；load-engine bounded output 不写原始 binding proof。

真实验证产物位于：

```text
artifacts/trtexec-bounded-runtime/identity
```

| 产物 | 状态 | SHA256 |
| --- | --- | --- |
| `identity.plan` | 3212 bytes | `64EC4A51BAF2B22918EED397C3781BD2680D4E466508A92441EBE884E641E23A` |
| `identity-build-report.json` | `identity-roundtrip` / `synthetic-input-runtime` / `OutputMatch=true` | `A8B073D402525272FDB71565A6FC4FA094DC2F440A954D005A2F298353E09505` |
| `identity-load-report.json` | `load-engine-identity-runtime` / `synthetic-input-runtime` / `OutputMatch=true` / `LoadedEngineDiagnostics=readonly-deserialize-succeeded` | `0B9E23F5A13DCB0C84EA691CA059F02A2DB7DDBAF8CF8BF76815144A48AD42B1` |
| `tensorrtexec-load-report.json` | `load-engine-identity-runtime` / `synthetic-input-runtime` / `OutputMatch=true` / CLI app path | `B56167040121B846E32D79E15CE345EF950DBDDC4131CE83AD50867CA27D43EC` |

该结果证明 source-tree 工具链的 engine save/load/readback/enqueue 路径已贯通，不是 real-model-runtime、不是 clean package-consumer-runtime，也不能替代发布 proof。

### Deferred 与 RNNv2 边界

- B-tier 45/45 已完成稳定 proof closure；不得扩大 selection 重复制造 B-tier 工作项。
- 当前 C-tier `design-gate-required` 行为 360。
- `IRNNv2Layer::getDataLength` 以及五个 borrowed-state/gate-weight getter 均已有真实 native/header/source、generated/manual interop 和高层 C# API。
- 已完成的五个 RNNv2 getter 为：
  - `getBiasForGate`
  - `getCellState`
  - `getHiddenState`
  - `getSequenceLengths`
  - `getWeightsForGate`
- `getCellState`、`getHiddenState`、`getSequenceLengths` 返回 owner-bound tensor wrapper；gate weights/bias 使用 copied snapshot，公开 API 不暴露裸 `IntPtr` 或悬空 borrowed pointer。
- RNNv2 getter 的 10 条版本行已移动到 `implemented-with-deferred-history`；当前 RNNv2 剩余 C-tier 为 16 条 setter 版本行，覆盖 `setBiasForGate`、`setCellState`、`setDirection`、`setHiddenState`、`setInputMode`、`setOperation`、`setSequenceLengths`、`setWeightsForGate`。

### 验证结论

- bindings：3681 API records / 148 manifests。
- binding generator outputs：通过。
- interface coverage：已刷新；TRT8 为 698 implemented / 149 deferred-only，TRT10 为 709 / 170，TRT11 为 768 / 133；CUDA 13.2 为 207 / 123。
- bridge package consumer：成功，最新 RNNv2 五个 getter、owner-bound tensor 和 copied snapshot API 均通过独立 package consumer 编译。
- TRT10 bridge package runtime consumer：仓库外 `PackageReference` consumer 成功完成 identity engine build/serialize/deserialize/enqueue/output compare，`smokeStatus=passed`、`identityOutputMatch=true`、9 个 native asset 均记录 SHA256；严格保持 `isPackageConsumerRuntimeProof=false`。
- TensorRtExec / OnnxToEngine bounded load-engine runtime：TRT10/CUDA12.9 identity engine save/load/readback/enqueue/output compare 在 source-tree 路径通过，`identity.plan` SHA256 为 `64EC4A51BAF2B22918EED397C3781BD2680D4E466508A92441EBE884E641E23A`；该结果严格保持 `synthetic-input-runtime`，不冒充 real-model 或 package-consumer proof。
- full package consumer：restore/build/native assets 19/19 成功，runtime smoke 被 CUDA error 35 分类为 `runtime-smoke-driver-blocked`，未冒充 runtime proof。
- TRT10/CUDA12.9 MNIST 0–9：10/10 `Success=true`、`InferenceRan=true`、`OutputMatch=true`、`ProofClassification=real-model-runtime`。
- TRT10/CUDA11.8 MNIST digit 7：`ExpectedDigit=7`、`PredictedDigit=7`、`Confidence=0.99999285`、`ProofClassification=real-model-runtime`，engine 为 366,348 bytes。
- TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA11.8、TRT10/CUDA12.9 identity：4/4 `OutputMatch=true`，严格分类为 `synthetic-input-runtime`。
- 多版本矩阵 exporter 与质量测试已新增；矩阵 19 个案例中 15 passed、4 blocked，所有已存在的 report/log/engine/model/input/output/tensor/bridge 证据均记录 SHA256。
- final dry-run：`ready-needs-manual-approval`、自动 blocker 0；final pre-publish audit matrix 严格验证通过，`FailedBlockers=0`，但矩阵仍保留 10 条 owner-proof blocker。
- Solution Debug build：0 warnings / 0 errors。
- release-quality workflow 同口径分组测试：33/33 passed；MNIST、TRT8 binding-size fallback、OnnxToEngine 和多版本矩阵定向分组：29/29 passed。
- ProjectQuality inventory 已刷新为 1034 个测试、308 个测试类；首字母分片为 A-F 107 类、G-M 9 类、N-S 165 类、T-Z 27 类。`eng/Export-ProjectQualityShardCoverage.ps1` 只接受 `state=passed`、TRX 文件存在且 SHA256 与 summary 一致的执行单元，当前累计类覆盖为 308/308、严格 TRX 59、缺失类 0、无效证据 0。
- A-F、G-M、N-S、T-Z 的缺口类均已补跑。重型 `ReleaseCandidateReadinessTests` 单类 56/56 passed，耗时约 880.590 秒；runner 支持 batch/class filter、独立 log/TRX/hash、timeout kill-tree，并禁用 MSBuild server/node reuse。本轮超时残留的 14 个孤儿 MSBuild 节点已按父进程和命令行精确清理，修复后未再产生新残留。
- `Test-ReleaseQualityGate.ps1 -Strict`：`release-quality-gate-passed`、required failure 0。
- `Test-ReleaseEvidenceClassificationAudit.ps1 -Strict`、`Test-PublicProofClaimBoundaryAudit.ps1 -Strict` 与 `Test-RealProofImportBoundaryAudit.ps1 -Strict`：finding count 均为 0。
- `Test-StaleReleaseClaims.ps1`：finding count 0。
- TRT11/CUDA13 native Release configure/build：成功；保留 CUDA deprecated API、MSVC constant-condition 和 delay-load 等既有 warning，不能写成 native 0 warning。
- `git diff --check`：通过；仅输出工作树既有 CRLF/LF 转换提示。

### 尚未完成

1. 在 CUDA 13 capable compatible host 上使用当前 TRT11 managed/runtime package 精确 SHA256 完成 clean consumer runtime smoke，补齐 external-runtime-proof-record 并通过 strict validator。现有 TRT10/CUDA12.9 bridge proof 只覆盖兼容主机路线，不得替代或解除该 blocker。
2. TRT8 MNIST 仍需要匹配 CUDA11.8/CUDA12.1 的 `cudnn64_8.dll` runtime；TRT11/CUDA12.9 仍需要匹配版本的 `nvinfer_11.dll`、`nvinfer_plugin_11.dll`、`nvonnxparser_11.dll`。缺失资产不得通过混用 CUDA13.2 runtime 或伪造通过状态绕过。
3. YoloVision 仍缺 det/cls/seg/obb/pose/sem 全任务真实模型 proof；模型矩阵、候选资源、模板和 source-tree build 不得冒充 real-case evidence。
4. TensorRtExec CLI/WinForms 已完成 load-engine readonly readback 与 bounded identity runtime 的共享服务路径；仍需继续补齐外部真实模型 expected output、dynamic shape 多 profile 案例、precision/timing 真实模型报告、WinForms 端人工/自动执行截图证据和 package-consumer executable path。
5. final release close dashboard 当前仍为 `blocked-final-release-close-owner-action-required`，需要 owner proof、Linux runner、redistribution disposition、signing decision 和 post-publish clean consumer 记录。
6. NuGet.org/GitHub Packages/GitHub Release 尚未执行，本阶段没有使用 publish token，也没有创建或上传公开 release。

## 2026-06-18 历史覆盖基线

以下表格保留首次完成度审查时的基线口径，用于观察阶段变化。涉及当前 readiness、包集合、deferred triage 和 release gate 时，应以 2026-07-10 最新机器可读 artifacts 为准。

## 已发布 Runtime 目标矩阵

| 平台目标 | TensorRT/CUDA/cuDNN 组合数 | 当前状态 |
| --- | ---: | --- |
| `win-x64` | 6 | 按方案已完成 |
| `linux-x64.ubuntu22.04` | 6 | 按方案已完成 |
| `linux-x64.ubuntu24.04` | 3 | 已完成 NVIDIA 官方支持的现代组合 |
| `linux-x64.ubuntu20.04` | 3 | 已完成建模的 legacy 组合 |
| ARM / Jetson / 非 Ubuntu | 0 | 仅作为未来独立包线保留 |

## Manifest 级 API 完成度

`已实现` 表示 manifest 中的非 deferred 入口。`Deferred` 表示接口已经识别并登记，但当前仍是安全边界占位实现，或者尚未提升为正式可用实现。

| 模块 | 版本线 | Manifest 总数 | 已实现 | Deferred | 已实现比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| common | common | 11 | 11 | 0 | 100.0% |
| cuda | common | 524 | 355 | 169 | 67.7% |
| tensorrt | 8 | 793 | 536 | 257 | 67.6% |
| tensorrt | 10 | 873 | 632 | 241 | 72.4% |
| tensorrt | 11 | 1069 | 887 | 182 | 83.0% |
| tensorrt | common | 1 | 1 | 0 | 100.0% |
| total | all | 3271 | 2422 | 849 | 74.0% |

## TensorRT 头文件到桥接层覆盖

头文件扫描显示，所有被扫描到的官方 TensorRT 接口都能匹配到 manifest/source 记录。表格后半部分进一步区分了真实非 deferred 覆盖和 deferred 占位项。

| TensorRT 包 | CUDA 变体 | 扫描到的官方接口 | Manifest/source 已匹配 | 非 deferred | Deferred |
| --- | --- | ---: | ---: | ---: | ---: |
| TensorRT 8.6.1.6 | CUDA 11.8 | 847 | 847 | 580 | 267 |
| TensorRT 8.6.1.6 | CUDA 12.1 | 847 | 847 | 580 | 267 |
| TensorRT 10.11.0.33 | CUDA 11.8 | 879 | 879 | 594 | 285 |
| TensorRT 10.11.0.33 | CUDA 12.9 | 879 | 879 | 594 | 285 |
| TensorRT 11.0.0.114 | CUDA 12.9 | 901 | 901 | 688 | 213 |
| TensorRT 11.0.0.114 | CUDA 13.2 | 901 | 901 | 688 | 213 |

## CUDA Runtime 头文件到桥接层覆盖

本机安装了 CUDA Toolkit `11.6` 和 `12.3`，所以扫描结果中也包含它们；但这两个版本不是当前方案中已发布 runtime 矩阵的一部分。

| CUDA Toolkit | 扫描到的 Runtime 函数 | Manifest/source 已匹配 | 非 deferred | Deferred |
| --- | ---: | ---: | ---: | ---: |
| 11.6 | 268 | 268 | 157 | 111 |
| 11.8 | 273 | 273 | 160 | 113 |
| 12.1 | 277 | 277 | 162 | 115 |
| 12.3 | 292 | 292 | 163 | 129 |
| 12.9 | 307 | 307 | 165 | 142 |
| 13.2 | 330 | 330 | 175 | 155 |

## 主要 Deferred 接口组

下面是当前规模最大的未完成或刻意延后接口组。

| 领域 | Deferred 数量信号 | 说明 |
| --- | ---: | --- |
| TensorRT plugins | TRT8: 69, TRT10: 88, TRT11: 88 | Plugin V2/V3、plugin creator、plugin registry 修改/资源所有权，以及 runtime/build plugin 回调都需要专门设计安全 plugin 桥接层。 |
| CUDA graph 高级 API | 80 | Graph node 参数修改/查询、graph user object、dependency 变体、memory node 和较新的 graph 功能都需要更完整的对象模型和生命周期建模。 |
| CUDA 较新 runtime/driver-adjacent API | 59 | 包括 library/kernel 查询、driver entrypoint/export table、green/execution context、日志 API 和 device resource API。 |
| TensorRT ONNX/parser 高级 API | TRT8: 39, TRT10: 17, TRT11: 34 | ONNX config、parser refitter、model-proto/weight-descriptor 路径需要处理模型 buffer 所有权和诊断对象生命周期。 |
| TensorRT 诊断/回调 | TRT8: 21, TRT10: 32, TRT11: 22 | ErrorRecorder、Profiler、DebugListener、ProgressMonitor、Logger 的回调和生命周期桥接尚未完全提升为正式 API。 |
| TensorRT algorithm/timing API | TRT8: 29, TRT10: 26 | IAlgorithm、IAlgorithmContext、IAlgorithmIOInfo、IAlgorithmSelector 大多仍处于 deferred 状态。 |
| TensorRT allocator/output allocator | TRT8: 12, TRT10: 15, TRT11: 10 | GPU allocator、async allocator、output allocator 需要 callback/vtable 加托管生命周期设计。 |
| TensorRT calibration API | TRT8: 10, TRT10: 14 | INT8 calibrator 系列仍处于 deferred 状态。 |
| TensorRT builder/config/runtime 边缘 API | mixed | 部分旧版/新版 builder config、runtime serialization、stream reader/writer 和 execution context 兼容性 API 仍处于 deferred 状态。 |

## 高层 C# Wrapper 信号

当前底层 native/PInvoke 覆盖明显宽于高层 C# 易用封装。启发式扫描显示，TensorRT 扫描接口中大约 68-72% 能找到疑似高层 wrapper 引用；CUDA 的高层覆盖则明显更薄，因为目前大多数 CUDA 覆盖仍停留在 native interop 加少量面向用户的 helper。

| 领域 | 当前判断 |
| --- | --- |
| TensorRT 核心部署路径 | 基本可用：runtime、builder、builder config、network、engine、execution context、ONNX parser、dynamic shape/profile、refit、timing cache，以及大量 layer API 已存在。 |
| TensorRT 高级扩展路径 | 尚不完整：plugins、自定义 allocators、output allocators、error recorder/profiler callbacks、stream reader/writer、calibrators、algorithm selector。 |
| CUDA 部署路径 | 常用 stream/event/memory/device 操作，以及部分 graph/module/kernel helper 可用。 |
| CUDA 完整 runtime 对齐 | 尚不完整：大量高级 graph、IPC、texture/surface/external memory、library、execution context、log 和 CUDA 13 时代 API 仍处于 deferred 状态。 |

## 审查备注

- `eng/Export-InterfaceCoverageMatrix.ps1` 对本地 TensorRT 与 CUDA 头文件给出了 100% manifest/source 匹配，因此在已扫描头文件范围内，没有发现明显未纳入追踪的官方接口。
- `eng/Export-ApiInventory.ps1` 报告有 165 个 manifest 条目没有匹配到 source，全部位于 TensorRT 10 deferred manifest。手工抽查显示，这些条目通过 `JYPPX_TRT10_PLUGIN_DEFERRED_STUB` 等 deferred stub 宏实现，所以脚本低估了宏生成/宏展开形式的 source export。从功能角度看，它们仍然是 deferred，不能算作已完成。
- 已生成的 interop 产物存在，且足以支撑当前审计脚本：148 个 manifest 文件、3681 条 manifest API 记录、生成的 NativeMethods、生成的 entrypoint names，以及生成的 API catalog 文件。

## 建议的下一批工作

1. 第一优先级仍是在 CUDA 13 capable compatible host 上消费当前四角色 split set，采集真实 host metadata、stdout/stderr、package/log SHA256，并通过 strict runtime-proof validator。
2. 扩展 YoloVision det/cls/seg/obb/pose/sem 真实模型证据，优先完成至少 det、cls、seg 三条可重复运行路径，并保持 source-tree `real-model-runtime` 与 package-consumer proof 分离。
3. 在不混用 ABI/runtime 的前提下补齐 TRT8 cuDNN8 runtime 与 TRT11/CUDA12.9 runtime assets；资产不可获得时生成精确 acquisition/owner-action 记录，不重复执行必然失败的 runtime。
4. 收口 TensorRtExec CLI/WinForms 与 OnnxToEngine 的 trtexec-like 参数、engine load/readback、报告 schema、动态 shape/profile 和 runtime 路径一致性。
5. 扩展 RNNv2 setter、algorithm selector、calibrator、allocator/output allocator 等剩余 C-tier 时，必须先完成 owner/lifetime/callback 设计门禁，不允许公开裸 pointer。
6. 后续代码变化只补跑受影响类并刷新 308 类覆盖汇总；compatible-host proof、owner authorization、Linux runner proof 和 post-publish proof 未全部完成前，保持 `canPublishPublicly=false`、`canCloseReleaseIssue=false`，继续禁止自动 publish。

## 2026-07-18 实际 Vendor Symbol Safe-Deferred Uplift

本阶段从 deferred inventory 先完成 vendor 头文件、import library/DLL 和跨版本差异审计，再选择两个内聚模块：built-in plugin 初始化（TRT8/10/11）与 legacy ONNX `parseWithWeightDescriptors`（TRT8/10；TRT11 已移除）。候选审计记录在 `artifacts/interface-coverage/trt8-safe-deferred-candidate-audit.md` 与 `.json`。

### Promotion 结果

- 新增 5 个非 deferred manifest entry；原有 5 个 deferred entry 全部保留，通过 `Global::initLibNvInferPlugins` 和 `IParser::parseWithWeightDescriptors` 显式 alias 归并为 `implemented-with-deferred-history`。
- native ABI 使用 caller-buffer 字符串和 pinned/caller-buffer 模型字节；plugin 初始化只在同步 vendor 调用期间借用 logger，不创建、返回或接管 plugin 对象。
- C++ exception 与 Windows SEH 均在 native helper 内转换为 bridge status；TRT8、TRT10、TRT11 translation unit 各自保留独立 version guard。
- TRT11 不新增已经被 vendor 删除的 parser 方法，继续使用已有 model-proto 生命周期路径。
- public C# surface 只暴露 `InitializeBuiltInPlugins`、`TryInitializeBuiltInPlugins`、`ParseWithWeightDescriptors` 和 copied diagnostics，不公开裸 `IntPtr`、`nint`、`UIntPtr`、SafeHandle 或 vendor pointer。

### Verification

- binding generator 幂等验证：`184 manifests / 3924 records`。
- interface coverage：TRT8 `753 implemented / 94 deferred-only`，TRT10 `761 / 118`，TRT11 `814 / 87`。
- native build：TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 通过；TRT8 parser 因本机缺 `cudnn64_8.dll` 由既有 CMake 条件禁用，但 plugin entry 完成真实 vendor link。
- ABI declaration/export parity：TRT8、TRT10、TRT11 均 `MissingDeclarations=0 MissingExports=0`。
- `TrtSafeDeferredUpliftTests`：3/3；TRT10/CUDA12.9 PluginRegistry smoke 真实初始化 built-in plugins；OnnxToEngine smoke 真实完成 legacy weight-descriptor parse、engine round-trip、enqueue 与 output match。
- bridge-only package consumer 的 `PackageReference` compile surface 新增验证两个公开 wrapper；分类仍为 `compile-surface-proof`，不是 package-consumer runtime proof。

本阶段不改变长期安全边界：callback trampoline、plugin create/register/deregister/load、allocator/resource acquire/release、device/borrowed pointer、RNNv2 setter 和 consistency checker 继续 deferred。未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

### 本阶段收尾复核（2026-07-18）

- Release solution build：成功，0 error；输出保留 5 个既有 nullable warning，未新增本阶段 warning。
- native rebuild：六套配置全部成功：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA11.8、TRT10/CUDA12.9、TRT11/CUDA12.9、TRT11/CUDA13.2。
- ABI declaration/export parity：上述六套配置均 `MissingDeclarations=0 MissingExports=0`。
- package：`JYPPX.TensorRT.CSharp.API 4.0.0` managed pack 成功；TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 三个 bridge-only `PackageReference` consumer 均 restore/build 成功，0 warning / 0 error，分类保持 `compile-surface-proof`。
- focused ProjectQuality：`TrtSafeDeferredUpliftTests` 与 `BridgePackageConsumerTests` 共 7/7 通过；bindings 重新生成并幂等校验为 `184 manifests / 3924 records`。
- strict classification：`classification-audit-passed-non-proof-boundaries-intact`，`FindingCount=0`；标准 strict release quality gate：`release-quality-gate-passed`，`RequiredFailureCount=0`。
- owner convergence：structural `9/9`、accepted `0/9`、gates `2/3`、validation blocker `0`，仍是 owner-action blocked；`canPublishPublicly=false`、`canCloseReleaseIssue=false`。
- 带 `-RequirePackageInventory` 的额外 gate 仍报告本地 split package inventory 缺少 3 个角色包；这是当前本地资产门禁的独立阻塞，不改变标准 strict gate 结果，也不构成公开发布许可。
- 本轮未执行 NuGet push、GitHub Packages publish、GitHub Release upload、issue close 或其它公开发布副作用；commit `5d63087` 已通过 SSH 推送，GitHub Actions run `29647634173` 已 `completed/success`（source-quality 及 bounded ProjectQuality shard smoke 通过，条件大任务 skipped）。

## 2026-07-18 IBuilderConfig Scalar Alias-History Promotion Review

本次继续审计上一批已经存在真实实现、但 deferred inventory 仍保留占位入口的四个 scalar controls：`getAvgTimingIterations`、`setAvgTimingIterations`、`getBuilderOptimizationLevel` 和 `setBuilderOptimizationLevel`。审计记录位于 `artifacts/interface-coverage/trt-builder-config-scalar-candidate-audit.md` 与 `.json`。

### Promotion decision

- TRT8、TRT10、TRT11 的 `NvInfer.h` 均有对应 scalar vtable method；现有 native implementation、version guard、manifest 和 C# wrapper 已完整存在，不重复添加第二套 ABI。
- `eng/Export-InterfaceCoverageMatrix.ps1` 现在把真实 manifest IDs 放在 explicit alias map，把 `*-deferred` IDs 单独放在 `deferredHistoryAliasMap`；两者同时命中时状态仍为 `implemented-with-deferred-history`。
- TRT8/10/11 的旧 twenty-third-batch deferred manifest 与 diagnostic stub 均保留；未把 deferred stub 误报为正式支持路径，也没有删除历史记录。
- scalar 输入/输出只经过 typed opaque builder-config owner 和 primitive integer；无 callback、borrowed pointer、数组、外部资源或 ownership transfer。

### Verification

- Coverage exporter 重跑成功；四个接口在 TRT8/10/11 的 24 行均为 `implemented-with-deferred-history`，每行同时匹配 real entry 与 deferred history entry。
- `BuilderConfigScalarControlsTests` 专项为 `5/5` 通过；测试锁定 real/history alias 分离、TRT8/10/11 manifest、版本 guard、审计证据、scalar wrapper/smoke 和无裸指针 public surface。
- 本轮未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close；owner convergence、runtime proof 与公开发布边界保持不变。

## 2026-07-19 CUDA Stream-Capture Variant Safe-Deferred Uplift

本批在 CUDA runtime deferred inventory 中先核对了本机 CUDA 头文件、运行时导入库、跨版本 guard 和 native link 结果，再选择三个没有 callback、device pointer 或外部资源 ownership 的 stream-capture variant：`cudaStreamGetCaptureInfo_ptsz`、`cudaStreamUpdateCaptureDependencies_ptsz` 和 `cudaStreamUpdateCaptureDependencies_v2`。候选审计记录位于 `artifacts/interface-coverage/cuda-stream-capture-variants-candidate-audit.md` 与 `.json`。

### Promotion 结果

- 新增 3 个非 deferred CUDA manifest entry；原有 stream-device-boundary deferred records 保留，通过 explicit alias/history 归并为 `implemented-with-deferred-history`。
- `cudaStreamGetCaptureInfo_ptsz` 只返回 copied capture status/id；dependency variants 只接受 managed graph-node token，并在同步 vendor call 内 pinned/copy edge data。
- CUDA 11.x、12.x、13.x 分别使用独立 guard；CUDA 11.8 头文件未声明的 `cudaStreamUpdateCaptureDependencies_ptsz` 通过受 guard 保护的 vendor declaration 处理，没有对 CUDA 13.2 伪造 12.x API。
- public C# surface 不暴露 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、device pointer 或 borrowed vendor pointer。

### Verification

- binding generator/output validation：`188 manifests / 3945 records`，幂等通过。
- coverage export 重跑成功；三个函数的适用 toolkit rows 均为 `implemented-with-deferred-history`。
- 新增 `CudaStreamCaptureVariantsUpliftTests`：`4/4`；相关 CUDA graph/stream tests：`30/30`。
- 四套 native configuration 成功：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2；ABI declaration/export parity 均为 `MissingDeclarations=0 MissingExports=0`。
- C 盘审查未发现本批下载的 CUDA、TensorRT、cuDNN 或模型；构建输出位于 E 盘 `build-out`，C:\Users\guoji\AppData\Local\Temp 下无本批同名构建目录。用户已有 Downloads、NuGet 和工具缓存未删除。

本批仍不改变长期安全边界：callback trampoline、allocator/resource、borrowed/device pointer、plugin lifecycle、RNNv2 setter 和 consistency checker 继续 deferred；未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-19 CUDA Stream-Capture-To-Graph Owner-Safe Uplift

本批继续从 CUDA deferred inventory 做 vendor-first 审计，核对
`cudaStreamBeginCaptureToGraph` 在 CUDA 12.3、12.9、13.2 的
`cuda_runtime_api.h`、import library、runtime DLL symbol 和独立
`CUDART_VERSION >= 12030` guard；CUDA 11.x/12.1 没有该入口，继续保持
deferred。候选审计记录在
`artifacts/interface-coverage/cuda-stream-capture-to-graph-candidate-audit.md`
及 `.json`。

### Promotion 结果

- 新增 `cudaStreamBeginCaptureToGraph` 的 owner-safe begin entry，并复用
  `cudaStreamEndCapture` 作为 session terminator；旧 deferred manifest 没有删除。
- `CudaStreamCaptureToGraphSession` 在 capture 期间保留 stream/graph owner，
  两个 wrapper 的 `Dispose()` 会阻止提前释放；End 验证 CUDA 返回的是同一
  graph handle，不创建第二个 managed graph owner。
- dependency token 和 `CudaGraphEdgeData` 只在同步 native call 内复制/pin；
  C++ exception、分配失败与 Windows SEH 在 bridge 内转换为 status。
- public C# surface 仅公开 typed `CudaStream`、`CudaGraph` 和 session，未暴露
  `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、device pointer 或 borrowed pointer。

### Verification

- binding generator/output validation：`189 manifests / 3947 API records`，
  第二次生成幂等通过；coverage export 的 CUDA 12.3/12.9/13.2 行均为
  `implemented-with-deferred-history`。
- TRT8、TRT10、TRT11 ABI declaration/PE export parity 均为
  `MissingDeclarations=0 MissingExports=0`；四套 native Release preset
  TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2 成功。
- `CudaStreamCaptureToGraphUpliftTests`、相邻 stream/graph 专项共 `13/13`；
  managed Release solution 为 `0 errors`，保留既有 `5` 个 nullable warnings。
- managed pack 成功；TRT8/CUDA12.1、TRT10/CUDA12.9、TRT11/CUDA13.2
  bridge-only consumer 均 restore/build `0 warning / 0 error`，分类仍为
  `compile-surface-proof`。
- TRT10/CUDA12.9 `CudaGraphSmokeRunner` 完成 graph round trip，并输出
  `ToGraph=True Nodes=1`；`ptsz`/CUDA13-only 路径按版本能力记录受控 skip。
- strict classification audit：`classification-audit-passed-non-proof-boundaries-intact`
  且 `FindingCount=0`；标准 strict release quality gate：
  `release-quality-gate-passed` 且 `RequiredFailureCount=0`。

### C/E 盘清理

- 构建与 package consumer 输出均位于 E 盘仓库；本批删除了
  `E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\build-out`。
- smoke 产生的唯一可归因 C 盘目录
  `C:\Users\guoji\AppData\Local\Temp\jyppx-cuda-graph-smoke\48a961776a64499fbd34c9ba34026d86`
  已删除；`C:\jyppx-pkgcache\...` package restore 临时目录由脚本自动删除。
- 未在 C 盘下载或保留 CUDA、TensorRT、cuDNN、模型或 nupkg；
  `C:\Users\guoji\Downloads`、用户 NuGet 缓存、Codex/工具缓存及系统 CUDA
  安装目录均未删除。

本批仍不改变长期安全边界：callback trampoline、allocator/resource、borrowed/device pointer、plugin lifecycle、RNNv2 setter 和 consistency checker 继续 deferred；未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。上述 smoke/build/package consumer 结果不等于 clean public package-consumer runtime proof、post-publish proof 或 release-close approval。

## 2026-07-19 CUDA Conditional Graph Owner-Safe Uplift

本批从 CUDA conditional graph deferred inventory 选择 v1/v2 handle、conditional
node 与 body topology 查询，先完成 CUDA 12.3、12.9、13.2 header、import
library、DLL export 和版本 guard 核对，再实现 bridge-owned metadata。审计记录位于
`artifacts/interface-coverage/cuda-conditional-graph-candidate-audit.md` 与
`.json`。

### Promotion 结果

- 新增 v1 handle、CUDA 13.2 v2 handle、conditional node、body count/topology
  查询和 body-local empty-node 插入；原 deferred manifest 保留，并通过 real
  alias/history 归并为 `implemented-with-deferred-history`。
- child/body graph 句柄不离开 native bridge；`CudaGraphConditionalHandle` 与
  `CudaGraphConditionalNode` 只持有 bridge metadata。parent graph 在 metadata
  wrapper 活跃时拒绝 Dispose，generic owner-scoped node destroy 不会误接管
  conditional node。
- public C# surface 未暴露 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、device
  pointer、callback 或 borrowed child graph。body capture、kernel/raw-pointer
  node、外部资源和 callback ownership 继续 deferred。

### Verification

- binding generator/output validation：`190 manifests / 3958 API records`，幂等
  通过；coverage export 在 CUDA 12.3/12.9/13.2 目标行保持
  `implemented-with-deferred-history`。
- native build 通过：TRT8/CUDA11.8、TRT8/CUDA12.1、TRT10/CUDA12.9、
  TRT11/CUDA12.9、TRT11/CUDA13.2；ABI declaration/PE export parity 未新增
  missing。
- CUDA 12.9 conditional smoke 通过 IF node、两个 body、body topology、default
  value、instantiate/launch 和 active-owner dispose rejection；CUDA 13.2 本机
  smoke 在 CUDA error 35 处受 driver/runtime 边界阻断，未冒充 13.2 runtime proof。
- `CudaConditionalGraphUpliftTests` `4/4` 通过；与 ABI/coverage/相邻 graph
  专项组合为 `59/59` 通过；binding output validation 通过。

### C/E 盘清理与发布边界

- 删除本批 E 盘 `build-out` 和 `artifacts/test-temp-cuda-conditional-build.log`。
- 删除本批 .NET 临时文件：`C:\Users\guoji\AppData\Local\Temp` 下 9 个
  可明确归属本轮的随机占位/`Microsoft.NET.Workload_*.log` 文件。
- 未发现本批下载到 C 盘的 CUDA、TensorRT、cuDNN、模型或 nupkg；用户
  Downloads、NuGet、Codex、工具缓存和系统 CUDA 安装目录均未删除。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue
  close；上述 native/build/smoke/compile-surface 结果不等于 clean public
  package-consumer runtime proof、post-publish proof 或 release-close approval。

## 2026-07-19 CUDA 13 Device-Resource Pointer-Free Snapshot Uplift

本阶段继续从 CUDA 13 deferred inventory 做 vendor-first 审计，选择
`cudaDeviceGetDevResource`、`cudaExecutionCtxGetDevResource` 和
`cudaStreamGetDevResource` 三条查询入口。官方返回值包含 tagged union、opaque
workqueue 状态和 `nextResource` 链指针；本桥只复制 type、SM/workqueue 标量、opaque
workqueue 存在性和 `HasNextResource` 标记，绝不把链指针或 union padding 交给托管层。

### Promotion 与边界

- 新增 3 个 CUDA 13 real manifest entry；原有 device/stream/execution-context
  deferred entries 保留，通过 real alias 与 history alias 归并为
  `implemented-with-deferred-history`。
- native 使用独立 `CUDART_VERSION >= 13000` guard；CUDA 11/12 返回
  `NotSupported`。green context、resource split、descriptor 生成和 opaque
  workqueue 复用继续 deferred。
- public C# 只暴露 `CudaDevResourceSnapshot` 值类型和 typed owner 查询；没有
  `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、vendor union 或 `nextResource`。

### Verification

- binding generator/output validation：`191 manifests / 3961 API records`，幂等通过。
- coverage export：CUDA 13.2 三条目标行均为
  `implemented-with-deferred-history`；旧 deferred manifest 保留。
- focused `CudaDevResourceSnapshotUpliftTests`：`5/5`；CUDA 相关
  ProjectQuality 集合：`114/114`。
- TRT11/CUDA13.2 native configure/build 成功，ABI surface
  `MissingDeclarations=0 MissingExports=0`；保留既有 vendor deprecation 与
  constant-condition warnings。
- CudaSmokeRunner 在本机最早的 `cudaRuntimeGetVersion` 因 CUDA error 35 停止，
  因此只记录 compatible-host blocker，不声称 device-resource runtime proof。

### C/E 盘与发布边界

- 构建输出位于 E 盘 `build-out`；本次未下载 CUDA、TensorRT、cuDNN、模型或 nupkg
  到 C 盘。smoke 输出日志位于 E 盘 `artifacts/smoke`。
- 未删除 `C:\Users\guoji\Downloads`、NuGet 缓存、Codex/工具缓存或系统 CUDA
  安装；只核查并清理本轮明确可归因的临时文件。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
  native/build/compile-surface 结果不等于 clean package-consumer runtime proof。

## 2026-07-19 TensorRtExec Timing-Iteration Builder Readback Uplift

本阶段从应用层高价值 parity 缺口选择 `--avgTiming` / `--minTiming`，复用已经
存在且跨 TRT8/10/11 可用的 `TensorRtBuilderConfig` timing setter/getter。该批不改变
TensorRT/CUDA manifest 或 ABI surface，而是让 TensorRtExec 的 CLI、WinForms 共享
build service 真正应用并回读 builder 配置，同时修复 native bridge/vendor runtime
缺失时未处理异常的问题。

### Promotion 与边界

- `--avgTiming` 在 TRT8、TRT10、TRT11 真实 build 中调用
  `SetAverageTimingIterations` / `GetAverageTimingIterations`，日志记录
  `RequestedIterations`、`ReadbackIterations`、`ReadbackMatch` 和
  `EvidenceBoundary=builder-config-readback-only`。
- `--minTiming` 只在 TRT8 使用 legacy compatibility setter/getter；TRT10/11 保持
  parse-only，并在 diagnostics 中明确版本原因。两者都不能证明 tactic quality、benchmark
  performance、模型正确性或 package-consumer-runtime。
- 无法加载 `jyppxtrtbridge`，或 bridge 能加载但 vendor runtime 创建失败时，
  TensorRtExec 现在生成 `dependency-probe-only` report/sidecar，而不是抛出未处理的
  `DllNotFoundException` 或结构化 TensorRT runtime exception。
- parity matrix、feature matrix、release gap list、外部 ONNX report、option layering
  和 getting-started 文档已同步；没有新增裸指针、callback、borrowed handle 或 ownership
  surface。

### Verification

- binding generator/output validation：`191 manifests / 3961 API records`，重复生成幂等通过；
  本批没有 manifest 改动。
- 完整 solution Debug build：`0 warnings / 0 errors`。
- TensorRtExec/OnnxToEngine/文档/质量定向集合：`41/41` 通过。
- CLI no-bridge smoke：生成 `dependency-probe-only` report，exit code 0，未崩溃；
  TRT8 bridge-only smoke：在 vendor runtime 创建结构化异常处生成同类 skip report，
  保留 `TrtexecTiming` diagnostics。两次均把 engine/report 路径放在 E 盘。
- 全量 ProjectQuality 曾启动并暴露现有 release-package inventory 与 owner-input
  artifact 断言失败；该套件执行大量共享发布脚本后已停止，不能报告为全量通过。失败集中在
  `ReleaseCandidatePackageInventoryTests` 与 `FinalOwnerExecution*`，与本批定向测试和
  managed build 无关，仍需独立 release-artifact 环境收口。

### C/E 盘与发布边界

- 本批 smoke 临时目录 `artifacts/test-temp/timing-iterations` 只位于 E 盘，完成后删除；
  不使用 C 盘作为 engine/report 输出路径。
- 本批没有下载 CUDA、TensorRT、cuDNN、模型或 nupkg 到 C 盘；不删除用户 Downloads、
  NuGet/Codex/工具缓存、历史 CI runner 或系统 CUDA 安装。
- 未执行 NuGet push、GitHub Packages/Release 上传或 issue close；builder readback、
  dependency skip、managed build 和定向质量门禁均不等于 clean public package-consumer
  runtime proof、post-publish proof 或 release-close approval。
## 2026-07-19 TRT11 profiler interface-info proof closure

本批收口 `IProfiler::getInterfaceInfo [TRT11]` 的已有 safe alternative：native caller-buffer/scalar copy、managed `TryGet...` 和 pointer-free `GetInterfaceMetadataSnapshot` 已由专项质量门覆盖。TRT8/TRT10 保持 controlled unsupported；`trt11-profiler-get-interface-info-deferred` 与 deferred source 历史记录保留。该批 evidence kind 为 `build-and-source-quality-proof`，`isRuntimeExecutionProof=false`、`isPackageConsumerRuntimeProof=false`，不能替代真实 host、真实模型、package consumer 或公开发布证据。

## 2026-07-19 Parser preflight copied readback

本批将已有 `IParser::getError`、`IParser::getErrorCount` 和 `IParser::isSubgraphSupported` safe wrapper 接入 `OnnxEngineBuildResult`，新增 pointer-free `ParserPreflightSnapshot`。构建 parse 后报告复制的 error count、diagnostic summary、Identity operator support，以及 TRT10/TRT11 模型/子图支持计数；TRT8 在 vendor 未暴露子图查询时保留 controlled `unavailable`。旧 deferred manifest/history 保留，未通过报告字段改变 coverage。

证据 artifact：`artifacts/interface-coverage/parser-preflight-readback-proof.json` 和 `.md`。Tools、TensorRtExec build 通过，TensorRtExec report schema 定向测试 `5/5` 通过。该批 evidence kind 为 `copied-parser-preflight`，`PointerFreeCopiedSnapshot=true`，但 `CanPromoteRuntimeProof=false`、`CanPromoteReleaseProof=false`；不能替代 real-model-runtime、package-consumer-runtime、post-publish 或 release-owner proof。

## 2026-07-19 B-tier Work Package 状态纠偏与 Proof Ledger

本批审计发现 `deferred-btier-implementation-work-package.json` 仍把 `btier-001` 到
`btier-045` 标成下一批工程任务，但这些项目早已由四批文档与 ProjectQuality 门禁完成
source-quality proof closure。该状态漂移会让后续开发重复选择 `btier-006` 到
`btier-021` 等旧任务，因此新增 `deferred-btier-work-item-proof-closure-ledger.json/.md`
作为可机读事实源，并让 work-package 生成器输出逐项 `workItemState`、
`closedWorkItemCount=45` 和 `remainingWorkItemCount=0`。

### 状态与边界

- 45 项全部标记为 `source-quality-proof-closed`，每项继续保留 safe alternative manifest、
  deferred history manifest、native/source、pointer-free managed wrapper、docs 和测试证据。
- 最终 blocker dashboard 不再提示重复执行 `btier-041..045`，而是要求从新的 candidate
  audit、真实 external model/runtime gap 或已通过 ownership design gate 的候选中选批。
- ledger 和 work package 均保持 `canDeleteDeferredRecords=false`、
  `isRuntimeExecutionProof=false`、`isPackageConsumerRuntimeProof=false`、
  `canPromoteReleaseProof=false`；源码质量闭环不等于真实运行或公开发布许可。

### Verification

- binding generator/output validation：`191 manifests / 3961 API records`，幂等通过。
- 完整 solution Debug build：`0 warnings / 0 errors`；定向状态门禁 `5/5`，扩展
  B-tier/scalar/parser/public-handle/classification 集合 `23/23`。
- TRT8/CUDA12、TRT10/CUDA12、TRT11/CUDA12、TRT11/CUDA13 native 增量构建通过。
- ABI declaration/PE export parity：TRT8 `991/991`、TRT10 `1086/1086`、TRT11
  `1233/1233`，missing declaration/export 均为 `0`。
- strict classification audit：finding `0`；strict release quality gate：required failure `0`。

### C 盘与发布边界

- 本批没有下载 TensorRT、CUDA、cuDNN、模型或 NuGet 包到 C 盘，Downloads 时间窗新增为
  `0`。清理本批 dotnet/MSBuild 产生的 9 个 workload 小日志和 19 个空临时目录。
- 3 个由正在运行的 Codex/桌面进程锁定的 0 字节 `.tmp` 保留，不强制终止共享进程；
  NuGet、CUDA、.NET、Codex 与系统缓存未触碰。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-19 官方 YOLOX-S 源码树真实运行闭环

本阶段不再复用来源不清的本地 YOLOv8s 资产，而是固定 YOLOX 官方
`0.1.1rc0` release 与提交 `e1052df71842031413f6030723c3607b839c80ce`，完成
Apache-2.0 provenance、E 盘 acquisition、官方预处理、TRT10 build、真实图片 enqueue、
raw grid/stride 解码、NMS、JSON/SVG 与严格 sample-run evidence 的端到端闭环。

### 实现与真实结果

- 新增 `YoloModelFamily.YoloX`，保持旧枚举值稳定；profile 支持 `yolox/x`，并明确
  限定为 detection-only。
- `YoloXOutputDecoder` 对 `[1,8400,85]` boxes-first raw output 按 stride
  `8/16/32` 执行 `(xy+grid)*stride` 与 `exp(wh)*stride`，之后复用 objectness、
  class-aware NMS 和 top-k；rank、layout、input shape、box count、class count 与非有限值
  都有显式错误边界。
- YOLOX profile 默认 `NCHW + BGR + float32 0..255 + fill 114 + top-left
  letterbox`；普通 family 继续使用既有居中 letterbox。alignment 已进入控制台、output
  JSON、preflight 和 schema。
- acquisition manifest 固定 model、LICENSE、dog image、COCO classes 与官方预/后处理
  参考源码的 URL、length、SHA256；脚本拒绝 C 盘输出，并可离线复核派生 PPM/labels。
- TRT10.11 FP32 engine build `PASSED`：input `images [1,3,640,640]`，output
  `output [1,8400,85]`，engine `48,241,100` bytes。
- YoloVision 真实运行 `Passed=True`，elapsed `10.012 ms`，检测 5 项：最高
  `bicycle=0.954841`，并命中 `dog=0.913382`。output report strict blocker `0`；
  sample-run validator 为 `real-model-runtime`、`CanPromoteRealModelRuntime=True`、
  owner-action `0`。

### Verification

- binding generator/output：`191 manifests / 3961 API records`，重复生成幂等。
- solution Release build：`0 errors`；保留 5 个既有 test nullable warning。
- YoloVision 全相关 ProjectQuality：`88/88`；新增合成 grid/stride、官方预处理、
  unsupported contract、acquisition/publish boundary 防回归门禁。
- sample asset manifest audit 现在同时扫描 template 与 example：`9` 份 manifest、finding
  `0`；YOLOX example 的 sidecar/sample-run cross-check 均为 `checked`。
- native 增量 build：TRT8/CUDA12、TRT10/CUDA12、TRT11/CUDA12、
  TRT11/CUDA13 全部通过。
- ABI declaration/PE parity：TRT8 `991/991`、TRT10 `1086/1086`、TRT11
  `1233/1233`，missing declaration/export 均为 `0`。
- TRT10 bridge-only PackageReference consumer restore/build/probe：`0 warning / 0 error`，
  `NativeDependencyStatus=ready`；证据仍是 compile-surface/dependency proof。
- strict classification audit finding `0`；strict release quality required failure `0`；
  final dry-run 在 `-AllowRuntimeSmokeBlocked` 边界下通过。未带该开关时继续因真实外部
  public package runtime proof 缺失而阻断，未把长期 owner blocker 改写成通过。

### C/E 盘与发布边界

- 模型、图片、engine、labels 与 tensor 共约 `90.7 MB`，全部位于外层 E 盘
  `downloads/yolox-apache`，未提交到仓库。
- C 盘全用户树/Temp/Downloads 审计未发现 YOLOX 资产；清理 6 个空测试目录与 bridge
  consumer 用完的空 `C:\jyppx-pkgcache` 根目录。Visual Studio/Tencent 临时文件、
  CUDA、NuGet、Codex 与用户文件未触碰。
- `real-model-runtime` 仅描述本次源码树真实 TensorRT 运行。`isPackageConsumerRuntime=false`、
  `publicRedistributionOwnerApproval=false`、`canPublishPublicly=false`；没有伪造 reviewer、
  owner signature 或公开发布批准。
- 未执行 NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。

## 2026-07-23 Dual Package Public Article Expansion

本阶段继续公开文章矩阵收口，扩写 `docs/articles/zh-cn/publishing/package-strategy-public-article.md`，
把 GitHub full runtime 包与 NuGet small bridge/core 包的双路线、runtime split roles、
runtime package key、local candidate evidence 和真实 package-consumer-runtime proof 边界写成外部
读者可直接理解的文章。

### 实现

- 文章明确 `JYPPX.TensorRT.CSharp.API` 托管 API、C++ bridge DLL、`Bridge`、
  `CudaCudnn`、`TensorRt` split runtime 包，以及 full runtime 包之间的交付关系。
- 增加 runtime package key 示例：`win-x64-trt10.11-cuda12.9-cudnn9.22`、
  `win-x64-trt11.0-cuda13.2-cudnn9.22`、`win-x64-trt8.6-cuda11.8-cudnn8.9`。
- 将 `Export-DualPackagePublishPreflightMatrix.ps1`、
  `Export-FinalOwnerExecutionChecklist.ps1`、`release-docs-and-nuget-metadata-audit.json`
  和 `release-candidate-package-inventory.md` 定位为本地 candidate/pre-publish evidence。
- 明确 local feed、ProjectReference、direct `.nupkg`、build-only、
  dependency-probe-only、dashboard、dry-run、template JSON 和 `failedBlockerCount=0`
  都不能替代 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加专项质量门，防止双路线、runtime key 与 proof
  边界在后续文章维护中丢失。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `4/4` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Release Evidence Ladder Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/release-evidence-ladder-public-article.md`，将 release evidence
ladder 从短说明扩展为完整的发布证据分层文章。文章把 template/preflight/build-only/
readonly diagnostics、real-model runtime、package-consumer runtime、post-publish verification
和 release close 分层，并明确 forbidden substitute scan 与 owner input 字段的边界。

### 实现

- 文章补齐 `package-consumer-runtime-proof-owner-input.template.json`、
  `package-consumer-runtime-proof-owner-input.schema.json`、
  `package-consumer-runtime-proof-forbidden-substitute-scan.json`、
  `public-docs-package-metadata-gate.json`、`final-owner-execution-package.json` 与
  `Test-PackageConsumerRuntimeProofRecord.ps1` / `Test-ReleaseIssueCloseRecord.ps1`
  的证据路径。
- 写入 owner input schema 当前 `fieldCount=80`、`requiredFieldCount=49`，并列出 clean
  consumer、公开包源、managed/runtime package、host metadata、commands、runtime result、
  logs 和 side-effect guards 等关键字段。
- 公开说明 forbidden substitute scan 当前 `blocked-forbidden-substitute-detected`，其中
  `repository path leakage` 与 `template placeholder` 是 blocker；local feed、
  ProjectReference、direct `.nupkg`、build-only、dry-run、queued GitHub Actions run、
  missing self-hosted runner、GitHub Actions dry-run `.nupkg`、dashboard、GUI screenshot 和
  TensorRtExec build report only 均不能替代 package-consumer-runtime proof。
- 明确 GitHub Actions dry-run 只能作为 context evidence：`isDryRunOnly=true`、
  `isPublishedPackageProof=false`、`isPackageConsumerRuntimeProof=false`、
  `manualWorkflowDispatchNotPerformed=true`、`performsPublish=false`、
  `canPublishPublicly=false`、`canCloseReleaseIssue=false`、`canPromoteProof=false`。
- 明确 `public-docs-package-metadata-gate.json` 的 `failedBlockerCount=0` 只表示文档没有
  forbidden overclaim，不是 ready-to-publish，也不是 release close。
- 为 `PublishingPublicArticleTests` 增加
  `ReleaseEvidenceLadderPublicArticleCoversOwnerInputForbiddenSubstitutesAndCloseBoundaries`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `7/7` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Deferred Boundary Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/deferred-boundary-public-article.md`，将 deferred boundary
从短说明扩展为完整的完成度/风险边界文章。文章明确 manifest/source 100% 不等于可发布
100%，并用 evidence ladder、A/B/C/D 风险层级和 runtime deserialization 案例说明哪些
API 可以继续提升、哪些必须保留 deferred history。

### 实现

- 文章补齐 `project-completion-review.md`、`tensorrt-interface-comparison.csv`、
  `deferred-boundary-risk-tier-gate.md`、`deferred-manual-design-groups.md` 与
  `runtime-deserialization-deferred-boundary-audit.md/json` 的证据路径。
- 增加 manifest/source match -> non-deferred native bridge -> generated interop/header
  parity -> typed C# wrapper -> quality test/smoke -> runtime smoke ->
  package-consumer-runtime proof 的完成度梯子。
- 明确 A-tier copied value、B-tier safe alternative、C-tier design-gate-required、
  D-tier keep-deferred 的提升策略，并写入默认低风险 deferred 候选为 `0` 的当前边界。
- 公开说明 algorithm selector borrowed objects、allocator/output allocator/debug listener
  callback、plugin lifecycle、execute/enqueue、IDimensionExpr/IExprBuilder 与
  runtime deserialization ownership 不能机械提升。
- 以 runtime-deserialization audit 为案例，明确 direct `deserializeCudaEngineV2` 和
  `loadRuntime` 共 5 条 TRT8/TRT10/TRT11 行继续 `deferred-only`，安全替代面只限
  pointer-free diagnostics/design gate。
- 为 `PublishingPublicArticleTests` 增加
  `DeferredBoundaryPublicArticleCoversRiskTiersRuntimeDeserializationAndNoSubstituteProof`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `6/6` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 Builder Config Readback Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/builder-config-readback-public-article.md`，将 builder
config readback 从短说明扩展为可发布的教程型文章。文章围绕 TensorRtExec/trtexec 对齐场景，
解释 `Requested`、`Readback`、`ReadbackMatch` 与
`EvidenceBoundary=builder-config-readback-only` 的意义，并明确 build/report evidence 与
package-consumer-runtime proof 的边界。

### 实现

- 文章补齐 `TensorRtBuilderConfig.cs`、`TensorRtBuilderConfig.Trt11Diagnostics.cs`、
  `TrtexecLikeDeploymentOptions.cs`、`OnnxEngineBuildService.cs`、
  `OnnxEngineBuildDiagnostics.cs` 和
  `applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json` 的证据路径。
- 公开说明 `--workspace`、`--memPoolSize`、`--avgTiming`、`--minTiming`、
  `--tacticSources`、`--profilingVerbosity`、`--exportTimingCache`、DLA/GPU fallback、
  DirectIO、sparsity、strongly typed、engine packaging、timing cache 和 scalar controls 的
  readback 边界。
- 明确 TRT8 legacy compatibility、TRT10/11 timing 差异、TRT11 removed/changed API、
  progress monitor/calibrator/algorithm selector presence probe 与 plugin lifecycle 的高风险边界。
- 为 `PublishingPublicArticleTests` 增加
  `BuilderConfigReadbackPublicArticleCoversTrtexecControlsVersionGuardsAndProofBoundaries`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `5/5` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。

## 2026-07-23 OnnxToEngine Trtexec Parity Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/onnxtoengine-trtexec-parity-public-article.md`，将
OnnxToEngine 与 TensorRtExec/trtexec parity 从短说明扩展为可审计的教程型文章。文章明确
CLI、WinForms、YoloVision 与 owner proof input 的关系，并把 parity matrix、gap list、
report、command preview 和 bounded runtime output 的证据边界写清楚。

### 实现

- 文章补齐 `applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json`、
  `applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json`、
  `samples/OnnxToEngine/Program.cs`、`TensorRtExecOptions.cs`、
  `TensorRtExecService.cs`、`TensorRtExecReport.cs`、`TensorRtExecCommand.cs`、
  `MainForm.cs`、`TrtexecLikeParser`、`TrtexecLikeOptions`、
  `OnnxEngineBuildOptions.FromTrtexecLikeOptions` 与 `OnnxEngineBuildService` 的证据路径。
- 公开说明 `implemented`、`implemented-report`、`implemented-build-readback`、
  `implemented-bounded-runtime`、`parse-report-only`、
  `diagnostic-alias-compatible` 与 `checklist-backed-command-preview` 等 matrix 状态。
- 补充 trtexec-like 参数覆盖，包括 ONNX/engine path、dynamic shape、precision、
  workspace/memory pool、DLA/GPU fallback、tactic source、IO format、precision policy、
  timing cache、profiling、layer info、runtime loop、CUDA graph、input/output dump 与 timing export。
- 明确 `runtimeProofItems = 0` 与 `packageConsumerRuntimeProofItems = 0` 的当前发布边界；
  GUI 截图、command preview、parity matrix、gap list、local feed、ProjectReference consumer、
  direct `.nupkg` install、GitHub Actions dry-run 和 post-publish verification 都不能替代
  package-consumer-runtime proof。
- 将 YoloVision 的 YOLOv5/YOLOv6/YOLOv7/YOLOv8/YOLOv9/YOLOv10/YOLO11/YOLO26 与
  det/cls/seg/obb/pose/sem real-model-runtime 证据路径写入文章，指向 sample run evidence 与
  owner backfill pack。
- 为 `PublishingPublicArticleTests` 增加
  `OnnxToEngineTrtexecParityPublicArticleCoversMatrixStatusesGuiYoloAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `13/13` 通过。
- 编译阶段未出现本批新增错误；此前项目存在的 nullable warning 不属于本批改动范围。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-24 Builder Config Readback Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/builder-config-readback-public-article.md`，将 Builder Config Readback
文章从配置项说明扩展为覆盖 readback 行级 schema、失败分类、report promotion flags 和跨版本分类的公开教程。
文章明确本阶段只是文档和质量门，不是 engine runtime proof、real-model-runtime proof、package-consumer-runtime
proof 或发布授权。

### 实现

- 文章新增 readback 行级 schema：OptionName、OptionGroup、TensorRtLine、RequestedValue、
  NormalizedRequestedValue、Applied、ReadbackValue、ReadbackMatch、UnsupportedReason、DiagnosticCode、
  EvidenceBoundary、CanPromoteRuntimeProof、CanPromotePackageConsumerProof 和 CanDeleteDeferredRecord。
- 增加失败分类：unsupported-on-trt-line、removed-in-trt11、legacy-trt8-only、dependency-probe-only、
  builder-config-unavailable、setter-rejected、readback-mismatch、parse-only 和 dry-run-only。
- 补充 report promotion flags：BuilderConfigReadbackEvidence、BuilderConfigCreatedEngine、
  BuilderConfigRanInference、BuilderConfigValidatedOutputs、BuilderConfigIsRealModelRuntimeProof、
  BuilderConfigIsPackageConsumerRuntimeProof、BuilderConfigCanPromoteRuntimeProof 和
  BuilderConfigCanPromoteReleaseProof。
- 明确 ReadbackMatch=True 是构建配置证据，不是模型输出证据；build readback section 不能继承 runtime section 的 proof 状态。
- 增加跨版本失败分类表，覆盖 trt8-legacy-compatibility、trt10-typed-setter、trt11-removed-setter、
  always-strongly-typed 和 callback-lifecycle-required。
- 继续明确 timing cache SHA256、tactic source mask readback、profiling verbosity readback 和
  BuilderConfigReadbackEvidence=true 不能替代 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `BuilderConfigReadbackPublicArticleCoversStructuredReadbackRowsPromotionFlagsAndFailureClassification`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `27/27` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 engine runtime proof、real-model-runtime proof、package-consumer-runtime proof、Linux runner proof、
  post-publish verification 或 owner authorization，因此不改变 `canPublishPublicly=false`、
  `canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-28 External Model Evidence And Project Release Story Closure

本阶段完成技术文章路线图 80/81 的长文收口，把外部模型证据回填和项目能力/发布边界从短篇提纲扩展为可独立发布、可由仓库事实重新验证的完整教程。

### 实现

- `external-model-evidence-case-study.md` 扩为 550 行 / 15,226 字符，覆盖 acquisition、build-only、output review、`real-model-runtime`、`package-consumer-runtime` 和 post-publish 六层证据边界。
- 外部模型文章绑定官方 YOLOX-S 与 YOLOv10n 的来源、许可证、SHA256、TensorRtExec/YoloVision 命令、运行 closure、owner pack 和严格 validator，并明确 E 盘资产隔离与 C 盘拒绝策略。
- `project-release-story-and-boundaries.md` 扩为 611 行 / 15,423 字符，覆盖 manifest -> C ABI -> generated interop -> wrapper -> samples/tools -> release evidence 架构、三代 TensorRT coverage、TensorRtExec、YoloVision、双 package 路线、18 条 runtime matrix、文章矩阵和五个最终 blocker。
- 项目文章直接引用 freeze blocker ID，并保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。
- 路线图 80/81 更新为“完整教程已收口”，根 README 增加两篇前台入口。
- `TechnicalArticleRoadmapTests` 新增两项专项门禁，从权威 JSON/Markdown 动态核对文章长度、CLI/validator 名称、模型 identity、coverage、85 个 GUI/CLI fields、20 个 gap items、60/55/5 capability、18 个 runtime keys、103/44 篇文章和五个 blocker，并拒绝过期参数或发布完成声明。

### Verification

- 新增专项门禁：2/2 通过。
- `TechnicalArticleRoadmapTests` + `PublishingPublicArticleTests`：79/79 通过。
- 完整 `TensorRtSharp.sln` Debug build：0 warning / 0 error。首次并行验证因两个残留 build 同时写 `obj` 出现一次 `CS2012`，关闭 build server 后使用 `-m:1` 串行复跑通过；不归类为源码失败。
- stale release claims audit：1118 files scanned / 0 findings。
- `git diff --check` 通过，仅有 README 既有行尾转换提示；两篇文章未出现过期 CLI 参数或 `canPublishPublicly=true` 等错误声明。

### C 盘与发布边界

- `C:\Users\guoji\Downloads` 未发现本阶段新增 `.onnx`、`.engine`、`.plan`、`.nupkg` 等重资产。
- Temp 未发现本项目命名的残留目录；扫描到的 DLL 位于 Visual Studio Setup 随机临时目录，来源与本阶段无关，未擅自删除。
- 仓库内一次性 TestResults/TRX 已在记录结果后删除。
- 未执行 push、GitHub Actions、workflow dispatch、NuGet/GitHub Packages/GitHub Release 发布或 issue close。
- 本阶段只收口文章与 source-quality 门禁，不创建新的 package-consumer、Linux runner、post-publish 或 owner authorization proof，最终发布状态保持 `blocked-real-proof-required`。

## 2026-07-26 TensorRT Execution Context NVTX Verbosity Deferred Alias Closure

本阶段将 `IExecutionContext::getNvtxVerbosity` 与 `IExecutionContext::setNvtxVerbosity`
作为成对的 scalar diagnostics 工作包收口。TRT8、TRT10、TRT11 的真实 native
implementation、version-specific manifest、C# interop、pointer-free wrapper 和
`TensorRtSmokeRunner` 调用均已存在；本阶段补齐 coverage exporter 的显式 deferred-history
alias，并新增候选审计记录与回归门禁。旧 deferred manifest 未删除。

### 实现与证据

- `eng/Export-InterfaceCoverageMatrix.ps1` 显式归并
  `IExecutionContext::getNvtxVerbosity` / `setNvtxVerbosity` 的 TRT8/TRT10 deferred
  history；TRT11 没有对应历史 deferred record，保持真实 entry 为 `implemented`。
- 新增 `artifacts/interface-coverage/trt-execution-context-nvtx-verbosity-candidate-audit.md`
  与 `.json`，记录 `NvInferRuntime.h`、native ABI/export parity、version guards、ownership
  和 public pointer-free 结论。
- `ExecutionContextReadonlyControlsTests` 增加 alias、跨版本、审计字段和删除 deferred
  禁止项的断言。

### Verification

- `Export-InterfaceCoverageMatrix.ps1`：成功；TRT8/10 rows 为
  `implemented-with-deferred-history`，TRT11 rows 为 `implemented`，所有 package
  `manifest matched` 与 `native source present` 保持完整。
- 未删除 TRT8/TRT10 deferred records；未新增 TRT11 虚构 deferred record。
- 本阶段没有 callback、allocator、plugin lifecycle、borrowed pointer、device pointer
  或 external resource API uplift。

### C 盘与发布边界

- 未下载模型、ONNX、engine、plan、TensorRT、CUDA、cuDNN 或 NuGet 包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、
  GitHub Release upload 或 issue close。
- 本阶段不构成 runtime proof、package-consumer-runtime proof、release proof 或公开发布授权，
  `canDeleteDeferredRecords=false`、`canPublishPublicly=false` 保持不变。

## 2026-07-26 TensorRT Compatibility Diagnostics Candidate Proof Batch

本阶段继续 B-tier safe-alternative 路线，审计并收口以下已有真实 route 的 compatibility/
diagnostics 接口：`IBuilderConfig::getTilingOptimizationLevel`、
`ICudaEngine::hasImplicitBatchDimension`、`IUffParser::getUffRequiredVersionPatch`、
`IParser::getError` 和 `IParserRefitter::getError`。本阶段没有新增 ABI 签名；重点是显式
deferred-history alias、跨版本差异、pointer-free public surface、既有 smoke 和候选审计。

### 实现与文档

- `Export-InterfaceCoverageMatrix.ps1` 新增 tiling、implicit-batch、parser/refitter 的
  显式 deferred aliases；UFF patch alias 已有并由本批补齐 tuple proof。
- 新增 `artifacts/interface-coverage/trt-compatibility-diagnostics-candidate-audit.md/.json`。
- 新增 `DeferredCompatibilityDiagnosticsProofTests`，锁定 manifest/source/wrapper/smoke、
  coverage rows、version guards 和 public pointer-free 约束。
- 新增公开技术文章
  `docs/articles/zh-cn/deferred-compatibility-diagnostics-proof.md` 并加入 `docs/toc.yml`。

### 跨版本与边界

- TRT10 tiling/implicit-batch rows 为 `implemented-with-deferred-history`；TRT11 tiling
  为独立 `implemented`，implicit-batch public route 保持 `NotSupported`。
- TRT8 UFF version tuple 为 copied readonly snapshot；TRT8 engine implicit-batch 是 legacy
  alias；TRT8/10/11 parser diagnostics 均不暴露 parser/refitter error pointer。
- 所有 deferred manifest 保留，`canDeleteDeferredRecords=false`；本批不构成 runtime proof、
  package-consumer runtime proof、release proof 或公开发布授权。

## 2026-07-24 OnnxToEngine Trtexec Parity Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写 docs/articles/zh-cn/publishing/onnxtoengine-trtexec-parity-public-article.md，将 OnnxToEngine 与 TensorRtExec 的 trtexec-like 对齐文章从参数矩阵说明推进为包含 conversion playbook、shape/profile 归一化、typed readback、artifact hash、runtime candidate 和 proof promotion criteria 的公开教程。文章明确本阶段只是文档和质量门，不是 engine runtime proof、package-consumer-runtime proof、post-publish proof 或发布授权。

### 实现

- 文章新增 Conversion Playbook 与 Parity 晋级标准章节，分层说明 parse-normalized、profile-normalized、builder-applied、builder-readback、artifact-written、bounded-runtime-output、real-model-runtime candidate 和 package-consumer-runtime remains external。
- 明确 TrtexecLikeParser.Parse、TrtexecLikeOptions.ToArgumentLine、EngineBuildProfile.Parse、EngineBuildShape、TrtexecLikeDeploymentOptions、TrtexecLikeBuildPolicy 和 BuilderConfigDeploymentSnapshot 的证据位置。
- 补充 AcceptedAlias、ParsedOnlyReason、AppliedByTypedWrapper、VersionGuard、ReadbackMatch、ArtifactWritten、ArtifactSha256、RuntimeExecuted、OutputValidationPerformed、OwnerReviewed 和 ProofClassification 等 promotion 字段。
- 写清 --mnist、--mnistInput、--expectedDigit、--exportPreprocessedInput、MnistOnnxRuntime、MnistOnnxRuntimeResult 和 OutputMatch 只能进入 real-model-runtime candidate，不能替代 package-consumer-runtime proof。
- 保持 calibrator-owner-evidence-required、plugin-lifecycle-owner-evidence-required 和 borrowed-pointer-disallowed 的高风险边界。
- 为 PublishingPublicArticleTests 增加 OnnxToEngineTrtexecParityPublicArticleCoversConversionPlaybookPromotionCriteriaAndMnistRuntimeCandidate 专项门禁。

### Verification

- dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter FullyQualifiedName~PublishingPublicArticleTests：30/30 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、GitHub Release upload、issue close 或 push。
- 本批没有执行真实模型转换、没有生成 engine/plan/nupkg、没有 public package proof、package-consumer-runtime proof、post-publish verification、Linux runner proof 或 owner authorization。

## 2026-07-24 Release Evidence Ladder Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
docs/articles/zh-cn/publishing/release-evidence-ladder-public-article.md，将 release evidence ladder 从证据
分层说明扩展为覆盖 release close lanes、non-proof flags、owner decision 和 post-publish hard gate 的公开发布边界文章。
文章明确本阶段只是文档和质量门，不是 package proof、runtime proof、post-publish proof、owner authorization 或发布动作。

### 实现

- 文章新增 Release Close Lanes 与 Non-Proof Flags 章节，分开说明 package-consumer-runtime proof、
  real-model-runtime proof、Linux runner proof、public package download proof、post-publish clean consumer proof、
  owner authorization 和 final release close decision。
- 增加 non-proof flags：isRuntimeExecutionProof=false、isPackageConsumerRuntimeProof=false、
  isPostPublishProof=false、isReleaseCloseProof=false、performsPublish=false、canPublishPublicly=false、
  canCloseReleaseIssue=false、canPromoteProof=false 和 ownerDecisionRequired=true。
- 明确 strictValidatorPassed=true 只能说明对应 lane 的结构和禁止项通过；如果 ownerDecisionRequired=true 或
  releaseCloseBlockedReason 仍存在，仍不能公开发布、关闭 issue 或把 owner execution package/runbook/template/import
  result 写成 proof。
- 明确 failedBlockerCount=0、public package download hash、dashboard、GUI screenshot、local feed、ProjectReference、
  direct .nupkg、build-only、dry-run、queued Actions 或 missing self-hosted runner 都不能替代 post-publish clean consumer
  proof 或 final close decision。
- 为 PublishingPublicArticleTests 增加
  ReleaseEvidenceLadderPublicArticleCoversCloseLanesPublishFlagsAndNonProofOwnerRunbooks 专项门禁。

### Verification

- dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"：29/29 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。
- dotnet build-server shutdown：已成功关闭 MSBuild 与 VB/C# compiler server。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、GitHub Release upload、issue close 或 push。
- 本批没有 public package proof、package-consumer-runtime proof、post-publish verification、Linux runner proof、
  real-model-runtime proof 或 owner authorization，因此不改变 canPublishPublicly=false、canCloseReleaseIssue=false 或 release blocker 状态。

## 2026-07-24 NuGet Runtime Install Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md`，将 NuGet/runtime 安装文章从
runtime key 和 clean consumer 边界说明扩展为覆盖 package source、NuGet cache、PackageReference-only
consumer、split/full runtime package、native asset manifest 和 proof boundary 的公开教程。文章明确本阶段只是文档
和质量门，不是 public package proof、package-consumer-runtime proof、post-publish verification 或发布授权。

### 实现

- 文章新增 package source 与缓存边界，覆盖 `dotnet nuget list source`、`dotnet nuget add source`、
  `dotnet nuget locals all --list`、NuGet.org、GitHub Packages、GitHub Release asset、企业内网 feed 和
  public package source 的区别。
- 明确 NuGet global packages cache 可能落在 C 盘 `%UserProfile%\.nuget\packages`，可用
  `NUGET_PACKAGES=E:\NuGetPackages` 调整缓存位置，但这只改变缓存位置，不改变 proof 语义。
- 增加 PackageReference-only consumer 示例，覆盖 `RuntimeIdentifier=win-x64`、`PlatformTarget=x64`、
  managed package 和 split runtime Bridge/CudaCudnn/TensorRt package references。
- 写清 full runtime collection package 与 split components 的区别，collection package 不能替代每个组件的
  nupkg SHA256、native asset listing 和 runtime smoke。
- 增加 native asset manifest 字段：NativeAssetsCopied、BridgeAssetPresent、CudaCudnnAssetsPresent、
  TensorRtAssetsPresent、RuntimePackageKey、RestoreSourceMode、PackageReferenceOnly、UsesProjectReference、
  UsesLocalFeed、UsesDirectNupkg、DependencyProbeOnly、RuntimeSmokeAttempted 和 PackageConsumerRuntimeProof。
- 明确 NuGet cache 命中、`NUGET_PACKAGES` 改到 E 盘、PackageReference-only 但无 runtime smoke、
  NativeAssetsCopied=true 但 dependency probe 失败、dependency probe passed 但无 enqueue/output validation
  都不能替代 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `NuGetInstallRuntimePackagePublicArticleCoversPackageSourcesCacheBoundariesPackageReferenceOnlyAndNativeAssetManifest`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `28/28` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 public package proof、package-consumer-runtime proof、post-publish verification、Linux runner proof、
  real-model-runtime proof 或 owner authorization，因此不改变 `canPublishPublicly=false`、
  `canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 Deferred Boundary Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/deferred-boundary-public-article.md`，将 deferred 边界文章从风险分层和
runtime deserialization 示例扩展为覆盖 uplift 作业单、机器可读候选清单、安全正例、必交质量门和 proof
promotion flags 的公开教程。文章明确本阶段只是文档和质量门，不是 deferred API 实装、runtime smoke、
package-consumer-runtime proof 或发布授权。

### 实现

- 文章新增 Uplift 作业单，要求从 `deferred-readonly-candidate-list.json`、
  `deferred-candidate-safety-triage.json`、`tensorrt-interface-comparison.csv` 和
  `tensorrt-interface-coverage.json` 出发，而不是凭直觉提升。
- 明确每条候选应保留 candidateId、apiArea、riskLevel、outputMode、native/managed/smoke required、
  implementationStatus、nativeSources、managedSources、smokeSources、qualityTests、publicSurface、
  ownershipBoundary、DeferredRowsStillRequired 和 `CanDeleteDeferredRecord=false`。
- 补充 `implemented-with-deferred-history` 语义：安全替代面可用，但旧 deferred row 仍作为边界记录保留。
- 增加安全正例：Plugin Registry、Engine Inspector、ONNX Parser/Refitter diagnostics、BuilderConfig readback、
  CUDA device/memory/graph-memory summary 和 OnnxEngine parser preflight snapshot。
- 增加 report flags：`RuntimeEvidenceKind=copied-readonly-summary`、`PointerFreeCopiedSummary=true`、
  `IsRuntimeExecutionProof=false`、`IsPackageConsumerRuntimeProof=false`、
  `CanPromoteRuntimeProof=false`、`CanPromoteReleaseProof=false` 和 `CanDeleteDeferredRecord=false`。
- 补充必交质量门清单，覆盖 readonly diagnostics、public API handle、native ABI parity、plugin inventory、
  refitter/engine inspector、runtime deserialization、callback/allocator、algorithm snapshot、builder config 和
  public article gates。
- 继续明确 graph/external resource ownership 与 callback proof attempt 不能替代真实 owner/lifecycle/runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `DeferredBoundaryPublicArticleCoversUpliftWorkOrderSafeExamplesQualityGatesAndPromotionFlags`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `26/26` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 deferred API 实装、runtime smoke、package-consumer-runtime proof、real-model-runtime proof、
  Linux runner proof、post-publish verification 或 owner authorization，因此不改变
  `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 CUDA TensorRT DLL Troubleshooting Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md`，将 CUDA/TensorRT DLL
加载排查文章从基础决策树扩展为覆盖 Windows native loader、resolver 候选路径、PATH 污染、临时本地复制、
最小复现记录和非 proof 边界的公开教程。文章明确本阶段只是文档和质量门，不是 dependency/runtime smoke、
package-consumer-runtime proof 或发布授权。

### 实现

- 文章新增 Windows native loader 检查表，覆盖 `where`、`Get-Command`、`dumpbin /dependents`、
  `jyppxtrtbridge.dll`、`nvinfer.dll`、`nvinfer_10.dll`、`nvinfer_plugin.dll`、`nvonnxparser.dll`、
  `cudart64_12.dll` 和 `cudnn64_9.dll`。
- 补充 `NativeBridgePathResolver candidate paths`、`NativeBridgeLibraryLoader load result`、
  `NativeBridgeLoadException message`、`NativeDependencyProbeStatus`、`ResolvedBridgePath`、
  `ResolvedVendorDllDirectory`、ProcessArchitecture、RuntimeIdentifier 和 VC++ runtime installed 字段。
- 写清 bridge DLL 自身找不到与 vendor dependency 找不到的区别。
- 增加 `temporary-local-diagnostic-copy` 与 `path-contamination` 边界，说明临时复制 DLL 或清理 PATH 后通过
  不能成为 package content proof 或 runtime proof。
- 增加最小可复现记录字段：MinimalReproProjectOutsideRepository、PackageReferenceOnly、UsesLocalFeed、
  UsesProjectReference、UsesDirectNupkg、NativeAssetsCopied、DependencyProbeOnly、RuntimeSmokeAttempted、
  RuntimeSmokePassed、DriverBlocked、PathContaminationSuspected 和 TemporaryLocalDiagnosticCopyUsed。
- 为 `PublishingPublicArticleTests` 增加
  `CudaTensorRtDllTroubleshootingPublicArticleCoversWindowsLoaderResolverPathContaminationAndTemporaryCopyBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `25/25` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 dependency/runtime smoke、package-consumer-runtime proof、real-model-runtime proof、Linux runner proof、
  post-publish verification 或 owner authorization，因此不改变 `canPublishPublicly=false`、
  `canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 Plugin Inventory Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/plugin-inventory-public-article.md`，将 Plugin Inventory 只读文章从
registry/source/creator 字段说明扩展为覆盖 PluginCreatorV3 metadata、PluginV2/V3 layer metadata、
source-only smoke、report 字段和 plugin lifecycle proof boundary 的公开教程。文章明确本阶段只是文档
和质量门，不是 plugin create/clone/serialize/deserialize/enqueue、runtime smoke、package-consumer-runtime
proof 或发布授权。

### 实现

- 文章补充 `TensorRtPluginCreatorInfo`、`TensorRtPluginCreatorSummary`、`TensorRtPluginFieldInfo`、
  `TensorRtPluginFieldSummary`、`TensorRtPluginCreatorV3MetadataDesignGate`、
  `TensorRtPluginCreatorV3MetadataDesignGateResult` 和 `TensorRtVersionedInterfaceMetadata` 的只读边界。
- 增加 `TensorRtPluginV2LayerMetadata`、`TensorRtPluginV3LayerMetadata` 和
  `TensorRtPluginV3SerializationFieldInventory` 说明，强调 layer/serialization metadata 是 pointer-free
  inspection，不是 plugin lifecycle。
- 明确 `PluginInventorySourceOnlySmokeTests` 与 source-only evidence 只能证明源码、wrapper、文档和门禁链路存在，
  不能删除 deferred lifecycle 记录。
- 增加建议 report 字段：`EvidenceKind = plugin-inventory-readonly-diagnostics`、
  `RuntimeEvidenceKind = readonly-metadata`、`PluginLifecycleProof = false`、
  `PluginCreateProof = false`、`PluginEnqueueProof = false`、
  `PackageConsumerRuntimeProof = false` 和 `ForbiddenSubstitutes`。
- 明确 registry exists、creator/field metadata、parent-search round-trip、source-only smoke、local feed、
  ProjectReference、direct `.nupkg`、dashboard、dry-run 和 release checklist 都不能替代真实 plugin enqueue
  或 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `PluginInventoryPublicArticleCoversV3MetadataLayerMetadataSourceOnlyReportsAndLifecycleBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `23/23` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 plugin lifecycle proof、plugin enqueue proof、runtime smoke、package-consumer-runtime proof、
  real-model-runtime proof、Linux runner proof、post-publish verification 或 owner authorization，因此不改变
  `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 Native Bridge Build Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/native-bridge-build-public-article.md`，将 native bridge 构建文章从
manifest/generated/CMake 导读扩展为覆盖 deferred uplift 验收、native/source/generated/wrapper 对齐、
loader 路径解析、package 双路线和 proof boundary 的公开教程。文章明确本阶段只是文档和质量门，不是
native ABI 实装、CMake release build、runtime smoke、package-consumer-runtime proof 或发布授权。

### 实现

- 文章新增 Bridge 实现验收表，覆盖 `native/manifests/tensorrt/v8/v10/v11`、`native/manifests/cuda`、
  `native/src/tensorrt/v8/api.cpp`、`native/src/tensorrt/v10/api.cpp`、
  `native/src/tensorrt/v11/api.cpp`、`native/src/cuda/api.cpp`、generated headers、generated C#
  interop、high-level wrapper 和质量门。
- 补充 native 到 wrapper 的提升规则：manifest entry、native implementation、generated interop、
  `NativeBridgeApi` helper、public wrapper、smoke/quality gate 和 public article 必须成链。
- 明确只读 copied snapshot 是优先提升路线，覆盖 plugin registry inventory、engine inspector、
  parser diagnostics、builder config readback、runtime dependency diagnostics 和 CUDA device/memory/stream
  状态。
- 强化高风险边界：callback trampoline、allocator ownership、plugin lifecycle、borrowed pointer、
  external resource 和 runtime deserialization ownership 必须保留 lifecycle/no-throw/in-flight
  drain/copied-before-interop/loader 证据，不能直接伪装成低风险 API。
- 增加 loader 与运行时解析说明，覆盖 `NativeBridgePathResolver`、`NativeBridgeLibraryLoader`、
  `NativeBridgePathResolver.EnumerateCandidatePaths`、`NativeBridgeLoadException`、`runtimes/<rid>/native`、
  `jyppxtrtbridge.dll`、`jyppxcudabridge.dll`、`nvinfer_plugin.dll`、`nvonnxparser.dll` 和
  `cudart64_*.dll`。
- 补充 native bridge 最小验证组合，强调文章测试不能替代 ABI/wrapper 门禁。
- 写清 split runtime Bridge 组件、本地 bridge consumer、package inventory、runtime readiness、
  public package download template、queued workflow 和 GitHub Actions dry-run 都不能替代 TensorRT
  runtime 真实执行或 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `NativeBridgeBuildPublicArticleCoversImplementationAcceptanceWrapperUpliftLoaderAndNonProofBridgeConsumers`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `22/22` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 native ABI 实装、CMake release build、runtime smoke、package-consumer-runtime proof、
  real-model-runtime proof、Linux runner proof、post-publish verification 或 owner authorization，因此不改变
  `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 Windows Source Build Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/source-build-windows-public-article.md`，将 Windows 源码构建文章从
CMake/环境导读扩展为覆盖 Developer PowerShell、C++ bridge、manifest/generated/native ABI、version
guard、package layout、本地候选检查、DLL loader 排查和 forbidden proof substitute 的公开教程。
文章明确本阶段只是文档和质量门，不是 native release build、runtime smoke、package consumer proof、
post-publish verification 或发布授权。

### 实现

- 文章补齐 Developer PowerShell for VS 2022、`where cl/link/cmake/dotnet/pwsh`、
  VCToolsVersion、WindowsSDKVersion、DOTNET_ROOT、CUDA_PATH、TensorRT include/lib root、
  cuDNN include/lib/bin root 和多 CUDA 版本对齐要求。
- 增加 manifest/generated/native ABI 三层核对路径：`native/manifests/tensorrt`、
  `native/manifests/cuda`、generated C# interop、`native/generated/bridge_api_catalog.g.h` 和
  `native/generated/bridge_entrypoints.g.h`。
- 补充 `Test-TensorRtNativeAbiSurface.ps1`、单批 CMake preset 命令、`build-out/<preset>` 输出边界、
  `Test-ManagedPackageContent.ps1`、`Test-RuntimePackageReadiness.ps1` 与
  `Validate-SplitRuntimePackages.ps1` 的证据角色。
- 写清 TRT8/TRT10/TRT11 version guard 核对表，覆盖 legacy parser/network flags、strongly typed
  network、precision/layer policy、refit/stripped plan 和 debug listener/callback 高风险面。
- 明确只读 API 要返回 copied data，callback、allocator、plugin lifecycle、borrowed pointer、
  external resource 和 runtime deserialization ownership 不能在缺少生命周期/smoke 证据时从 deferred
  伪装成普通低风险 API。
- 补充 Windows loader 排查和 DLL 来源一致性：`jyppxtrtbridge.dll`、`nvinfer.dll`、
  `nvinfer_plugin.dll`、`nvonnxparser.dll`、`cudart64_*.dll`、`cudnn*.dll`。
- 增加 forbidden proof substitutes：CMake 成功、dotnet build、binding/ABI test、`build-out` DLL、
  `dumpbin`、dependency-probe-only、blocked-by-cuda-driver、local feed、ProjectReference、direct
  `.nupkg`、package inventory/runtime readiness、public package download template、GitHub Actions
  dry-run 和 queued workflow 都不能替代 release proof。
- 为 `PublishingPublicArticleTests` 增加
  `SourceBuildPublicArticleCoversGeneratedAbiVersionGuardsPackageLayoutAndForbiddenProofSubstitutes`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `21/21` 通过。
- 编译阶段未出现本批新增错误；此前 nullable warning 不属于本批改动范围。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 native release build、runtime smoke、package-consumer-runtime proof、real-model-runtime
  proof、Linux runner proof、post-publish verification 或 owner authorization，因此不改变
  `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 Package Strategy Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/package-strategy-public-article.md`，将 GitHub full runtime 与 NuGet
small bridge/core 双路线从概念说明扩展为可追踪到 `pack` 入口、runtime manifest、split package、
hash proof、clean consumer 和 release close boundary 的公开文章。文章明确本阶段只是文档和质量门，
不是 package build、runtime proof、public package download proof、post-publish verification 或发布授权。

### 实现

- 文章补齐 `pack/JYPPX.TensorRT.CSharp.API`、`pack/runtime`、`pack/runtime-split` 的固定入口，
  包括 managed/core package、full runtime manifest、Linux runtime targets、smoke command template、
  local path example、split runtime manifest 和 split README。
- 公开说明 full runtime 与 split runtime 的取舍，覆盖 Bridge、CudaCudnn、TensorRtRuntime、
  TensorRtBuilder SM 分片、split meta package pins、TRT8/TRT10/TRT11 runtime key 和组件 hash 记录。
- 增加本地候选包与矩阵脚本说明：`Invoke-LocalRuntimePackage.ps1`、
  `Invoke-LocalSplitRuntimePackage.ps1`、`Resolve-SplitPackagePins.ps1`、
  `Validate-SplitRuntimePackages.ps1`、`Test-RuntimePackageReadiness.ps1`、
  `Export-ReleaseCandidatePackageInventory.ps1`、`Export-PreReleasePackageProofReadinessMatrix.ps1`
  和 `Export-PackageConsumerDualRouteProofPlan.ps1`。
- 补充 `dual-package-publish-preflight-matrix`、`pre-release-package-proof-readiness-matrix`、
  `package-consumer-dual-route-proof-plan`、`release-package-proof-bundle` 与
  `public-package-url-hash-verification-candidate` 的证据边界。
- 写清 public package source/hash proof 字段：publicPackageSource、publicPackageUrl、package id/version、
  runtimePackageKey、downloaded/expected nupkg SHA256、packageHashMatch、cleanConsumerRoot、
  packageReferenceOnly、smokeExitCode、nativeAssetsCopied、mergedTranscriptSha256 和 owner review。
- 明确文章、README、dashboard、matrix ready、candidate inventory ready、`failedBlockerCount=0`、
  dry-run output、本地 `.nupkg`、local feed、ProjectReference 和 template 都不能覆盖
  `canPublishPublicly=false` 或 `canCloseReleaseIssue=false`。
- 为 `PublishingPublicArticleTests` 增加
  `PackageStrategyPublicArticleCoversRepositoryPackEntrypointsSplitPackagesHashProofAndReleaseCloseBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `20/20` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 public package download proof、package-consumer-runtime proof、post-publish verification、
  Linux runner proof、real-model-runtime proof 或 owner authorization，因此不改变
  `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或 release blocker 状态。

## 2026-07-24 TensorRtExec CLI Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/tensorrtexec-cli-public-article.md`，把 TensorRtExec CLI、
WinForms、trtexec-like parser、shared build service、report schema、runtime artifact 和 release proof
boundary 串成一篇可公开发布的使用与评审文章。文章明确本阶段只是文档和质量门，不是 runtime proof、
package consumer proof、发布授权或 release close 证据。

### 实现

- 文章补齐 `applications/TensorRtExec/README.md`、`TensorRtExec.csproj`、CLI command、
  options、service、report、WinForms main form、parity matrix、gap list、GUI/CLI field map 和 report
  schema 的职责边界。
- 公开说明 `TrtexecLikeParser`、`TrtexecLikeOptions`、`OnnxEngineBuildOptions`、
  `OnnxEngineBuildService`、`OnnxEngineBuildDiagnostics` 与
  `OnnxEngineRuntimeArtifactWriter` 如何支撑 TensorRtExec CLI、GUI 和 shared service 的一致行为。
- 覆盖 ONNX/engine 输入输出、build-only、dry-run、preview-only、report/evidence sidecar、dynamic
  shapes、precision、workspace/memory pool、timing、runtime iterations、profile/layer info、plugin、
  safe/consistency、builder cache、deployment、refit、stripped plan 和 weight streaming 等 trtexec-like
  参数族。
- 写清 report 字段、normalized command、deployment snapshot、parser preflight、runtime options、
  loaded engine diagnostics、capability probe、readback fingerprint、runtime output artifact 和 proof
  classification 的边界。
- 明确 TensorRtExec report、GUI screenshot、command preview、synthetic-input runtime、dependency probe、
  capability probe、build-only evidence、sidecar-only、blocked-by-cuda-driver、Linux runner proof、owner
  authorization 和 post-publish verification 之间不能互相替代。
- 补充 TensorRtExec 与 `samples/OnnxToEngine`、`samples/YoloVision`、real-case proof pack、YOLO
  多系列多任务输出契约、real-model-runtime proof 和 package-consumer-runtime proof 的关系。
- 为 `PublishingPublicArticleTests` 增加
  `TensorRtExecCliPublicArticleCoversCliGuiParityReportsRuntimeArtifactsAndProofBoundary` 专项门禁，并将
  `tensorrtexec-cli-public-article.md` 纳入公开文章基础清单。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `19/19` 通过。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 owner 真实 runtime proof、package-consumer-runtime proof、Linux runner proof 或
  post-publish verification，因此不改变 `canPublishPublicly=false`、`canCloseReleaseIssue=false` 或
  release blocker 状态。

## 2026-07-24 NuGet Install Runtime Package Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md`，将 NuGet
安装说明从短骨架扩展为 managed package、runtime package、split runtime package、clean
consumer proof 和 troubleshooting 边界指南。文章强调 NuGet 安装教程本身不是
package-consumer-runtime proof，也不能授权发布。

### 实现

- 文章补齐 `pack/runtime/runtime-packages.manifest.json`、
  `pack/runtime-split/split-runtime-packages.manifest.json`、
  `pack/runtime/runtime-package-smoke-command-template.json`、`pack/runtime/README.md`、
  `pack/runtime-split/README.md` 与 runtime package 相关文章的证据路径。
- 公开说明整包 runtime key 与 split runtime package 的区别，覆盖 `role = bridge`、
  `role = cuda-cudnn`、`role = tensorrt` 以及典型 Bridge/CudaCudnn/TensorRt package id。
- 补充 runtime key 选择顺序：RID、TensorRT line、CUDA line、cuDNN major、GPU driver
  compatibility，并列出 `key`、`packageId`、`rid`、`tensorRtVersion`、`cudaVersion`、
  `cudnnVersion`、`distributionTier`、`validationState`、`tensorRtFiles`、`cudaFiles`、
  `cudnnFiles` 等 manifest 字段。
- 增加 clean consumer proof 最低字段、strict validator 路径、native asset listing、
  dependency probe、runtime smoke log、nupkg SHA256、host metadata 和 owner review 边界。
- 明确 local feed、ProjectReference、direct `.nupkg`、GitHub Actions dry-run、collection
  package、owner execution package、template、input draft、build-only、dependency-probe-only
  和 blocked-by-cuda-driver 都不是 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `NuGetInstallRuntimePackagePublicArticleCoversRuntimeKeysSplitPackagesCleanConsumerAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `14/14` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-24 CUDA TensorRT DLL Troubleshooting Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md`，将 Windows
CUDA/TensorRT DLL 加载排查从短说明扩展为 native load 决策树、采集字段和 proof boundary
指南。文章明确 troubleshooting guide 不是 package-consumer-runtime proof，也不能授权发布或关闭
release issue。

### 实现

- 文章补齐 `NativeBridgePathResolver.cs`、`NativeBridgeLibraryLoader.cs`、`BridgeConstants.cs`、
  `CudaEnvironmentProbe.cs`、`TensorRtEnvironmentProbe.cs`、`TensorRtToolSupport.cs` 与
  `OnnxEngineBuildService.cs` 的证据路径。
- 公开说明 `NativeLibrary.SetDllImportResolver`、`NativeLibrary.TryLoad`、
  `BridgeConstants.NativeBridgeLibraryName`、`jyppxtrtbridge.dll` 以及 TensorRT/CUDA/cuDNN DLL
  依赖链。
- 增加 native load 决策树：`dotnet --info`、`nvidia-smi`、进程位数、runtime package manifest、
  output directory、PATH/current directory、最小 probe、dependency probe 和 runtime smoke。
- 覆盖 `DllNotFoundException`、`BadImageFormatException`、CUDA error 35、
  `CUDA driver/runtime mismatch`、`native initialization failed`、blocked-by-cuda-driver、
  TRT8/TRT10/TRT11 ABI/DLL 命名差异和 cuDNN 8/9 差异。
- 明确 local feed、ProjectReference、direct `.nupkg` install、GitHub Actions dry-run、
  dependency-probe-only、build-only TensorRtExec report、OnnxToEngine report、YoloVision matrix、
  sidecar-only metadata、GUI screenshot、command preview 和无 hash 日志都不是
  package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `CudaTensorRtDllTroubleshootingPublicArticleCoversNativeLoadDecisionTreeAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `15/15` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-24 YoloVision Overview Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/yolovision-overview-public-article.md`，将 YoloVision 总览从短说明
扩展为 YOLO 多系列、多任务、preflight、preprocess、output JSON/SVG、官方资产、owner proof 与
package proof 边界的完整公开文章。

### 实现

- 文章补齐 `samples/YoloVision` 的核心路径：`Program.cs`、`YoloVision.csproj`、
  `YoloSampleRunner.cs`、`YoloVisionResult.cs`、`YoloVisionOutputReport.cs`、
  `YoloVisionPreflightReport.cs`、`YoloImagePreprocessor.cs`、runtime output set/tensor/role
  resolver、multi-output metadata 和 visualization writer。
- 覆盖 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26、YOLOX、
  custom，以及 det/cls/seg/obb/pose/sem 六任务。
- 公开说明 `yolo-model-matrix.json/md`、`yolovision-task-output-contract.json`、
  `yolovision-output.schema.json`、`yolovision-preflight.schema.json`、
  `yolovision-family-task-real-asset-roadmap.json`、article case pack 和 real asset owner backfill
  pack 的证据角色。
- 增加 `--self-test-end2end`、preflight、preprocess-only、六任务命令骨架、dedicated output role
  options、`yolovision-output.v1`、`bindingMetadata`、example output JSON、SVG visualization 和
  `eng/Test-YoloVisionOutputReport.ps1` 验证边界。
- 补充 YOLOv10n 官方 `[1,300,6]` end-to-end、YOLOX-S `[1,8400,85]` detection-only、
  YOLOv8n 六任务模板、owner backfill scripts 和 local PackageReference consumer 边界。
- 明确 support matrix、task contract、preflight report、preprocessing output、output JSON、SVG、
  local feed、ProjectReference、direct `.nupkg`、GitHub Actions dry-run、TensorRtExec build-only
  report、OnnxToEngine report、dependency-probe-only、GUI screenshot、command preview 和
  sample evidence 中的 package-consumer-runtime 字符串都不能替代 package-consumer-runtime proof。
- 为 `PublishingPublicArticleTests` 增加
  `YoloVisionOverviewPublicArticleCoversFamiliesTasksReportsOwnerEvidenceAndProofBoundary`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `16/16` 通过。
- 编译阶段未出现本批新增错误；此前 nullable warning 不属于本批改动范围。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-24 Project Overview Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/project-overview-public-article.md`，将项目总览从短介绍扩展为面向
微信公众号、博客和项目主页的入口文章。文章把 native bridge、generated interop、C# wrapper、
samples/applications、runtime package 双路线、article matrix 和 release proof boundary 串成统一叙事。

### 实现

- 文章补齐 `src/JYPPX.TensorRtSharp`、`src/JYPPX.CudaSharp`、`src/JYPPX.TensorRtSharp.Tools`、
  `src/JYPPX.Shared`、`native/src/tensorrt`、`native/src/cuda`、TRT8/TRT10/TRT11/CUDA manifest、
  `samples/OnnxToEngine`、`samples/YoloVision`、`applications/TensorRtExec`、
  `pack/runtime`、`pack/runtime-split` 和 `artifacts/final-release` 的入口关系。
- 公开说明 `TensorRtBuilder`、`TensorRtBuilderConfig`、`TensorRtRuntime`、`TensorRtEngine`、
  `TensorRtExecutionContext`、`TensorRtOnnxParser`、`TensorRtOnnxParserRefitter`、
  `TensorRtPluginRegistryInventory`、`TensorRtEngineInspector`、CUDA/TensorRT probe 等高层 wrapper
  价值。
- 增加 native/generated 与 generated C# interop 证据路径，并强调 manifest/source match 不等于
  runtime proof，generated interop 不等于 high-level wrapper。
- 补充 TensorRtExec CLI/WinForms、OnnxToEngine、YoloVision 多系列多任务、runtime package 双路线、
  30+ 文章矩阵、typical command path 与 owner validator 路径。
- 明确 build-only、dry-run、template、input draft、local feed、ProjectReference、direct `.nupkg`、
  GitHub Actions dry-run、dependency-probe-only、GUI screenshot、command preview、TensorRtExec report、
  OnnxToEngine report、YoloVision matrix、output JSON、SVG、sidecar-only 和 blocked-by-cuda-driver 都不是
  package-consumer-runtime proof。
- 将 `project-overview-public-article.md` 纳入 `PublishingPublicArticleTests` 基础文章清单，并新增
  `ProjectOverviewPublicArticleCoversArchitectureSamplesPackagesArticleMatrixAndReleaseBoundary` 专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `17/17` 通过。
- 编译阶段未出现本批新增错误；此前 nullable warning 不属于本批改动范围。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-24 Package Consumer Proof Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/package-consumer-proof-public-article.md`，将 package consumer
runtime proof 从短说明扩展为 release owner 和评审者可执行、可判断的 clean external consumer proof
文章。文章明确本阶段只是文档与质量门，不是 owner 真实 proof、发布授权或 release close 证据。

### 实现

- 文章补齐 package consumer owner input、record、validation、forbidden substitute scan、execution pack、
  clean consumer checklist、external runtime proof、post-publish verification、release close preflight 和
  blocker dashboard 的证据路径。
- 公开说明 clean consumer 必须包含 public package source、managed/runtime package id/version/key、
  nupkg SHA256、restore/build/smoke command、exitCode、smokeStatus、nativeAssetsCopied、log SHA256、
  stdout/stderr summary、host metadata、owner review 和 strict validator。
- 增加推荐执行顺序：导出/校验 owner execution pack，owner 在仓库外 clean consumer 安装公开包，导入
  `package-consumer-runtime-proof-owner-input`，再执行 `Test-PackageConsumerRuntimeProofRecord.ps1`
  与 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
- 明确 strict validator 必须拒绝仓库内 consumer、ProjectReference、local feed、direct `.nupkg`、
  dependency-probe-only、build-only、hash mismatch、host metadata 缺失、dry-run、queued Actions、
  missing runner、dashboard、template 和 skipped run。
- 写清 YoloVision、TensorRtExec、OnnxToEngine 与 package-consumer-runtime 的边界；real-model-runtime、
  sample-run evidence、TensorRtExec report 和 OnnxToEngine report 都不能替代 clean external consumer。
- 明确 release close 仍需要 owner authorization、package-consumer-runtime、Linux runner proof、
  real-model-runtime、post-publish verification、release close preflight 和 release issue close record。
- 为 `PublishingPublicArticleTests` 增加
  `PackageConsumerProofPublicArticleCoversCleanConsumerOwnerInputValidatorsAndForbiddenSubstitutes`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `18/18` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。

## 2026-07-25 YOLOv8n Semantic Segmentation Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-semantic-segmentation-map-guide.md，补齐统一文章骨架、可复用 E 盘 case workspace、浮点 semantic map 输出契约、argmax/resize-back/palette 证据、JSON/SVG 和 owner 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 补齐适用读者、解决问题、背景与场景、操作路径、代码与文件入口、图示建议、边界说明和下一步章节。
- 将案例明确为 owner-provided、兼容 `YoloVision sem` decoder 的 semantic model，不假设存在官方 `yolov8n-sem.pt`。
- 固定 E:\TensorRtSharpAssets\cases\yolov8n-sem 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/palette/input 来源、license、SHA256、classCount、map shape/layout、argmax、resize-back、ignoreIndex 和 paletteSha256 回填字段。
- 使用当前 CLI 实际支持的 `--semantic-output semantic`、`--class-count 21`、`--output-json`、`--visualization-svg` 命令，移除不存在的 map/palette CLI 参数示例。
- 明确 decoder 支持 `[C,H,W]`、`[1,C,H,W]` 和按 class count 识别的 `[1,H,W,C]` 浮点 map，不把预先 argmax 的整数索引图当作等价输入。
- 同步修正 `yolovision-article-case-pack.json`、semantic candidate template、owner backfill pack 和 generated projection，移除不存在的 semantic CLI flags；projection gate 达到 `projection-aligned`。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nSemanticArticleCoversMapWorkspaceRolesHashesAndValidation 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests、YoloVisionRealAssetCandidatePackTests、YoloVisionRealAssetCandidateValidatorTests 定向测试合计：17/17 通过。
- Export-YoloVisionRealAssetOwnerBackfillPack.ps1：`ValidationState=projection-aligned FailedCount=0`。
- git diff --check：通过；仅提示 project-completion-review.md 与 generated owner pack 的既有 CRLF/LF 规范化提示。
- C 盘本批唯一关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无文章名、批次名、candidate 名或 semantic case 命中。
- 测试期间曾观察到 C:\Users\guoji\AppData\Local\Temp 下短生命周期 OpenCV preflight nupkg；测试自身已清理，最终复查无残留，未删除用户既有缓存。
- 今日 .onnx/.engine/.plan/.nupkg 审计：最终 C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YOLOv8n Classification Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-classification-yolov8n-labels-topk-guide.md，补齐统一文章骨架、可复用 E 盘 case workspace、labels/score 语义、Top-K JSON/SVG 和 owner 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 补齐适用读者、解决问题、背景与场景、代码与文件入口、图示建议、边界说明和下一步章节。
- 固定 E:\TensorRtSharpAssets\cases\yolov8n-cls 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/input 来源、license、SHA256、classCount、classificationOutput、outputShape、labelsPath、topK、classScoreField 和 activation 回填字段。
- 增加 preprocess-only、显式 `--classification-output logits`、`--top-k 5`、output JSON、visualization SVG 命令。
- 明确标准 classification 可能需要 resize + center crop，通用 letterbox 命令只有在 owner 确认模型契约一致时才可使用；否则必须回填 owner-approved preprocess pipeline 和 tensor hash。
- 明确程序输出的 `postprocess.topK`、`classId`、`className`、`score`，以及 owner 必须回填的 logits/probability、softmaxApplied、labels locale 和 score precision 边界。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nClassificationArticleCoversLabelsTopKWorkspaceHashesAndValidation 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests 定向测试：6/6 通过。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘本批唯一关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无文章名、批次名或 candidate 名命中。
- 通用 `top-k` 词命中 C:\Users\guoji\Downloads\PaddleOCR-main 中 3 个 2026-01-20 已有源码文件，确认与本批无关，未删除用户既有下载。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中；本批命名资产也无命中。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YOLOv8n OBB Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-obb-angle-output-guide.md，新增可复用 E 盘 case workspace、angle 输出角色、单位/范围/坐标空间 metadata、OBB JSON/SVG 和 owner 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 固定 E:\TensorRtSharpAssets\cases\yolov8n-obb 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/input 来源、license、SHA256、outputRoleMap、boxFormat、rotatedBoxLayout、angleUnit、coordinateSpace 和 angleRange 回填字段。
- 增加 preprocess-only、显式 boxes/angles role map、angle-radians、output JSON、visualization SVG 命令和 OBB 输出必填 metadata。
- 明确程序输出 `angleUnit=radian`、`angleRange=owner-record-required`，四点坐标、旋转方向和 rotated NMS 仍属于 owner 语义证据，不能从截图推断。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nObbArticleCoversAngleWorkspaceRolesHashesAndValidation 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests 定向测试：5/5 通过。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YOLOv8n Pose Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-pose-keypoint-output-guide.md，新增可复用 E 盘 case workspace、keypoint 输出角色、坐标/骨架 metadata、pose JSON/SVG 和 owner 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 固定 E:\TensorRtSharpAssets\cases\yolov8n-pose 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/input 来源、license、SHA256、outputRoleMap、keypointCount、keypointStride、coordinateLayout、keypointLayout 和 keypointScoreField 回填字段。
- 增加 preprocess-only、显式 boxes/keypoints role map、keypoint count/stride、output JSON、visualization SVG 命令和 pose 输出必填 metadata。
- 明确程序输出的 keypoint index/x/y/score 与 owner 回填的名称、visibility、skeleton map 边界，避免把 COCO 17 点假设写成通用事实。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nPoseArticleCoversKeypointWorkspaceRolesHashesAndValidation 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests 定向测试：4/4 通过。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YOLOv8n Segmentation Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-segmentation-mask-postprocess-guide.md，新增可复用 E 盘 case workspace、prototype 输出角色、mask 后处理证据字段和 owner 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 固定 E:\TensorRtSharpAssets\cases\yolov8n-seg 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/input 来源、license、SHA256、outputRoleMap、prototypeShape、maskCoefficientCount 和 maskResizePolicy 回填字段。
- 增加 preprocess-only、显式 boxes/proto role map、output JSON、visualization SVG 命令和 mask 输出必填 metadata。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nSegmentationArticleCoversPrototypeWorkspaceRolesHashesAndValidation 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests 定向测试：3/3 通过。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YOLOv8n Detection Case Public Article Expansion

本阶段继续具体模型文章质量提升，扩写 docs/articles/zh-cn/yolovision-detection-yolov8n-download-export-run.md，新增可复用 E 盘 case workspace、真实图片预处理、输出 JSON/SVG 和 owner evidence 验证顺序。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 固定 E:\TensorRtSharpAssets\cases\yolov8n-det 的 models/labels/images/tensors/engines/reports/logs 布局。
- 补充 candidate template 的 model/labels/input 来源、license、export command 和 SHA256 回填字段。
- 增加 preprocess-only、output-json、visualization-svg 命令和输出 JSON 必填 metadata。
- 增加 Test-YoloVisionOutputReport.ps1、Test-YoloVisionRealAssetCandidate.ps1、Test-SampleRunEvidenceRecord.ps1 验证顺序。
- 更新 TechnicalArticleCampaignFourthBatchBodyTests，新增 YoloVisionYolov8nDetectionArticleCoversAssetWorkspacePreprocessOutputValidationAndCandidateFields 专项门禁。

### Verification

- TechnicalArticleCampaignFourthBatchBodyTests 定向测试：2/2 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 YoloVision Case Matrix Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写 docs/articles/zh-cn/publishing/yolovision-overview-public-article.md，新增案例矩阵与证据回填顺序章节。文章明确本阶段只是文档与质量门，不是真实模型运行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 对齐 yolo-model-matrix.json 与 yolovision-task-output-contract.json，说明 family/task 选择、模型来源、license/hash、preflight、build-only、真实运行和 owner review 顺序。
- 细化 det/cls/seg/obb/pose/sem 六任务必填输出 metadata，保留 box、score、NMS、prototype、angle、keypoint、semantic map 等任务专属字段。
- 明确 YOLOv10 [1,300,6]、YOLOX [1,8400,85] 和 YOLO26 owner-approved output contract 的差异，不以“支持全部 YOLO”替代具体证据。
- 固定 owner-action-required、canPromoteRealModelRuntime=false、canPromotePackageConsumerRuntime=false，禁止本地样例直接晋级 package-consumer-runtime。
- 更新 PublishingPublicArticleTests，新增 YoloVisionOverviewPublicArticleCoversCaseMatrixAcquisitionOrderTaskMetadataAndPromotionBoundaries 专项门禁。

### Verification

- PublishingPublicArticleTests 定向测试：33/33 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 TensorRtExec CLI WinForms Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写 docs/articles/zh-cn/publishing/tensorrtexec-cli-public-article.md，新增 CLI/WinForms 操作闭环、共享数据流和 CUDA/TensorRT DLL 排障决策树。文章明确本阶段只是文档与质量门，不是 runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 新增准备、构建、诊断/运行、归档四阶段工作流，区分 dry-run、build-only、readonly diagnostics、bounded runtime 和 evidence archive。
- 明确 WinForms -> TensorRtExecOptions -> ToArgumentLine -> TensorRtExecService -> TensorRtExecReport 的共享数据流，防止 GUI/CLI 参数语义漂移。
- 新增 jyppxtrt、jyppxcudabridge、nvinfer、nvinfer_plugin、nvonnxparser、cudart64 DLL 排障表。
- 固定 TRT8/TRT10/TRT11 manifest、native bridge、托管 version guard、架构和 PATH 对齐要求。
- 更新 PublishingPublicArticleTests，新增 TensorRtExecCliPublicArticleCoversOperationalWorkflowWinFormsDataFlowAndDllTroubleshooting 专项门禁。

### Verification

- PublishingPublicArticleTests 定向测试：32/32 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有生成 engine/plan/nupkg，没有 real-model-runtime proof、package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-25 OnnxToEngine Case Tutorial Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写 docs/articles/zh-cn/publishing/onnx-to-engine-public-article.md，新增从模型获取到可复核案例的完整教程清单。文章明确本阶段只是文档与质量门，不是模型转换执行、runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 新增模型来源与许可证记录、外部 E:\TensorRtSharpAssets\cases\<case-id> workspace、models/inputs/engines/reports/logs/packages 布局说明。
- 新增 ONNX/input/build report/engine/runtime log SHA256、modelSourceUrl、license、downloadedAtUtc、host metadata、ownerReviewed 和 proofClassification 字段。
- 明确 parser dry-run、build-only、模型特定 runtime 和 clean external consumer 四个阶段的证据边界。
- 更新 PublishingPublicArticleTests，新增 OnnxToEnginePublicArticleCoversModelAcquisitionHashIsolationAndCaseEvidenceChecklist 专项门禁。

### Verification

- PublishingPublicArticleTests 定向测试：31/31 通过。
- 编译阶段仍有 5 条既有 nullable warning，位置在 FinalPublishProofGateAndOwnerExecutionPackTests.cs 与 ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs，非本批引入。
- git diff --check：通过；仅提示 project-completion-review.md 的既有 CRLF/LF 规范化提示。
- C 盘关键词审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无本批关键词命中。
- 今日 .onnx/.engine/.plan/.nupkg 审计：C:\Users\guoji\Downloads 与 C:\Users\guoji\AppData\Local\Temp 均无命中。

### C 盘与发布边界

- 本阶段未下载模型、ONNX、engine、TensorRT、CUDA、cuDNN、Python/pip 资产或 NuGet 临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、push、NuGet push、GitHub Packages publish、GitHub Release upload 或 issue close。
- 本批没有执行真实模型转换、没有生成 engine/plan/nupkg、没有 package-consumer-runtime proof、post-publish verification 或 owner authorization。

## 2026-07-24 Engine Inspector Public Article Expansion

本阶段继续公开文章矩阵质量提升，扩写
`docs/articles/zh-cn/publishing/engine-inspector-public-article.md`，将 Engine Inspector 只读文章从
engine/readback/report 基础说明扩展为覆盖 inspector 边界分层、execution context 关联、error recorder copied
snapshot、readback artifact 字段、bounded runtime 分界和 proof promotion flags 的公开教程。文章明确本阶段只是文档
和质量门，不是 engine runtime proof、real-model-runtime proof、package-consumer-runtime proof 或发布授权。

### 实现

- 文章新增 Inspector 边界分层，区分 engine-level metadata、layer-level metadata 和 association state。
- 补充 `SetEngineInspectorExecutionContext`、binding address、device buffer、stream synchronization、
  `TensorRtErrorRecorderSnapshot`、`TensorRtErrorRecorderSummary` 和 `TensorRtErrorRecord` 的只读边界。
- 增加 `.engine-readback.json` 建议字段：ArtifactKind、ArtifactBoundary、EngineInspectorApiAvailable、
  DiagnosticsState、SkippedReason、InspectorInformationLength、ReadbackFingerprint、ReadbackSha256、
  RuntimeOutputCaptured 和 OutputValidationPerformed。
- 明确 `EngineInspectorApiAvailable=True` 可能仍只是 capability-probe-only，`ReadbackSha256` 不是 runtime log hash 或 public package hash。
- 补充 bounded runtime 与 inspector 的分界，说明 `LoadEngineReadonlyDiagnostics` 成功不等于 `LoadEngineBoundedRuntime` 或 package proof。
- 增加 promotion flags：EngineInspectorReadonlyMetadata、EngineInspectorCreatedExecutionBindings、
  EngineInspectorEnqueuedInference、EngineInspectorValidatedOutputs、EngineInspectorCanPromoteRuntimeProof 和
  EngineInspectorCanPromotePackageConsumerProof。
- 为 `PublishingPublicArticleTests` 增加
  `EngineInspectorPublicArticleCoversInspectorBoundaryLevelsArtifactsBoundedRuntimeAndPromotionFlags`
  专项门禁。

### Verification

- `dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~PublishingPublicArticleTests"`：
  `24/24` 通过。
- 编译阶段仍出现 5 条既有 nullable warning，位置在
  `FinalPublishProofGateAndOwnerExecutionPackTests.cs` 与
  `ReleasePublishReadinessEvidencePackAndPublicDocsGateTests.cs`，本批未改动这些文件。

### C 盘与发布边界

- 本阶段未下载 TensorRT、CUDA、cuDNN、模型、ONNX、engine、Python/pip 资产或 NuGet
  临时包到 C 盘。
- 未执行 GitHub Actions、workflow dispatch、NuGet push、GitHub Packages publish、
  GitHub Release upload、issue close 或 push。
- 本批没有 engine runtime proof、real-model-runtime proof、package-consumer-runtime proof、Linux runner proof、
  post-publish verification 或 owner authorization，因此不改变 `canPublishPublicly=false`、
  `canCloseReleaseIssue=false` 或 release blocker 状态。
## 2026-07-28 TensorRtExec Multi-Input And Reference Output Validation

本阶段将 generic bounded runtime 从单 float input 扩展为按 engine binding 顺序处理全部 float inputs，并新增 copied、
pointer-free 的逐输入工件。`--loadInputs` 映射在提供时必须完整覆盖全部输入，缺失、重复和未知 tensor name fail closed；
未提供时每个 input 使用独立确定性数据。

新增 `--referenceOutputs`、absolute/relative tolerance、NaN/Infinity policy 与 `OnnxEngineReferenceTensorData` JSON
合同。校验覆盖 mapping、文件、name、shape、count 和全部 values；工件记录 reference path/hash/source、逐 tensor mismatch、
first mismatch 与最大误差。`IdentityOutputMatch` 与 `OutputValidated` 已拆分，只有全部 structured references 通过才设置后者。

TRT10.11/CUDA12.9 真实 synthetic Add/Sub smoke 完成 2 inputs、2 outputs、build/serialize/deserialize/enqueue/readback，
两个 output 的 16 个 float 值全部通过 reference；combined raw SHA256 为
`05080c5591955c003b781dba0170ad3aca244e3033b89e81569a39abe305e2c5`。独立 load-engine 负例将
`difference[7]` 修改 `0.25`，得到 1 个 mismatch、`OutputValidated=false` 与
`load-engine-reference-validation-failed`。分类保持 synthetic runtime，不是 real-model、package-consumer、public package、
post-publish、Linux、Owner accepted 或 release proof。

定向合同、parser、artifact、TensorRtExec application、GUI/CLI、report schema、capability、公开文章、public material 与
release scaffold 测试 `109/109` 通过；完整 `TensorRtSharp.sln` Debug build 为 `0 warning / 0 error`。GUI/CLI strict
checklist 与真实 validated report strict validator 均为 0 blocker；public API compiler documentation 与双语审计均为
0 finding；8 份本批 JSON 均可解析。

## 2026-07-28 TensorRtExec MNIST Structured Reference Regression Candidate

本阶段把仓库已有 TensorRT MNIST digit-7 真模型接入 structured reference-output 合同，同时不把同一运行时派生的
reference 伪装成独立 golden output 或 Owner 认可的真实模型证明。

### 实现与证据

- 新增 `mnist-trt10-7.reference.json`，固定 `Plus214_Output_0` 的 `[1,10]` logits、`schemaVersion=1` 和
  `repository-mnist-runtime-output-derived-unreviewed` 来源分类；sidecar 同时记录 ONNX Model Zoo 来源说明、model/input/
  source-output hashes、TensorRT sample license 审查状态和空缺的 Owner review。
- TRT10.11/CUDA12.9 source-tree build 后输出 `external-onnx-reference-validated-runtime`，独立 `--loadEngine` 输出
  `load-engine-reference-validated-runtime`；两条路径均为 10/10 比较、0 mismatch、`OutputValidated=true`，最大绝对/
  相对误差为 `9.536743e-07` / `1.3443339e-06`，使用 `1e-4` absolute/relative tolerance。
- `samples/RefittedPlan.PackageConsumer` 增加独立 structured reference parser/comparer，验证 schema/name/shape/count、
  tolerance、NaN/Infinity policy 和全部值；现有 local-only feed consumer 仍保留 raw SHA256 精确比较。真实 run 的
  reference comparison 为 10/10、0 mismatch，strict validator 为 `53/53`。
- 新增 `tensorrtexec-mnist-reference-validation-evidence.json` 与 strict validator；要求本机 runtime artifacts 时为
  `44/44`，交叉检查 reference/sidecar、build/load engine/output/report/raw hashes、package consumer evidence 和
  许可/Owner/proof flags。

### Proof Boundary

- 模型 README 只能说明 TensorRT 样例指向 ONNX Model Zoo，样例 license 文本只能作为 Owner 审查输入；当前没有
  仓库再分发批准，也没有独立 ONNX Runtime reference 或 Owner golden-output 接受记录。
- generic 工具报告保持 `synthetic-input-runtime` 的受限分类；紧凑记录单独标为
  `real-model-reference-candidate-runtime`，两者均不能升级为 independent numerical correctness、Owner accepted
  real-model、public package、post-publish 或 release-close proof。
- 本阶段未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release 操作或 issue close。

## 2026-07-29 MNIST Reference Consumer And Cross-Version ABI Closure

本阶段完成 MNIST structured reference、隔离本地 PackageReference consumer 与多版本 evidence matrix 的最终验证，
并纠正一次由 vendor header 类型误判造成的 `IExecutionContext::getErrorBuffer` 虚假实现声明。

### `getErrorBuffer` 事实纠正

- `native/src/tensorrt/v8/api.cpp` 参与 TRT8 与 TRT10 bridge 编译；两套实际 vendor headers 上的标准
  `nvinfer1::IExecutionContext` 都没有 `getErrorBuffer()`。不能把 `NvInferSafeRuntime.h` 中安全运行时类型的同名接口
  直接映射到标准 execution-context owner。
- 保留 `jyppx_trt8_execution_context_get_error_buffer_copy` C ABI 与托管 `TryGetErrorBuffer` 兼容面，但 native 入口
  不再访问 vendor 对象或 borrowed pointer：清零 required size 后无条件返回明确的 `NotImplemented` deferred diagnostic。
- manifest ID 改为 `trt8-execution-context-get-error-buffer-copy-deferred`；旧 deferred history 继续保留。
  coverage 中两个 TRT8 CUDA 变体的 `IExecutionContext::getErrorBuffer` 均恢复为 `deferred-only`，不再声称
  `implemented-with-deferred-history`。
- 若未来接入 TensorRT safe runtime，必须新增独立 safe execution-context owner/lifetime/runtime proof，不能复用标准
  context 指针或只靠 header 同名方法晋级。

### 生成、native 与 ABI 验证

- binding generator 两次幂等通过：`201 manifests / 4001 API records`。
- 当前本机 coverage：TRT8 `760 implemented / 120 deferred-only`、TRT10 `761 / 118`、TRT11 `814 / 87`。
- `win-x64-trt8-cuda11-release` 与 `win-x64-trt10-cuda11-release` 均成功重建 DLL；保留既有 C4127/C4244
  compiler warnings，不将 native build 记为 zero-warning。
- ABI declaration parity：TRT8 `994/994`、TRT10 `1087/1087`、TRT11 `1234/1234`；新重建 TRT8/CUDA11
  与 TRT10/CUDA11 bridge 的 PE export missing 均为 `0`。

### Evidence 与测试

- MNIST strict validator 使用 PowerShell 7 和 `-RequireRuntimeArtifacts`：`44/44`；local package-consumer strict
  validator：`53/53`。Windows PowerShell 5.1 不支持脚本使用的 `ConvertFrom-Json -Depth`，该次调用失败，不计入通过结果。
- `ExecutionContextErrorBufferCopyTests`、multi-version matrix、MNIST evidence、package-consumer 四类窄集合：`11/11`。
- 扩展 TensorRtExec/trtexec 集合首次为 `122/124`，暴露 b5390b5 多输入/reference 重构后两处过期结构断言；同步
  release gap Markdown/status 和 benchmark worker 的 `input.Binding.Name + input.Shape` 断言后，局部 `5/5`、扩展集合
  `124/124` 通过。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。本阶段没有运行完整 ProjectQuality 全集，因此不声明
  全量 ProjectQuality 通过。

### Proof 与发布边界

- source-tree build、独立 load-engine 与 local-only PackageReference consumer 的 reference comparison 均通过，但
  reference 仍是 `repository-mnist-runtime-output-derived-unreviewed`，不是独立 ONNX Runtime golden 或 Owner accepted output。
- local feed、真实本机 GPU、ABI/export parity 和 retained runtime artifacts 都不是 public package、post-publish、Linux
  runner 或 release-close proof。
- C 盘 `Downloads`/用户 Temp 的本批关键词审计无命中；隔离 consumer workspace 已删除。系统与用户目录两套
  `dotnet build-server shutdown` 均已执行，并精确终止本轮 CMake 留下的 orphan MSBuild node；最终 build/test/compiler
  进程残留为 `0`。
- 本阶段未 push、未触发 GitHub Actions，未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 Independent ONNX Runtime Reference And Fail-Closed Negative Runtime

本阶段为 TensorRtExec MNIST structured reference 增加真正独立于 TensorRT 执行的 ONNX Runtime CPU 候选，并将
name/shape/value-count/NaN/Infinity 五类受控负例贯通 source-tree CLI 与隔离 local PackageReference consumer。

### 独立 ORT CPU 证据

- 隔离 producer 使用 ONNX Runtime `1.23.2`、显式 `CPUExecutionProvider` 和单线程顺序执行；profiling trace 仅包含
  CPU provider。两次运行的 float32 bytes 完全一致，raw SHA256 为
  `a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75`，预测 digit 7。
- runner 不下载包，只复制本机 NuGet cache 中已有的四个 `.nupkg` 到 E 盘临时 feed，清空远程源并使用隔离 restore
  cache；主 solution 不增加 ONNX Runtime 依赖，workspace 已删除。
- 新 reference 分类为 `onnxruntime-cpu-1.23.2-derived-unreviewed`，SHA256 为
  `1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571`。与旧 TensorRT reference 比较 10/10、
  0 mismatch，最大绝对/相对误差 `5.722046e-06` / `5.7323444e-07`，在 `1e-4` tolerance 内。
- ORT strict validator 在要求 raw/profile/log 时为 `51/51`。clean clone 只依赖检入的 reference、sidecar、compact
  evidence 和 validation summary，不要求本机重运行工件。

### 受控负向运行

- 新增五类 reference：tensor name mismatch、shape mismatch、value count mismatch、NaN/reject、Infinity/reject。
- source-tree CLI 与隔离 local-feed PackageReference consumer 各执行五次，共 10 次真实 TensorRT enqueue/readback。
  source-tree 均以退出码 2、consumer 均以退出码 1 fail closed；两边均记录 `OutputValidated=false`，raw output SHA256
  仍为 `0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5`。
- consumer 的 metadata mismatch 改为返回不可比较的 validation result，确保先记录 enqueue/output/diagnostic，再非零
  退出；owner scope 在退出前已释放并记录 `OwnerScopeExited=true`。metadata 三例为 `Completed=false`，特殊值两例为
  `Completed=true`、1 mismatch、first mismatch 0。
- 负向 strict validator 为 `72/72`；multi-version matrix 保留原 19 cases / 15 passed / 4 blocked，同时新增 ORT CPU
  candidate 与 5/5 + 5/5 fail-closed 摘要，不重复增加版本案例计数。

### Proof 与发布边界

- ORT CPU profile 证明执行路径独立于 TensorRT，但 Owner 尚未接受它为 golden，也未批准 model/input/reference 的仓库
  再分发；`independent-framework-reference-candidate-runtime` 不能晋级为 Owner accepted real-model proof。
- 受控畸形 reference 只证明真实 enqueue/readback 后 fail closed；本地 PackageReference feed 不是 public package proof。
- 本阶段未 push、未触发 GitHub Actions，未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

### 最终验证

- ORT / same-runtime / negative / local consumer strict validators：`51/51`、`44/44`、`72/72`、`53/53`。
- TensorRtExec、trtexec、OnnxToEngine parity、multi-version matrix 与 publishing article 扩展定向集合：`155/155`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- ORT、negative 与正向 package consumer 的 E 盘隔离 workspace 均已删除；本批未运行完整 ProjectQuality 全集。
- C 盘 Downloads/用户 Temp 本批关键词与当日 `.onnx/.engine/.plan/.nupkg` 命中均为 0；dotnet build server 已关闭，
  最终相关 build/test/compiler 进程残留为 0。

## 2026-07-29 Cross-Task Reference Provenance Contract

本阶段把 Classification 与 YoloVision 的 independent-reference 准入字段收敛为一个机器可读合同，同时保持任务专属
preprocess/postprocess/output semantics，不允许通过复用 MNIST reference 或通用 output hash 消除真实缺口。

### 合同与矩阵

- 新增 `samples/assets/cross-task-reference-provenance-contract.json`。公共层覆盖 asset identity、tensor contract、
  execution identity、reference identity、comparison policy 与 Owner decision；复用要求 model/input/preprocess/output/
  labels/task-semantics 六个 SHA256 fingerprint 全部存在且完全一致。
- 任务层包含 7 个 profile：generic Classification，以及 YoloVision det/cls/seg/obb/pose/sem。Classification 固定
  resize/crop/color/scale/mean/std、raw logits/probabilities、score transform、labels、Top-K/argmax；六类 YOLO 分别固定
  box/NMS、class score、mask composition、angle、keypoint、semantic-map 语义。
- exporter 交叉读取 Classification manifest、Yolo task contract、六任务 Owner input template 与 MNIST ORT compact
  evidence。当前矩阵为 7 rows / 0 ready / 7 owner-action-required；每行保留 ready/required/missing 字段与确切缺口。
- MNIST ORT candidate 明确 `eligibleTaskIds=[mnist]`，对 7 个矩阵 task 全部 ineligible；其 model/input/preprocess/
  output/labels/task semantics 不匹配，且 Owner golden/redistribution 仍为 false。

### 验证与文档

- `Test-CrossTaskReferenceProvenanceMatrix.ps1 -Strict` 校验合同结构、四个源文件 hash、字段集合、任务隔离、candidate
  provenance 与全部 promotion flags：`93/93`。
- 新增 4 项 ProjectQuality 门禁并实际执行 exporter/validator；Classification/YoloVision 资产、schema、Owner template、
  candidate validator 与 publishing article 扩展集合：`65/65`。
- Classification、YoloVision、samples/assets README 与两篇中文资产文章已同步 reuse fingerprint 和任务语义边界。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`；本批未运行完整 ProjectQuality 全集。

### Proof 边界

- readiness matrix 是缺口审计，不生成 reference、不批准 license、不接受 Owner golden，也不证明 real-model runtime。
- independent framework candidate 不是 Owner golden；local PackageReference 不是 public package；hash match 不能绕过任务
  semantics 或 redistribution decision。
- 所有 `canPromoteRealModelRuntime`、`canPromotePackageConsumerRuntime`、`canPublishPublicly`、`canCloseReleaseIssue`
  继续保持 false。
- C 盘 Downloads/用户 Temp 的本批关键词与当日重资产命中均为 0；dotnet build server 已关闭，最终相关进程为 0。

## 2026-07-29 Classification Real Input And Reference Artifact Closure

本阶段补齐 generic Classification 的托管真实图片输入、可审计输出 JSON 与 task-specific structured reference
comparison。它解决了“将图片原始 bytes 当作 tensor”的歧义，但未提供分类模型、许可资产、独立 framework
reference 或 Owner golden。

### 实现

- 抽取 `JYPPX.SampleSupport.SampleRgbImageDecoder`，供 Classification 与 YoloVision 共用；支持 P3/P6 PPM/PNM 和
  uncompressed 24/32-bit BMP，覆盖 CRLF P6 头、max-value、截断与不支持格式的 fail-closed 行为。
- Classification 新增 `--image`、`--preprocessed-output`、stretch 与 shorter-side-center-crop、NCHW/NHWC、RGB/BGR、
  scale/mean/std 和稳定的 preprocessing contract SHA256。raw `--input` 仍仅表示一元素一字节的归一化 tensor，
  `--input-data` 表示调用方已经预处理的 float32/text tensor。
- 输入 fingerprint 现在基于实际送入 TensorRT 的 float32 值，而非 external raw/text 源文件 bytes；外部预处理的
  reference comparison 必须显式提供合法小写 `--preprocess-contract-sha256`。
- 新增 raw/softmax 输出、稳定的 score-first/index-second Top-K、`classification-output.v1` 报告、
  `classification-reference.schema.json` 与 `classification-output.schema.json`。reference 验证严格检查
  name/shape/value count/value kind 与 model/input/preprocess/output/labels/task semantics 六个 SHA256 fingerprint，
  并覆盖 absolute/relative tolerance、NaN reject/equal、Infinity exact/reject 和 UTF-8 BOM。
- metadata mismatch 返回 `Completed=false`；value mismatch 返回 `Completed=true, Passed=false` 且进程以 1 退出。
  所有 report boundary flags 继续是 false。
- cross-task matrix 为 Classification 行新增 `implemented-managed-contract-owner-assets-required`，列出四个运行时
  合同文件，同时仍保持 `7 rows / 0 ready / 7 owner-action-required`；strict validator 从 `93/93` 增至 `94/94`。

### 验证

- Classification/YoloVision/CrossTask/output-schema 定向 ProjectQuality 集合：`63/63` 通过。
- `Export-CrossTaskReferenceProvenanceMatrix.ps1` 后 strict validator：`94/94`；matrix 保持 0 ready、7 owner action required。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 完整 `JYPPX.ProjectQuality.Tests` 两次均在 604 秒内没有完成最终汇总；已精确终止两棵遗留 testhost 进程树，
  因此不将其表述为全量通过。

### Proof Boundary

- 本批的 BMP/PPM decoder、preprocessing tensor、output JSON、reference schema、hash 或 managed tests 都不是
  real-model-runtime、independent-framework golden、Owner accepted golden、package-consumer、public package、
  post-publish、Linux runner 或 release-close proof。
- Owner 仍需提供可审查 model/labels/image/license/hash、独立 provider reference、reference source classification
  及 golden/redistribution decision；在此之前所有 promotion/publication flags 保持 false。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。


## 2026-07-29 CUDA Driver Managed Source Module Closure

本阶段继续收口托管公开 API 的目录职责，把 CUDA Driver capability、module owner 与 typed launch owner 从通用
`Core`/`Kernels` 目录统一归入独立 `Drivers` 模块。该变更仅调整源码组织，不改变 namespace、类型名、public API、
owner 生命周期、interop 声明或 C ABI。

### 实现与门禁

- `CudaDriver.cs` 从 `Core` 移至 `Drivers`；`CudaDriverModule.cs` 与 `CudaDriverKernelLaunch.cs` 从 `Kernels` 移至
  `Drivers`。`Kernels` 继续负责 CUDA Runtime kernel library、argument 与 launch configuration。
- `ManagedSourceModuleLayoutTests` 将 `Drivers` 纳入 `JYPPX.CudaSharp` 约定模块，并固定当前三份 Driver owner 文件的
  职责集合，同时拒绝旧 `Core`/`Kernels` 路径回流。
- CUDA RTC owner 测试的源码审计路径和中英文 source-organization 文档已同步更新。

### 验证与边界

- layout、CUDA RTC owner 与 roadmap 定向 ProjectQuality 集合：`20/20` 通过。
- `JYPPX.CudaSharp` 全目标框架 Debug build：`0 warning / 0 error`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 本批未修改 native、manifest 或 generated bindings，因此未重跑 native/ABI/export/generator；也未运行会长时间挂起的
  完整 ProjectQuality 全集，不将定向结果表述为全量通过。
- 文件重定位不是 CUDA kernel correctness、Linux、package consumer、public package、post-publish、Owner accepted 或
  release proof；所有相关 promotion 边界保持不变。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0；用户 Temp
  递归枚举在 30 秒超时，因此不把该项表述为完整递归审计。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Interfaces And Weights Managed Module Closure

本阶段继续拆分 `JYPPX.TensorRtSharp/Core` 中的跨职责公开 API，将版本化 interface metadata 与 weights 类型归入
独立模块。六份文件均为逐字节纯移动，namespace、类型名、public surface、partial owner surface 与行为不变。

### 实现与门禁

- `Interfaces` 包含 `TensorRtInterfaceInfo`、`TensorRtVersionedInterfaceMetadata` 与 owner-scoped metadata query/surface；
  它们共同负责 copied、pointer-free 的版本化接口信息。
- `Weights` 包含托管 immutable weights payload、复制型 weights metadata 与 refit weights role。
- `Core` 由 12 份文件收敛为 6 份，只保留共享异常、dims、enums 与 `TensorRtSharpInfo`。
- 布局测试精确固定两个新模块各三份文件，并拒绝它们回流 `Core`；三处版本化接口源码路径合同已同步到新目录。
- 中英文 source-organization 模块表和职责说明已同步更新。

### 验证与边界

- layout、progress-monitor boundary、owner-scoped versioned metadata 与 API-language readonly 定向集合：`22/22` 通过。
- `JYPPX.TensorRtSharp` 全目标框架 Debug build：`0 warning / 0 error`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 六组新旧文件 Git blob hash 完全一致；本批未修改 native、manifest、generated bindings 或 ABI。
- 未运行完整 ProjectQuality 全集；源码目录归类不是 runtime correctness、Linux、real model、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Callback Interface Diagnostics Managed Interop Module Closure

本阶段继续整理 TensorRtSharp 手写 interop，将 callback、owner-scoped interface metadata 与 error-code diagnostics
共 8 份单一职责文件归入对应模块。Generated、namespace、partial type、delegate signature、entrypoint、owner 行为与
public API 均未改变。

### 实现与门禁

- `Callbacks` 包含 allocator dry-run、callback interface/state copied operations，以及 logger/profiler/progress-monitor
  三种 unmanaged delegate signature，共 6 份文件。
- `Interfaces` 包含 owner-scoped versioned-interface metadata copy；`Diagnostics` 包含 error-code metadata bound。
- 精确 internal layout 门禁固定三个模块的 8 份文件并拒绝根目录回流；所有源码路径合同和中英文文档已同步。
- `SafeDeferredUplift.cs` 保留根目录，因为它混合 plugin initialization 与 ONNX weight-descriptor parsing；
  `GlobalRuntimePluginProbe.cs` 的跨职责边界也保持不变。

### 验证与边界

- layout、callback allocator/interface/state、logger/profiler/progress-monitor 与 owner-scoped metadata 定向集合：
  `67/67` 通过。
- `JYPPX.TensorRtSharp` 全目标框架 Debug build：`0 warning / 0 error`；完整 solution build：`0 warning / 0 error`。
- 8 组新旧 Git blob hash 完全一致；Generated/native/manifest/ABI 未修改，未重跑 generator/native/export parity。
- 未运行完整 ProjectQuality；前批已确认的 `pwsh` 缺失边界保持。本批不是 callback native invocation、trampoline、
  in-flight accounting、detach-before-release、lifetime、Linux、package 或发布 proof。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Parsing And Plugins Managed Interop Module Closure

本阶段将 TensorRtSharp 手写 interop 中 13 份单一职责 partial API 归入 `Parsing` 与 `Plugins`。Generated、namespace、
partial type、method、P/Invoke/entrypoint、owner 行为与 public API 均未改变。

### 实现与门禁

- `Parsing` 包含 legacy parser diagnostics、ONNX config/model buffer/support、builder-config attachment、layer-output
  metadata 与 parser-refitter diagnostics，共 7 份文件。
- `Plugins` 包含 builder capability/runtime registry inventory、V2/V3 layer metadata 与 owner-scoped query snapshot，
  共 6 份文件。
- 精确 internal interop 布局门禁固定两个模块的 13 份文件并拒绝根目录回流；所有 ProjectQuality `ReadSource` 路径和
  中英文 source-organization 已同步。
- `NativeBridgeApi.GlobalRuntimePluginProbe.cs` 保留根目录，因为它混合 runtime version、logger、ONNX parser version
  与 plugin registry；需要单独拆文件，不能标记为纯 plugin。
- 修正两项既有测试漂移：versioned parser-refitter alias 断言对齐 exporter 当前三条真实 alias；plugin field copy
  断言改为验证 `TensorRtPluginFieldInfo` 构造，不再锁死 `fields`/`fieldList` 局部变量名。

### 验证与边界

- 首次扩展集合 `120/123`：两项既有过时断言失败，修正后均通过；另一项 B-tier exporter 测试因本机无 `pwsh`
  无法启动。未用 Windows PowerShell 5.1 替代 PS7。
- 排除该 PS7 环境项后，可执行的 layout、parser、plugin、ABI-contract 定向集合：`122/122` 通过。
- `JYPPX.TensorRtSharp` 全目标框架 Debug build：`0 warning / 0 error`；完整 solution build：`0 warning / 0 error`。
- 13 组新旧 Git blob hash 完全一致；Generated/native/manifest/ABI 未修改，未重跑 generator/native/export parity。
- 未运行完整 ProjectQuality；本批不是 ABI/export、parser/plugin runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 CUDA Managed Interop Feature Module Closure

本阶段继续整理 `JYPPX.CudaSharp/Internal/Interop` 的手写 partial API，将 8 份职责明确的文件与公开 owner feature area
对齐。Generated 文件、namespace、partial type、method、P/Invoke/entrypoint、owner 行为与 public API 均未改变。

### 实现与门禁

- `Devices`：device-resource snapshot 与 primary execution-context 操作。
- `Diagnostics`：copied runtime log 操作；`Drivers`：optional Driver capability/module/typed launch 操作。
- `IPC`：owner-safe export/import token 操作；`Kernels`：Runtime kernel-library owner/launch 操作。
- `RuntimeCompilation`：optional NVRTC program capability/create/compile/artifact/lowered-name 操作。
- 精确 interop 布局门禁固定六个 feature module 的 8 份文件，并检查它们不再回流 interop 根目录；六处源码路径测试和
  中英文 source-organization 已同步。
- `NativeCudaApi.Deployment.cs` 仍保留根目录，因为它横跨 error、PCI、stream/event、pinned memory、atomic capability
  与 device selection；需要另立拆文件批次，不能错误归入 `Devices`。

### 验证与边界

- layout、device context/resource、IPC、kernel library 与 CUDA RTC owner 定向集合：`61/61` 通过。
- `JYPPX.CudaSharp` 全目标框架 Debug build：`0 warning / 0 error`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 8 组新旧 Git blob hash 完全一致；未修改 Generated、manifest、native 或 ABI，因此未重跑 generator/native/export parity。
- 未运行完整 ProjectQuality；本批不是 ABI/export、CUDA kernel correctness、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Tools Refit Managed Module Closure

本阶段将 `JYPPX.TensorRtSharp.Tools/Build` 中两份纯 refit evidence model 归入独立 `Refit` 模块，使 build orchestration
与 refit lifecycle/persistence snapshot 的源码职责分开。文件内容、namespace、类型名、public API 与消费关系不变。

### 实现与门禁

- `OnnxEngineRefitSnapshot.cs` 与 `OnnxEngineRefitPersistenceSnapshot.cs` 从 `Build` 纯移动到 `Refit`。
- `Build` 继续保留 build profile/report/shape、options、service、diagnostics、result 与 parser preflight；它通过同一
  `JYPPX.TensorRtSharp.Tools` namespace 消费 refit snapshot，不需要项目文件或调用点修改。
- 布局测试精确固定 `Refit` 的两份文件并拒绝旧 `Build` 路径；两条源码合同和中英文组织文档已同步。

### 验证与边界

- layout、ONNX refit lifecycle 与 refitted-plan persistence 定向集合：`11/11` 通过。
- `JYPPX.TensorRtSharp.Tools` Debug build：`0 warning / 0 error`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 两组新旧 Git blob hash 完全一致；未修改 refit 实现、native、manifest、generated bindings 或 ABI。
- 未运行完整 ProjectQuality；本批不是 refit runtime、persisted engine correctness、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads/用户 Temp 顶层当日重资产与本批关键词命中为 0，项目相关 build/test 进程残留为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Feature Partial Managed Interop Module Closure

本阶段继续整理 `JYPPX.TensorRtSharp/Internal/Interop` 的手写 partial API，将 11 份职责单一的文件归入
`Builder`、`ControlFlow`、`Inference`、`Layers`、`Network` 与 `Weights`。所有文件均为纯目录移动；namespace、
partial type、method、P/Invoke/entrypoint、owner 行为与 public API 均未改变。

### 实现与门禁

- `Builder` 包含 timing-cache 操作；`ControlFlow` 包含 loop/conditional 操作。
- `Inference` 包含同步 execute/enqueue 操作；`Weights` 包含复制型 layer-weight metadata。
- `Layers` 包含 quantization、attention、fill-int64、tensor metadata、transformer 与 RNNv2 操作，共 6 份文件。
- `Network` 包含 safe network-v2 layer 创建操作。
- 精确 internal interop 布局门禁固定六个模块的 11 份文件；RNN 与 synchronous inference 的源码路径合同已同步，
  旧根路径引用扫描为 0，中英文 source-organization 已补齐模块职责。
- `Trt11Diagnostics`、`Trt11Dims64`、`Trt11RuntimeControls`、`Trt11RuntimeSerializationRefit`、
  `GlobalRuntimePluginProbe`、`SafeDeferredUplift` 与 `DeploymentMetadata` 继续保留根目录，因为其方法集合跨 owner 或
  feature；不能仅按 `Trt11` 文件名前缀机械分类。

### 验证与边界

- layout、engine/RNN readonly diagnostics、synchronous inference、RNN borrowed-state design 与 inference binding fallback
  定向集合：`35/35` 通过。
- `JYPPX.TensorRtSharp` 全目标框架 Debug build：`0 warning / 0 error`。
- 完整 `TensorRtSharp.sln` Debug build：`0 warning / 0 error`。
- 11 组唯一新旧 Git blob hash 完全一致；`git diff --check` 通过，Generated/native/manifest/ABI 未修改。
- 未运行完整 ProjectQuality；本批不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件为 0，用户 Temp 顶层本批关键词命中为 0，项目相关 build/test 进程残留为 0；
  Downloads 中既存的历史 CUDA/TensorRT 安装包与运行时包未改动。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Layer Attributes And Build Probe Managed Interop Module Closure

本阶段继续清理 TensorRT interop 根目录，将单一 layer-owner 的兼容/部署属性操作归入 `Layers`，并将仅由
`TensorRtEnvironmentProbe` 调用的两条 TRT11 build-probe 入口归入 `Diagnostics`。两份文件内容均保持不变；后者只将
含混的 `NativeBridgeApi.Trt11.cs` 文件名明确为 `NativeBridgeApi.Trt11BuildProbe.cs`。

### 实现与门禁

- `Layers/NativeBridgeApi.ThirtyThirdBatchLayerAttributes.cs` 包含 convolution/deconvolution padding、slice axes、
  normalization compute precision、resize align-corners、TopK indices type 与 dequantize block-shape 操作，共 14 个
  public static 方法，全部接收 layer owner。
- `Diagnostics/NativeBridgeApi.Trt11BuildProbe.cs` 只包含 minimal build chain 与 serialized-network-only 两条诊断 probe；
  仓库内调用点仅位于 `TensorRtEnvironmentProbe`。
- 精确布局门禁将这两份文件固定到 `Layers` 与 `Diagnostics`，旧根路径引用扫描为 0；中英文
  source-organization 同步记录职责和剩余跨 owner 文件边界。
- `Trt11DeploymentAdditions` 仍保留根目录，因为它同时包含 network layer 创建和 layer attribute 操作，需要单独拆分。

### 验证与边界

- 首次扩展集合 `29/30`：唯一失败是 PS7 专用测试无法启动本机不存在的 `pwsh`，没有代码断言失败，也未用 Windows
  PowerShell 5.1 替代。
- 排除该 PS7 环境项后，layout、TRT11 compatible-host 与 runtime-create diagnostics 定向集合：`29/29` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- 两组新旧 Git blob hash 完全一致；`git diff --check` 通过，Generated/native/manifest/ABI 未修改。
- 未运行完整 ProjectQuality；本批不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Deployment Additions Partial Split

本阶段将历史 `NativeBridgeApi.Trt11DeploymentAdditions.cs` 按方法 owner 拆为 Network 与 Layers 两份 partial 文件，
不再让 network layer 创建和 layer attribute getter/setter 共存于 interop 根目录。拆分仅移动原有连续代码块，未改动
namespace、partial type、方法签名/方法体、P/Invoke entrypoint、版本守卫、异常文案或 helper 行为。

### 实现与门禁

- `Network/NativeBridgeApi.DeploymentNetworkLayers.cs` 为 454 行，包含 20 个 public static 方法：16 个 `Add*Layer`
  创建入口，以及 refittable-weight 的 mark/unmark/query/name 四个 network 操作。
- `Layers/NativeBridgeApi.DeploymentLayerAttributes.cs` 为 506 行，包含 66 个 public static layer attribute 方法，覆盖
  gather/scatter/one-hot/cumulative/assertion/grid-sample/normalization/dynamic-quantize/NMS/einsum/reverse-sequence 等职责；
  其中 `Add*Layer` 方法为 0。
- `ManagedSourceModuleLayoutTests` 同时固定两份新文件的目录、20/66 方法数量和 owner 命名边界，并拒绝旧根文件回流。
- 两份新文件按原第 454 行边界重组后的 Git blob 为 `4230870fb121047b18c894a8935156c431db1825`，与 HEAD 中
  拆分前原文件完全一致；旧文件名仅保留在“文件必须不存在”的负向门禁中。

### 验证与边界

- layout 与 deployment owner 方法集合定向测试：`24/24` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- partial 文件拆分不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Runtime Serialization Refit Partial Split And Evidence Path Closure

本阶段将 776 行 `NativeBridgeApi.Trt11RuntimeSerializationRefit.cs` 按 owner 拆入 `Runtime`、`Serialization`、
`Execution` 与 `Refit`。原文件中 serialization 与 execution/runtime-config 方法交错，本批按原始片段边界机械提取，
没有改动方法签名/方法体、版本分支、异常文案、entrypoint 或 helper 行为。

### 实现与门禁

- `Runtime/NativeBridgeApi.RuntimeDeploymentControls.cs`：264 行、17 个 runtime control/copied diagnostic 方法。
- `Serialization/NativeBridgeApi.EngineSerialization.cs`：126 行、8 个 engine serialization/config-flag 方法。
- `Execution/NativeBridgeApi.ExecutionContextCreation.cs`：70 行、5 个 context/runtime-config 创建与 allocation-strategy 方法。
- `Refit/NativeBridgeApi.RefitterControls.cs`：352 行、23 个 async refit、weights/dynamic-range 与 copied diagnostic 方法。
- 方法级布局门禁固定 17/8/5/23 数量和 owner 命名规则，并拒绝旧根文件回流；6 处硬编码源码合同已按断言职责读取
  对应新文件，旧消费路径扫描为 0。
- 四文件按原交错片段顺序重组后的 Git blob 为 `1acfd88f713d5bbf7b22ea990a29246fc1b023fd`，与 HEAD 原文件一致。

### Evidence 路径校准

- 首次定向集合为 `58/59`，唯一失败来自本机忽略 artifact `deferred-readonly-candidate-list.json` 中的旧 managedSources
  路径，而非代码断言失败。
- 审计发现 69 条失效路径、48 个唯一旧路径；47 个可按文件名唯一映射，复合 interop 条目映射为 Runtime 与 Refit 两条。
- 本机 artifact 完成 69 条原位替换后，所有候选的 native/managed/smoke/quality evidence 路径缺失为 0；该 artifact
  受 `.gitignore` 管理且从未受 Git 跟踪，本批没有用 `git add -f` 改变其版本控制边界。
- 新增通用 ProjectQuality 门禁，遍历所有带 `implementationEvidence` 的候选和证据桶，防止后续目录移动留下断链。

### 验证与边界

- 受影响 layout、callback/runtime diagnostics、readonly evidence、runtime serialization 与 refitter diagnostics 集合：
  最终 `60/60` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- partial 拆分与 evidence path consistency 不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Deployment Metadata Owner Partial Split

本阶段将 1478 行 `NativeBridgeApi.DeploymentMetadata.cs` 拆为 Engine、Refit、Execution、Layers 与 Shared 五份 partial。
所有 delegate 和跨 owner private helper 统一保留在根目录 Shared 文件；公共方法按原 owner 区段机械提取，没有复制 helper，
也没有改动方法签名/方法体、entrypoint、版本守卫或异常文案。

### 实现与门禁

- `NativeBridgeApi.DeploymentMetadataShared.cs`：482 行、0 个 public static 方法，保存 25 个 delegate 与跨 owner helper。
- `Engine/NativeBridgeApi.EngineDeploymentMetadata.cs`：197 行、22 个 engine/tensor/profile metadata 方法。
- `Refit/NativeBridgeApi.RefitterDeploymentMetadata.cs`：106 行、6 个 refitter entry/weights/refit 方法。
- `Execution/NativeBridgeApi.ExecutionContextDeploymentMetadata.cs`：362 行、28 个 context shape/debug/profile/memory 方法。
- `Layers/NativeBridgeApi.LayerDeploymentMetadata.cs`：383 行、74 个 classic layer attribute 方法。
- 方法级布局门禁固定 22/6/28/74/0 数量与 owner 规则，并拒绝旧根文件回流；三处源码合同分别改为读取实际需要的
  owner 文件，B-tier 聚合合同显式读取五份完整源码，旧消费路径为 0。
- 五文件按原七段顺序重组后的 Git blob 为 `6ef1d699b455403ee73ccbdc7a50b860d0f5934e`，与 HEAD 原文件一致。

### 验证与边界

- 首次扩展集合 `41/42`：唯一失败是 B-tier 测试在进入源码断言前无法启动本机不存在的 `pwsh`；未使用 Windows
  PowerShell 5.1 替代。
- 排除该 PS7 环境项后，layout、execution-context error-buffer、engine/RNN diagnostics 与 readonly evidence 定向集合
  `41/41` 通过；五个新源码路径均存在，旧消费路径为 0。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的 engine metadata 路径已在本机校准，全 evidence 路径缺失保持 0；该文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- partial 拆分不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Dims64 Owner Partial Split

本阶段将 273 行 `NativeBridgeApi.Trt11Dims64.cs` 按 owner 拆入 Network、Engine、Execution、Profiles 与 Layers。
三个 layer getter delegate 和三个私有 helper 只服务 layer 方法，因此随 Layers 文件移动，不需要新增 Shared partial。
所有方法签名/方法体、entrypoint、版本守卫与异常行为保持不变。

### 实现与门禁

- `Network/NativeBridgeApi.Dims64NetworkTensor.cs`：57 行、6 个 tensor/network shape 与 extent 方法。
- `Engine/NativeBridgeApi.Dims64EngineMetadata.cs`：49 行、4 个 engine tensor/profile shape 方法。
- `Execution/NativeBridgeApi.Dims64ExecutionContext.cs`：49 行、4 个 context shape/stride 方法。
- `Profiles/NativeBridgeApi.Dims64OptimizationProfile.cs`：29 行、2 个 optimization-profile shape 方法。
- `Layers/NativeBridgeApi.Dims64LayerMetadata.cs`：125 行、27 个 layer slot/feature Dims64 方法及专属 delegate/helper。
- 方法级布局门禁固定 6/4/4/2/27 数量、`*64` 后缀与 owner 排斥规则，并拒绝旧根文件回流。
- 五文件按原七段顺序重组后的 Git blob 为 `4d686dae4a948d978328c0d0e9da98d452f4adb9`，与 HEAD 原文件一致；
  旧文件名仅保留在负向门禁中，不存在源码、文档或 evidence 消费路径。

### 验证与边界

- layout 与 parser layer-output metadata 定向集合：`39/39` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的全 evidence 路径缺失保持 0；本批无需更新该 artifact。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- Dims64 partial 拆分不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Runtime Controls Owner Partial Split

本阶段将 404 行 `NativeBridgeApi.Trt11RuntimeControls.cs` 按真实 owner 拆入 Engine、Execution 与 Builder。
原文件由 Engine、Execution、Builder 三个连续职责区段组成；本批只移动完整方法块，没有改动方法签名/方法体、
版本分支、entrypoint、异常文案、UTF-8 转换或返回值处理。

### 实现与门禁

- `Engine/NativeBridgeApi.EngineRuntimeControls.cs`：123 行、9 个 weight-streaming、engine stat、hardware compatibility
  与 implicit-batch compatibility 方法。
- `Execution/NativeBridgeApi.ExecutionContextRuntimeControls.cs`：57 行、4 个 input-consumed event、output tensor address、
  output allocator 与 temporary-storage allocator presence 方法。
- `Builder/NativeBridgeApi.BuilderConfigRuntimeControls.cs`：242 行、19 个 flags、device/DLA、tiling、max tactics、
  quantization flags 与 remote auto-tuning 方法。
- 方法级布局门禁固定 19/9/4 数量和 BuilderConfig/Engine/ExecutionContext owner 命名边界，并拒绝旧根文件回流；
  7 处受影响源码合同已改为读取实际 owner 文件，旧 interop 消费路径为 0。
- 三文件按原 `Builder header + Engine + Execution + Builder body` 片段顺序重组后的 Git blob 为
  `896499a3ed2ee5ec9bb3fab8a962e2b6306ab9cc`，与 HEAD 原文件完全一致。

### 验证与边界

- layout、runtime serialization、execution-context readonly、engine/RNN diagnostics、B-tier 41-45、BuilderConfig scalar
  与 readonly evidence 定向集合：`59/59` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 已在本机迁移 Builder interop 路径；243 条 evidence 引用、133 个唯一路径缺失为 0，
  且该文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Diagnostics Owner Partial Split

本阶段将 422 行 `NativeBridgeApi.Trt11Diagnostics.cs` 按真实 owner 拆入 Builder、Network、Engine 与 Execution。
原文件的四个职责区段连续且没有共享 delegate/private helper；本批只移动完整方法块，没有改动方法签名/方法体、
版本分支、entrypoint、异常文案、UTF-8 buffer 读取或地址诊断的整数转换。

### 实现与门禁

- `Builder/NativeBridgeApi.BuilderConfigDiagnostics.cs`：116 行、9 个 reset、timing-cache/DLA、plugin serialization 与
  progress-monitor 方法。
- `Network/NativeBridgeApi.NetworkDiagnostics.cs`：95 行、7 个 debug tensor、unfused debug 与 shape-output 方法。
- `Engine/NativeBridgeApi.EngineInspectorDiagnostics.cs`：50 行、4 个 layer information、execution-context 与
  error-recorder presence 方法。
- `Execution/NativeBridgeApi.ExecutionContextDiagnostics.cs`：188 行、15 个 address diagnostic、allocator/debug-listener/
  profiler/runtime-config/NVTX/aux-stream/unfused-debug 方法。
- 方法级布局门禁固定 9/7/4/15 数量和 BuilderConfig/Network/EngineInspector/ExecutionContext 命名边界，并拒绝旧根文件回流；
  7 处受影响源码合同已改为只读取实际 owner 文件，旧文件名只保留在负向门禁中。
- 四文件按原 Builder、Network、Engine、Execution 片段顺序重组后的 Git blob 为
  `89f756495b67e847d11dab8569dd26aa0e8229d2`，与 HEAD 原文件完全一致。

### 验证与边界

- layout、BuilderConfig scalar/B-tier、engine/RNN readonly、execution-context aux-stream/readonly、engine-inspector 与
  plugin serialization 定向集合：`71/71` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 已在本机迁移 Builder diagnostics 路径；243 条 evidence 引用、133 个唯一路径
  缺失为 0，且该文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Global Runtime Plugin Probe Behavior Split

本阶段将 788 行 `NativeBridgeApi.GlobalRuntimePluginProbe.cs` 按行为拆入 Runtime、Parsing、Plugins 与 helper-only
Shared partial。原文件的 runtime version/logger、ONNX parser version 与 global plugin registry 片段交错；本批按原
八段边界机械提取，没有改动方法签名/方法体、版本分支、entrypoint、异常文案、UTF-8 转换或 copied snapshot 逻辑。

### 实现与门禁

- `Runtime/NativeBridgeApi.GlobalRuntimeVersion.cs`：148 行、7 个 composite/infer-lib version 与 global logger 方法。
- `Parsing/NativeBridgeApi.GlobalOnnxParserVersion.cs`：32 行、1 个 global ONNX parser version 方法。
- `Plugins/NativeBridgeApi.GlobalPluginRegistry.cs`：618 行、8 个公开 inventory/lookup/registry 方法与 17 个私有
  creator/field helper；TRT8 optional field failure helper 随 Plugins 移动。
- `NativeBridgeApi.GlobalProbeShared.cs`：17 行、0 个公开方法，只保存 1 个跨三类行为共用的 unsupported-line helper。
- 布局门禁固定 7/1/8/0 公开方法数量、行为命名边界、Shared helper-only 约束和旧根文件禁止回流；5 个插件源码合同
  已改为读取 Plugins owner 文件。
- 四文件按原八段顺序重组后的 Git blob 为 `ecb073f64a8dbe0b0f9248275e6ce32e9ba99531`，与 HEAD 原文件完全一致。

### 验证与边界

- 首次定向集合 `55/56`：唯一失败是 plugin lookup 测试仍以拆分前紧邻的 runtime 方法作为字符串截取终点；改为
  Plugins 文件中的下一公开方法后，同一集合最终 `56/56` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的 3 条 global plugin registry 路径已在本机迁移；243 条 evidence 引用、
  133 个唯一路径缺失为 0，且该文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的兼容主机 exporter 测试。
- behavior split 不是 ABI/export、TensorRT runtime、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Safe Deferred Uplift Behavior Split

本阶段将 75 行 `NativeBridgeApi.SafeDeferredUplift.cs` 按行为拆入 Plugins 与 Parsing。原文件只包含彼此独立的
plugin initialization 和 ONNX weight-descriptor parsing 两个方法，没有共享 delegate/private helper；本批只移动
完整方法块，没有改动 logger/parser owner 校验、UTF-8 namespace、pinned byte[] 生命周期、版本分支或异常文案。

### 实现与门禁

- `Plugins/NativeBridgeApi.PluginInitialization.cs`：33 行、1 个 `InitializeLibNvInferPlugins` 方法，保留有效 logger
  要求及 TRT8/10/11 分支。
- `Parsing/NativeBridgeApi.OnnxWeightDescriptorParsing.cs`：52 行、1 个 `ParseOnnxWithWeightDescriptors` 方法，保留
  null/empty model 校验、`GCHandle` finally 释放及 TRT11 removed-by-vendor guard。
- 布局门禁固定两个文件各自唯一的方法并拒绝旧根文件回流；唯一硬编码源码合同改为拼接实际 Plugins/Parsing 文件。
- 两文件按原片段顺序重组后的 Git blob 为 `7051679f0e12a8e152e803dc5e8f14b1958d9561`，与 HEAD 原文件完全一致。

### 验证与边界

- 首次定向集合 `50/51`：唯一失败是 Plugins 预期文件数组未保持字典序；调整门禁顺序后，同一集合最终
  `51/51` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 不引用原文件，无需迁移；243 条 evidence 引用、133 个唯一路径缺失保持 0。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- behavior split 不会提升任何 deferred API，也不是 plugin/parser lifetime、ABI/export、TensorRT runtime、Linux、
  package consumer、public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Boundary Controls Owner Partial Split

本阶段将 441 行 `NativeBridgeApi.Trt11BoundaryControls.cs` 按 owner 拆入 Builder、Engine、Execution、Network 与
helper-only Shared partial。EngineInspector 与 Network 的方法在原文件中存在第二段交错，本批按原七段边界机械提取，
没有改动方法签名/方法体、版本分支、entrypoint、异常文案、UTF-8 buffer 或 copied error-recorder snapshot 逻辑。

### 实现与门禁

- `Builder/NativeBridgeApi.BuilderBoundaryControls.cs`：178 行、12 个 builder compatibility/callback/error-recorder/
  network-support 方法。
- `Engine/NativeBridgeApi.EngineBoundaryControls.cs`：114 行、5 个 engine/inspector error-recorder 与 aliased-input 方法。
- `Execution/NativeBridgeApi.ExecutionContextBoundaryControls.cs`：74 行、3 个 execution-context error-recorder 方法。
- `Network/NativeBridgeApi.NetworkBoundaryControls.cs`：88 行、5 个 network error-recorder、remove-tensor 与 TopK V2 方法。
- `NativeBridgeApi.OwnerErrorRecorderSnapshotShared.cs`：31 行、0 个公开方法，只保存 1 个跨 owner copied snapshot mapper。
- 布局门禁固定 12/5/3/5/0 公开方法数量、`IsNetworkSupported`/`AddTopKV2Layer` 的历史命名例外、Shared helper-only
  约束和旧根文件禁止回流；8 处测试源码合同已按断言职责读取 owner 文件或显式聚合五份文件。
- 五文件按原七段顺序重组后的 Git blob 为 `7df064d599c044c32479aeecfe29e75957388374`，与 HEAD 原文件完全一致。

### 验证与边界

- 首次定向集合 `70/71`：唯一失败是 Network 命名门禁未允许历史 `AddTopKV2Layer`；将其固定为唯一显式例外后，
  同一集合最终 `71/71` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的复合 boundary path 已在本机展开为五条 owner/Shared 路径；evidence 现为
  247 条引用、137 个唯一路径、0 缺失，且该文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 owner/lifetime、ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Fourteenth Batch Owner And Feature Partial Split

本阶段将 341 行 `NativeBridgeApi.Trt11FourteenthBatch.cs` 按 owner/feature 拆入两个 Builder 文件以及 Serialization、
Profiles、Execution。原文件的 Builder build、host-memory metadata、optimization-profile shape values、BuilderConfig
plugin serialization 与 ExecutionContext address/aux-stream 方法连续交错；本批只移动完整区段，没有改动签名/方法体、
版本守卫、entrypoint、UTF-8/GCHandle 生命周期、SafeHandle 返回值或异常文案。

### 实现与门禁

- `Builder/NativeBridgeApi.BuilderBuildOutputs.cs`：61 行、2 个 build 方法；顶层
  `NativeTensorRtSerializedNetworkWithKernelText` struct 随唯一使用它的 Builder build-output 方法移动。
- `Builder/NativeBridgeApi.BuilderConfigPluginSerialization.cs`：85 行、2 个 flag/plugin serialization 方法。
- `Serialization/NativeBridgeApi.HostMemoryMetadata.cs`：26 行、1 个 host-memory data-type 方法。
- `Profiles/NativeBridgeApi.OptimizationProfileShapeValues.cs`：112 行、3 个 shape-value V2 set/count/copy 方法。
- `Execution/NativeBridgeApi.ExecutionContextAddressAndAuxStreams.cs`：105 行、6 个 address/device-memory/event/aux-stream
  方法与 1 个 Execution 专属私有 helper。
- 布局门禁固定 2/2/1/3/6 方法集合、struct 归属和旧根文件禁止回流；3 个硬编码源码合同已切到实际 owner 文件。
- 五文件按原区段顺序重组后的 Git blob 为 `32ad72873f1c5051fd7b31132acec548c08737a2`，与 HEAD 原文件完全一致。

### 验证与边界

- layout、execution aux-stream、plugin serialization 与 host-memory stream 定向集合：`56/56` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 不引用原文件，无需迁移；evidence 维持 247 条引用、137 个唯一路径、0 缺失。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality。
- partial 拆分不是 owner/lifetime、ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层当日本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Fifteenth Batch Engine And Execution Metadata Split

本阶段将 161 行 `NativeBridgeApi.Trt11FifteenthBatch.cs` 按 owner/feature 拆入两个 Engine 文件与一个 Execution
文件。原文件包含 engine profile tensor values、engine-inspector error-recorder control 与 execution-context engine
metadata 三段连续职责；本批只移动完整方法块及其专属 helper，没有改动签名/方法体、版本守卫、entrypoint、
UTF-8/GCHandle 生命周期、SafeHandle 参数、返回值或异常文案。

### 实现与门禁

- `Engine/NativeBridgeApi.EngineProfileTensorValues.cs`：102 行、2 个 profile tensor value 方法与 2 个专属
  validation/version helper。
- `Engine/NativeBridgeApi.EngineInspectorErrorRecorder.cs`：22 行、1 个 inspector error-recorder clear 方法。
- `Execution/NativeBridgeApi.ExecutionContextEngineMetadata.cs`：57 行、6 个 context event/runtime-config/engine metadata
  方法。
- 布局门禁固定 2/1/6 方法集合、2 个 helper 归属和旧根文件禁止回流；两个 readonly diagnostics 源码合同已切到
  实际 Engine 文件。
- 三文件按原片段顺序重组后的 Git blob 为 `cc9f64caaf277a1b922d6ad0868f532f0508ad20`，与 HEAD 原文件完全一致。

### 验证与边界

- layout、engine/RNN readonly、engine-inspector 与 readonly candidate evidence 定向集合：`54/54` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 已在本机将 profile tensor evidence 迁移到实际 Engine 文件；evidence 为 247 条引用、
  137 个唯一路径、0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 owner/lifetime、ABI/export、TensorRT runtime、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Timing Cache Lifecycle Partial Split

本阶段开始对 4,882 行根 `NativeBridgeApi.cs` 做小批、可重组的 owner/feature 瘦身，先将其中连续 79 行的跨版本
timing-cache lifecycle 片段移入 Builder。该片段只有 create/set/serialize 三个公开方法，没有专属跨段 helper；本批
没有改动签名/方法体、TRT8/10/11 分支、entrypoint、GCHandle finally 释放、返回值或异常文案。

### 实现与门禁

- `Builder/NativeBridgeApi.TimingCacheLifecycle.cs`：89 行，其中 79 行为原始方法片段，包含 `CreateTimingCache`、
  `SetTimingCache`、`SerializeTimingCache`。
- 根 `NativeBridgeApi.cs` 从 4,882 行降至 4,803 行；布局门禁固定三个方法只归属 Builder lifecycle partial，并禁止
  方法体回流根文件。
- 将 79 行片段插回根文件原第 3546 行位置后，重组 Git blob 为
  `ffd4c81df9d553a1a24575539130576566662206`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、BuilderConfig scalar 与 runtime serialization stream 定向集合：`49/49` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 timing-cache runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root ONNX Parser Core Partial Split

本阶段继续对根 `NativeBridgeApi.cs` 做可重组的 owner/feature 瘦身，将顶部 1 行 parser string delegate 与原根文件
中连续 409 行 ONNX parser core 拆入四个 Parsing partial。parser core 的生命周期/输入、错误诊断、flags/operator
support 本身边界清晰；通用复制字符串 delegate/helper 已被 parser、parser-refitter 与 parser support 共同使用，因此
显式归入 helper-only Shared partial，没有复制实现或改变 allocator 生命周期。

### 实现与门禁

- `Parsing/NativeBridgeApi.OnnxParserLifecycleAndInput.cs`：105 行、3 个 create/file/memory parse 方法，保留 byte[]
  pinning 与 `finally` 释放。
- `Parsing/NativeBridgeApi.OnnxParserDiagnostics.cs`：205 行、5 个 error/diagnostic/local-stack/clear 方法与 1 个
  parser 专属 variable-string selector helper。
- `Parsing/NativeBridgeApi.OnnxParserFlagsAndOperatorSupport.cs`：99 行、6 个 flags 与 operator-support 方法。
- `Parsing/NativeBridgeApi.ParserStringReadShared.cs`：40 行、0 个公开方法，只保存 1 个 delegate 与 1 个跨 Parsing
  copied-string allocator/helper。
- 根 `NativeBridgeApi.cs` 从 4,803 行降至 4,393 行；布局门禁固定 3/5/6/0 方法分布、helper-only Shared 约束和
  delegate/helper 禁止回流根文件；ONNX input 与 B-tier diagnostics 源码合同改读实际 feature 文件。
- 按 `root header + Shared delegate + root prefix + lifecycle + diagnostics part 1 + flags + diagnostics helper + Shared helper +
  root suffix` 原顺序重组后的 Git blob 为 `78f8f0e43aec19b6c6c28c92487d00e1f1570f1e`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、ONNX managed input、B-tier 41-45、parser-refitter、ONNX support 与 engine-inspector 定向集合：`61/61` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 parser lifetime/runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Tail Engine Execution Owner Partial Split

本阶段将根 `NativeBridgeApi.cs` 尾部连续的 EngineInspector、ExecutionContext、HostMemory 与 Engine metadata 方法按
真实 owner 拆入 Engine、Execution、Serialization。后置 helper 审计确认 engine-information 与 engine IO-tensor-name
getter 各只服务一个新 owner；BuilderConfig bit-flag 以及 Network/Tensor/Layer name helper 仍服务根文件方法，因此保留
原位，没有制造模糊 Shared 文件或复制实现。

### 实现与门禁

- `Engine/NativeBridgeApi.EngineInspectorCore.cs`：92 行、3 个 inspector create/context/information 方法与 1 个专属
  copied-string getter helper。
- `Execution/NativeBridgeApi.ExecutionContextBindingsAndEnqueue.cs`：122 行、6 个 shape/binding/address/enqueue 方法。
- `Serialization/NativeBridgeApi.HostMemoryBuffer.cs`：59 行、2 个 size/copy 方法，保留 byte-count 完整性校验。
- `Engine/NativeBridgeApi.EngineCoreMetadata.cs`：281 行、14 个 IO tensor/device-memory/profile/debug metadata 方法与
  1 个 IO-tensor-name getter helper。
- 根 `NativeBridgeApi.cs` 从 4,393 行降至 3,878 行；布局门禁固定 3/6/2/14 方法分布、两个 owner helper 归属，并确认
  尚未迁移的 bit-flag 与 Network/Tensor/Layer name helper 继续存在根文件。Inspector 与 HostMemory 源码合同已切到实际文件。
- 按 `root prefix + Inspector + Execution + HostMemory + Engine + Inspector helper + root bit helper + Engine helper + root suffix`
  原顺序重组后的 Git blob 为 `3a5a8c67e9debdc35dc4164112cfd500cfdd3c9d`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、engine-inspector、serialization stream、engine/RNN readonly、safe lifecycle 与 execution aux-stream 定向集合：
  `68/68` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 engine/context/host-memory lifetime 或 runtime correctness，也不是 ABI/export、Linux、package
  consumer、public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Cross Version Bindings And Build Chain Probe Split

本阶段将根 `NativeBridgeApi.cs` 中的跨版本 line bindings 与 environment build-chain probe 按“共享路由”与“诊断行为”
拆开。`TensorRtLineBindings` 同时被根 core 方法、生成 bindings/helper 和 TRT11 build probe 使用，因此归入 Runtime
helper-only partial；六个 `Try*` probe 与两个执行 helper 只服务 environment diagnostics，因此归入 Diagnostics。
没有修改生成文件、路由条件、SafeHandle using 链、失败 fallback 文案或 build/deserialization/context 创建顺序。

### 实现与门禁

- `Runtime/NativeBridgeApi.CrossVersionLineBindings.cs`：81 行、0 个公开方法，保存 9 个 delegate、私有
  `TensorRtLineBindings` class、`GetBindings` 与 TRT11 bridge-build 判定。
- `Diagnostics/NativeBridgeApi.BuildChainProbe.cs`：167 行、6 个公开 runtime/builder/TRT8/10 build-chain probes 与
  2 个私有 minimal-build/serialized-build helper；既有 TRT11 probe 继续调用同一 helper。
- 根 `NativeBridgeApi.cs` 从 3,878 行降至 3,649 行；布局门禁固定 6/0 方法分布、9 delegate、helper-only Runtime
  约束，并禁止 bindings class/helper 与 build-chain helper 定义回流根文件。
- 按 `root header + bindings first delegates + root prefix + probe public + bindings body + probe helpers + root suffix`
  原顺序重组后的 Git blob 为 `bbd2f2227ad3b3eeca69bf0099d023a0ff8b7b5a`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、TRT11 runtime-create diagnostic 与 runtime deserialization precheck 定向集合：`51/51` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 build/runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Network Definition Core Partial Split

本阶段从根 `NativeBridgeApi.cs` 提取基础 Network definition core，包括 input/output ownership、layer lookup、network
name、creation flags 与 implicit-batch metadata。`GetNetworkNameNative` 只由 core name getter 使用，因此随 Network
移动；`AddIdentityLayer` 起始的 layer feature、Tensor/Layer name helper 与 optional-weight helper 保留待后续小批拆分。
本批没有改动 SafeHandle 返回、Dims 转换、UTF-8 allocator、版本分支、entrypoint 或异常文案。

### 实现与门禁

- `Network/NativeBridgeApi.NetworkCore.cs`：263 行、14 个 input/output/layer/name/flags 方法与 1 个专属 UTF-8 name
  getter helper。
- 根 `NativeBridgeApi.cs` 从 3,649 行降至 3,396 行；布局门禁固定 14 方法和 helper 归属，禁止回流，并明确
  `AddIdentityLayer`、Tensor/Layer name helper 与 optional-weight helper 仍留根文件。
- 按 `root prefix + Network core + root middle + Network name helper + root suffix` 原顺序重组后的 Git blob 为
  `c9a2593a94949f9321c1783952011ca54dbe593a`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- 首次 layout、BuilderConfig scalar、safe lifecycle 与 readonly candidate 定向集合 `57/58`；唯一失败是上一批尾部
  门禁仍要求 `GetNetworkNameNative` 留在根文件。删除该过期正向断言、由新 Network gate 接管后，同一集合最终
  `58/58` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 network/runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Weighted Layer Creation Feature Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Identity/Constant、Convolution、Deconvolution、Scale layer creation 按 feature
拆入 Layers。helper 调用关系显示 pin 与 data-type validation 被 convolution/deconvolution/scale 共用，进入 helper-only
Shared；data-type selector 只被 scale 使用，随 Scale 文件移动。所有方法体、TRT8/10/11 分支、entrypoint、weights
pinning、`finally` 释放顺序、SafeHandle 返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.IdentityAndConstant.cs`：51 行、2 个 identity/constant 方法。
- `Layers/NativeBridgeApi.Convolution.cs` 与 `Deconvolution.cs`：各 62 行、各 1 个 weighted creation 方法。
- `Layers/NativeBridgeApi.Scale.cs`：69 行、1 个 scale creation 方法与 1 个专属 data-type selector。
- `Layers/NativeBridgeApi.OptionalWeightsShared.cs`：29 行、0 个公开方法，只保存共用 pin 与 data-type validation helper。
- 根 `NativeBridgeApi.cs` 从 3,396 行降至 3,166 行；布局门禁固定 2/1/1/1/0 方法分布、helper-only Shared、kernel/bias
  pin/dispose 与 Scale power→scale→shift 逆序释放，并将下一根边界推进到 `AddPaddingLayer`。
- 按 `root prefix + Identity + Convolution + Deconvolution + Scale body + root middle + Shared pin + Scale selector +
  Shared validation + root close` 原顺序重组后的 Git blob 为 `b20082518f87fd16449b41149c8dc35591f1ddd0`，与拆分前
  HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`57/57` 一次通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer/weights runtime correctness、weights lifetime、ABI/export、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Basic Layer Operations Partial Split

本阶段继续按 feature 缩减根 `NativeBridgeApi.cs`，将连续的 Padding、ElementWise 与 MatrixMultiply creation/attributes
区段分别移入 Layers。三个区段没有私有 helper；所有方法体、TRT8/10/11 分支、entrypoint、SafeHandle 返回、枚举转换
与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.Padding.cs`：36 行、1 个 `AddPaddingLayer` 方法。
- `Layers/NativeBridgeApi.ElementWise.cs`：29 行、1 个 `AddElementWiseLayer` 方法。
- `Layers/NativeBridgeApi.MatrixMultiply.cs`：58 行、3 个 MatrixMultiply creation/set/get 方法。
- 根 `NativeBridgeApi.cs` 从 3,166 行降至 3,070 行；布局门禁固定 1/1/3 方法分布、禁止方法定义回流，并将下一根
  feature 边界推进到 `AddShuffleLayer`。
- 按 `root prefix + Padding + ElementWise + MatrixMultiply + root suffix` 原顺序重组后的 Git blob 为
  `1d53faa5f698f82d89b58390172ceb864d6cc618`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`58/58` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Shuffle Feature Partial Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Shuffle creation、reshape/transpose attributes 与 zero-placeholder controls
移入单独 Layers feature partial。该区段没有私有 helper；所有方法体、TRT8/10/11 分支、entrypoint、Dims 转换、
bool/int 转换、SafeHandle 返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.Shuffle.cs`：154 行、9 个 Shuffle creation/set/get 方法。
- 根 `NativeBridgeApi.cs` 从 3,070 行降至 2,925 行；布局门禁固定 9 个方法的精确集合、禁止定义回流，并将下一根
  feature 边界推进到 `AddReduceLayer`。
- 按 `root prefix + Shuffle + root suffix` 原顺序重组后的 Git blob 为
  `b74da23472cb112ba4b90f1c3c040b8103e05340`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`59/59` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Reduce Feature Partial Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Reduce creation 与 readonly operation/axes/keep-dimensions attributes 移入
单独 Layers feature partial。该区段没有私有 helper；所有方法体、TRT8/10/11 分支、entrypoint、枚举/bool 转换、
SafeHandle 返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.Reduce.cs`：75 行、4 个 Reduce creation/get 方法。
- 根 `NativeBridgeApi.cs` 从 2,925 行降至 2,859 行；布局门禁固定 4 个方法的精确集合、禁止定义回流，并将下一根
  feature 边界推进到 `AddSoftMaxLayer`。
- 按 `root prefix + Reduce + root suffix` 原顺序重组后的 Git blob 为
  `23be07836c33128bfb8415e94f8498bf3a70c4a0`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`60/60` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root SoftMax Unary TopK Gather Feature Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 SoftMax、Unary、TopK、Gather creation/attributes 按 feature 移入四份
Layers partial。四个区段均无私有 helper；所有方法体、TRT8/10/11 分支、entrypoint、枚举/axes 转换、SafeHandle
返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.SoftMax.cs`：52 行、3 个 creation/axes 方法。
- `Layers/NativeBridgeApi.Unary.cs`：39 行、2 个 creation/operation 方法。
- `Layers/NativeBridgeApi.TopK.cs`：75 行、4 个 creation/operation/k/axes 方法。
- `Layers/NativeBridgeApi.Gather.cs`：44 行、2 个 creation/axis 方法。
- 根 `NativeBridgeApi.cs` 从 2,859 行降至 2,685 行；布局门禁固定 3/2/4/2 方法分布、禁止定义回流，并将下一根
  feature 边界推进到 `AddActivationLayer`。
- 按 `root prefix + SoftMax + Unary + TopK + Gather + root suffix` 原顺序重组后的 Git blob 为
  `e36add4f9fda15648c60d10b0215f41bbeb4c154`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`61/61` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Activation Pooling LRN Feature Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Activation、Pooling、LRN creation/attributes 按 feature 移入三份 Layers
partial。三个区段均无私有 helper；所有方法体、TRT8/10/11 分支、entrypoint、Dims/枚举/标量转换、SafeHandle
返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.Activation.cs`：39 行、2 个 creation/type 方法。
- `Layers/NativeBridgeApi.Pooling.cs`：147 行、8 个 creation/type/window/stride/padding 方法。
- `Layers/NativeBridgeApi.Lrn.cs`：136 行、9 个 creation/window/alpha/beta/k 方法。
- 根 `NativeBridgeApi.cs` 从 2,685 行降至 2,390 行；布局门禁固定 2/8/9 方法分布、禁止定义回流，并将下一根
  feature 边界推进到 `AddResizeLayer`。
- 按 `root prefix + Activation + Pooling + LRN + root suffix` 原顺序重组后的 Git blob 为
  `940ec83e927ddf79d60c116735b3303b4a625fca`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`62/62` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Resize Concatenation Slice Feature Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Resize、Concatenation、Slice creation/attributes 按 feature 移入三份
Layers partial。三个区段均无私有 helper；Resize 的 scales 数组 pin/finally、Concatenation handle 数组校验与 Slice
Dims 转换均随完整方法块移动，TRT8/10/11 分支、entrypoint、SafeHandle 返回与异常文案保持不变。

### 实现与门禁

- `Layers/NativeBridgeApi.Resize.cs`：149 行、7 个 creation/dimensions/mode/scales 方法。
- `Layers/NativeBridgeApi.Concatenation.cs`：73 行、3 个 creation/axis 方法。
- `Layers/NativeBridgeApi.Slice.cs`：178 行、9 个 creation/start/size/stride/mode 方法。
- 根 `NativeBridgeApi.cs` 从 2,390 行降至 2,018 行；布局门禁固定 7/3/9 方法分布、禁止定义回流，并将下一根
  feature 边界推进到 `AddShapeLayer`。
- 按 `root prefix + Resize + Concatenation + Slice + root suffix` 原顺序重组后的 Git blob 为
  `3008fc07e6939a044a6e63e29771dab1603790aa`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`63/63` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Shape Select Fill Feature Split

本阶段将根 `NativeBridgeApi.cs` 中连续的 Shape、Select、Fill creation/attributes 按 feature 移入三份 Layers partial。
三个区段均无私有 helper；Fill 的 Dims/operation/alpha/beta 转换、TRT8/10/11 分支、entrypoint、SafeHandle 返回与
异常文案均保持不变，并在 `GetLayerOutput` 前停止，不带走通用 Layer metadata。

### 实现与门禁

- `Layers/NativeBridgeApi.Shape.cs`：24 行、1 个 creation 方法。
- `Layers/NativeBridgeApi.Select.cs`：29 行、1 个 creation 方法。
- `Layers/NativeBridgeApi.Fill.cs`：148 行、9 个 creation/dimensions/operation/alpha/beta 方法。
- 根 `NativeBridgeApi.cs` 从 2,018 行降至 1,844 行；布局门禁固定 1/1/9 方法分布、禁止定义回流，并将下一根边界
  推进到 `GetLayerOutput` 通用 Layer metadata。
- 按 `root prefix + Shape + Select + Fill + root suffix` 原顺序重组后的 Git blob 为
  `8527b3a51f62533af746f11f38a1a7b3264632ef`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`64/64` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 仍引用存在的根文件，无需迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 拆分不是 layer runtime correctness、owner/lifetime、ABI/export、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Layer And Tensor Metadata Owner Split

本阶段将根 `NativeBridgeApi.cs` 中通用 Layer 与 Tensor metadata 按 owner 拆入 Layers 与 Network。初始提示词将 Tensor
段记为 367 行；消费审计确认后 16 行实际为 BuilderConfig 使用的 `GetSingleBitFlagIndex`，因此真实 Tensor 主体为
351 行，该 helper 继续留根。Layer/Tensor name helper 位于根尾，分别随消费 owner 非连续迁移。

### 实现与门禁

- `Layers/NativeBridgeApi.LayerCoreMetadata.cs`：332 行、15 个公开方法、`MapLayerType` 与专属 name getter。
- `Network/NativeBridgeApi.TensorCoreMetadata.cs`：377 行、22 个公开方法与专属 name getter。
- 根 `NativeBridgeApi.cs` 从 1,844 行降至 1,155 行；37 个 owner 方法与 3 个专属 helper 禁止回流，BuilderConfig
  bit-flag helper 固定留根，历史下一边界门禁回到仍存在的 `GetTacticSources`。
- 按 `root prefix + Layer body + Tensor body + bit-flag helper + Tensor name helper + Layer name helper + root close`
  重组后的 Git blob 为 `efaa13a8f6352354866248651ae24c56c523d6f2`，与拆分前 HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle 与 BuilderConfig scalar 定向集合：`65/65` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的根路径仍由 BuilderConfig scalar controls 真实消费，无需迁移；evidence 保持
  247 条引用、137 个唯一路径、0 缺失，该 ignored 文件未强制提交。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- owner 拆分不是 Layer/Tensor runtime correctness、ABI/export、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Root Remaining Owner Final Split

本阶段将根 `NativeBridgeApi.cs` 剩余 1,142 行实现按 adapter、callback、runtime、builder、network、serialization、
execution、profile 与 BuilderConfig owner/feature 全部迁出。根文件保留 13 行 partial 声明 shell，不含 public/private
static 实现；所有方法体、数组 pin/finally、版本路由、entrypoint、SafeHandle 返回与异常文案保持不变。

### 实现与门禁

- Runtime：`AdapterInfo.cs` 14 行/1 方法，`RuntimeCreation.cs` 51 行/2 方法。
- Callbacks：`ManagedDiagnostics.cs` 127 行/7 方法。
- Builder：`BuilderCore.cs` 85 行/6 方法，`SerializedNetworkBuild.cs` 21 行/1 方法，
  `BuilderConfigCore.cs` 567 行/38 方法并接管 bit-flag helper。
- Network：`NetworkCreation.cs` 17 行/1 方法；Serialization：`EngineDeserialization.cs` 65 行/2 方法。
- Execution：`ExecutionContextCore.cs` 32 行/2 方法；Profiles：`OptimizationProfileCore.cs` 258 行/9 方法。
- 布局门禁固定 `1/7/2/6/1/1/2/2/9/38` 方法分布，并要求根 shell 无任何 static 实现。
- 十个主体按原顺序插回 13 行 shell 后的 Git blob 为 `c95369e9723bf92c2f0100133c4756710598c3b3`，与拆分前
  HEAD 根文件完全一致。

### 验证与边界

- layout、safe lifecycle、BuilderConfig、runtime deserialization/stream IO 与 TRT11 runtime diagnostic 定向集合：
  `76/76` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 的 root evidence 已本机迁移到 `BuilderConfigCore.cs`；evidence 仍为 247 条引用、
  137 个唯一路径、0 缺失，该 ignored 文件未强制提交。
- 非 publishing 技术文档与 4 份硬编码测试合同已改读真实 owner 文件；8 份 publishing 用户变更未触碰。
- `git diff --check` 通过；Generated/native/manifest/ABI 未修改，未运行完整 ProjectQuality，也未运行依赖本机缺失
  `pwsh` 的 B-tier 聚合测试。
- partial 收口不是 runtime correctness、ABI/export、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- C 盘 Downloads 顶层本批相关文件、用户 Temp 顶层本批关键词与项目相关 build/test 进程残留均为 0。
- 未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages 发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Managed Wrapper Feature Partial Split

本阶段将 `TensorRtLayer.cs` 与 `TensorRtNetworkDefinition.cs` 的连续高层 feature API 拆入职责明确的 partial，
不修改 public 签名、参数校验、TensorRT API line、SafeHandle/owner lease、native entrypoint 或异常文案。

### 实现与门禁

- `TensorRtLayer.cs` 从 1,371 行降至 173 行，只保留构造、通用 metadata、owner lease、output index 校验、
  共享 Dims 校验与释放；118 个方法进入 Shuffle 到 Fill 的 17 份 feature partial。
- `TensorRtNetworkDefinition.cs` 从 755 行降至 211 行，只保留 input/output/layer ownership、mark/unmark output、
  共享 tensor 校验与释放；21 个 `Add*` 方法进入 Identity 到 Fill 的 21 份 feature partial。
- `ManagedWrapperFeatureLayoutTests` 固定 38 份新 partial 的精确方法集合、两个 core 的 helper/lifetime 归属，
  并按原顺序重组规范化源码。
- 拆分前 Layer/Network Git blob 分别为 `faa5fa87fb24247a86f9d166073cf3858bad9ac2` 与
  `933e7b253ecfefa00034da2544912b6e20c353ae`；重组后的 normalized SHA-256 分别保持
  `a87256e88735c3896b4c75227e108191b5ae85a20a035af08bc3fa55442c1f4f` 与
  `ac973a03e4580ddaed35c5d83e5d23a84dd981b46b475b323fe9c30a87e5accc`。
- B-tier focused managed evidence 改为读取 `TensorRtLayer*.cs`；RNNv2 owner-lifetime 测试继续读取 core，
  因其消费的 owner lease/helper 仍由 core 真实持有。

### 验证与边界

- wrapper layout/recomposition、managed layout、safe lifecycle、BuilderConfig、runtime deserialization/stream IO、
  TRT11 runtime diagnostic 与 RNNv2 owner lease 定向集合：`124/124` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate artifact 无旧大文件 feature 路径需要迁移；evidence 保持 247 条引用、137 个唯一路径、
  0 缺失，该 ignored 文件未强制提交。
- Generated/native/manifest/ABI 未修改；未运行依赖本机缺失 `pwsh` 的 B-tier 聚合测试，也未把 source split
  表述为 runtime correctness、package consumer、public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT Engine And Parser Managed Wrapper Split

本阶段继续收口高层 managed wrapper，将通用 Engine 与 ONNX Parser 按实际 feature 拆分，同时把 native handle、
logger/config/initializer lifetime、Dispose 与跨 feature 共享校验固定在 core。

### 实现与门禁

- `TensorRtEngine.cs` 从 659 行降至 124 行；36 个 tensor/profile metadata、binding report、
  execution-context creation、refit 与 inspection 方法进入 5 份 feature partial。
- `TensorRtEngine.BindingReports.cs` 同时接管 `CreateBindingReport` 与 `TryGetProfileShape` 两个私有 helper；
  Engine core 仅保留 handle、通用标量属性与 Dispose。
- `TensorRtOnnxParser.cs` 从 685 行降至 198 行；32 个 model parsing/loading、TryParse、diagnostics、
  operator-support 与 flags 方法进入 6 份 feature partial。
- Parser core 继续持有 logger/config/initializer ownership、Dispose、flags validation 与 model segment/stream copy
  helper；这些 helper 均存在跨 partial 消费，未错误归入单一 feature。
- `ManagedEngineParserFeatureLayoutTests` 固定 11 份 partial 的精确方法/重载集合、helper/core 归属与非连续片段
  重组。拆分前 Git blob 为 `fd6907a9e03eab3b6f9a1b5820eea9e6e1e82ea9`、
  `d8e3f135b71f9b2fd893146776da7d538db5d020`；normalized SHA-256 分别保持
  `2f96a53b0b3c0f5032b0108d684c8a6377e02d6d59923a24ad74ec953334dab3` 与
  `3bce180397b2eb158c2fe1a6c79d16807d77e9eb3ba0ef8d169247b9a0f7897e`。

### 验证与边界

- managed wrapper/layout/recomposition、safe lifecycle、BuilderConfig、runtime IO、Engine/RNN readonly、
  parser input/model-buffer/diagnostics/flags 与 logger borrower 定向集合：`174/174` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- 10 份直接读取旧 wrapper 的质量测试已改读真实 core/feature owner；B-tier dashboard 改为模块级 owner 路径，
  tracked evidence map 与 ignored candidate artifact 分别改指 `Inspection` / `BindingReports`。
- ignored evidence 保持 247 条引用，唯一路径因两个重复 core 路径被拆成两个实际 owner 而从 137 增至 138，
  缺失保持 0；ignored 文件未强制提交。
- Generated/native/manifest/ABI 未修改；未运行依赖本机缺失 `pwsh` 的 B-tier 聚合测试，也未把 source split
  表述为 runtime correctness、package consumer、public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT BuilderConfig And ExecutionContext Managed Wrapper Split

本阶段将 BuilderConfig 与 ExecutionContext 通用 wrapper 按功能拆分，同时保留被既有 TRT11 partial 共用的
progress-monitor、profiler、aux-stream 与 layer-validation 生命周期 helper。

### 实现与门禁

- `TensorRtBuilderConfig.cs` 从 583 行降至 102 行；34 个 profile/flag/compatibility/layer-device/memory-pool/
  scalar/tactic/timing-cache 方法与 7 个 feature 属性进入 8 份 partial。
- `ValidateLayer` 仍被 `Trt11Diagnostics` 消费，disposed-state 与 progress-monitor helper 也跨 partial 使用，
  因此与 handle、Dispose 一起留在 BuilderConfig core。
- `TensorRtExecutionContext.cs` 从 410 行降至 176 行；20 个 shape/address/device-memory/event/enqueue 方法进入
  5 份 partial。Profiler 与 auxiliary-stream cleanup helper 继续留 core。
- `ManagedBuilderExecutionFeatureLayoutTests` 固定 13 份 partial 的方法/属性归属、两个 core 的共享 helper，
  并重组非连续 Shapes/Addresses 片段。拆分前 Git blob 为
  `592ce09c4da5fb4f7a376800cd5a6b309a22481b`、`1614e46a4b0175ff9889302f977a0520be404b29`；
  normalized SHA-256 保持 `180b5e504f28a7203115392d21386f44bfe6951e49cc88bc3be883db0f080d52` 与
  `0f6e52115efb010129148b9a731d30d8f6aa65ea263bb82922b8c62f6381f856`。
- Builder scalar audit、TensorRtExec gap list 与 7 份直接消费旧 core 的测试已改读实际 feature；B-tier dashboard
  改用 Builder/Execution 模块级 owner 路径。

### 验证与边界

- 三组 managed wrapper layout/recomposition、managed module layout、safe lifecycle、BuilderConfig、runtime IO、
  Engine/Parser、callback lifetime 与 TensorRtExec gap 定向集合：`227/227` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored builder readback 的重复 core 证据已展开为 `MemoryPools`、`ScalarControls` 与
  `CompatibilityPresence`；evidence 现为 248 条引用、140 个唯一路径、0 缺失，ignored 文件未强制提交。
- Generated/native/manifest/ABI 未修改；未运行依赖本机缺失 `pwsh` 的 B-tier 聚合测试，也未把 source split
  表述为 runtime correctness、package consumer、public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT ParserRefitter And InferenceBindings Managed Wrapper Split

本阶段继续按调用阶段收口高层 managed wrapper，将 ParserRefitter 的 model/initializer/diagnostic 路径与
InferenceBindings 的 geometry/buffer/address/execution 路径分离，同时保持 owner、buffer 与 Dispose 边界不变。

### 实现与门禁

- `TensorRtOnnxParserRefitter.cs` 从 460 行降至 114 行；21 个 refit/model-proto/initializer/diagnostic 方法进入
  Refitting、ModelLoading、Initializers、Diagnostics 四份 partial。
- ParserRefitter core 继续持有 native SafeHandle、refitter/logger borrower、initializer pin lifetime、Dispose 与被
  多个 input feature 共用的 model segment/stream copy helper；diagnostic summary helper 随 Diagnostics 移动。
- `TensorRtInferenceBindings.cs` 从 588 行降至 124 行；14 个 tensor geometry、buffer ownership、host transfer、
  address binding、execution 与 diagnostic 方法进入 6 份 partial。
- InferenceBindings 的 `GetTensor` 与 `RefreshReport` 被多个 feature 共用，因此与 engine/context/buffer owner、Dispose
  和 disposed-state 留在 core；size estimation、buffer replacement 与 readiness helper 分别随 feature owner 移动。
- `ManagedParserRefitterInferenceFeatureLayoutTests` 固定 10 份 partial 的精确方法/重载集合、两个 core 的 owner/helper
  归属，并处理 diagnostics/model-loading 与多个 helper 的非连续原片段重组。
- 拆分前 Git blob 为 `07975ca6274fb64fcce06ca6e967515f07aa734c`、
  `6257b4c8ad3c0f8c4ec349208e8583b670359d0a`；normalized SHA-256 保持
  `f4b7d422744849ead2bebc899000f495a2c27d96c120922228ff81451c4b88b6` 与
  `021eb1ba7d4f1bd632a9f142adb99806ceb96c99e7895df0b1ced3125590a753`。
- 8 份直接读取旧 core 的质量测试改读真实 feature 或完整 partial 集合；两份 technical-article exporter anchor
  与两篇 InferenceBindings 教程同步到新 owner 文件。

### 验证与边界

- 新增 layout/recomposition 门禁：`14/14` 通过；包含前三批 wrapper layout、managed module layout、parser-refitter、
  inference execution、model buffer 与文章 source marker 的可归因扩展集合：`160/160` 通过。
- 更宽的探索性集合实际为 `163 passed / 4 failed / 167 total`：其中 3 项因本机缺失 `pwsh` 无法启动 exporter，
  另 1 项为既有文章基线缺少 `4001` marker；未将该集合宣称为通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 248 条引用、140 个唯一路径、0 缺失；ignored 文件未强制提交。
- 本批无自有 JSON 改动；Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过。
- build 结束后复核并清理本批遗留编译进程，dotnet/MSBuild/VBCSCompiler/testhost 残留为 0。
- source split 不构成 runtime correctness、real model、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 TensorRT EnvironmentProbe And Plugin Inventory Source Split

本阶段继续整理高层 C# 诊断与复制型 plugin metadata，将超大的静态环境探测类按能力域拆分，并把一个文件中的
多个独立 public copied model 类型分离到各自源码文件，不改变 namespace、public 签名或 proof 分类。

### 实现与门禁

- `TensorRtEnvironmentProbe.cs` 从 1,497 行降至 61 行，只保留跨 feature 的 probe exception 分类、诊断格式化与
  generic stage helper；43 个公开静态入口进入 PluginInitialization、RuntimeMetadata、GlobalPluginRegistry、
  BuilderPluginRegistry、DependencyProbes、ObjectCreation、BuildChains 七份 partial。
- `TensorRtPluginRegistryInventory.cs` 从 887 行降至 294 行，仅保留主 inventory 聚合类；两个 enum 进入
  `TensorRtPluginRegistryTypes.cs`，field/creator info、creator/field summary 与 inventory diagnostics 进入另外五份文件。
- `ManagedEnvironmentPluginInventoryLayoutTests` 固定 7 份 EnvironmentProbe partial 的精确方法/重载集合、共享 helper
  core、7 个 plugin 类型文件的精确 top-level type 集合，并规范化重组两份原源码。
- 拆分前 Git blob 为 `bcd1301777f81d9de32625b4c7be952de3126d92`、
  `c39d9ec853987f50f452b0dfebc2658120aa95d2`；normalized SHA-256 保持
  `532720104c57a6316bf47b7a2c1e83eacaa8749fe4515a5ea83a4aeecff210a2` 与
  `2af6a19673875d36b0c0143f9c25343a3a1c989e89e5eb346562f4f766dedd93`。
- 12 份直接读取旧大文件的质量测试改读具体 feature/type owner；两份 exporter 与 runtime/plugin 教程补充真实路径。

### 验证与边界

- 新增 layout/recomposition 门禁：`18/18` 通过；合并前四批 wrapper layout、managed module layout 与本批消费测试的
  扩展定向集合：`214/214` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 校准为 260 条引用、147 个唯一路径、0 缺失；JSON 可解析，ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过；本机仍缺少 `pwsh`，未运行相关 exporter/B-tier 聚合测试。
- source split 与 copied metadata 文件归类不构成 plugin load/register/create/enqueue runtime proof，也不构成 Linux、
  package consumer、public package、post-publish 或 Owner acceptance proof。
- solution build 留下的 14 个 node-reuse 子进程已按父进程与启动时间确认并终止，build/test 进程残留为 0。
- Temp 顶层命中既有 `WrapPlugin.dll.log.logdat`；其创建时间早于本批且来源不明确，因此保留未删除。Downloads 顶层本批关键词命中为 0。
- 最终进程表中的 `codex-powershell-7-tool` 与 MSBuild 命令行指向另一个 `OpenCV-CSharp-API` 工作区，属于并行任务；
  不计入 TensorRT 本批残留，也未再终止或删除其工具目录。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 CUDA Graph And Memory Managed Wrapper Split

本阶段继续整理 `JYPPX.CudaSharp` 高层 owner wrapper，将 graph node/topology/diagnostics 与 device-memory
allocation/transfer/range 操作从两个超大文件拆到可定位的 feature partial，同时保持 public API、SafeHandle、owner
计数、registered-host 扩展与 Dispose 语义不变。

### 实现与门禁

- `CudaGraph.cs` 从 1,630 行降至 206 行，只保留 graph handle、Create、capture/conditional/allocation owner 计数、
  跨 feature validation、Dispose 与 disposed-state；公开能力进入 ConditionalHandles、GraphComposition、NodeCreation、
  TopologyDiagnostics、NodeInspection、NodeMutation、NodeRelations、Instantiation 八份 partial。
- `CudaMemory.cs` 从 951 行降至 114 行，只保留 allocation handle、构造、size/IPC-import metadata、Dispose、range/advice
  validation 与同步 allocation helper；公开能力进入 Ipc、RangeDiagnostics、AsyncAllocation、Fill、PrefetchAdvice、
  HostTransfers、DeviceTransfers、AsyncFree、ArrayConversion 九份 partial，既有 RegisteredHost partial 保持不变。
- `ManagedCudaGraphMemoryFeatureLayoutTests` 固定 17 份 partial 的精确公开方法/重载集合、两个 core 的 owner/helper
  归属，并规范化重组两份拆分前源码。
- 拆分前 Git blob 为 `5b7e06e4e59d6961c9c848f88d1f9ace6a9c0450`、
  `1d2085d8ffc527cfd280c2ba68836d14bb2c6334`；normalized SHA-256 保持
  `be9ccd67654b6797e7aeab63bf8baddf23a6cd8381eb03e4733f139dc76ab377` 与
  `3036e5c57cac5684c816a317b905580b3aa6ed0960c6f7f460144ab882f24760`。
- 19 份直接读取旧 core 的质量测试改读实际 feature 或明确的 core+feature 组合；memory 教程、Graph Event roadmap
  source artifact 与双语 source-organization 已同步到真实 owner。

### 验证与边界

- 新布局/重组门禁：`21/21` 通过；相关 CUDA 能力与文章消费集合：`91/91` 通过；全部 managed 源码布局门禁
  合并集合：`182/182` 通过。
- `JYPPX.CudaSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；JSON 可解析且未强制提交。
- 本机仍缺少仓库认可的 `pwsh`，未运行相关 exporter/roadmap/B-tier 聚合测试；探索性 technical-article 集合中的
  既有 `4001` marker 不一致也未伪装为本批通过。
- Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过；进程审计未清理或终止其他工作区进程。
- source split 不构成 CUDA runtime correctness、real model、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行 NuGet/GitHub Packages
  发布、Release/tag/issue 远程操作。

## 2026-07-29 CUDA Device And Module Enum Source Split

本阶段继续清理 `JYPPX.CudaSharp` 的跨职责大文件，将 device-wide static helper 按调用域拆开，并把原
`Core/CudaFlags.cs` 中混合的 stream/event/memory/device/array/graph enum 移入实际模块目录，不改变 namespace、
public enum 数值、device context/P2P/error 状态语义或 proof 分类。

### 实现与门禁

- `CudaDevice.cs` 从 793 行降至 144 行，仅保留 runtime/driver version、device count/current、Set/Use、基础 info 与
  property snapshot；graph resource、runtime configuration、initialization/selection、peer capability、memory pool、
  cache/RDMA、synchronization/error diagnostics 进入七份 partial。
- `CopyDeviceOrdinals` 随 InitializationSelection 移动，`CopyAtomicOperations` 随 PeerCapabilities 移动；core 不再保留
  仅由单一 feature 消费的 helper。
- 原 912 行 `Core/CudaFlags.cs` 已删除；24 个 public enum 按 Streams、Events、Memory、Devices、Graphs 归入
  `CudaStreamFlags.cs`、`CudaEventFlags.cs`、`CudaHostMemoryFlags.cs`、`CudaMemoryAdvice.cs`、
  `CudaDeviceExecutionEnums.cs`、`CudaPeerAccessEnums.cs`、`CudaArrayFlags.cs`、`CudaGpuDirectRdmaEnums.cs`、
  `CudaGraphNodeType.cs`、`CudaManagedMemoryAttachmentFlags.cs` 十份文件。
- `ManagedCudaDeviceFlagsLayoutTests` 固定 7 份 Device partial 加 core 的精确方法/属性集合、两个 helper owner、
  10 个 enum 文件的精确 top-level type 集合，并按原交错顺序重组两份源码。
- 拆分前 Git blob 为 `df16e51427f82c9b99fa867819015539dd65ac0c`、
  `dca1aa594002506ce47bd247f47141201af6591d`；normalized SHA-256 保持
  `d864d0cf8626bb59b75b7c5a2b30013ab3652afa0f0e2b5d5ee9b57960b7436d` 与
  `ec815281a28b92d8320d644a89ccc54b6e23d901de39bd353cbf22a766fa9d43`。
- 8 份直接读取旧 Device/Flags 文件的质量测试改读真实 feature/type owner；双语 source-organization 同步。

### 验证与边界

- 新布局/重组门禁：`22/22` 通过；相关 Device/enum 消费集合：`52/52` 通过；全部 managed 源码布局门禁合并集合：
  `204/204` 通过。
- `JYPPX.CudaSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- tracked/ignored 搜索未发现旧 `CudaFlags.cs` 或错误 Device core owner 路径；deferred candidate evidence 路径校准
  继续保持 260 条引用、147 个唯一路径、0 缺失。
- Generated/native/manifest/ABI 改动为 0；本机仍无仓库认可的 `pwsh`，未运行 exporter/B-tier 聚合测试。
- type/source relocation 不构成 device initialization、P2P、graph memory、runtime correctness、real model、Linux、
  package consumer、public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-29 CUDA Pitched Memory And Array Owner Split

本阶段把 `CudaPitchedMemory` 与 `CudaArray` 两个大 owner wrapper 按 transfer dimensionality 与 diagnostics 分类，
保持 SafeHandle、descriptor/extent/flags、pitch/byteCount checked arithmetic、pinned host lifetime 与 Dispose 边界不变。

### 实现与门禁

- `CudaPitchedMemory.cs` 从 678 行降至 181 行，仅保留 allocation/metadata、Allocate3D、Dispose 与共享
  pitch/2D/3D extent/pinned-buffer validation；fill、2D transfer、3D transfer、array conversion 进入四份 partial。
- `CudaArray.cs` 从 609 行降至 208 行，仅保留 allocation/metadata、Create3D、Info/ChannelDescriptor、Dispose 与共享
  stream/byteCount/2D/3D/pinned validation；requirements/sparse diagnostics、1D/2D/3D transfer、array conversion
  进入五份 partial。
- `ManagedCudaArrayPitchedLayoutTests` 固定 9 份 feature partial 的精确重载集合、两个 core 的 metadata/helper owner，
  并规范化重组两份原源码。
- 拆分前 Git blob 为 `e30004cce7cc55c7b62e19478de913d68be3c591`、
  `6de4bc82180a8539e3b6d6fa610c485c042d043e`；normalized SHA-256 保持
  `92238c1fb314ca978e09318573ff60c8413f172f7d505d23a6c26e5551147221` 与
  `98814300efb40ed16dc9b95482eb875a3af4f4dfb63085167b0e6b637557e2bf`。
- 现有 tests/docs/eng/artifacts 没有直接读取这两个旧大文件的质量门禁；memory owner 教程补充 Pitched 2D/3D
  partial 的真实路径，publishing 用户产物继续保留未触碰。

### 验证与边界

- 新布局/重组门禁：`13/13` 通过；全部 managed 源码布局门禁合并集合：`217/217` 通过；memory wrapper 文章
  authoritative marker：`1/1` 通过。
- `JYPPX.CudaSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- Generated/native/manifest/ABI 改动为 0；本机仍无仓库认可的 `pwsh`，未运行 exporter/B-tier 聚合测试。
- owner source split 不构成 2D/3D CUDA runtime、pinned async completion、real model、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-29 CUDA Stream And Memory Pool Owner Split

本阶段继续整理 `JYPPX.CudaSharp` 高层 stream 与 memory-pool owner，将 diagnostics、capture、同步、分配、访问和
属性操作移入可直接定位的 feature 文件，同时保持 SafeHandle、capture owner count、pool value/owned 语义与 Dispose
边界不变。

### 实现与门禁

- `CudaStream.cs` 从 439 行降至 126 行，仅保留 handle/property、capture-to-graph owner count、Dispose 与内部
  Enter/Exit helper；一般 diagnostics、capture diagnostics、capture dependencies、synchronization/event 与 capture
  lifecycle 进入五份 partial。
- `CudaMemoryPool.cs` 从 372 行降至 31 行，仅保留 pool handle/value core；factory、allocation、access 与 attributes
  进入四份 partial，独立 `CudaOwnedMemoryPool` owner 和两个 pool enum 分别移入专用文件。
- `ManagedCudaStreamMemoryPoolLayoutTests` 固定两个 core 与九份 feature partial 的精确成员集合、capture owner
  lifecycle、owned-pool Dispose/helper 与 enum type owner，并规范化重组两份原源码。
- 拆分前 Git blob 为 `379fd60f08beca36711507a6c83a2192d20b6562`、
  `7028220b628b3d2af9c3f007dda75ec4ff2f5fd3`；normalized SHA-256 保持
  `8b6117416cd402dc7bbed964431822652be7ef19fe61a94c7e3924b8f780fd52` 与
  `df79524f844e67591ee942abdca0a9a8ce996cbcc32729695dcc7754722f707e`。
- 四份直接读取旧 Stream core 的质量测试改读 CaptureDiagnostics、CaptureDependencies、CaptureLifecycle 或明确的
  core+feature 组合；stream-capture audit、两篇 CUDA 文章与双语 source-organization 同步到真实 owner。

### 验证与边界

- 新布局/重组门禁：`15/15` 通过；四组 CUDA 消费门禁：`18/18` 通过；两者合并聚焦集合：`33/33` 通过；
  全部 managed 源码布局门禁合并集合：`232/232` 通过。
- `JYPPX.CudaSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；22 份 ignored JSON 均可解析，
  ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0；本机仍无仓库认可的 `pwsh`，未运行 exporter/B-tier 聚合测试。
- owner/source relocation 不构成 CUDA runtime、stream capture、async pool allocation、real model、Linux、package
  consumer、public package、post-publish、Owner acceptance 或 release proof。
- 进程审计识别出其他工作区的 PowerShell/dotnet 任务并原样保留；8 份 publishing 用户变更未触碰、未暂存；
  未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 ONNX Engine Build Service And Result Source Split

本阶段继续整理 `JYPPX.TensorRtSharp.Tools` 的 ONNX engine build domain，将构建配置、缓存、refit、诊断、runtime、
benchmark、输入与 reference validation 从单一超大 service 中分离，并把 build result 文件中混放的独立 copied
evidence/summary 类型移入专用文件。public API、selected-device thread、worker lifetime、proof classification 与执行顺序不变。

### 实现与门禁

- `OnnxEngineBuildService.cs` 从 3,214 行降至 540 行，仅保留 `Execute`、selected-device thread 与 `ExecuteCore`
  build orchestration；BuilderConfiguration、TimingCache、ResultCreation、Refit、Diagnostics、RuntimeExecution、
  Benchmarking、RuntimeInputs、ReferenceValidation、DeploymentConfiguration 进入十份 partial。
- timing-cache lease、benchmark worker/run/warm-up state 与 runtime input/output state 跟随各自 feature；
  `ReferenceJsonOptions` 随 reference tensor 读取和比较逻辑移动，不再留在 orchestration core。
- `OnnxEngineBuildResult.cs` 从 970 行降至 304 行主 result；`OnnxEngineTimingCacheArtifact`、
  `OnnxEngineCapabilityProbe`、`OnnxLoadedEngineDiagnostics`、`OnnxEnginePreflightMetadata`、
  `OnnxEngineBuildModelEvidence` 与 `OnnxEngineBenchmarkSummary` 六个独立 public 类型各自成文件。
- `ManagedOnnxEngineBuildLayoutTests` 固定 11 份 service core/feature 的精确 method 与 nested-type owner、七个 result
  type 文件的精确 public property 集合、文档/exporter owner marker，并按原顺序重组两份拆分前源码。
- 拆分前 Git blob 为 `31f2c170c9c74b7278b4bc266eca76405dc33067`、
  `97f6e992fe582513fcf77b10ce05de4de32af1a5`；normalized SHA-256 保持
  `433cd0e3ffdf2da39f8bb345eb96d39423885e5046119edb11b9f11c5bc4d4cb` 与
  `683fd2ce1579286a222cd61842b754b7731b94bf6b30440da4bfb653cd2dbc2b`。
- 九个直接读取旧 service core 的能力测试类改读实际 feature 或明确的 core+feature 组合；ONNX roundtrip 博客、
  builder-config/engine-inspector 文章、foundations exporter 与双语 source-organization 同步到真实 owner。

### 验证与边界

- 新布局/重组/文档 marker 门禁：`22/22` 通过；受影响的既有 build/runtime/refit/benchmark 消费门禁：`80/80`
  通过；两篇 publishing 文章专项：`2/2` 通过；全部 managed 源码布局门禁合并集合：`254/254` 通过。
- `JYPPX.TensorRtSharp.Tools` 与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；22 份 ignored JSON 均可解析，
  ignored 文件未强制提交。
- 本机仍无仓库认可的 `pwsh`，因此未运行 foundations exporter 与 B-tier 聚合测试；未把源码 marker 门禁冒充 exporter pass。
- Generated/native/manifest/ABI 改动为 0；进程审计保留其他工作区的 PowerShell/dotnet 任务。
- source/type relocation 不构成 TensorRT/CUDA runtime、real model、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 TensorRT Public Enum Module Split

本阶段移除 `JYPPX.TensorRtSharp/Core/TensorRtEnums.cs` 这一跨模块聚合文件，将其中 64 个 public enum 按实际
API owner 归档，同时保持 namespace、名称、underlying type、数值、`Flags` 属性与 XML 文档逐字不变。

### 实现与门禁

- 三个跨 owner tensor 基础类型 `TensorRtDataType`、`TensorRtIOMode`、`TensorRtTensorLocation` 进入
  `Core/TensorRtTensorCoreEnums.cs`；Core 不再承载 builder/layer/parser/runtime 专属 enum。
- Network、Parsing、Execution、Serialization、Engine、Runtime、Builder、Profiles、ControlFlow 分别获得模块 enum 文件；
  32 个 layer enum 进一步按 RNN、operation、resize、metadata、attention 分成五个文件，没有形成新的 layer 聚合大文件。
- Serialization、TensorFormat、Quantization、BuilderFlag 等单值/flags 配对保持同文件；八个显式 `uint` enum 与
  八个 `[Flags]` enum 的集合保持不变。
- `ManagedTensorRtEnumModuleLayoutTests` 固定 15 个模块文件的精确 type/Flags owner、64 个类型唯一性、显式
  underlying type 集合，并按原声明顺序重组拆分前源码。
- 拆分前 Git blob 为 `861460b834d7d415fc0497f6c5b05fecaba03707`；normalized SHA-256 保持
  `f369f7a9d557889dee633d8e7c39de77231c9c61ca45a5b35476d95d8e429dff`。
- Builder scalar、trtexec deployment、Engine/RNN diagnostics 三个直接读取旧聚合文件的测试类已迁移到
  Builder、Network、Layers/RNN 的真实 owner；双语 source-organization 已同步。

### 验证与边界

- 新 enum owner/value/Flags/重组门禁：`18/18` 通过；受影响消费门禁：`49/49` 通过；合并聚焦集合：`67/67`
  通过；全部 managed 源码布局门禁合并集合：`272/272` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0；本机仍无仓库认可的 `pwsh`，未运行 exporter/B-tier 聚合测试。
- enum/source relocation 不构成 TensorRT runtime、real model、Linux、package consumer、public package、post-publish、
  Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 ONNX Runtime Artifact And Build Diagnostics Source Split

本阶段继续整理 `JYPPX.TensorRtSharp.Tools` 的 runtime artifact 与 build report 输出层，将结构化输出、profile、
engine readback、raw binding、proof boundary、JSON/Markdown projection 和 option status 从两个大文件中分离。
public API、JSON/Markdown 字段、artifact bytes/hash、文件写入顺序与 proof classification 保持不变。

### 实现与门禁

- `OnnxEngineRuntimeArtifactWriter.cs` 从 1,052 行降至 85 行 dispatch/hash core；Times、Output、Profile、
  EngineReadback、RawBindings、ProofBoundary 与 FileIO 进入七份 partial。
- `OnnxEngineRuntimeArtifactData` 与 `OnnxEngineRuntimeOutputArtifact` 成为独立 public model 文件；raw binding segment
  与 runtime proof boundary nested type 继续跟随各自 feature owner。
- `OnnxEngineBuildDiagnostics.cs` 从 776 行降至 37 行 report format dispatch core；Json、Markdown、OptionStatus、
  ReportBoundary 进入四份 partial，build option implementation status 与 report boundary 成为独立 public model 文件。
- `ManagedOnnxEngineArtifactDiagnosticsLayoutTests` 固定两个 core/十一份 feature partial 的精确 method/nested-type
  owner、四个 model 的精确 public property 集合，并按原顺序重组两份拆分前源码。
- 拆分前 Git blob 为 `5269b4685e87d8eb0e65b337499963b12de71aaa`、
  `c7c832f862d8c3d4bff2b93666e9f842b24cdd91`；normalized SHA-256 保持
  `ef0a5dbcb12d1924a2c764fc30c8997adc704f165053e8c11cd9a74ec525a5a7` 与
  `149fad1b78428b823b8f4f14424491e66ea215e4885175ae6886b49e3fcb7329`。
- 八个直接读取旧 writer/diagnostics core 的能力测试类改读实际 feature 或明确组合；六篇 publishing 文章、
  PublishingPublicArticleTests 与双语 source-organization 同步到真实 owner。

### 验证与边界

- 新布局/重组门禁：`19/19` 通过；布局加受影响 build/runtime/refit/benchmark/article 消费聚焦集合：`127/127`
  通过；全部 managed 源码布局门禁合并集合：`291/291` 通过。
- `JYPPX.TensorRtSharp.Tools` 与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；当前 92 份 ignored JSON 均可解析，
  ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过；本机仍无仓库认可的 `pwsh`，未运行
  exporter/B-tier 聚合测试。
- 进程审计保留其他父进程启动的 PowerShell 与 MSBuild node-reuse 任务，未终止、删除或借用其他工作区进程。
- source/model relocation 不构成 TensorRT/CUDA runtime、real model、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 MNIST Runtime And Trtexec Parser Source Split

本阶段继续整理 `JYPPX.TensorRtSharp.Tools` 的模型特定 runtime 与 CLI parser，将 MNIST options/model/result、
PGM preprocessing、classification、environment、diagnostics、service helper，以及 trtexec argument/scalar/build/memory
解析从两个大文件中分离。public API、参数优先级、异常文本、bytes/hash、runtime 执行顺序与 proof classification 不变。

### 实现与门禁

- 原 932 行 `Runtime/MnistOnnxRuntime.cs` 已删除；`MnistOnnxRuntimeOptions`、`MnistPgmImage`、`MnistPgmReader`、
  `MnistClassification`、`MnistOutputClassifier`、`MnistRuntimeEnvironment`、`MnistOnnxRuntimeResult` 与
  `MnistOnnxRuntimeDiagnostics` 各自成文件。
- `MnistOnnxRuntimeService.cs` 为 202 行 execution core；artifact/hash、tensor/shape 与 option validation 进入
  Artifacts、Tensors、Validation 三份 partial，service 最大文件不再承载跨职责 helper。
- `TrtexecLikeParser.cs` 从 731 行降至 262 行 Parse orchestration core；Arguments、ScalarParsing、
  BuildOptionValues、MemoryUnits 四份 partial 分别承载参数集合、数值/reference policy、build 值归一化与 checked
  memory unit parsing。
- `ManagedMnistRuntimeParserLayoutTests` 固定 17 个 MNIST/parser 文件的精确 method owner、五个 model 文件的精确
  public property 集合、旧 MNIST 聚合文件删除，并按原顺序重组两份拆分前源码。
- 拆分前 Git blob 为 `116ff0707028cd234819b023f6b2b5e0ff3cbb43`、
  `3c94e7b5fe1ed741b722f71eb10f2888f76567b7`；normalized SHA-256 保持
  `bd8b5e607b5f966dc6be455adc2ecf6794db86669f1d9597dcd6a7c3a9998ff5` 与
  `333c37056b216785710d0e32ee431b24e8d4261ad8acd9694a222b7bd2ae940e`。
- MNIST 直接消费测试改读 result owner；ONNX roundtrip 博客、三篇 publishing 文章、
  PublishingPublicArticleTests 与双语 source-organization 同步到真实 parser/runtime owner。

### 验证与边界

- 新布局/owner/property/重组门禁：`23/23` 通过；布局加受影响 MNIST/parser/application/article 消费聚焦集合：
  `133/133` 通过；全部 managed 源码布局门禁合并集合：`314/314` 通过。
- `JYPPX.TensorRtSharp.Tools` 与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；当前 92 份 ignored JSON 均可解析，
  ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过；本机仍无仓库认可的 `pwsh`，未运行
  exporter/B-tier 聚合测试。
- 进程审计保留其他父进程启动的 PowerShell 与 MSBuild node-reuse 任务，未终止、删除或借用其他工作区进程。
- source/type relocation 不构成新的 TensorRT/CUDA runtime、real model、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 Trtexec Deployment Options And Build Policy Source Split

本阶段继续整理 `JYPPX.TensorRtSharp.Tools/Trtexec` 的 deployment projection 与 build policy，将 argument、diagnostics、
tactic、memory pool、IO format、precision、layer policy、rule validation 与 data type parsing 从两个大文件中分离。
参数顺序、quoted argument、diagnostic 文本、version guard、policy precedence、wildcard 与 fail-closed 行为不变。

### 实现与门禁

- `TrtexecLikeDeploymentOptions.cs` 从 644 行降至 225 行 constructor/default/property core；Arguments、Diagnostics、
  Tactics、MemoryPools、ProjectionHelpers 进入五份 partial，`TrtexecLikeMemoryPoolSize` 独立成文件。
- tactic-source XML 文档随 `ResolveTacticSources` 进入 Tactics owner，未在 diagnostics 文件留下悬空文档。
- `TrtexecLikeBuildPolicy.cs` 从 568 行降至 93 行 normalization/Apply orchestration core；Parsing、IoFormats、
  Precision、Layers、Rules、DataTypes 进入六份 partial。
- `TrtexecLikeIoFormatSpec` 与 `TrtexecLikeLayerTypeRule` 两个 internal model 各自成文件。
- `ManagedTrtexecDeploymentBuildPolicyLayoutTests` 固定 16 个 deployment/policy 文件的精确 method owner、四个 model/core
  的精确 public property 集合，并按原顺序重组两份拆分前源码。
- 拆分前 Git blob 为 `d55d835f5dd934975af9a302391415e49030a14d`、
  `ef5b4bf488124bb1d37090cbf5cf1b75cf345dfb`；normalized SHA-256 保持
  `cd3154c7ed1bc1fc75023dead6f623b460378ed25898766c7f4bbebf50432e43` 与
  `68f57328142b52c81d08a6c15f6f7165e55fef7b1da39e5d101fb9c6646c3468`。
- memory-pool、build-policy、release-readiness 消费测试改读真实 owner；builder-config publishing 文章、gap list、
  PublishingPublicArticleTests 与双语 source-organization 同步。

### 验证与边界

- 新布局/owner/property/重组门禁：`22/22` 通过；布局加受影响 deployment/policy/article/gap 消费聚焦集合：
  `115/115` 通过；全部 managed 源码布局门禁合并集合：`336/336` 通过。
- `JYPPX.TensorRtSharp.Tools` 与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- ignored deferred candidate evidence 保持 260 条引用、147 个唯一路径、0 缺失；当前 92 份 ignored JSON 均可解析，
  ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0，`git diff --check` 通过；本机仍无仓库认可的 `pwsh`，未运行
  exporter/B-tier 聚合测试。
- 进程审计保留其他父进程启动的 PowerShell 与 MSBuild node-reuse 任务，未终止、删除或借用其他工作区进程。
- source/model relocation 不构成新的 TensorRT/CUDA runtime、real model、Linux、package consumer、public package、
  post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 DebugListener Precheck And Allocator Callback Owner Source Split

本阶段继续整理 `JYPPX.TensorRtSharp/Callbacks`，将 DebugListener runtime-proof prerequisite chain 与 allocator
callback owner 的 copied model、生命周期、dry-run、ledger 和 internal prototype 从两个大文件中分离。全部 public/internal
签名、诊断文本、proof classification、GCHandle/delegate keep-alive 与释放顺序保持不变。

### 实现与门禁

- `TensorRtDebugListenerRuntimeProofPrecheck.cs` 从 1,774 行降为 379 行最终 evaluation core；前 15 个 overload 按
  DesignPrerequisites、NativeAttachDesign、OwnerLifecycle、RuntimeScaffold、FinalRuntimeGates 进入五份 partial，最终
  19-parameter gate 保留在 core，`TensorRtDebugListenerRuntimeProofPrecheckResult` 独立成文件。
- `TensorRtAllocatorCallbackOwner.cs` 从 1,163 行降为 120 行 state/constructor/property core；Lifecycle、ManagedDryRun、
  NativeDryRun、StateLedger、InternalPrototype、ResultMapping 进入六份 partial。
- allocator request、managed/native/state result、snapshot、delegate 与 internal prototype result 七个 top-level 类型各自成文件；
  `Dispose`、callback drain、runtime delegate GCHandle、callback state GCHandle、release hook 与 `GC.KeepAlive` 顺序逐字不变。
- `ManagedCallbackPrecheckAllocatorLayoutTests` 固定 16 个 `Evaluate` overload 的完整 prerequisite type prefix、allocator
  method/property/type owner、delegate owner、两份旧源码的规范化重组 SHA，以及 readiness/test source-set 枚举。
- 拆分前 Git blob 为 `c33e5795a35e4ef04836486ac0fb64552f608681`、
  `7c71b544fe40b1a4f5c77b42b3900e1ce62f899c`；normalized SHA-256 保持
  `d7ede0aca1090be578050e43b80e1ed737189f415f9c1663465457e3bd7cc67c` 与
  `7b1556a06acdf5dce2fff5af49852df5f7490ec8201fba083abbfa548ce1b993`。
- 18 个直接读取旧聚合文件的测试类统一改读真实 core/partial/model 组合；`Test-RuntimePackageReadiness.ps1` 的
  28 个 evidence 聚合点统一经 `Get-EvidenceSourceText` 展开两套实际文件，PowerShell UTF-8 source parse 为 0 error。
- callback/allocator 路线图与双语 source-organization 同步；文档明确 core 单文件不再代表完整实现，也不改变
  pointer-free readiness 与 real callback runtime proof 的边界。

### 验证与边界

- 新 overload/owner/property/重组/evidence 组合门禁：`23/23` 通过；DebugListener、allocator、deferred boundary
  直接消费聚焦集合：`145/145` 通过；callback readiness/closure 聚合：`6/6` 通过；全部 managed layout 合并集合：
  `359/359` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`。
- 额外 article 测试切片实际为 `52/55`：两个失败来自本机不存在仓库认可的 `pwsh`，另一个来自既有 release story
  仍断言旧 `Manifest API count: 3976`；未将该切片或 exporter 宣称通过，也未生成新的 exporter evidence。
- ignored deferred candidate evidence 保持 `260` 条引用、`147` 个唯一路径、`0` 缺失；其中引用的 `22` 份 JSON
  全部可解析，ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0；`git diff --check` 通过，readiness 脚本未运行。
- 进程审计快照只发现当前审计 PowerShell；Downloads 与用户 Temp 顶层近三小时没有本批 TensorRT/JYPPX/CUDA/
  NVRTC/ONNX/engine/nupkg 重资产匹配项，未终止、删除或借用其他工作区进程。
- source/type relocation 不构成新的 callback runtime、TensorRT/CUDA runtime、real model、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。

## 2026-07-30 DebugListener Callback Owner And Closure Matrix Source Split

本阶段继续整理 callback owner 与 pointer-free closure aggregation，将 DebugListener copied request/snapshot、design
diagnostic、snapshot mapping、lifecycle、trampoline/state、shape formatting，以及 closure matrix 的五个 family row、统一
row construction、blocker aggregation 与 result model 从两个大文件中分离。所有行为和 non-proof 分类保持不变。

### 实现与门禁

- `TensorRtDebugListenerCallbackOwner.cs` 从 841 行降为 68 行 owner state/constructor/property core；DesignDiagnostic、
  Snapshots、Lifecycle、Trampoline、ShapeFormatting 进入五份 partial，request 与 snapshot model 各自成文件。
- `TensorRtDebugListenerDesignGateCallback` 与 nested `CallbackState` 共同留在 Trampoline owner；`Dispose`、active gate
  drain、delegate GCHandle、callback-state GCHandle、release hook 与 `GC.KeepAlive` 顺序逐字不变。
- `TensorRtCallbackOwnerClosureMatrix.cs` 从 648 行降为 47 行 Evaluate core；Allocators、OutputDebug、StreamIo、
  RowConstruction、Blockers 进入五份 partial，row 与 result model 各自成文件。
- matrix 的 `GpuAllocator -> GpuAsyncAllocator -> OutputAllocator -> DebugListener -> StreamReaderWriter` 行顺序、
  15 个 closure column、source blocker 去重、deferred-row-required 与 package-consumer proof 计算均保持不变。
- `ManagedCallbackOwnerClosureLayoutTests` 固定 12 份 partial/core 的方法归属、四个 model 的精确 public property、
  nested state/delegate owner、readiness/test source-set 枚举，并按原顺序重组两份拆分前源码。
- 拆分前 Git blob 为 `f5ed356884eb6aa4b5721501a77e3a299ebdd771`、
  `d36f22e8cf3040f169885637ce5be7801551bbf0`；normalized SHA-256 保持
  `d1ce0f25a783b7a251312d376d3f171412e9777ae1868de434b14b926d30595e` 与
  `85a441af723e7f924ec5d3534d61353b56553465e17efb349fac56b79e36869f`。
- 两个直接读取旧聚合文件的测试改读真实八文件组合；`Test-RuntimePackageReadiness.ps1` 与测试 reader 都显式
  展开两套 source set；callback/allocator 路线图和双语 source-organization 同步。

### 验证与边界

- 新 method/property/nested-owner/重组/evidence 组合门禁：`20/20` 通过；owner、closure、readiness 与前批布局
  聚焦集合：`54/54` 通过；全部 managed layout 合并集合：`379/379` 通过。
- `JYPPX.TensorRtSharp` 全目标框架与完整 `TensorRtSharp.sln` Debug build 均为 `0 warning / 0 error`；
  RuntimePackageReadiness UTF-8 source parse 为 `0 error`，但因本机无仓库认可的 `pwsh` 未运行 exporter。
- ignored deferred candidate evidence 保持 `260` 条引用、`147` 个唯一路径、`0` 缺失；其中引用的 `22` 份 JSON
  全部可解析，ignored 文件未强制提交。
- Generated/native/manifest/ABI 改动为 0；`git diff --check` 通过。
- 进程审计快照只发现当前审计 PowerShell；Downloads 与用户 Temp 顶层近三小时没有本批 TensorRT/JYPPX/CUDA/
  NVRTC/ONNX/engine/nupkg 重资产匹配项，未终止、删除或借用其他工作区进程。
- source/type relocation 不构成新的 callback runtime、TensorRT/CUDA runtime、real model、Linux、package consumer、
  public package、post-publish、Owner acceptance 或 release proof。
- 8 份 publishing 用户变更未触碰、未暂存；未 push、未触发 GitHub Actions、未执行远程发布操作。
