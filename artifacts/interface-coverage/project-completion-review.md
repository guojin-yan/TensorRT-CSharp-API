# TensorRtSharp4.0 完成情况审查

生成日期：2026-07-17

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
