# CUDA Deferred Candidate Safety Audit

最新 `cuda-runtime-interface-coverage.json` 包含 311 条 `deferred-only` 行、64 个唯一函数。按当前
typed owner 与 pointer-free public API 边界重新分类后，immediate-safe 候选为 0。因此本批不以
零散标量接口凑 uplift，按阶段规则转向 TensorRtExec/OnnxToEngine 完整度。

| 分类 | 唯一函数 | 决策 | 边界 |
|---|---:|---|---|
| callback / user object lifecycle | 12 | keep-deferred | callback trampoline、线程异常、unregister、retain/release |
| external / graphics resource | 26 | keep-deferred | foreign owner、import/destroy、map state、async semaphore use |
| generic graph params / kernel | 9 | keep-deferred | generic union、kernel args、device pointer、node-specific owner |
| raw symbol / entry point | 12 | keep-deferred | function/export-table/symbol pointer 与 symbol identity |
| resource / context ownership | 5 | keep-deferred | resource partition 与 create/get/destroy ownership |

版本行数：CUDA 11.6/11.8/12.1 各 49，12.3 为 51，12.9 为 54，13.2 为 59。分类并集精确覆盖
64 个唯一函数，没有 unclassified 行。

本审计不是 API 实现许可，也不是 runtime/publication proof。旧 deferred manifest 必须保留；
public API 继续禁止 `IntPtr`、`nint`、`UIntPtr`、`SafeHandle`、device pointer 和 function pointer。
