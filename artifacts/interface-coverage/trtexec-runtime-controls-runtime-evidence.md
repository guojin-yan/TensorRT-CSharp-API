# Trtexec-Like Runtime Controls Evidence

TRT10.11/CUDA12.9、RTX 3060 Laptop、driver 576.02 上完成两条本地真实 smoke。

| Run | 请求 | 实际结果 |
|---|---|---|
| threads + spin + CUDA graph | `infStreams=2`、`threads`、`useSpinWait`、`useCudaGraph`、3 rounds | 2 个独立 driver threads、`[3,3]` per-context rounds、6 samples；event query 生效；两个 context 均 capture/instantiate/launch 成功，fallback reason 为空；output match |
| no data transfers | `noDataTransfers`、2 rounds，同时请求 output/raw dump | 2 次真实 enqueue；0 次 input H2D、0 次 output D2H/readback；`OutputMatch=false`、output count 0、raw proof false，三个 output 选项保持 parse-only |

`--threads` 按官方布尔开关处理；旧 `--threads N` 只作为兼容输入，normalized command 输出裸
`--threads`。每个 worker 在独立线程恢复创建资源时的 CUDA device，并拥有独立 context、bindings、
stream、events 和可选 graph/graph-exec owner。

`--useSpinWait` 通过 `CudaEvent.IsReady()` 主动查询完成状态。`--useCudaGraph` 在计时前进行一次
direct enqueue 初始化，再捕获 TensorRT enqueue、实例化 graph 并执行 graph launch；任何 context
失败都会释放已建 graph 并统一回退 direct enqueue，报告 fallback reason，该次 option 不标 applied。

`--noDataTransfers` 只分配并绑定 device buffers，不执行 input host copy，也不读取 output；times
artifact 可证明 scheduler/enqueue 行为，但不能证明 tensor correctness。`--sleepTime` 继续 parse-only，
因为普通 CPU sleep 不是官方 device-side launch-to-compute gap。

本证据仍是 ProjectReference synthetic/local runtime：`isRealModelRuntimeProof=false`、
`isPackageConsumerRuntimeProof=false`、`canPublishPublicly=false`，不授权任何公开发布操作。
