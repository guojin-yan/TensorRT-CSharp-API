# Trtexec-Like Bounded Benchmark Scheduler Runtime Evidence

TRT10/CUDA12.9、RTX 3060 Laptop、driver 576.02 上完成两条真实本地 smoke。

| Run | 请求 | 实际结果 |
|---|---|---|
| concurrent contexts | iterations=3, warmUp=5ms, infStreams=2, idleTime=1ms, avgRuns=2, p90 | 2 contexts、3 rounds、6 raw samples、3 average windows、warmup 5.1009ms、output match |
| duration floor | iterations=1, duration=1s | 6440 rounds、measurement 1000.1398ms、output match |

每个 inference stream 都拥有独立的 `TensorRtExecutionContext`、`TensorRtInferenceBindings`、
`CudaStream` 和 start/stop `CudaEvent`。测量轮次先向所有 stream 提交，再同步 stop event，因此不是
重复使用单 context 的串行别名。worker dispose 前会 drain stream；构造失败也会释放已创建 owner。

`iterations` 与 `duration` 使用双下限；`warmUp` 在测量前执行；`idleTime` 只放在连续测量轮次间；
`avgRuns` 生成连续窗口平均值；percentile 使用 raw GPU event samples。

`sleepTime`、`useSpinWait`、`threads`、`useCudaGraph`、`noDataTransfers` 仍是 unapplied/parse-only。
本证据是 ProjectReference synthetic identity scheduler runtime，不是 real-model-runtime、公开包
consumer 或发布许可；`canPromotePublicProof=false`、`canPublishPublicly=false`。
