# Real Model And Package Proof Input Package

`Export-RealModelAndPackageProofInputPackage.ps1` 用来把真实模型资产、package-consumer-runtime、compatible host、Linux runner 和 post-publish verification 的 owner 输入材料集中到一个可复制执行的清单中。它不是 proof，不发布包，也不关闭 release issue。

核心边界固定为：

- `recordKind=real-model-and-package-proof-input-package`
- `packageState=owner-action-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelAndPackageProofInputPackage.ps1
```

输出文件：

- `artifacts/final-release/real-model-and-package-proof-input-package.json`
- `artifacts/final-release/real-model-and-package-proof-input-package.md`

## Package Consumer Runtime 输入

`package-consumer-runtime` 只能来自 clean consumer runtime smoke。owner 需要回填真实 package、真实主机和真实日志信息：

- managed/runtime nupkg SHA256
- `runtimePackageKey`
- repository 外部 clean consumer identity
- host owner、machine、OS、GPU、driver、CUDA、TensorRT、cuDNN metadata
- restore/build/dependency probe/runtime smoke commands
- reviewed `stdoutSummary` 和 `stderrSummary`
- `smokeLogPath` 和 `smokeLogSha256`
- `no ProjectReference`

验证命令必须使用：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
```

`ProjectReference`、local feed、helper scan、DependencyProbe、build-only、parse-only、sidecar-only、template、draft、runbook、collection package 和 `blocked-by-cuda-driver` 都不能替代 `package-consumer-runtime`。

## Real Model Runtime 输入

`real-model-runtime` 是样例级真实模型运行证明。它可以证明 Classification 或 YoloVision 在真实模型、真实输入和真实日志上跑通，但不能替代 NuGet package consumer release proof。

Classification 输入至少包括：

- model path、labels path、input asset path
- model/license/labels/input 的 SHA256
- TensorRtExec build report 和 sidecar
- sample runner command 和 sample runner log
- reviewed `stdoutSummary` 和 `stderrSummary`
- sample-run-evidence record

YoloVision 范围固定为 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。同样需要模型、labels、输入资产、license、SHA256、TensorRtExec sidecar、sample runner log 和 sample-run-evidence record。

样例 proof 只能晋级 `real-model-runtime`，不能晋级 `package-consumer-runtime`。

## Post-Publish Verification 输入

`post-publish verification` 只能在 owner 完成真实渠道发布后执行。输入需要来自真实 channel package，而不是本地包或草稿：

- real channel package URL
- downloaded managed/runtime nupkg SHA256
- clean consumer root outside repository
- no `ProjectReference`
- native assets listing
- dependency probe log
- runtime smoke log
- reviewed `stdoutSummary` 和 `stderrSummary`

验证命令必须使用：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

只有真实渠道包、下载 hash、clean consumer restore/build/probe/smoke 日志和 validator 一起通过后，post-publish proof 才能进入 release close preflight。

## Copyable Owner Order

1. 刷新 `compatible-host-proof-backfill-package`。
2. 生成或复制 `external-runtime-proof-record.input-template.json`。
3. 在兼容 CUDA/TensorRT 主机上运行 clean package consumer smoke。
4. 填写 `external-runtime-proof-record.json` 并运行 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
5. 回填 Classification/YoloVision 真实模型资产、license、SHA256 和 sample-run-evidence。
6. owner 完成真实发布后，填写 `post-publish-verification-record.json`。
7. 运行 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
8. 刷新 `Export-ReleaseEvidenceBundle.ps1` 和 `Export-ReleaseClosePreflight.ps1`。

## 不可替代材料

以下材料只能作为辅助或输入，不能写成 proof：

- template
- draft
- runbook
- collection package
- local feed
- `ProjectReference`
- build-only
- parse-only
- sidecar-only
- DependencyProbe
- dependency-probe-only
- `blocked-by-cuda-driver`
- owner guidance without real logs
- owner-action-required without validator pass

这个输入包的价值是减少 owner 在真实主机上来回翻文件的成本，同时保留 proof 边界：没有真实日志、真实 hash、真实主机 metadata 和 validator 通过，就不能把状态写成完成。
