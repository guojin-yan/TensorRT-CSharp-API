# ProjectQuality 分片质量门禁：如何验证大规模发布工程

TensorRT/CUDA C# 绑定项目进入发布收口阶段后，测试不再只是“能不能编译”的问题。一个完整发布候选需要同时覆盖 API 边界、native bridge、C# 高层封装、sample、runtime package、clean consumer、post-publish、owner proof 和 release close。

这会让 `ProjectQuality` 测试数量快速增长。当前项目已经有 300 多个测试类，如果强行每次都跑 whole-suite，很容易遇到超时、长耗时 proof 测试互相拖累、日志难以追踪等问题。因此项目采用 **class-level shard** 作为持续开发阶段的质量证据。

## 当前证据状态

当前分片覆盖证据位于：

- `artifacts/test-analysis/project-quality-shard-class-coverage.json`
- `artifacts/test-analysis/project-quality-shard-class-coverage.md`
- `artifacts/test-analysis/project-quality-shards/**/summary.json`
- `artifacts/test-analysis/project-quality-shards/**/*.trx`

当前状态：

- `coverageState=complete-class-coverage`
- `inventoryClassCount=309`
- `coveredClassCount=309`
- `missingClassCount=0`
- `invalidEvidenceCount=0`
- `strictPassedTrxCount=88`

这说明所有登记的 ProjectQuality 测试类都有 hash-verified passed TRX 证据，但它仍然不是一次性 whole-suite pass。

## 为什么不直接依赖 whole-suite

发布工程里的测试存在几类不同性质：

1. **轻量结构测试**：检查 JSON schema、文档链接、manifest、artifact 字段。
2. **生成器测试**：检查 binding generator、coverage matrix、release bundle。
3. **release proof 测试**：验证 clean consumer、post-publish、owner input、release close 边界。
4. **runtime smoke 测试**：依赖 CUDA/TensorRT/cuDNN/driver/host 状态。
5. **owner-action 测试**：检查真实 owner 输入是否存在，通常会保持 blocked。

这些测试混在一次 whole-suite 里时，失败定位成本很高。分片运行可以让每个阶段只验证相关测试类，同时把通过证据记录为可复查的 TRX。

## 推荐命令

先构建测试项目：

```powershell
Set-Location E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0

dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore
```

运行单个测试类：

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug `
  --no-build `
  --filter "FullyQualifiedName~JYPPX.ProjectQuality.Tests.FinalProofReadinessBlockerDashboardTests"
```

运行阶段相关测试类：

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug `
  --no-build `
  --filter "FullyQualifiedName~FinalProofReadinessBlockerDashboardTests|FullyQualifiedName~ReleaseEvidenceNewProofInputsTests|FullyQualifiedName~YoloVisionAssetLicenseApprovalTests"
```

生成 TRX：

```powershell
$results = "artifacts\test-analysis\project-quality-shards\YYYYMMDD-targeted-run"
New-Item -ItemType Directory -Path $results -Force | Out-Null

dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug `
  --no-build `
  --filter "FullyQualifiedName~JYPPX.ProjectQuality.Tests.FinalProofReadinessBlockerDashboardTests" `
  --logger "trx;LogFileName=project-quality-final-proof-readiness.trx" `
  --results-directory $results `
  /p:UseSharedCompilation=false `
  /nr:false
```

刷新 coverage：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ProjectQualityShardCoverage.ps1
```

## 证据接收标准

`Export-ProjectQualityShardCoverage.ps1` 只接受严格证据：

- execution unit 必须是 passed。
- TRX 文件必须存在。
- TRX SHA256 必须与 summary 中记录一致。
- 对应测试类必须存在于 ProjectQuality inventory。
- failed、timed-out、preview-only、missing、hash mismatch 都不能计入 coverage。

这也是为什么 `complete-class-coverage` 比简单的“跑过一批测试”更可靠。

## 与发布 proof 的关系

ProjectQuality shard coverage 是质量证据，但发布 proof 仍要看：

- clean public package consumer proof。
- post-publish clean consumer proof。
- TRT runtime smoke proof。
- owner real input。
- package hash。
- host metadata。
- release close strict validator。

因此以下说法都不成立：

- shard coverage 等于 package-consumer-runtime proof。
- shard coverage 等于 post-publish proof。
- shard coverage 等于 public publish approval。
- shard coverage 等于 release close approval。
- shard coverage 等于 whole-suite pass。

最终发布判断应查看：

- `artifacts/final-release/release-evidence-bundle.json`
- `artifacts/final-release/final-proof-readiness-blocker-dashboard.json`
- `artifacts/final-release/final-release-close-blocker-dashboard.json`

## 实战建议

开发 deferred API 时，建议每批至少运行：

1. binding generator 输出测试。
2. interface coverage matrix。
3. 本批新增 wrapper / docs / artifact 测试。
4. release evidence 关键测试。

发布 proof 阶段，建议每批至少运行：

1. 当前 proof dashboard 测试。
2. clean consumer 或 post-publish 对应测试。
3. owner input strict validator 测试。
4. release evidence bundle 测试。

这样可以让项目持续向“可发布”逼近，而不是在 whole-suite 超时中失去可追踪证据。
