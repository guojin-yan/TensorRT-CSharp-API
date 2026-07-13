[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$record = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-record-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  proofClassification = "template-only"
  proofState = "template-only"
  templateOnly = $true
  isRuntimeExecutionEvidence = $false
  isDependencyProbeOnly = $true
  canPromoteRuntimeProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  cleanExternalConsumerRoot = "<owner-fill-clean-external-consumer-root-outside-repository>"
  consumerProjectPath = "<owner-fill-clean-consumer-csproj-path>"
  consumerProjectScan = [pscustomobject]@{
    projectExists = $false
    usesProjectReference = $true
    usesLocalFeed = $true
    usesDirectNupkg = $true
    noProjectReference = $false
    noLocalFeed = $false
    noDirectNupkg = $false
  }
  publicPackageSource = "<owner-fill-public-package-source-url-or-id>"
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  managedPackageVersion = "<owner-fill-managed-package-version>"
  managedNupkgPath = "<owner-fill-public-managed-nupkg-path>"
  managedNupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
  runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  runtimeNupkgPath = "<owner-fill-public-runtime-nupkg-path>"
  runtimeNupkgSha256 = "<owner-fill-runtime-nupkg-sha256>"
  host = [pscustomobject]@{
    ownerName = "<owner-fill-owner-name>"
    machineName = "<owner-fill-machine-name>"
    osDescription = "<owner-fill-host-os>"
    hostArchitecture = "<owner-fill-host-architecture>"
    gpuName = "<owner-fill-gpu-name>"
    driverVersion = "<owner-fill-cuda-driver-version>"
    cudaDriverSupportedRuntime = "<owner-fill-cuda-driver-supported-runtime>"
    cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
    tensorRtRuntimeVersion = "<owner-fill-tensorrt-version>"
    cudnnVersion = "<owner-fill-cudnn-version>"
    tensorRtLine = "<owner-fill-tensorrt-line>"
  }
  command = [pscustomobject]@{
    restoreCommand = "<owner-fill-restore-command>"
    buildCommand = "<owner-fill-build-command>"
    smokeCommand = "dotnet run --project <clean-consumer-project> -- --runtime-package-key $RuntimePackageKey"
    exitCode = $null
    startedAtUtc = "<owner-fill-started-at-utc>"
    finishedAtUtc = "<owner-fill-finished-at-utc>"
    logPath = "<owner-fill-package-consumer-smoke-log-path>"
    logSha256 = "<owner-fill-smoke-log-sha256>"
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = "pending"
    smokeStatus = "pending-compatible-host-execution"
    nativeAssetsCopied = $null
    stdoutSummary = "<owner-fill-stdout-summary>"
    stderrSummary = "<owner-fill-stderr-summary-or-no-stderr-emitted>"
    failureDiagnostic = ""
  }
  externalRuntimeProofRecordPath = "artifacts/final-release/external-runtime-proof-record.json"
  strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  externalRuntimeProofValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof"
  requiredRealInputRules = @(
    "recordKind=package-consumer-runtime-proof-record",
    "proofClassification=package-consumer-runtime",
    "cleanExternalConsumerRoot outside repository",
    "consumer project exists and has no ProjectReference to repository",
    "no local feed or direct .nupkg as public proof",
    "publicPackageSource is public/non-local",
    "managed/runtime package SHA256 values are real and matching",
    "compatible host metadata is non-placeholder",
    "smokeCommand includes --runtime-package-key and runtimePackageKey",
    "smoke log exists and SHA256 matches",
    "stdout/stderr summaries are reviewed"
  )
  safetyBoundary = "Template only. It does not publish packages, close the release issue, or promote runtime proof."
}

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record.template.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record.template.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$ruleLines = $record.requiredRealInputRules | ForEach-Object { "- ``$_``" }
$markdown = @"
# Package Consumer Runtime Proof Record Template

生成时间：$($record.generatedAtUtc)

## 用途

该模板用于把真实 clean external consumer smoke evidence 记录为可验证 proof record。它与 `external-runtime-proof-record` schema 对齐，但本文件本身仍是 template-only，不是 proof。

## 当前状态

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($record.recordKind)`` |
| proofClassification | ``$($record.proofClassification)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Required Real Input Rules

$($ruleLines -join "`r`n")

## 验证

```powershell
$($record.strictValidationCommand)
```

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof record template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
