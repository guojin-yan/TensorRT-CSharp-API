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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $DefaultValue
}

function Find-PackageConsumerRuntimeProofPreflightEntry {
  param(
    [AllowNull()][object]$Matrix,
    [string]$RuntimePackageKey
  )

  if ($null -eq $Matrix) {
    return $null
  }

  foreach ($entry in @($Matrix.entries)) {
    if ([string]::Equals([string]$entry.runtimePackageKey, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $entry
    }
  }

  return $null
}

$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$runtimeReadiness = Read-JsonOrNull "artifacts\package-readiness\runtime-package-readiness-summary.json"
$runtimeProofPreflightMatrix = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
$runtimeProofPreflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $runtimeProofPreflightMatrix -RuntimePackageKey $RuntimePackageKey
$preflightEntryFound = $null -ne $runtimeProofPreflightEntry
$preflightRuntimePackageId = [string](Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "runtimePackageId" -DefaultValue "")
$preflightRestoreSourceMode = [string](Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "restoreSourceMode" -DefaultValue "")
$preflightNativeAssetCopyExpected = Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "nativeAssetCopyExpected" -DefaultValue $null

$currentStatus = if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofStatus") {
  [string]$runtimeReadiness.runtimeProofStatus
}
elseif ($packageConsumer) {
  [string]$packageConsumer.status
}
else {
  "missing-runtime-proof-source"
}

$evidenceClassifications = @(
  "precheck",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$classificationRules = @(
  "precheck records prove command/report readiness only and cannot be promoted as runtime proof.",
  "build-only records prove restore/build/conversion only and cannot be promoted as runtime proof.",
  "dependency-probe-only records prove bridge/load diagnostics only and cannot be promoted as runtime proof.",
  "synthetic-input-runtime records prove pipeline execution with synthetic input only; they are not real model proof.",
  "real-model-runtime records require model/hash/license/input/log evidence but are still separate from clean package-consumer proof.",
  "package-consumer-runtime is the required classification for external runtime proof promotion."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-record-template"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = $true
  proofState = "template-only"
  proofClassification = "template-only"
  evidenceClassifications = @($evidenceClassifications)
  isRuntimeExecutionEvidence = $false
  isDependencyProbeOnly = $true
  canPromoteRuntimeProof = $false
  currentRuntimeProofStatus = $currentStatus
  host = [pscustomobject]@{
    ownerName = ""
    machineName = ""
    osDescription = ""
    gpuName = ""
    driverVersion = ""
    cudaDriverSupportedRuntime = ""
    cudaRuntimeVersion = ""
    tensorRtRuntimeVersion = ""
    cudnnVersion = ""
    tensorRtLine = ""
  }
  packageSource = [pscustomobject]@{
    managedPackageSource = ""
    runtimePackageSource = ""
    runtimePackageKey = $RuntimePackageKey
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    consumerProjectName = ""
    consumerProjectPath = ""
    managedNupkgSha256 = ""
    runtimeNupkgSha256 = ""
    noProjectReference = $null
  }
  command = [pscustomobject]@{
    restoreCommand = ""
    buildCommand = ""
    smokeCommand = ""
    exitCode = $null
    startedAtUtc = $null
    finishedAtUtc = $null
    logPath = ""
    logSha256 = ""
  }
  modelEvidence = [pscustomobject]@{
    modelName = ""
    modelSha256 = ""
    modelLicense = ""
    inputAssetName = ""
    inputAssetSha256 = ""
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = "pending"
    smokeStatus = "pending-compatible-host-execution"
    nativeAssetsCopied = $null
    nativeAssetsExpected = $preflightNativeAssetCopyExpected
    nativeAssetsFound = $null
    stdoutSummary = ""
    stderrSummary = ""
    failureDiagnostic = ""
  }
  runtimeProofPreflight = [pscustomobject]@{
    matrixFound = $null -ne $runtimeProofPreflightMatrix
    entryFound = $preflightEntryFound
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    nativeAssetCopyExpected = $preflightNativeAssetCopyExpected
    ownerActionRequired = $true
    canPromotePackageConsumerRuntimeProof = $false
    boundary = "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source."
  }
  classificationRules = @($classificationRules)
  promotionRules = @(
    "A template-only record is not runtime proof.",
    "isRuntimeExecutionEvidence=true requires a real smoke command, compatible CUDA driver/GPU host, exitCode=0, and smokeStatus=passed.",
    "runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound must align with RuntimeProofPreflight.",
    "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.",
    "consumerProjectName and consumerProjectPath must identify the clean package consumer project.",
    "host CUDA runtime, TensorRT runtime, cuDNN runtime, driver, GPU, OS, and TensorRT line must be filled.",
    "smokeCommand must include --runtime-package-key for the release target runtime package key.",
    "managedNupkgSha256 and runtimeNupkgSha256 must be 64-character SHA256 hashes of the consumed packages.",
    "logSha256 must be a 64-character SHA256 hash of the smoke log.",
    "stdoutSummary must summarize reviewed stdout from the real smoke log.",
    "stderrSummary must summarize reviewed stderr, or explicitly state no-stderr-emitted when stderr is empty.",
    "proofClassification must be package-consumer-runtime before this record can be promoted.",
    "DependencyProbe output is useful diagnostics but is not runtime execution proof.",
    "blocked-by-cuda-driver is not smoke passed."
  )
  ownerActionSummary = @(
    "复制本模板为 external-runtime-proof-record.json 后再填写真实字段。",
    "在兼容 CUDA/TensorRT 主机上运行 clean package consumer smoke。",
    "保留 smoke log、logSha256、stdoutSummary、stderrSummary 和 host metadata。",
    "按 RuntimeProofPreflight 填写 packageSource.runtimePackageId、packageSource.restoreSourceMode、results.nativeAssetsExpected 和 results.nativeAssetsFound。",
    "用 Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof 验证真实记录。",
    "template、draft、example、dependency-probe-only、blocked-by-cuda-driver 都不能关闭 release issue。"
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-record-template.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-record-template.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Record Template")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Proof state: ``template-only``")
$lines.Add("")
$lines.Add("Proof classification: ``template-only``")
$lines.Add("")
$lines.Add("Runtime execution evidence: ``false``")
$lines.Add("")
$lines.Add("Dependency probe only: ``true``")
$lines.Add("")
$lines.Add("Current runtime proof status: ``$currentStatus``")
$lines.Add("")
$lines.Add("## Runtime Proof Preflight")
$lines.Add("")
$lines.Add("- matrix found: ``$($record.runtimeProofPreflight.matrixFound)``")
$lines.Add("- entry found: ``$($record.runtimeProofPreflight.entryFound)``")
$lines.Add("- runtime package id: ``$($record.runtimeProofPreflight.runtimePackageId)``")
$lines.Add("- restore source mode: ``$($record.runtimeProofPreflight.restoreSourceMode)``")
$lines.Add("- native assets expected: ``$($record.runtimeProofPreflight.nativeAssetCopyExpected)``")
$lines.Add("- can promote from preflight: ``$($record.runtimeProofPreflight.canPromotePackageConsumerRuntimeProof)``")
$lines.Add("")
$lines.Add("This template records what an external CUDA-compatible host must fill before runtime proof can be promoted. It is not runtime proof by itself.")
$lines.Add("")
$lines.Add("## Required Host Evidence")
$lines.Add("")
$lines.Add("- owner name")
$lines.Add("- machine name")
$lines.Add("- OS description")
$lines.Add("- GPU name")
$lines.Add("- driver version")
$lines.Add("- CUDA driver supported runtime")
$lines.Add("- CUDA runtime version")
$lines.Add("- TensorRT runtime version")
$lines.Add("- TensorRT line")
$lines.Add("- managed/runtime package source")
$lines.Add("- runtime package key")
$lines.Add("- runtime package id and restore source mode from RuntimeProofPreflight")
$lines.Add("- managed/runtime nupkg SHA256")
$lines.Add("- no ProjectReference confirmation")
$lines.Add("- restore/build/smoke commands")
$lines.Add("- exit code, log path, and log SHA256")
$lines.Add("- native assets expected/found aligned to RuntimeProofPreflight")
$lines.Add("- stdout/stderr summaries")
$lines.Add("")
$lines.Add("## Evidence Classifications")
$lines.Add("")
foreach ($item in $record.evidenceClassifications) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Classification Rules")
$lines.Add("")
foreach ($rule in $record.classificationRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $record.promotionRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $record.ownerActionSummary) {
  $lines.Add("- $item")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof record template written to $jsonPath"
Write-Host "External runtime proof record template written to $markdownPath"
