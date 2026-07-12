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
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$classificationRules = @(
  "build-only records prove restore/build/conversion only and cannot be promoted as runtime proof.",
  "dependency-probe-only records prove bridge/load diagnostics only and cannot be promoted as runtime proof.",
  "synthetic-input-runtime records prove pipeline execution with synthetic input only; they are not real model proof.",
  "real-model-runtime records require model/hash/license/input/log evidence but are still separate from clean package-consumer proof.",
  "package-consumer-runtime is the required classification for external runtime proof promotion."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-record-input-template"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = $true
  exampleOnly = $false
  proofState = "input-template"
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
  requiredEvidenceSummary = @(
    "Change recordKind to external-runtime-proof-record only for a real filled record.",
    "Set templateOnly=false and exampleOnly=false.",
    "Set proofClassification=package-consumer-runtime for promotable runtime proof; build-only, dependency-probe-only, synthetic-input-runtime, and real-model-runtime are non-promotable external proof classifications.",
    "Keep runtimePackageKey equal to the release target runtime package key.",
    "Keep packageSource.runtimePackageKey equal to the release target runtime package key.",
    "Keep packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound aligned with RuntimeProofPreflight.",
    "Fill owner, machine, OS, GPU, NVIDIA driver, CUDA driver supported runtime, CUDA runtime, TensorRT runtime, cuDNN runtime, and TensorRT line.",
    "Use package sources, not ProjectReference or local bin output, and record packageSource.consumerProjectName plus packageSource.consumerProjectPath.",
    "Record managedNupkgSha256 and runtimeNupkgSha256 as 64-character hashes for the exact consumed packages.",
    "Record restore/build/smoke commands, ensure smokeCommand includes --runtime-package-key for the release target, exitCode=0, and smokeStatus=passed.",
    "Record model/hash/license/input evidence when the smoke uses a real model.",
    "Record stdoutSummary and stderrSummary instead of relying on an unreviewed log path only; if stderr is empty, stderrSummary must explicitly state no-stderr-emitted.",
    "Record nativeAssetsCopied=true, logPath, and logSha256 for the compatible host run.",
    "Record results.nativeAssetsExpected and results.nativeAssetsFound so the strict validator can compare them with package-consumer-runtime-proof-preflight-matrix.json.",
    "Set isDependencyProbeOnly=false before requesting runtime proof promotion."
  )
  validationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\final-release\external-runtime-proof-record.json"
  classificationRules = @($classificationRules)
  promotionRules = @(
    "This input template is not runtime proof.",
    "runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound must align with RuntimeProofPreflight.",
    "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.",
    "consumerProjectName and consumerProjectPath must identify the clean package consumer project.",
    "host CUDA runtime, TensorRT runtime, cuDNN runtime, driver, GPU, OS, and TensorRT line must be filled.",
    "smokeCommand must include --runtime-package-key for the release target runtime package key.",
    "managedNupkgSha256 and runtimeNupkgSha256 must be 64-character SHA256 hashes of the consumed packages.",
    "logSha256 must be a 64-character SHA256 hash of the smoke log.",
    "stdoutSummary and stderrSummary must both be reviewed; stderrSummary may use no-stderr-emitted only after checking the preserved smoke log.",
    "proofClassification must be package-consumer-runtime before this record can be promoted.",
    "DependencyProbe output is useful diagnostics but is not runtime execution proof.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Real runtime proof requires compatible host smoke, exitCode=0, native assets copied, and no ProjectReference."
  )
  ownerActionSummary = @(
    "Owner must copy this input template to external-runtime-proof-record.json before filling real proof.",
    "Owner must run the package consumer on a compatible CUDA/TensorRT host, not only a dependency probe.",
    "Owner must preserve the smoke log and compute logSha256 from the exact file referenced by command.logPath.",
    "Owner must preserve RuntimeProofPreflight alignment fields when copying this template to a real record.",
    "Owner must fill stdoutSummary and stderrSummary after reviewing real output; use no-stderr-emitted only after checking stderr.",
    "Owner must validate with Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof before promotion."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-record.input-template.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-record.input-template.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Record Input Template")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Proof state: ``input-template``")
$lines.Add("")
$lines.Add("Proof classification: ``template-only``")
$lines.Add("")
$lines.Add("Runtime execution evidence: ``false``")
$lines.Add("")
$lines.Add("Can promote runtime proof: ``false``")
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
$lines.Add("This is a human-fill input template for a compatible CUDA/GPU host. It is not runtime proof and cannot approve publication.")
$lines.Add("")
$lines.Add("## Required Evidence")
$lines.Add("")
foreach ($item in $record.requiredEvidenceSummary) {
  $lines.Add("- $item")
}
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
$lines.Add("## Validation")
$lines.Add("")
$lines.Add("After copying this file to a real ``external-runtime-proof-record.json`` and filling it, run:")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add($record.validationCommand)
$lines.Add('```')
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

Write-Host "External runtime proof record input template written to $jsonPath"
Write-Host "External runtime proof record input template written to $markdownPath"
