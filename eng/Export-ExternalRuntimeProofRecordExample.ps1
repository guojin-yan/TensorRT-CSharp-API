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

$runtimeProofPreflightMatrix = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
$runtimeProofPreflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $runtimeProofPreflightMatrix -RuntimePackageKey $RuntimePackageKey
$preflightEntryFound = $null -ne $runtimeProofPreflightEntry
$preflightRuntimePackageId = [string](Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "runtimePackageId" -DefaultValue "example-runtime-package-id")
$preflightRestoreSourceMode = [string](Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "restoreSourceMode" -DefaultValue "clean-consumer-package-source-required")
$preflightNativeAssetCopyExpected = Get-PropertyOrDefault -Object $runtimeProofPreflightEntry -Name "nativeAssetCopyExpected" -DefaultValue 0

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
  recordKind = "external-runtime-proof-record-example"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = $false
  exampleOnly = $true
  publicationState = "example-not-for-publication"
  proofState = "example-not-for-publication"
  proofClassification = "dependency-probe-only"
  evidenceClassifications = @($evidenceClassifications)
  isRuntimeExecutionEvidence = $false
  isDependencyProbeOnly = $true
  canPromoteRuntimeProof = $false
  host = [pscustomobject]@{
    ownerName = "example-owner-not-for-publication"
    machineName = "example-compatible-host"
    osDescription = "Windows 11 example host"
    gpuName = "Example NVIDIA GPU"
    driverVersion = "example-driver-version"
    cudaDriverSupportedRuntime = "example-cuda-driver-supported-runtime"
    cudaRuntimeVersion = "example-cuda-runtime"
    tensorRtRuntimeVersion = "example-tensorrt-runtime"
    cudnnVersion = "example-cudnn-runtime"
    tensorRtLine = "11"
  }
  packageSource = [pscustomobject]@{
    managedPackageSource = "example-local-feed-not-for-publication"
    runtimePackageSource = "example-local-feed-not-for-publication"
    runtimePackageKey = $RuntimePackageKey
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    consumerProjectName = "ExampleConsumer"
    consumerProjectPath = "artifacts/final-release/example-consumer/ExampleConsumer.csproj"
    noProjectReference = $true
  }
  command = [pscustomobject]@{
    restoreCommand = "dotnet restore ExampleConsumer.csproj --source example-local-feed"
    buildCommand = "dotnet build ExampleConsumer.csproj -c Release --no-restore"
    smokeCommand = "dotnet run --project ExampleConsumer.csproj -- --runtime-package-key $RuntimePackageKey"
    exitCode = 35
    startedAtUtc = $null
    finishedAtUtc = $null
    logPath = "artifacts/final-release/example-external-runtime-proof.log"
    logSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
  modelEvidence = [pscustomobject]@{
    modelName = "example-model-not-for-publication"
    modelSha256 = "example-sha256-not-for-publication"
    modelLicense = "example-license-not-for-publication"
    inputAssetName = "example-input-not-for-publication"
    inputAssetSha256 = "example-input-sha256-not-for-publication"
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = "dependency-probe-only"
    smokeStatus = "blocked-by-cuda-driver"
    nativeAssetsCopied = $false
    nativeAssetsExpected = $preflightNativeAssetCopyExpected
    nativeAssetsFound = 0
    stdoutSummary = "example DependencyProbe output only"
    stderrSummary = "example blocked-by-cuda-driver diagnostic"
    failureDiagnostic = "example-not-for-publication; blocked-by-cuda-driver is not smoke passed"
  }
  runtimeProofPreflight = [pscustomobject]@{
    matrixFound = $null -ne $runtimeProofPreflightMatrix
    entryFound = $preflightEntryFound
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    nativeAssetCopyExpected = $preflightNativeAssetCopyExpected
    nativeAssetsExpectedMatches = $preflightEntryFound
    nativeAssetsFoundMatches = $false
    ownerActionRequired = $true
    canPromotePackageConsumerRuntimeProof = $false
    boundary = "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source."
  }
  requiredEvidenceSummary = @(
    "This example intentionally remains non-proof.",
    "A real record must use recordKind=external-runtime-proof-record.",
    "A real record must set proofClassification=package-consumer-runtime before promotion.",
    "A real record must keep runtimePackageKey equal to the release target runtime package key.",
    "A real record must keep packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound aligned with RuntimeProofPreflight.",
    "A real record must include host CUDA/TensorRT/cuDNN versions and clean consumer project identity.",
    "A real record must report exitCode=0 and smokeStatus=passed.",
    "A real smokeCommand must include --runtime-package-key for the release target runtime package key.",
    "A real record must set isDependencyProbeOnly=false.",
    "A real record must set nativeAssetsCopied=true and include a real log path plus logSha256.",
    "This example keeps nativeAssetsFound below RuntimeProofPreflight and therefore remains non-promotable."
  )
  classificationRules = @($classificationRules)
  promotionRules = @(
    "Examples are not runtime proof.",
    "runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound must align with RuntimeProofPreflight.",
    "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.",
    "consumerProjectName and consumerProjectPath must identify the clean package consumer project.",
    "host CUDA runtime, TensorRT runtime, cuDNN runtime, driver, GPU, OS, and TensorRT line must be filled.",
    "smokeCommand must include --runtime-package-key for the release target runtime package key.",
    "logSha256 must be a 64-character SHA256 hash of the smoke log.",
    "proofClassification must be package-consumer-runtime before this record can be promoted.",
    "DependencyProbe-only evidence is not runtime execution proof.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Do not copy example values into release evidence."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-record.example.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-record.example.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Record Example")
$lines.Add("")
$lines.Add("This is an example only. It is not runtime proof, does not approve publication, and must not be copied into a release evidence bundle as proof.")
$lines.Add("")
$lines.Add("- record kind: ``external-runtime-proof-record-example``")
$lines.Add("- publication state: ``example-not-for-publication``")
$lines.Add("- proof classification: ``dependency-probe-only``")
$lines.Add("- runtime execution evidence: ``false``")
$lines.Add("- can promote runtime proof: ``false``")
$lines.Add("- smoke status: ``blocked-by-cuda-driver``")
$lines.Add("")
$lines.Add("## Runtime Proof Preflight")
$lines.Add("")
$lines.Add("- matrix found: ``$($record.runtimeProofPreflight.matrixFound)``")
$lines.Add("- entry found: ``$($record.runtimeProofPreflight.entryFound)``")
$lines.Add("- runtime package id: ``$($record.runtimeProofPreflight.runtimePackageId)``")
$lines.Add("- restore source mode: ``$($record.runtimeProofPreflight.restoreSourceMode)``")
$lines.Add("- native assets expected: ``$($record.runtimeProofPreflight.nativeAssetCopyExpected)``")
$lines.Add("- native assets found matches: ``$($record.runtimeProofPreflight.nativeAssetsFoundMatches)``")
$lines.Add("- can promote from preflight: ``$($record.runtimeProofPreflight.canPromotePackageConsumerRuntimeProof)``")
$lines.Add("")
$lines.Add("## Why This Example Is Blocked")
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
$lines.Add('```powershell')
$lines.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\final-release\external-runtime-proof-record.example.json")
$lines.Add('```')
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $record.promotionRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof record example written to $jsonPath"
Write-Host "External runtime proof record example written to $markdownPath"
