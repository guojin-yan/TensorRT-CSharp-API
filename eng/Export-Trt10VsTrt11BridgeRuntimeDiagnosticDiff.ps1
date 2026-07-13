[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrThrow {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required artifact '$RelativePath' was not found."
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
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Write-Utf8File {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject
  )

  $content = @($InputObject) -join [Environment]::NewLine
  [System.IO.File]::WriteAllText($LiteralPath, $content + [Environment]::NewLine, $script:utf8)
}

function Get-NativeAssetSummary {
  param([AllowNull()][object[]]$NativeAssets)

  $assets = @($NativeAssets)
  $bridge = $assets | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") -eq "jyppxtrtbridge.dll" } | Select-Object -First 1
  $nvinfer = @($assets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).StartsWith("nvinfer", [StringComparison]::OrdinalIgnoreCase) })
  $cudart = @($assets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).StartsWith("cudart", [StringComparison]::OrdinalIgnoreCase) })
  $cudnn = @($assets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).StartsWith("cudnn", [StringComparison]::OrdinalIgnoreCase) })

  return [pscustomobject]@{
    totalCount = $assets.Count
    bridgeSha256 = [string](Get-PropertyOrDefault -Object $bridge -Name "sha256" -DefaultValue "")
    bridgeLength = [int64](Get-PropertyOrDefault -Object $bridge -Name "length" -DefaultValue 0)
    nvinferCount = $nvinfer.Count
    cudartCount = $cudart.Count
    cudnnCount = $cudnn.Count
    cudnnFileVersions = @($cudnn | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "fileVersion" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
  }
}

function Get-ProofSummary {
  param(
    [string]$Label,
    [object]$Proof
  )

  $hostRecord = Get-PropertyOrDefault -Object $Proof -Name "host" -DefaultValue $null
  $roots = Get-PropertyOrDefault -Object $Proof -Name "roots" -DefaultValue $null
  $logs = Get-PropertyOrDefault -Object $Proof -Name "logs" -DefaultValue $null
  $nativeAssets = @((Get-PropertyOrDefault -Object $Proof -Name "nativeAssets" -DefaultValue @()))
  $assetSummary = Get-NativeAssetSummary -NativeAssets $nativeAssets

  return [pscustomobject]@{
    label = $Label
    sourceRuntimeKey = [string](Get-PropertyOrDefault -Object $Proof -Name "sourceRuntimeKey" -DefaultValue "")
    proofClassification = [string](Get-PropertyOrDefault -Object $Proof -Name "proofClassification" -DefaultValue "")
    smokeStatus = [string](Get-PropertyOrDefault -Object $Proof -Name "smokeStatus" -DefaultValue "")
    exitCode = [int](Get-PropertyOrDefault -Object $Proof -Name "exitCode" -DefaultValue -1)
    runtimeEnvironmentLine = [string](Get-PropertyOrDefault -Object $hostRecord -Name "runtimeEnvironmentLine" -DefaultValue "")
    gpuName = [string](Get-PropertyOrDefault -Object $hostRecord -Name "gpuName" -DefaultValue "")
    driverVersion = [string](Get-PropertyOrDefault -Object $hostRecord -Name "driverVersion" -DefaultValue "")
    tensorRtRoot = [string](Get-PropertyOrDefault -Object $roots -Name "tensorRtRoot" -DefaultValue "")
    cudaRoot = [string](Get-PropertyOrDefault -Object $roots -Name "cudaRoot" -DefaultValue "")
    cudnnRoot = [string](Get-PropertyOrDefault -Object $roots -Name "cudnnRoot" -DefaultValue "")
    searchDirectories = @((Get-PropertyOrDefault -Object $roots -Name "searchDirectories" -DefaultValue @()) | ForEach-Object { [string]$_ })
    searchDirectoryCount = @((Get-PropertyOrDefault -Object $roots -Name "searchDirectories" -DefaultValue @())).Count
    nativeAssets = $assetSummary
    stdoutSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "stdoutSha256" -DefaultValue "")
    stderrSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "stderrSha256" -DefaultValue "")
    combinedSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "combinedSha256" -DefaultValue "")
    stdoutSummary = @((Get-PropertyOrDefault -Object $logs -Name "stdoutSummary" -DefaultValue @()) | ForEach-Object { [string]$_ })
    stderrSummary = @((Get-PropertyOrDefault -Object $logs -Name "stderrSummary" -DefaultValue @()) | ForEach-Object { [string]$_ })
    runtimeCreateDiagnostic = Get-PropertyOrDefault -Object $Proof -Name "runtimeCreateDiagnostic" -DefaultValue $null
    cudaPreflight = Get-PropertyOrDefault -Object $Proof -Name "cudaPreflight" -DefaultValue $null
    enqueueCompleted = [bool](Get-PropertyOrDefault -Object $Proof -Name "enqueueCompleted" -DefaultValue $false)
    identityOutputMatch = [bool](Get-PropertyOrDefault -Object $Proof -Name "identityOutputMatch" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Proof -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $Proof -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
    canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Proof -Name "canPromoteRuntimeProof" -DefaultValue $false)
  }
}

function New-DiffItem {
  param(
    [string]$Field,
    [AllowNull()][object]$Trt10,
    [AllowNull()][object]$Trt11,
    [string]$Impact
  )

  [pscustomobject]@{
    field = $Field
    trt10 = $Trt10
    trt11 = $Trt11
    differs = ([string]$Trt10) -ne ([string]$Trt11)
    impact = $Impact
  }
}

$trt10RelativePath = "artifacts\package-consumer\bridge-runtime\win-x64-trt10.11-cuda12.9-cudnn9.22\bridge-package-runtime-consumer-proof.json"
$trt11RelativePath = "artifacts\package-consumer\bridge-runtime\win-x64-trt11.0-cuda13.2-cudnn9.22\bridge-package-runtime-consumer-proof.json"
$rootCauseRelativePath = "artifacts\final-release\trt11-runtime-smoke-root-cause-report.json"

$trt10Proof = Read-JsonOrThrow $trt10RelativePath
$trt11Proof = Read-JsonOrThrow $trt11RelativePath
$trt11RootCause = Read-JsonOrThrow $rootCauseRelativePath

$trt10 = Get-ProofSummary -Label "trt10-passed-bridge" -Proof $trt10Proof
$trt11 = Get-ProofSummary -Label "trt11-failed-bridge" -Proof $trt11Proof

$diffItems = @(
  New-DiffItem -Field "sourceRuntimeKey" -Trt10 $trt10.sourceRuntimeKey -Trt11 $trt11.sourceRuntimeKey -Impact "runtime line differs"
  New-DiffItem -Field "proofClassification" -Trt10 $trt10.proofClassification -Trt11 $trt11.proofClassification -Impact "TRT10 passed compatible-host bridge smoke while TRT11 failed"
  New-DiffItem -Field "smokeStatus" -Trt10 $trt10.smokeStatus -Trt11 $trt11.smokeStatus -Impact "TRT11 runtime smoke is the active blocker"
  New-DiffItem -Field "exitCode" -Trt10 $trt10.exitCode -Trt11 $trt11.exitCode -Impact "non-zero TRT11 exit blocks promotion"
  New-DiffItem -Field "runtimeEnvironmentLine" -Trt10 $trt10.runtimeEnvironmentLine -Trt11 $trt11.runtimeEnvironmentLine -Impact "both report TensorRT/CUDA available, so failure is after dependency preflight"
  New-DiffItem -Field "tensorRtRoot" -Trt10 $trt10.tensorRtRoot -Trt11 $trt11.tensorRtRoot -Impact "vendor TensorRT root differs"
  New-DiffItem -Field "cudaRoot" -Trt10 $trt10.cudaRoot -Trt11 $trt11.cudaRoot -Impact "CUDA major/minor differs"
  New-DiffItem -Field "cudnnRoot" -Trt10 $trt10.cudnnRoot -Trt11 $trt11.cudnnRoot -Impact "cuDNN package root differs"
  New-DiffItem -Field "searchDirectoryCount" -Trt10 $trt10.searchDirectoryCount -Trt11 $trt11.searchDirectoryCount -Impact "DLL search order differs and must be checked before rerun"
  New-DiffItem -Field "nativeAssetCount" -Trt10 $trt10.nativeAssets.totalCount -Trt11 $trt11.nativeAssets.totalCount -Impact "native dependency set differs"
  New-DiffItem -Field "bridgeSha256" -Trt10 $trt10.nativeAssets.bridgeSha256 -Trt11 $trt11.nativeAssets.bridgeSha256 -Impact "bridge binaries differ by runtime line"
  New-DiffItem -Field "cudnnAssetCount" -Trt10 $trt10.nativeAssets.cudnnCount -Trt11 $trt11.nativeAssets.cudnnCount -Impact "cuDNN asset bundle differs"
  New-DiffItem -Field "stdoutSha256" -Trt10 $trt10.stdoutSha256 -Trt11 $trt11.stdoutSha256 -Impact "stdout evidence differs"
  New-DiffItem -Field "stderrSha256" -Trt10 $trt10.stderrSha256 -Trt11 $trt11.stderrSha256 -Impact "TRT11 stderr contains failure signature"
  New-DiffItem -Field "runtimeCreateDiagnostic.available" -Trt10 ([bool](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "available" -DefaultValue $false)) -Trt11 ([bool](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "available" -DefaultValue $false)) -Impact "TRT11 native create-runtime diagnostic should be present after rebuilding the TRT11 bridge"
  New-DiffItem -Field "runtimeCreateDiagnostic.attempted" -Trt10 ([bool](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "attempted" -DefaultValue $false)) -Trt11 ([bool](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "attempted" -DefaultValue $false)) -Impact "shows whether createInferRuntime was reached before the failure"
  New-DiffItem -Field "runtimeCreateDiagnostic.returnedNull" -Trt10 ([bool](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "returnedNull" -DefaultValue $false)) -Trt11 ([bool](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "returnedNull" -DefaultValue $false)) -Impact "captures guarded native null-return evidence without exposing runtime pointers"
  New-DiffItem -Field "runtimeCreateDiagnostic.lastStatus" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "lastStatus" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "lastStatus" -DefaultValue "")) -Impact "captures native bridge status around createInferRuntime"
  New-DiffItem -Field "runtimeCreateDiagnostic.phase" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "phase" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "phase" -DefaultValue "")) -Impact "captures the native createInferRuntime diagnostic phase"
  New-DiffItem -Field "runtimeCreateDiagnostic.loggerMessageCount" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "loggerMessageCount" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "loggerMessageCount" -DefaultValue "")) -Impact "captures whether TensorRT emitted logger messages during runtime creation"
  New-DiffItem -Field "runtimeCreateDiagnostic.lastLoggerMessage" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.runtimeCreateDiagnostic -Name "lastLoggerMessage" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "lastLoggerMessage" -DefaultValue "")) -Impact "captures the copied TensorRT logger message without exposing logger pointers"
  New-DiffItem -Field "cudaPreflight.available" -Trt10 ([bool](Get-PropertyOrDefault -Object $trt10.cudaPreflight -Name "available" -DefaultValue $false)) -Trt11 ([bool](Get-PropertyOrDefault -Object $trt11.cudaPreflight -Name "available" -DefaultValue $false)) -Impact "captures whether CUDA preflight evidence was collected before TensorRT runtime creation"
  New-DiffItem -Field "cudaPreflight.driverVersion" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.cudaPreflight -Name "driverVersion" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.cudaPreflight -Name "driverVersion" -DefaultValue "")) -Impact "compares CUDA driver version evidence visible to each compatible-host consumer"
  New-DiffItem -Field "cudaPreflight.runtimeVersion" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.cudaPreflight -Name "runtimeVersion" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.cudaPreflight -Name "runtimeVersion" -DefaultValue "")) -Impact "compares CUDA runtime version evidence visible to each compatible-host consumer"
  New-DiffItem -Field "cudaPreflight.deviceCount" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.cudaPreflight -Name "deviceCount" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.cudaPreflight -Name "deviceCount" -DefaultValue "")) -Impact "captures visible CUDA device count before createInferRuntime"
  New-DiffItem -Field "cudaPreflight.initStatus" -Trt10 ([string](Get-PropertyOrDefault -Object $trt10.cudaPreflight -Name "initStatus" -DefaultValue "")) -Trt11 ([string](Get-PropertyOrDefault -Object $trt11.cudaPreflight -Name "initStatus" -DefaultValue "")) -Impact "captures CUDA device metadata initialization status before createInferRuntime"
  New-DiffItem -Field "canPromoteRuntimeProof" -Trt10 $trt10.canPromoteRuntimeProof -Trt11 $trt11.canPromoteRuntimeProof -Impact "neither bridge-only line is public clean package-consumer proof"
)

$trt11FailureSignature = [string](Get-PropertyOrDefault -Object $trt11RootCause -Name "failureSignature" -DefaultValue "")
$trt11RootCauseCategory = [string](Get-PropertyOrDefault -Object $trt11RootCause -Name "rootCauseCategory" -DefaultValue "")
$trt11RootCauseSubcategory = [string](Get-PropertyOrDefault -Object $trt11RootCause -Name "rootCauseSubcategory" -DefaultValue "")
$trt11NativeCreateRuntimePhase = [string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "phase" -DefaultValue "")
$trt11NativeCreateRuntimeLoggerMessageCount = [string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "loggerMessageCount" -DefaultValue "")
$trt11NativeCreateRuntimeLastLoggerMessage = [string](Get-PropertyOrDefault -Object $trt11.runtimeCreateDiagnostic -Name "lastLoggerMessage" -DefaultValue "")
$trt11CudaPreflight = $trt11.cudaPreflight
$trt11CudaPreflightAvailable = [bool](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "available" -DefaultValue $false)
$trt11CudaPreflightAttempted = [bool](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "attempted" -DefaultValue $false)
$trt11CudaPreflightDriverVersion = [string](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "driverVersion" -DefaultValue "")
$trt11CudaPreflightRuntimeVersion = [string](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "runtimeVersion" -DefaultValue "")
$trt11CudaPreflightDeviceCount = [string](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "deviceCount" -DefaultValue "")
$trt11CudaPreflightInitStatus = [string](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "initStatus" -DefaultValue "")
$trt11CudaPreflightCanAttemptTensorRtRuntimeCreate = [bool](Get-PropertyOrDefault -Object $trt11CudaPreflight -Name "canAttemptTensorRtRuntimeCreate" -DefaultValue $false)

$record = [pscustomobject]@{
  schemaVersion = 1
  recordKind = "trt10-vs-trt11-bridge-runtime-diagnostic-diff"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reportState = "diagnostic-diff-ready-non-proof"
  trt10 = $trt10
  trt11 = $trt11
  trt11FailureSignature = $trt11FailureSignature
  trt11RootCauseCategory = $trt11RootCauseCategory
  trt11RootCauseSubcategory = $trt11RootCauseSubcategory
  trt11RuntimeCreateDiagnostic = $trt11.runtimeCreateDiagnostic
  trt11NativeCreateRuntimePhase = $trt11NativeCreateRuntimePhase
  trt11NativeCreateRuntimeLoggerMessageCount = $trt11NativeCreateRuntimeLoggerMessageCount
  trt11NativeCreateRuntimeLastLoggerMessage = $trt11NativeCreateRuntimeLastLoggerMessage
  trt11CudaPreflight = $trt11CudaPreflight
  trt11CudaPreflightAvailable = $trt11CudaPreflightAvailable
  trt11CudaPreflightAttempted = $trt11CudaPreflightAttempted
  trt11CudaPreflightDriverVersion = $trt11CudaPreflightDriverVersion
  trt11CudaPreflightRuntimeVersion = $trt11CudaPreflightRuntimeVersion
  trt11CudaPreflightDeviceCount = $trt11CudaPreflightDeviceCount
  trt11CudaPreflightInitStatus = $trt11CudaPreflightInitStatus
  trt11CudaPreflightCanAttemptTensorRtRuntimeCreate = $trt11CudaPreflightCanAttemptTensorRtRuntimeCreate
  trt10SmokePassed = $trt10.smokeStatus -eq "passed"
  trt11SmokeFailed = $trt11.smokeStatus -eq "failed"
  runtimeEnvironmentBothAvailable = $trt10.runtimeEnvironmentLine.Contains("TensorRtAvailable=True", [StringComparison]::OrdinalIgnoreCase) -and $trt11.runtimeEnvironmentLine.Contains("TensorRtAvailable=True", [StringComparison]::OrdinalIgnoreCase) -and $trt10.runtimeEnvironmentLine.Contains("CudaAvailable=True", [StringComparison]::OrdinalIgnoreCase) -and $trt11.runtimeEnvironmentLine.Contains("CudaAvailable=True", [StringComparison]::OrdinalIgnoreCase)
  differingFieldCount = @($diffItems | Where-Object { $_.differs }).Count
  diffItems = $diffItems
  nextDiagnosticCommands = @(
    "cmake --preset win-x64-trt11-cuda13-release",
    "cmake --build --preset win-x64-trt11-cuda13-release --parallel",
    "Capture cudaGetDeviceCount/cudaFree(0) status in the package consumer immediately before TensorRT createInferRuntime.",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageRuntimeConsumer.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RuntimeSmoke",
    "Capture native loader diagnostics and TensorRT logger output around createInferRuntime before promoting any TRT11 runtime proof."
  )
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  sourceArtifacts = @(
    ($trt10RelativePath -replace "\\", "/"),
    ($trt11RelativePath -replace "\\", "/"),
    ($rootCauseRelativePath -replace "\\", "/")
  )
  boundary = "This TRT10/TRT11 bridge diagnostic diff compares a passing TRT10 compatible-host bridge smoke with a failing TRT11 compatible-host bridge smoke. It is diagnostic blocker evidence only, not public clean package-consumer proof, not post-publish proof, and not release approval."
}

$jsonPath = Join-Path $OutputRoot "trt10-vs-trt11-bridge-runtime-diagnostic-diff.json"
$markdownPath = Join-Path $OutputRoot "trt10-vs-trt11-bridge-runtime-diagnostic-diff.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# TRT10 vs TRT11 Bridge Runtime Diagnostic Diff")
$lines.Add("")
$lines.Add("| Field | TRT10 | TRT11 | Impact |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $diffItems) {
  $lines.Add("| ``$($item.field)`` | $(ConvertTo-MarkdownCell $item.trt10) | $(ConvertTo-MarkdownCell $item.trt11) | $(ConvertTo-MarkdownCell $item.impact) |")
}
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- reportState: ``$($record.reportState)``")
$lines.Add("- TRT10 smoke passed: ``$($record.trt10SmokePassed)``")
$lines.Add("- TRT11 smoke failed: ``$($record.trt11SmokeFailed)``")
$lines.Add("- TRT11 failure signature: ``$trt11FailureSignature``")
$lines.Add("- TRT11 root-cause category: ``$trt11RootCauseCategory``")
$lines.Add("- TRT11 root-cause subcategory: ``$trt11RootCauseSubcategory``")
$lines.Add("- TRT11 native create-runtime phase: ``$(ConvertTo-MarkdownCell $trt11NativeCreateRuntimePhase)``")
$lines.Add("- TRT11 native logger message count: ``$(ConvertTo-MarkdownCell $trt11NativeCreateRuntimeLoggerMessageCount)``")
$lines.Add("- TRT11 native last logger message: ``$(ConvertTo-MarkdownCell $trt11NativeCreateRuntimeLastLoggerMessage)``")
$lines.Add("- TRT11 CUDA preflight: ``available=$trt11CudaPreflightAvailable; attempted=$trt11CudaPreflightAttempted; driver=$trt11CudaPreflightDriverVersion; runtime=$trt11CudaPreflightRuntimeVersion; deviceCount=$trt11CudaPreflightDeviceCount; init=$trt11CudaPreflightInitStatus; canAttemptTensorRT=$trt11CudaPreflightCanAttemptTensorRtRuntimeCreate``")
$lines.Add("- differing field count: ``$($record.differingFieldCount)``")
$lines.Add("")
$lines.Add("## Next Diagnostic Commands")
$lines.Add("")
foreach ($command in $record.nextDiagnosticCommands) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $command)``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8File -LiteralPath $markdownPath -InputObject $lines

Write-Host "TRT10 vs TRT11 bridge runtime diagnostic diff written to $jsonPath"
Write-Host "TRT10 vs TRT11 bridge runtime diagnostic diff written to $markdownPath"
