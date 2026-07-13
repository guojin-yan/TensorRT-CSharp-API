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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Read-LinesOrEmpty {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return @()
  }

  return @(Get-Content -LiteralPath $path -Encoding utf8)
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

$runtimeKey = "win-x64-trt11.0-cuda13.2-cudnn9.22"
$proofRelativePath = "artifacts\package-consumer\bridge-runtime\$runtimeKey\bridge-package-runtime-consumer-proof.json"
$stdoutRelativePath = "artifacts\package-consumer\bridge-runtime\$runtimeKey\runtime-smoke.stdout.log"
$stderrRelativePath = "artifacts\package-consumer\bridge-runtime\$runtimeKey\runtime-smoke.stderr.log"
$combinedRelativePath = "artifacts\package-consumer\bridge-runtime\$runtimeKey\runtime-smoke.combined.log"

$proof = Read-JsonOrNull $proofRelativePath
$stdoutLines = Read-LinesOrEmpty $stdoutRelativePath
$stderrLines = Read-LinesOrEmpty $stderrRelativePath

$proofClassification = [string](Get-PropertyOrDefault -Object $proof -Name "proofClassification" -DefaultValue "missing-trt11-bridge-runtime-proof")
$smokeStatus = [string](Get-PropertyOrDefault -Object $proof -Name "smokeStatus" -DefaultValue "missing-smoke-status")
$exitCode = [int](Get-PropertyOrDefault -Object $proof -Name "exitCode" -DefaultValue -1)
$sourceRuntimeKey = [string](Get-PropertyOrDefault -Object $proof -Name "sourceRuntimeKey" -DefaultValue $runtimeKey)
$canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $proof -Name "canPromoteRuntimeProof" -DefaultValue $false)
$isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $proof -Name "isRuntimeExecutionProof" -DefaultValue $false)
$isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $proof -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)

$runtimeEnvironmentLine = [string](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $proof -Name "host" -DefaultValue $null) -Name "runtimeEnvironmentLine" -DefaultValue "")
$tensorRtAvailable = $runtimeEnvironmentLine.Contains("TensorRtAvailable=True", [StringComparison]::OrdinalIgnoreCase)
$cudaAvailable = $runtimeEnvironmentLine.Contains("CudaAvailable=True", [StringComparison]::OrdinalIgnoreCase)
$nativeAssets = @((Get-PropertyOrDefault -Object $proof -Name "nativeAssets" -DefaultValue @()))
$nativeBridgePresent = @($nativeAssets | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "") -eq "jyppxtrtbridge.dll" }).Count -gt 0
$cudnnAssetCount = @($nativeAssets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).StartsWith("cudnn", [StringComparison]::OrdinalIgnoreCase) }).Count
$searchDirectories = @((Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $proof -Name "roots" -DefaultValue $null) -Name "searchDirectories" -DefaultValue @()) | ForEach-Object { [string]$_ })
$runtimeCreateDiagnostic = Get-PropertyOrDefault -Object $proof -Name "runtimeCreateDiagnostic" -DefaultValue $null
$cudaPreflight = Get-PropertyOrDefault -Object $proof -Name "cudaPreflight" -DefaultValue $null
$cudaPreflightAvailable = [bool](Get-PropertyOrDefault -Object $cudaPreflight -Name "available" -DefaultValue $false)
$cudaPreflightAttempted = [bool](Get-PropertyOrDefault -Object $cudaPreflight -Name "attempted" -DefaultValue $false)
$cudaPreflightDriverVersion = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "driverVersion" -DefaultValue "")
$cudaPreflightRuntimeVersion = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "runtimeVersion" -DefaultValue "")
$cudaPreflightDeviceCount = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "deviceCount" -DefaultValue "")
$cudaPreflightSelectedDevice = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "selectedDevice" -DefaultValue "")
$cudaPreflightDeviceName = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "deviceName" -DefaultValue "")
$cudaPreflightGetDeviceCountStatus = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "getDeviceCountStatus" -DefaultValue "")
$cudaPreflightInitStatus = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "initStatus" -DefaultValue "")
$cudaPreflightLastErrorName = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "lastErrorName" -DefaultValue "")
$cudaPreflightLastErrorMessage = [string](Get-PropertyOrDefault -Object $cudaPreflight -Name "lastErrorMessage" -DefaultValue "")
$cudaPreflightCanAttemptTensorRtRuntimeCreate = [bool](Get-PropertyOrDefault -Object $cudaPreflight -Name "canAttemptTensorRtRuntimeCreate" -DefaultValue $false)
$nativeCreateRuntimeDiagnosticAvailable = [bool](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "available" -DefaultValue $false)
$nativeCreateRuntimeAttempted = [bool](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "attempted" -DefaultValue $false)
$nativeCreateRuntimeReturnedNull = [bool](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "returnedNull" -DefaultValue $false)
$nativeCreateRuntimeReturnedNonNull = [bool](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "returnedNonNull" -DefaultValue $false)
$nativeCreateRuntimeLastStatus = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastStatus" -DefaultValue "")
$nativeCreateRuntimeDetectedVersion = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "detectedVersion" -DefaultValue "")
$nativeCreateRuntimeLoggerCallbackAvailable = [bool](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "loggerCallbackAvailable" -DefaultValue $false)
$nativeCreateRuntimeLoggerMessageCount = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "loggerMessageCount" -DefaultValue "")
$nativeCreateRuntimeLastLoggerSeverity = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastLoggerSeverity" -DefaultValue "")
$nativeCreateRuntimeLastLoggerMessage = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastLoggerMessage" -DefaultValue "")
$nativeCreateRuntimePhase = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "phase" -DefaultValue "")
$nativeCreateRuntimeNativeDetail = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "nativeDetail" -DefaultValue "")
$nativeCreateRuntimeDiagnosticMessage = [string](Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "diagnosticMessage" -DefaultValue "")

$stderrText = ($stderrLines -join [Environment]::NewLine)
$failureSignature = if ($stderrText.Contains("createInferRuntime returned a null TensorRT object", [StringComparison]::OrdinalIgnoreCase)) {
  "createInferRuntime-null"
}
elseif ($smokeStatus -eq "passed") {
  "none"
}
elseif ($stderrLines.Count -gt 0) {
  "runtime-smoke-failed-other"
}
else {
  "missing-runtime-smoke-stderr"
}

$hasNativeCudaRuntimeLoggerError =
  $nativeCreateRuntimeLastLoggerMessage.Contains("Cuda Runtime", [StringComparison]::OrdinalIgnoreCase) -or
  $nativeCreateRuntimeLastLoggerMessage.Contains("catchCudaError", [StringComparison]::OrdinalIgnoreCase)

$rootCauseCategory = if ($failureSignature -eq "createInferRuntime-null" -and $tensorRtAvailable -and $cudaAvailable -and $hasNativeCudaRuntimeLoggerError) {
  "trt11-create-runtime-null-cuda-runtime-error"
}
elseif ($failureSignature -eq "createInferRuntime-null" -and $tensorRtAvailable -and $cudaAvailable) {
  "trt11-create-runtime-null-after-dependency-preflight"
}
elseif (-not $tensorRtAvailable -or -not $cudaAvailable) {
  "dependency-preflight-not-ready"
}
elseif ($smokeStatus -eq "passed") {
  "no-runtime-smoke-failure"
}
else {
  "runtime-smoke-failed-unclassified"
}

$rootCauseSubcategory = if ($rootCauseCategory -ne "trt11-create-runtime-null-cuda-runtime-error") {
  "not-cuda-runtime-error"
}
elseif (-not $cudaPreflightAttempted) {
  "cuda-preflight-unavailable"
}
elseif ($cudaPreflightGetDeviceCountStatus -eq "Failed" -or $cudaPreflightInitStatus -eq "CudaPreflightFailed") {
  "cuda-preflight-failed"
}
elseif ($cudaPreflightDeviceCount -eq "0" -or $cudaPreflightInitStatus -eq "NoDevice") {
  "cuda-no-device"
}
elseif ($cudaPreflightInitStatus -eq "DeviceInfoFailed") {
  "cuda-device-init-failed"
}
elseif ($cudaPreflightAvailable -and $cudaPreflightCanAttemptTensorRtRuntimeCreate) {
  "cuda-runtime-error-after-device-preflight"
}
else {
  "cuda-runtime-error-unknown"
}

$reportState = if ($rootCauseCategory -eq "trt11-create-runtime-null-cuda-runtime-error" -or $rootCauseCategory -eq "trt11-create-runtime-null-after-dependency-preflight") {
  "classified-owner-action-required"
}
elseif ($rootCauseCategory -eq "no-runtime-smoke-failure") {
  "no-blocker-detected"
}
else {
  "blocked-root-cause-followup-required"
}

$classificationRationale = @(
  "TRT11 bridge package restore/build reached runtime smoke and exited with code $exitCode."
  "Dependency preflight reported TensorRtAvailable=$tensorRtAvailable and CudaAvailable=$cudaAvailable via runtimeEnvironmentLine."
  "Native bridge asset present=$nativeBridgePresent; cuDNN 9 asset count=$cudnnAssetCount; search directory count=$($searchDirectories.Count)."
  "CUDA preflight available=$cudaPreflightAvailable; attempted=$cudaPreflightAttempted; driverVersion=$cudaPreflightDriverVersion; runtimeVersion=$cudaPreflightRuntimeVersion; deviceCount=$cudaPreflightDeviceCount; selectedDevice=$cudaPreflightSelectedDevice; initStatus=$cudaPreflightInitStatus; canAttemptTensorRtRuntimeCreate=$cudaPreflightCanAttemptTensorRtRuntimeCreate."
  "The failing stack terminates at NativeBridgeApi.CreateRuntime/TensorRtRuntime constructor with createInferRuntime returning null."
  "Native create-runtime diagnostic available=$nativeCreateRuntimeDiagnosticAvailable; attempted=$nativeCreateRuntimeAttempted; returnedNull=$nativeCreateRuntimeReturnedNull; lastStatus=$nativeCreateRuntimeLastStatus."
  "Native create-runtime phase=$nativeCreateRuntimePhase; loggerCallbackAvailable=$nativeCreateRuntimeLoggerCallbackAvailable; loggerMessageCount=$nativeCreateRuntimeLoggerMessageCount; lastLoggerSeverity=$nativeCreateRuntimeLastLoggerSeverity."
  "Native create-runtime logger last message=$nativeCreateRuntimeLastLoggerMessage; nativeDetail=$nativeCreateRuntimeNativeDetail."
  "This report is diagnostic blocker evidence only and cannot promote runtime proof while smokeStatus is not passed."
)

$recommendedNextActions = @(
  "Run the TRT11 bridge smoke without AllowRuntimeSmokeFailure on the same host after checking CUDA runtime/device/driver initialization state.",
  "Use NativeCreateRuntimeDiagnostic* stdout markers to distinguish logger validation, logger callback/vendor messages, TensorRT availability, guarded native status, and null runtime return.",
  "Add or run a package-consumer preflight that captures cudaGetDeviceCount/cudaFree(0) status before TensorRT createInferRuntime.",
  "Capture loader diagnostics for TensorRT 11, CUDA 13.2, cuDNN 9.22, and any plugin DLLs resolved before createInferRuntime.",
  "Compare PATH/searchDirectories against the TRT10 passing bridge smoke and remove stale TensorRT/CUDA/cuDNN directories.",
  "Rebuild the TRT11 native bridge with cmake --preset win-x64-trt11-cuda13-release and rerun the bridge package consumer smoke."
)

$record = [pscustomobject]@{
  schemaVersion = 1
  recordKind = "trt11-runtime-smoke-root-cause-report"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reportState = $reportState
  sourceRuntimeKey = $sourceRuntimeKey
  proofClassification = $proofClassification
  smokeStatus = $smokeStatus
  exitCode = $exitCode
  failureSignature = $failureSignature
  rootCauseCategory = $rootCauseCategory
  rootCauseSubcategory = $rootCauseSubcategory
  tensorRtAvailable = $tensorRtAvailable
  cudaAvailable = $cudaAvailable
  nativeBridgePresent = $nativeBridgePresent
  cudnnAssetCount = $cudnnAssetCount
  searchDirectoryCount = $searchDirectories.Count
  runtimeEnvironmentLine = $runtimeEnvironmentLine
  cudaPreflight = $cudaPreflight
  cudaPreflightAvailable = $cudaPreflightAvailable
  cudaPreflightAttempted = $cudaPreflightAttempted
  cudaPreflightDriverVersion = $cudaPreflightDriverVersion
  cudaPreflightRuntimeVersion = $cudaPreflightRuntimeVersion
  cudaPreflightDeviceCount = $cudaPreflightDeviceCount
  cudaPreflightSelectedDevice = $cudaPreflightSelectedDevice
  cudaPreflightDeviceName = $cudaPreflightDeviceName
  cudaPreflightGetDeviceCountStatus = $cudaPreflightGetDeviceCountStatus
  cudaPreflightInitStatus = $cudaPreflightInitStatus
  cudaPreflightLastErrorName = $cudaPreflightLastErrorName
  cudaPreflightLastErrorMessage = $cudaPreflightLastErrorMessage
  cudaPreflightCanAttemptTensorRtRuntimeCreate = $cudaPreflightCanAttemptTensorRtRuntimeCreate
  runtimeCreateDiagnostic = $runtimeCreateDiagnostic
  nativeCreateRuntimeDiagnosticAvailable = $nativeCreateRuntimeDiagnosticAvailable
  nativeCreateRuntimeAttempted = $nativeCreateRuntimeAttempted
  nativeCreateRuntimeReturnedNull = $nativeCreateRuntimeReturnedNull
  nativeCreateRuntimeReturnedNonNull = $nativeCreateRuntimeReturnedNonNull
  nativeCreateRuntimeLastStatus = $nativeCreateRuntimeLastStatus
  nativeCreateRuntimeDetectedVersion = $nativeCreateRuntimeDetectedVersion
  nativeCreateRuntimeLoggerCallbackAvailable = $nativeCreateRuntimeLoggerCallbackAvailable
  nativeCreateRuntimeLoggerMessageCount = $nativeCreateRuntimeLoggerMessageCount
  nativeCreateRuntimeLastLoggerSeverity = $nativeCreateRuntimeLastLoggerSeverity
  nativeCreateRuntimeLastLoggerMessage = $nativeCreateRuntimeLastLoggerMessage
  nativeCreateRuntimePhase = $nativeCreateRuntimePhase
  nativeCreateRuntimeNativeDetail = $nativeCreateRuntimeNativeDetail
  nativeCreateRuntimeDiagnosticMessage = $nativeCreateRuntimeDiagnosticMessage
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  upstreamCanPromoteRuntimeProof = $canPromoteRuntimeProof
  upstreamIsRuntimeExecutionProof = $isRuntimeExecutionProof
  upstreamIsPackageConsumerRuntimeProof = $isPackageConsumerRuntimeProof
  stdoutSummary = @($stdoutLines | Select-Object -First 20)
  stderrSummary = @($stderrLines | Select-Object -First 20)
  classificationRationale = $classificationRationale
  recommendedNextActions = $recommendedNextActions
  sourceArtifacts = @(
    ($proofRelativePath -replace "\\", "/"),
    ($stdoutRelativePath -replace "\\", "/"),
    ($stderrRelativePath -replace "\\", "/"),
    ($combinedRelativePath -replace "\\", "/"),
    "eng/Test-BridgePackageRuntimeConsumer.ps1",
    "eng/Resolve-RuntimeRoots.ps1"
  )
  boundary = "TRT11 root-cause report is diagnostic blocker evidence only. It does not run runtime smoke, does not publish packages, does not approve release close, and cannot promote runtime proof while smokeStatus is not passed."
}

$jsonPath = Join-Path $OutputRoot "trt11-runtime-smoke-root-cause-report.json"
$markdownPath = Join-Path $OutputRoot "trt11-runtime-smoke-root-cause-report.md"

$json = $record | ConvertTo-Json -Depth 16
Write-Utf8File -LiteralPath $jsonPath -InputObject $json

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# TRT11 Runtime Smoke Root Cause Report")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| reportState | ``$($record.reportState)`` |")
$lines.Add("| sourceRuntimeKey | ``$($record.sourceRuntimeKey)`` |")
$lines.Add("| proofClassification | ``$($record.proofClassification)`` |")
$lines.Add("| smokeStatus | ``$($record.smokeStatus)`` |")
$lines.Add("| exitCode | ``$($record.exitCode)`` |")
$lines.Add("| failureSignature | ``$($record.failureSignature)`` |")
$lines.Add("| rootCauseCategory | ``$($record.rootCauseCategory)`` |")
$lines.Add("| rootCauseSubcategory | ``$($record.rootCauseSubcategory)`` |")
$lines.Add("| TensorRT available | ``$($record.tensorRtAvailable)`` |")
$lines.Add("| CUDA available | ``$($record.cudaAvailable)`` |")
$lines.Add("| CUDA preflight available | ``$($record.cudaPreflightAvailable)`` |")
$lines.Add("| CUDA preflight attempted | ``$($record.cudaPreflightAttempted)`` |")
$lines.Add("| CUDA preflight driver/runtime | ``$(ConvertTo-MarkdownCell $record.cudaPreflightDriverVersion)/$(ConvertTo-MarkdownCell $record.cudaPreflightRuntimeVersion)`` |")
$lines.Add("| CUDA preflight device | ``count=$(ConvertTo-MarkdownCell $record.cudaPreflightDeviceCount); selected=$(ConvertTo-MarkdownCell $record.cudaPreflightSelectedDevice); name=$(ConvertTo-MarkdownCell $record.cudaPreflightDeviceName)`` |")
$lines.Add("| CUDA preflight status | ``getDeviceCount=$(ConvertTo-MarkdownCell $record.cudaPreflightGetDeviceCountStatus); init=$(ConvertTo-MarkdownCell $record.cudaPreflightInitStatus); canAttemptTensorRT=$(ConvertTo-MarkdownCell $record.cudaPreflightCanAttemptTensorRtRuntimeCreate)`` |")
$lines.Add("| native create-runtime diagnostic available | ``$($record.nativeCreateRuntimeDiagnosticAvailable)`` |")
$lines.Add("| native create-runtime attempted | ``$($record.nativeCreateRuntimeAttempted)`` |")
$lines.Add("| native create-runtime returned null | ``$($record.nativeCreateRuntimeReturnedNull)`` |")
$lines.Add("| native create-runtime last status | ``$($record.nativeCreateRuntimeLastStatus)`` |")
$lines.Add("| native create-runtime detected version | ``$(ConvertTo-MarkdownCell $record.nativeCreateRuntimeDetectedVersion)`` |")
$lines.Add("| native create-runtime phase | ``$(ConvertTo-MarkdownCell $record.nativeCreateRuntimePhase)`` |")
$lines.Add("| native logger callback available | ``$($record.nativeCreateRuntimeLoggerCallbackAvailable)`` |")
$lines.Add("| native logger message count | ``$(ConvertTo-MarkdownCell $record.nativeCreateRuntimeLoggerMessageCount)`` |")
$lines.Add("| native last logger severity | ``$(ConvertTo-MarkdownCell $record.nativeCreateRuntimeLastLoggerSeverity)`` |")
$lines.Add("| native last logger message | ``$(ConvertTo-MarkdownCell $record.nativeCreateRuntimeLastLoggerMessage)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("")
$lines.Add("## Classification Rationale")
$lines.Add("")
foreach ($item in $classificationRationale) {
  $lines.Add("- $(ConvertTo-MarkdownCell $item)")
}
$lines.Add("")
$lines.Add("## Recommended Next Actions")
$lines.Add("")
foreach ($item in $recommendedNextActions) {
  $lines.Add("- $(ConvertTo-MarkdownCell $item)")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8File -LiteralPath $markdownPath -InputObject $lines

Write-Host "TRT11 runtime smoke root cause report written to $jsonPath"
Write-Host "TRT11 runtime smoke root cause report written to $markdownPath"
