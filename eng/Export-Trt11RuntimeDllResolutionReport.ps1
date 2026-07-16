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

function Get-FileEvidence {
  param(
    [string]$Name,
    [AllowNull()][string]$Path,
    [AllowNull()][object]$NativeAsset
  )

  $currentPathExists = -not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path -PathType Leaf)
  $length = [int64](Get-PropertyOrDefault -Object $NativeAsset -Name "length" -DefaultValue 0)
  $sha256 = [string](Get-PropertyOrDefault -Object $NativeAsset -Name "sha256" -DefaultValue "")
  $fileVersion = [string](Get-PropertyOrDefault -Object $NativeAsset -Name "fileVersion" -DefaultValue "")
  $presentAtCapture = $null -ne $NativeAsset -and $length -gt 0 -and -not [string]::IsNullOrWhiteSpace($sha256)
  $exists = $currentPathExists -or $presentAtCapture

  if ($currentPathExists) {
    $item = Get-Item -LiteralPath $Path
    $length = [int64]$item.Length
    if ([string]::IsNullOrWhiteSpace($sha256)) {
      $sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
    if ([string]::IsNullOrWhiteSpace($fileVersion)) {
      $fileVersion = [string]$item.VersionInfo.FileVersion
    }
  }

  return [pscustomobject]@{
    name = $Name
    path = $Path
    exists = $exists
    presentAtCapture = $presentAtCapture
    currentPathExists = $currentPathExists
    length = $length
    sha256 = $sha256
    fileVersion = $fileVersion
  }
}

function Find-FirstFile {
  param(
    [string[]]$Directories,
    [string[]]$Names
  )

  foreach ($directory in $Directories) {
    if ([string]::IsNullOrWhiteSpace($directory) -or -not (Test-Path -LiteralPath $directory -PathType Container)) {
      continue
    }

    foreach ($name in $Names) {
      $candidate = Join-Path $directory $name
      if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return $candidate
      }
    }
  }

  return ""
}

function Find-DllOccurrences {
  param(
    [string[]]$Directories,
    [string[]]$Names
  )

  $items = @()
  foreach ($directory in $Directories) {
    if ([string]::IsNullOrWhiteSpace($directory) -or -not (Test-Path -LiteralPath $directory -PathType Container)) {
      continue
    }

    foreach ($name in $Names) {
      $candidate = Join-Path $directory $name
      if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $file = Get-Item -LiteralPath $candidate
        $items += [pscustomobject]@{
          name = $name
          directory = $directory
          path = $file.FullName
          length = [int64]$file.Length
          fileVersion = [string]$file.VersionInfo.FileVersion
        }
      }
    }
  }

  return $items
}

$runtimeKey = "win-x64-trt11.0-cuda13.2-cudnn9.22"
$proofRelativePath = "artifacts\package-consumer\bridge-runtime\$runtimeKey\bridge-package-runtime-consumer-proof.json"
$rootCauseRelativePath = "artifacts\final-release\trt11-runtime-smoke-root-cause-report.json"
$diffRelativePath = "artifacts\final-release\trt10-vs-trt11-bridge-runtime-diagnostic-diff.json"

$proof = Read-JsonOrNull $proofRelativePath
$rootCause = Read-JsonOrNull $rootCauseRelativePath
$diff = Read-JsonOrNull $diffRelativePath

$roots = Get-PropertyOrDefault -Object $proof -Name "roots" -DefaultValue $null
$hostRecord = Get-PropertyOrDefault -Object $proof -Name "host" -DefaultValue $null
$logs = Get-PropertyOrDefault -Object $proof -Name "logs" -DefaultValue $null
$nativeAssets = @((Get-PropertyOrDefault -Object $proof -Name "nativeAssets" -DefaultValue @()))
$searchDirectories = @((Get-PropertyOrDefault -Object $roots -Name "searchDirectories" -DefaultValue @()) | ForEach-Object { [string]$_ })
$pathDirectories = @(([string]($env:PATH)).Split([System.IO.Path]::PathSeparator) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$resolutionDirectories = @($searchDirectories + $pathDirectories | Select-Object -Unique)

$bridgeAsset = $nativeAssets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).Equals("jyppxtrtbridge.dll", [StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
$bridgePath = [string](Get-PropertyOrDefault -Object $bridgeAsset -Name "path" -DefaultValue "")

$requiredDllGroups = @(
  [pscustomobject]@{ key = "bridge"; names = @("jyppxtrtbridge.dll"); required = $true; category = "project-native-bridge" },
  [pscustomobject]@{ key = "tensorrt-runtime"; names = @("nvinfer_11.dll", "nvinfer.dll"); required = $true; category = "tensorrt" },
  [pscustomobject]@{ key = "tensorrt-plugin"; names = @("nvinfer_plugin_11.dll", "nvinfer_plugin.dll"); required = $true; category = "tensorrt" },
  [pscustomobject]@{ key = "tensorrt-lean"; names = @("nvinfer_lean_11.dll", "nvinfer_lean.dll"); required = $false; category = "tensorrt" },
  [pscustomobject]@{ key = "tensorrt-dispatch"; names = @("nvinfer_dispatch_11.dll", "nvinfer_dispatch.dll"); required = $false; category = "tensorrt" },
  [pscustomobject]@{ key = "cuda-runtime"; names = @("cudart64_13.dll", "cudart64_130.dll", "cudart64_12.dll"); required = $true; category = "cuda" },
  [pscustomobject]@{ key = "cublas"; names = @("cublas64_13.dll", "cublas64_12.dll"); required = $false; category = "cuda" },
  [pscustomobject]@{ key = "cublaslt"; names = @("cublasLt64_13.dll", "cublasLt64_12.dll"); required = $false; category = "cuda" },
  [pscustomobject]@{ key = "cudnn-runtime"; names = @("cudnn64_9.dll"); required = $true; category = "cudnn" },
  [pscustomobject]@{ key = "cudnn-ops"; names = @("cudnn_ops64_9.dll"); required = $true; category = "cudnn" },
  [pscustomobject]@{ key = "cudnn-adv"; names = @("cudnn_adv64_9.dll"); required = $false; category = "cudnn" },
  [pscustomobject]@{ key = "cudnn-cnn"; names = @("cudnn_cnn64_9.dll"); required = $false; category = "cudnn" }
)

$dllResolution = foreach ($group in $requiredDllGroups) {
  $nativeAsset = $nativeAssets | Where-Object {
    $assetName = [string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")
    @($group.names) -contains $assetName
  } | Select-Object -First 1

  $resolvedPath = [string](Get-PropertyOrDefault -Object $nativeAsset -Name "path" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($resolvedPath)) {
    $resolvedPath = Find-FirstFile -Directories $resolutionDirectories -Names @($group.names)
  }

  $occurrences = @(Find-DllOccurrences -Directories $resolutionDirectories -Names @($group.names))
  $fileEvidence = Get-FileEvidence -Name ([string]$group.key) -Path $resolvedPath -NativeAsset $nativeAsset

  [pscustomobject]@{
    key = [string]$group.key
    category = [string]$group.category
    names = @($group.names)
    required = [bool]$group.required
    resolvedPath = $resolvedPath
    found = [bool]$fileEvidence.exists
    presentAtCapture = [bool]$fileEvidence.presentAtCapture
    currentPathExists = [bool]$fileEvidence.currentPathExists
    length = [int64]$fileEvidence.length
    sha256 = [string]$fileEvidence.sha256
    fileVersion = [string]$fileEvidence.fileVersion
    occurrenceCount = $occurrences.Count
    duplicateAcrossSearchOrPath = $occurrences.Count -gt 1
    occurrences = @($occurrences)
  }
}

$missingRequired = @($dllResolution | Where-Object { $_.required -and -not $_.found })
$duplicateGroups = @($dllResolution | Where-Object { $_.duplicateAcrossSearchOrPath })
$bridgeResolution = $dllResolution | Where-Object { $_.key -eq "bridge" } | Select-Object -First 1
$nativeBridgePresent = $null -ne $bridgeResolution -and [bool]$bridgeResolution.found
$cudnnAssetCount = @($nativeAssets | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "name" -DefaultValue "")).StartsWith("cudnn", [StringComparison]::OrdinalIgnoreCase) }).Count
$tensorRtAvailable = ([string](Get-PropertyOrDefault -Object $hostRecord -Name "runtimeEnvironmentLine" -DefaultValue "")).Contains("TensorRtAvailable=True", [StringComparison]::OrdinalIgnoreCase)
$cudaAvailable = ([string](Get-PropertyOrDefault -Object $hostRecord -Name "runtimeEnvironmentLine" -DefaultValue "")).Contains("CudaAvailable=True", [StringComparison]::OrdinalIgnoreCase)
$failureSignature = [string](Get-PropertyOrDefault -Object $rootCause -Name "failureSignature" -DefaultValue "missing-failure-signature")
$rootCauseCategory = [string](Get-PropertyOrDefault -Object $rootCause -Name "rootCauseCategory" -DefaultValue "missing-root-cause-category")
$rootCauseSubcategory = [string](Get-PropertyOrDefault -Object $rootCause -Name "rootCauseSubcategory" -DefaultValue "missing-root-cause-subcategory")
$smokeStatus = [string](Get-PropertyOrDefault -Object $proof -Name "smokeStatus" -DefaultValue "missing-smoke-status")
$exitCode = [int](Get-PropertyOrDefault -Object $proof -Name "exitCode" -DefaultValue -1)
$runtimeCreateDiagnostic = Get-PropertyOrDefault -Object $proof -Name "runtimeCreateDiagnostic" -DefaultValue $null
$cudaPreflight = Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflight" -DefaultValue (Get-PropertyOrDefault -Object $proof -Name "cudaPreflight" -DefaultValue $null)
$cudaPreflightAvailable = [bool](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightAvailable" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "available" -DefaultValue $false))
$cudaPreflightAttempted = [bool](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightAttempted" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "attempted" -DefaultValue $false))
$cudaPreflightDriverVersion = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightDriverVersion" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "driverVersion" -DefaultValue ""))
$cudaPreflightRuntimeVersion = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightRuntimeVersion" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "runtimeVersion" -DefaultValue ""))
$cudaPreflightDeviceCount = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightDeviceCount" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "deviceCount" -DefaultValue ""))
$cudaPreflightSelectedDevice = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightSelectedDevice" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "selectedDevice" -DefaultValue ""))
$cudaPreflightDeviceName = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightDeviceName" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "deviceName" -DefaultValue ""))
$cudaPreflightGetDeviceCountStatus = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightGetDeviceCountStatus" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "getDeviceCountStatus" -DefaultValue ""))
$cudaPreflightInitStatus = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightInitStatus" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "initStatus" -DefaultValue ""))
$cudaPreflightLastErrorName = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightLastErrorName" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "lastErrorName" -DefaultValue ""))
$cudaPreflightLastErrorMessage = [string](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightLastErrorMessage" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "lastErrorMessage" -DefaultValue ""))
$cudaPreflightCanAttemptTensorRtRuntimeCreate = [bool](Get-PropertyOrDefault -Object $rootCause -Name "cudaPreflightCanAttemptTensorRtRuntimeCreate" -DefaultValue (Get-PropertyOrDefault -Object $cudaPreflight -Name "canAttemptTensorRtRuntimeCreate" -DefaultValue $false))
$nativeCreateRuntimeDiagnosticAvailable = [bool](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeDiagnosticAvailable" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "available" -DefaultValue $false))
$nativeCreateRuntimeAttempted = [bool](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeAttempted" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "attempted" -DefaultValue $false))
$nativeCreateRuntimeReturnedNull = [bool](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeReturnedNull" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "returnedNull" -DefaultValue $false))
$nativeCreateRuntimeReturnedNonNull = [bool](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeReturnedNonNull" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "returnedNonNull" -DefaultValue $false))
$nativeCreateRuntimeLastStatus = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeLastStatus" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastStatus" -DefaultValue ""))
$nativeCreateRuntimeDetectedVersion = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeDetectedVersion" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "detectedVersion" -DefaultValue ""))
$nativeCreateRuntimeLoggerCallbackAvailable = [bool](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeLoggerCallbackAvailable" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "loggerCallbackAvailable" -DefaultValue $false))
$nativeCreateRuntimeLoggerMessageCount = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeLoggerMessageCount" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "loggerMessageCount" -DefaultValue ""))
$nativeCreateRuntimeLastLoggerSeverity = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeLastLoggerSeverity" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastLoggerSeverity" -DefaultValue ""))
$nativeCreateRuntimeLastLoggerMessage = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeLastLoggerMessage" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "lastLoggerMessage" -DefaultValue ""))
$nativeCreateRuntimePhase = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimePhase" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "phase" -DefaultValue ""))
$nativeCreateRuntimeNativeDetail = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeNativeDetail" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "nativeDetail" -DefaultValue ""))
$nativeCreateRuntimeDiagnosticMessage = [string](Get-PropertyOrDefault -Object $rootCause -Name "nativeCreateRuntimeDiagnosticMessage" -DefaultValue (Get-PropertyOrDefault -Object $runtimeCreateDiagnostic -Name "diagnosticMessage" -DefaultValue ""))

$reportState = if ($missingRequired.Count -gt 0) {
  "diagnostic-owner-action-required-missing-required-dll"
}
elseif ($failureSignature -eq "createInferRuntime-null") {
  "diagnostic-owner-action-required-create-runtime-null"
}
else {
  "diagnostic-ready-non-proof"
}

$ownerActionRequired = $missingRequired.Count -gt 0 -or $failureSignature -eq "createInferRuntime-null" -or $duplicateGroups.Count -gt 0

$classificationRationale = @(
  "TRT11 bridge runtime smoke still reports smokeStatus=$smokeStatus and exitCode=$exitCode.",
  "Dependency preflight reports TensorRtAvailable=$tensorRtAvailable and CudaAvailable=$cudaAvailable, so DLL resolution must be checked before rerun.",
  "Search directory count=$($searchDirectories.Count); PATH directory count=$($pathDirectories.Count); native asset count=$($nativeAssets.Count).",
  "Required DLL groups missing=$($missingRequired.Count); duplicate DLL groups across search/PATH=$($duplicateGroups.Count).",
  "CUDA preflight available=$cudaPreflightAvailable; attempted=$cudaPreflightAttempted; deviceCount=$cudaPreflightDeviceCount; initStatus=$cudaPreflightInitStatus; lastError=$cudaPreflightLastErrorName.",
  "Native create-runtime diagnostic available=$nativeCreateRuntimeDiagnosticAvailable; attempted=$nativeCreateRuntimeAttempted; returnedNull=$nativeCreateRuntimeReturnedNull; lastStatus=$nativeCreateRuntimeLastStatus.",
  "Native create-runtime phase=$nativeCreateRuntimePhase; loggerCallbackAvailable=$nativeCreateRuntimeLoggerCallbackAvailable; loggerMessageCount=$nativeCreateRuntimeLoggerMessageCount; lastLoggerSeverity=$nativeCreateRuntimeLastLoggerSeverity.",
  "Native create-runtime logger last message=$nativeCreateRuntimeLastLoggerMessage; nativeDetail=$nativeCreateRuntimeNativeDetail.",
  "This report is diagnostic blocker evidence only and cannot promote runtime proof while TRT11 smoke is not passed."
)

$recommendedNextActions = @(
  "Review dllResolution rows with missingRequired=true or duplicateAcrossSearchOrPath=true before rerunning TRT11 smoke.",
  "Prefer the TRT11 TensorRT bin/lib and CUDA 13.2 bin/x64 directory ahead of stale TensorRT/CUDA/cuDNN PATH entries.",
  "Rerun eng/Test-BridgePackageRuntimeConsumer.ps1 for win-x64-trt11.0-cuda13.2-cudnn9.22 without AllowRuntimeSmokeFailure after DLL order cleanup.",
  "Use NativeCreateRuntimeDiagnostic* markers from the bridge runtime consumer proof to separate DLL-order issues from createInferRuntime null-return context and logger/vendor-message context."
)

$record = [pscustomobject]@{
  schemaVersion = 1
  recordKind = "trt11-runtime-dll-resolution-report"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reportState = $reportState
  sourceRuntimeKey = [string](Get-PropertyOrDefault -Object $proof -Name "sourceRuntimeKey" -DefaultValue $runtimeKey)
  proofClassification = [string](Get-PropertyOrDefault -Object $proof -Name "proofClassification" -DefaultValue "missing-trt11-bridge-runtime-proof")
  smokeStatus = $smokeStatus
  exitCode = $exitCode
  failureSignature = $failureSignature
  rootCauseCategory = $rootCauseCategory
  rootCauseSubcategory = $rootCauseSubcategory
  tensorRtAvailable = $tensorRtAvailable
  cudaAvailable = $cudaAvailable
  nativeBridgePresent = $nativeBridgePresent
  nativeBridgePath = $bridgePath
  nativeBridgeLength = [int64](Get-PropertyOrDefault -Object $bridgeAsset -Name "length" -DefaultValue 0)
  nativeBridgeSha256 = [string](Get-PropertyOrDefault -Object $bridgeAsset -Name "sha256" -DefaultValue "")
  nativeAssetCount = $nativeAssets.Count
  cudnnAssetCount = $cudnnAssetCount
  searchDirectoryCount = $searchDirectories.Count
  pathDirectoryCount = $pathDirectories.Count
  requiredDllGroupCount = $requiredDllGroups.Count
  foundRequiredDllGroupCount = @($dllResolution | Where-Object { $_.required -and $_.found }).Count
  missingRequiredDllGroupCount = $missingRequired.Count
  duplicateDllGroupCount = $duplicateGroups.Count
  ownerActionRequired = $ownerActionRequired
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
  searchDirectories = @($searchDirectories)
  dllResolution = @($dllResolution)
  missingRequiredDllGroups = @($missingRequired | ForEach-Object { $_.key })
  duplicateDllGroups = @($duplicateGroups | ForEach-Object { $_.key })
  stdoutSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "stdoutSha256" -DefaultValue "")
  stderrSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "stderrSha256" -DefaultValue "")
  combinedSha256 = [string](Get-PropertyOrDefault -Object $logs -Name "combinedSha256" -DefaultValue "")
  diffReportState = [string](Get-PropertyOrDefault -Object $diff -Name "reportState" -DefaultValue "missing-trt10-vs-trt11-bridge-runtime-diagnostic-diff")
  classificationRationale = $classificationRationale
  recommendedNextActions = $recommendedNextActions
  sourceArtifacts = @(
    ($proofRelativePath -replace "\\", "/"),
    ($rootCauseRelativePath -replace "\\", "/"),
    ($diffRelativePath -replace "\\", "/"),
    "eng/Test-BridgePackageRuntimeConsumer.ps1",
    "eng/Resolve-RuntimeRoots.ps1"
  )
  boundary = "TRT11 DLL resolution report is diagnostic blocker evidence only. It inspects local DLL paths and artifacts, does not run runtime smoke, does not publish packages, does not approve release close, and cannot promote runtime proof while TRT11 smokeStatus is not passed."
}

$jsonPath = Join-Path $OutputRoot "trt11-runtime-dll-resolution-report.json"
$markdownPath = Join-Path $OutputRoot "trt11-runtime-dll-resolution-report.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# TRT11 Runtime DLL Resolution Report")
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
$lines.Add("| CUDA preflight | ``available=$($record.cudaPreflightAvailable); attempted=$($record.cudaPreflightAttempted); driver=$($record.cudaPreflightDriverVersion); runtime=$($record.cudaPreflightRuntimeVersion); deviceCount=$($record.cudaPreflightDeviceCount); init=$($record.cudaPreflightInitStatus)`` |")
$lines.Add("| missingRequiredDllGroupCount | ``$($record.missingRequiredDllGroupCount)`` |")
$lines.Add("| duplicateDllGroupCount | ``$($record.duplicateDllGroupCount)`` |")
$lines.Add("| ownerActionRequired | ``$($record.ownerActionRequired)`` |")
$lines.Add("| native create-runtime diagnostic available | ``$($record.nativeCreateRuntimeDiagnosticAvailable)`` |")
$lines.Add("| native create-runtime attempted | ``$($record.nativeCreateRuntimeAttempted)`` |")
$lines.Add("| native create-runtime returned null | ``$($record.nativeCreateRuntimeReturnedNull)`` |")
$lines.Add("| native create-runtime last status | ``$($record.nativeCreateRuntimeLastStatus)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("")
$lines.Add("## Search Directories")
$lines.Add("")
foreach ($directory in $searchDirectories) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $directory)``")
}
$lines.Add("")
$lines.Add("## DLL Resolution")
$lines.Add("")
$lines.Add("| Key | Required | Found | Occurrences | Duplicate | Resolved Path |")
$lines.Add("| --- | --- | --- | ---: | --- | --- |")
foreach ($item in $dllResolution) {
  $lines.Add("| ``$($item.key)`` | ``$($item.required)`` | ``$($item.found)`` | ``$($item.occurrenceCount)`` | ``$($item.duplicateAcrossSearchOrPath)`` | ``$(ConvertTo-MarkdownCell $item.resolvedPath)`` |")
}
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

Write-Host "TRT11 runtime DLL resolution report written to $jsonPath"
Write-Host "TRT11 runtime DLL resolution report written to $markdownPath"
