[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\external-runtime-proof-record-template.json",
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$RequireExistingLog,
  [switch]$FailOnNotProof
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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  if ($null -eq $Object) {
    return $null
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $null
}

function Test-Truthy {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }

  if ($Value -is [bool]) {
    return [bool]$Value
  }

  $parsed = $false
  return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-HasRealValue {
  param([AllowNull()][object]$Value)

  return -not (Test-IsPlaceholder -Value $Value)
}

function Test-IntZero {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = 0
  return [int]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -eq 0
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail,
    [string]$OwnerAction = "",
    [string]$Boundary = ""
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
    ownerAction = $OwnerAction
    boundary = $Boundary
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-PackageConsumerRuntimeProofPreflightMatrix {
  $matrixPath = Join-Path $RepositoryRoot "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
  if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $matrixPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Find-PackageConsumerRuntimeProofPreflightEntry {
  param(
    [object]$Matrix,
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackageKey
  )

  if ($null -eq $Matrix -or $Matrix.PSObject.Properties.Name -notcontains "entries") {
    return $null
  }

  return @($Matrix.entries | Where-Object { $_.runtimePackageKey -eq $RuntimePackageKey } | Select-Object -First 1)[0]
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "External runtime proof record '$InputPath' was not found."
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$packageConsumerRuntimeProofPreflightMatrix = Get-PackageConsumerRuntimeProofPreflightMatrix
$runtimeProofPreflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $packageConsumerRuntimeProofPreflightMatrix -RuntimePackageKey $RuntimePackageKey
$recordKind = [string](Get-PropertyOrNull -Object $record -Name "recordKind")
$isDraftRecord = [string]::Equals($recordKind, "external-runtime-proof-record-draft", [System.StringComparison]::OrdinalIgnoreCase)
$recordRuntimePackageKey = [string](Get-PropertyOrNull -Object $record -Name "runtimePackageKey")
$templateOnly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "templateOnly")
$exampleOnly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "exampleOnly")
$proofState = [string](Get-PropertyOrNull -Object $record -Name "proofState")
$proofClassificationInput = [string](Get-PropertyOrNull -Object $record -Name "proofClassification")
$publicationState = [string](Get-PropertyOrNull -Object $record -Name "publicationState")
$isRuntimeExecutionEvidenceDeclared = Test-Truthy (Get-PropertyOrNull -Object $record -Name "isRuntimeExecutionEvidence")
$isDependencyProbeOnly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "isDependencyProbeOnly")
$canPromoteRuntimeProofDeclared = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPromoteRuntimeProof")

$hostInfo = Get-PropertyOrNull -Object $record -Name "host"
$packageSource = Get-PropertyOrNull -Object $record -Name "packageSource"
$command = Get-PropertyOrNull -Object $record -Name "command"
$results = Get-PropertyOrNull -Object $record -Name "results"
$modelEvidence = Get-PropertyOrNull -Object $record -Name "modelEvidence"

$ownerName = [string](Get-PropertyOrNull -Object $hostInfo -Name "ownerName")
$machineName = [string](Get-PropertyOrNull -Object $hostInfo -Name "machineName")
$osDescription = [string](Get-PropertyOrNull -Object $hostInfo -Name "osDescription")
$gpuName = [string](Get-PropertyOrNull -Object $hostInfo -Name "gpuName")
$driverVersion = [string](Get-PropertyOrNull -Object $hostInfo -Name "driverVersion")
$cudaDriverSupportedRuntime = [string](Get-PropertyOrNull -Object $hostInfo -Name "cudaDriverSupportedRuntime")
$cudaRuntimeVersion = [string](Get-PropertyOrNull -Object $hostInfo -Name "cudaRuntimeVersion")
$tensorRtRuntimeVersion = [string](Get-PropertyOrNull -Object $hostInfo -Name "tensorRtRuntimeVersion")
$cudnnVersion = [string](Get-PropertyOrNull -Object $hostInfo -Name "cudnnVersion")
$tensorRtLine = [string](Get-PropertyOrNull -Object $hostInfo -Name "tensorRtLine")
$managedPackageSource = [string](Get-PropertyOrNull -Object $packageSource -Name "managedPackageSource")
$runtimePackageSource = [string](Get-PropertyOrNull -Object $packageSource -Name "runtimePackageSource")
$packageSourceRuntimePackageKey = [string](Get-PropertyOrNull -Object $packageSource -Name "runtimePackageKey")
$packageSourceRuntimePackageId = [string](Get-PropertyOrNull -Object $packageSource -Name "runtimePackageId")
$packageSourceRestoreSourceMode = [string](Get-PropertyOrNull -Object $packageSource -Name "restoreSourceMode")
$managedNupkgSha256 = [string](Get-PropertyOrNull -Object $packageSource -Name "managedNupkgSha256")
$runtimeNupkgSha256 = [string](Get-PropertyOrNull -Object $packageSource -Name "runtimeNupkgSha256")
$noProjectReference = Test-Truthy (Get-PropertyOrNull -Object $packageSource -Name "noProjectReference")
$consumerProjectName = [string](Get-PropertyOrNull -Object $packageSource -Name "consumerProjectName")
$consumerProjectPath = [string](Get-PropertyOrNull -Object $packageSource -Name "consumerProjectPath")
$restoreCommand = [string](Get-PropertyOrNull -Object $command -Name "restoreCommand")
$buildCommand = [string](Get-PropertyOrNull -Object $command -Name "buildCommand")
$smokeCommand = [string](Get-PropertyOrNull -Object $command -Name "smokeCommand")
$exitCodeValue = Get-PropertyOrNull -Object $command -Name "exitCode"
$logPath = [string](Get-PropertyOrNull -Object $command -Name "logPath")
$logSha256 = [string](Get-PropertyOrNull -Object $command -Name "logSha256")
$dependencyProbeStatus = [string](Get-PropertyOrNull -Object $results -Name "dependencyProbeStatus")
$smokeStatus = [string](Get-PropertyOrNull -Object $results -Name "smokeStatus")
$nativeAssetsCopied = Test-Truthy (Get-PropertyOrNull -Object $results -Name "nativeAssetsCopied")
$nativeAssetsExpectedValue = Get-PropertyOrNull -Object $results -Name "nativeAssetsExpected"
$nativeAssetsFoundValue = Get-PropertyOrNull -Object $results -Name "nativeAssetsFound"
$stdoutSummary = [string](Get-PropertyOrNull -Object $results -Name "stdoutSummary")
$stderrSummary = [string](Get-PropertyOrNull -Object $results -Name "stderrSummary")
$failureDiagnostic = [string](Get-PropertyOrNull -Object $results -Name "failureDiagnostic")
$modelName = [string](Get-PropertyOrNull -Object $modelEvidence -Name "modelName")
$modelSha256 = [string](Get-PropertyOrNull -Object $modelEvidence -Name "modelSha256")
$modelLicense = [string](Get-PropertyOrNull -Object $modelEvidence -Name "modelLicense")
$inputAssetName = [string](Get-PropertyOrNull -Object $modelEvidence -Name "inputAssetName")
$inputAssetSha256 = [string](Get-PropertyOrNull -Object $modelEvidence -Name "inputAssetSha256")

$exitCodeIsZero = $false
if ($null -ne $exitCodeValue) {
  $exitCodeIsZero = Test-IntZero -Value $exitCodeValue
}

$resolvedLogPath = $null
if (-not [string]::IsNullOrWhiteSpace($logPath)) {
  $resolvedLogPath = Resolve-InputPath -Path $logPath
}

$runtimePackageKeyMatches = -not [string]::IsNullOrWhiteSpace($recordRuntimePackageKey) -and
  [string]::Equals($recordRuntimePackageKey, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)
$packageSourceRuntimeKeyMatches = -not [string]::IsNullOrWhiteSpace($packageSourceRuntimePackageKey) -and
  [string]::Equals($packageSourceRuntimePackageKey, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)
$runtimeProofPreflightMatrixFound = $null -ne $packageConsumerRuntimeProofPreflightMatrix
$runtimeProofPreflightEntryFound = $null -ne $runtimeProofPreflightEntry
$preflightRuntimePackageKeyMatches = $runtimeProofPreflightEntryFound -and
  [string]::Equals([string]$runtimeProofPreflightEntry.runtimePackageKey, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)
$preflightRuntimePackageId = if ($runtimeProofPreflightEntryFound) { [string]$runtimeProofPreflightEntry.runtimePackageId } else { "" }
$preflightRestoreSourceMode = if ($runtimeProofPreflightEntryFound) { [string]$runtimeProofPreflightEntry.restoreSourceMode } else { "" }
$preflightNativeAssetCopyExpected = if ($runtimeProofPreflightEntryFound) { [int]$runtimeProofPreflightEntry.nativeAssetCopyExpected } else { -1 }
$preflightCanPromotePackageConsumerRuntimeProof = $runtimeProofPreflightEntryFound -and [bool]$runtimeProofPreflightEntry.canPromotePackageConsumerRuntimeProof
$preflightOwnerActionRequired = (-not $runtimeProofPreflightEntryFound) -or [bool]$runtimeProofPreflightEntry.ownerActionRequired
$runtimePackageIdMatchesPreflight = -not [string]::IsNullOrWhiteSpace($packageSourceRuntimePackageId) -and
  $runtimeProofPreflightEntryFound -and
  [string]::Equals($packageSourceRuntimePackageId, $preflightRuntimePackageId, [System.StringComparison]::OrdinalIgnoreCase)
$restoreSourceModeMatchesPreflight = -not [string]::IsNullOrWhiteSpace($packageSourceRestoreSourceMode) -and
  $runtimeProofPreflightEntryFound -and
  [string]::Equals($packageSourceRestoreSourceMode, $preflightRestoreSourceMode, [System.StringComparison]::OrdinalIgnoreCase)
$nativeAssetsExpectedMatchesPreflight = $false
$nativeAssetsFoundMatchesPreflight = $false
if ($runtimeProofPreflightEntryFound -and -not (Test-IsPlaceholder -Value $nativeAssetsExpectedValue)) {
  $nativeAssetsExpectedParsed = 0
  $nativeAssetsExpectedMatchesPreflight = [int]::TryParse(([string]$nativeAssetsExpectedValue).Trim(), [ref]$nativeAssetsExpectedParsed) -and
    $nativeAssetsExpectedParsed -eq $preflightNativeAssetCopyExpected
}

if ($runtimeProofPreflightEntryFound -and -not (Test-IsPlaceholder -Value $nativeAssetsFoundValue)) {
  $nativeAssetsFoundParsed = 0
  $nativeAssetsFoundMatchesPreflight = [int]::TryParse(([string]$nativeAssetsFoundValue).Trim(), [ref]$nativeAssetsFoundParsed) -and
    $nativeAssetsFoundParsed -ge $preflightNativeAssetCopyExpected
}

$runtimeProofPreflightAligned = $runtimeProofPreflightMatrixFound -and
  $runtimeProofPreflightEntryFound -and
  $preflightRuntimePackageKeyMatches -and
  $runtimePackageIdMatchesPreflight -and
  $restoreSourceModeMatchesPreflight -and
  $nativeAssetsExpectedMatchesPreflight -and
  $nativeAssetsFoundMatchesPreflight -and
  -not $preflightCanPromotePackageConsumerRuntimeProof -and
  $preflightOwnerActionRequired
$managedNupkgSha256Ready = -not [string]::IsNullOrWhiteSpace($managedNupkgSha256) -and
  [System.Text.RegularExpressions.Regex]::IsMatch($managedNupkgSha256, "^[0-9a-fA-F]{64}$")
$runtimeNupkgSha256Ready = -not [string]::IsNullOrWhiteSpace($runtimeNupkgSha256) -and
  [System.Text.RegularExpressions.Regex]::IsMatch($runtimeNupkgSha256, "^[0-9a-fA-F]{64}$")
$logSha256FormatReady = -not [string]::IsNullOrWhiteSpace($logSha256) -and
  [System.Text.RegularExpressions.Regex]::IsMatch($logSha256, "^[0-9a-fA-F]{64}$")
$computedLogSha256 = ""
$logSha256Matches = $false
if ($RequireExistingLog.IsPresent -and
  $logSha256FormatReady -and
  -not [string]::IsNullOrWhiteSpace($resolvedLogPath) -and
  (Test-Path -LiteralPath $resolvedLogPath -PathType Leaf)) {
  $stream = [System.IO.File]::OpenRead($resolvedLogPath)
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      $computedLogSha256 = -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
      $sha.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }

  $logSha256Matches = [string]::Equals($computedLogSha256, $logSha256, [System.StringComparison]::OrdinalIgnoreCase)
}
elseif (-not $RequireExistingLog.IsPresent) {
  $logSha256Matches = $logSha256FormatReady
}

$logPathReady = -not [string]::IsNullOrWhiteSpace($logPath)
if ($RequireExistingLog.IsPresent) {
  $logPathReady = $logPathReady -and (Test-Path -LiteralPath $resolvedLogPath -PathType Leaf)
}

$consumerProjectIdentityReady = (Test-HasRealValue -Value $consumerProjectName) -and
  (Test-HasRealValue -Value $consumerProjectPath) -and
  $consumerProjectPath.EndsWith(".csproj", [System.StringComparison]::OrdinalIgnoreCase)

$smokeCommandRuntimeKeyReady = -not [string]::IsNullOrWhiteSpace($smokeCommand) -and
  $smokeCommand.IndexOf("--runtime-package-key", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
  $smokeCommand.IndexOf($RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -ge 0

$hostReady = (Test-HasRealValue -Value $ownerName) -and
  (Test-HasRealValue -Value $machineName) -and
  (Test-HasRealValue -Value $osDescription) -and
  (Test-HasRealValue -Value $gpuName) -and
  (Test-HasRealValue -Value $driverVersion) -and
  (Test-HasRealValue -Value $cudaDriverSupportedRuntime) -and
  (Test-HasRealValue -Value $cudaRuntimeVersion) -and
  (Test-HasRealValue -Value $tensorRtRuntimeVersion) -and
  (Test-HasRealValue -Value $cudnnVersion) -and
  (Test-HasRealValue -Value $tensorRtLine)

$packageSourceReady = (Test-HasRealValue -Value $managedPackageSource) -and
  (Test-HasRealValue -Value $runtimePackageSource) -and
  $packageSourceRuntimeKeyMatches -and
  $managedNupkgSha256Ready -and
  $runtimeNupkgSha256Ready -and
  $consumerProjectIdentityReady -and
  $noProjectReference

$commandsReady = (Test-HasRealValue -Value $restoreCommand) -and
  (Test-HasRealValue -Value $buildCommand) -and
  (Test-HasRealValue -Value $smokeCommand) -and
  $smokeCommandRuntimeKeyReady -and
  $exitCodeIsZero -and
  $logPathReady

$allowedProofClassifications = @(
  "template-only",
  "precheck",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$proofClassification = if ([string]::IsNullOrWhiteSpace($proofClassificationInput)) {
  if ($templateOnly -or $recordKind -in @("external-runtime-proof-record-template", "external-runtime-proof-record-input-template")) {
    "template-only"
  }
  elseif ($isDependencyProbeOnly) {
    "dependency-probe-only"
  }
  else {
    "missing"
  }
}
else {
  $proofClassificationInput
}

$proofClassificationKnown = $proofClassification -in $allowedProofClassifications
$proofClassificationPromotable = [string]::Equals($proofClassification, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)
$stdoutSummaryReady = Test-HasRealValue -Value $stdoutSummary
$stderrSummaryReady = Test-HasRealValue -Value $stderrSummary
$stdoutStderrSummariesReady = $stdoutSummaryReady -and $stderrSummaryReady
$realModelEvidenceReady = (Test-HasRealValue -Value $modelName) -and
  (Test-HasRealValue -Value $modelSha256) -and
  (Test-HasRealValue -Value $modelLicense) -and
  (Test-HasRealValue -Value $inputAssetName) -and
  (Test-HasRealValue -Value $inputAssetSha256)

$smokePassed = [string]::Equals($smokeStatus, "passed", [System.StringComparison]::OrdinalIgnoreCase)
$dependencyProbeOnlyBlocked = $isDependencyProbeOnly -or [string]::Equals($dependencyProbeStatus, "dependency-probe-only", [System.StringComparison]::OrdinalIgnoreCase)
$driverBlocked = $smokeStatus -like "*blocked-by-cuda-driver*" -or $failureDiagnostic -like "*blocked-by-cuda-driver*"
$isExampleOnly = $exampleOnly -or $recordKind -eq "external-runtime-proof-record-example" -or $publicationState -eq "example-not-for-publication" -or $proofState -eq "example-not-for-publication"

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ($recordKind -in @("external-runtime-proof-record", "external-runtime-proof-record-draft", "external-runtime-proof-record-template", "external-runtime-proof-record-input-template", "external-runtime-proof-record-example")) -Severity "blocker" -Detail "Record kind must identify an external runtime proof record, draft, input template, example, or template."
  New-ValidationItem -Id "proof-classification-known" -Passed $proofClassificationKnown -Severity "blocker" -Detail "proofClassification must be one of template-only, precheck, build-only, dependency-probe-only, synthetic-input-runtime, real-model-runtime, or package-consumer-runtime."
  New-ValidationItem -Id "runtime-package-key" -Passed $runtimePackageKeyMatches -Severity "proof-required" -Detail "runtimePackageKey must match the release target runtime package key."
  New-ValidationItem -Id "package-source-runtime-package-key" -Passed $packageSourceRuntimeKeyMatches -Severity "proof-required" -Detail "packageSource.runtimePackageKey must match the release target runtime package key."
  New-ValidationItem -Id "preflight-matrix-found" -Passed $runtimeProofPreflightMatrixFound -Severity "proof-required" -Detail "package-consumer-runtime-proof-preflight-matrix.json must be present before external runtime proof can be promoted."
  New-ValidationItem -Id "preflight-runtime-entry" -Passed $runtimeProofPreflightEntryFound -Severity "proof-required" -Detail "Runtime package key must exist in package-consumer-runtime-proof-preflight-matrix.json."
  New-ValidationItem -Id "preflight-runtime-package-id" -Passed $runtimePackageIdMatchesPreflight -Severity "proof-required" -Detail "packageSource.runtimePackageId must match the preflight matrix runtimePackageId."
  New-ValidationItem -Id "preflight-restore-source-mode" -Passed $restoreSourceModeMatchesPreflight -Severity "proof-required" -Detail "packageSource.restoreSourceMode must match the preflight matrix restoreSourceMode."
  New-ValidationItem -Id "preflight-native-assets-expected" -Passed $nativeAssetsExpectedMatchesPreflight -Severity "proof-required" -Detail "results.nativeAssetsExpected must match the preflight matrix nativeAssetCopyExpected."
  New-ValidationItem -Id "preflight-native-assets-found" -Passed $nativeAssetsFoundMatchesPreflight -Severity "proof-required" -Detail "results.nativeAssetsFound must be at least the preflight matrix nativeAssetCopyExpected."
  New-ValidationItem -Id "preflight-boundary-not-promotable" -Passed ((-not $preflightCanPromotePackageConsumerRuntimeProof) -and $preflightOwnerActionRequired) -Severity "proof-required" -Detail "Preflight matrix entries are owner-action-required planning records and cannot themselves promote package-consumer runtime proof."
  New-ValidationItem -Id "package-sha256" -Passed ($managedNupkgSha256Ready -and $runtimeNupkgSha256Ready) -Severity "proof-required" -Detail "managedNupkgSha256 and runtimeNupkgSha256 must be 64-character SHA256 hashes."
  New-ValidationItem -Id "not-example-only" -Passed (-not $isExampleOnly) -Severity "proof-required" -Detail "Example records are not runtime proof and cannot be promoted."
  New-ValidationItem -Id "real-record-kind" -Passed ($recordKind -eq "external-runtime-proof-record" -and -not $templateOnly) -Severity "proof-required" -Detail "Real proof requires recordKind=external-runtime-proof-record and templateOnly=false."
  New-ValidationItem -Id "proof-classification-promotable" -Passed $proofClassificationPromotable -Severity "proof-required" -Detail "Promotable external runtime proof requires proofClassification=package-consumer-runtime."
  New-ValidationItem -Id "host-metadata" -Passed $hostReady -Severity "proof-required" -Detail "Owner, machine, OS, GPU, driver, CUDA runtime, TensorRT runtime, cuDNN runtime, and TensorRT line are required."
  New-ValidationItem -Id "package-source" -Passed $packageSourceReady -Severity "proof-required" -Detail "Managed/runtime package sources and no ProjectReference confirmation are required."
  New-ValidationItem -Id "consumer-project-identity" -Passed $consumerProjectIdentityReady -Severity "proof-required" -Detail "External proof must identify the clean consumer project name and .csproj path."
  New-ValidationItem -Id "commands" -Passed $commandsReady -Severity "proof-required" -Detail "Restore/build/smoke commands, smoke runtime package key, exitCode=0, and logPath are required."
  New-ValidationItem -Id "smoke-command-runtime-package-key" -Passed $smokeCommandRuntimeKeyReady -Severity "proof-required" -Detail "Smoke command must include --runtime-package-key for the release target runtime package key."
  New-ValidationItem -Id "log-sha256" -Passed $logSha256FormatReady -Severity "proof-required" -Detail "logSha256 must be a 64-character SHA256 hash."
  New-ValidationItem -Id "log-sha256-match" -Passed $logSha256Matches -Severity "proof-required" -Detail "logSha256 must match the smoke log when -RequireExistingLog is used."
  New-ValidationItem -Id "stdout-summary" -Passed $stdoutSummaryReady -Severity "proof-required" -Detail "A reviewed stdoutSummary is required; a log path alone is not enough."
  New-ValidationItem -Id "stderr-summary" -Passed $stderrSummaryReady -Severity "proof-required" -Detail "A reviewed stderrSummary is required. Use an explicit no-stderr-emitted note when the process produced no stderr."
  New-ValidationItem -Id "stdout-stderr-summary" -Passed $stdoutStderrSummariesReady -Severity "proof-required" -Detail "Both stdoutSummary and stderrSummary must be reviewed for real package-consumer runtime proof."
  New-ValidationItem -Id "real-model-evidence" -Passed ($proofClassification -ne "real-model-runtime" -or $realModelEvidenceReady) -Severity "proof-required" -Detail "real-model-runtime records must include model/hash/license/input asset metadata."
  New-ValidationItem -Id "native-assets" -Passed $nativeAssetsCopied -Severity "proof-required" -Detail "Native assets must be copied in the clean consumer."
  New-ValidationItem -Id "smoke-passed" -Passed $smokePassed -Severity "proof-required" -Detail "Smoke status must be passed."
  New-ValidationItem -Id "not-dependency-probe-only" -Passed (-not $dependencyProbeOnlyBlocked) -Severity "proof-required" -Detail "DependencyProbe-only evidence is not runtime execution proof."
  New-ValidationItem -Id "not-driver-blocked" -Passed (-not $driverBlocked) -Severity "proof-required" -Detail "blocked-by-cuda-driver is not smoke passed."
  New-ValidationItem -Id "declared-runtime-proof" -Passed ($isRuntimeExecutionEvidenceDeclared -and $canPromoteRuntimeProofDeclared) -Severity "proof-required" -Detail "Record must explicitly declare runtime execution evidence and promotion readiness."
)

$validationOwnerGuidance = @{
  "record-kind" = @("确认 recordKind；真实证明必须改为 external-runtime-proof-record。", "template / input-template / draft / example 只能帮助填写，不能晋级为 release close proof。")
  "proof-classification-known" = @("选择受支持的 proofClassification；真实外部 runtime proof 只能使用 package-consumer-runtime。", "未知分类不能进入发布门禁。")
  "runtime-package-key" = @("回填与本次发布目标完全一致的 runtimePackageKey。", "其它 CUDA/TensorRT 组合的运行结果不能替代当前 runtime package proof。")
  "package-source-runtime-package-key" = @("确认 clean consumer 实际消费的 runtime 包 key 与发布目标一致。", "只改顶层 runtimePackageKey 不足以证明消费路径正确。")
  "preflight-matrix-found" = @("保留并提交 package-consumer-runtime-proof-preflight-matrix.json。", "没有预检矩阵时，外部 owner 输入不能被晋级为 package-consumer-runtime proof。")
  "preflight-runtime-entry" = @("确认 runtimePackageKey 已收录在预检矩阵中。", "未建模的 runtime key 不能绕过预检矩阵直接晋级。")
  "preflight-runtime-package-id" = @("在 packageSource.runtimePackageId 中填写实际消费的 runtime package id，并与预检矩阵对齐。", "runtime key 对齐但 package id 不对齐时，不能证明消费了正确包。")
  "preflight-restore-source-mode" = @("在 packageSource.restoreSourceMode 中填写 clean-consumer-package-source-required。", "local feed、direct nupkg 或 ProjectReference 不能替代 clean consumer package source。")
  "preflight-native-assets-expected" = @("在 results.nativeAssetsExpected 中填写 package consumer 预期 native asset 数量。", "native asset 数量必须与预检矩阵和 runtime manifest 同步。")
  "preflight-native-assets-found" = @("在 results.nativeAssetsFound 中填写 clean consumer 实际复制的 native asset 数量。", "只写 nativeAssetsCopied=true 不足以审计 asset copy 完整性。")
  "preflight-boundary-not-promotable" = @("保持 preflight matrix 条目为 owner-action-required，不要把预检矩阵本身改成 proof。", "RuntimeProofPreflight 是审计合同，不是 runtime proof。")
  "package-sha256" = @("对 clean consumer 实际消费的 managed/runtime nupkg 计算 SHA256 并回填。", "包名或本地路径不是可审计 proof。")
  "not-example-only" = @("不要用 example 文件提交发布证明；复制 input template 后回填真实记录。", "example-not-for-publication 永远不能关闭 release issue。")
  "real-record-kind" = @("真实记录必须设置 recordKind=external-runtime-proof-record 且 templateOnly=false。", "模板、草稿和 input template 不是真实 runtime proof。")
  "proof-classification-promotable" = @("真实外部 runtime proof 必须来自 package consumer smoke，并设置 proofClassification=package-consumer-runtime。", "precheck、build-only、dependency-probe-only、synthetic-input-runtime、real-model-runtime 都不能替代 package consumer proof。")
  "host-metadata" = @("在兼容 CUDA/TensorRT 主机上回填 owner、machine、OS、GPU、driver、CUDA、TensorRT、cuDNN 信息。", "没有 host metadata 的 log 不能跨机器复核。")
  "package-source" = @("记录 package source、nupkg hash、clean consumer identity，并确认 noProjectReference=true。", "ProjectReference 或源码直接引用不是包消费者 proof。")
  "consumer-project-identity" = @("填写源码仓库外 clean consumer 的项目名和 .csproj 路径。", "仓库内示例或测试项目不能替代 clean consumer。")
  "commands" = @("保留 restore/build/smoke 命令、exitCode=0 和 smoke log 路径。", "只保存命令计划或失败命令不是 runtime proof。")
  "smoke-command-runtime-package-key" = @("在 smokeCommand 中显式包含 --runtime-package-key 和目标 runtime key。", "未指定 runtime key 的 smoke 不能证明目标 runtime 包。")
  "log-sha256" = @("计算 smoke log 的 64 位 SHA256。", "stdout 摘要不能替代 log hash。")
  "log-sha256-match" = @("使用 -RequireExistingLog 校验记录中的 logSha256 与真实 log 文件一致。", "hash 不匹配的 log 不能作为 proof。")
  "stdout-summary" = @("人工复核真实 smoke stdout，并填写 stdoutSummary。", "logPath 单独存在不代表 reviewer 已确认输出。")
  "stderr-summary" = @("人工复核 stderr；为空时写 no-stderr-emitted。", "省略 stderrSummary 不能进入 close gate。")
  "stdout-stderr-summary" = @("同时保留 stdout/stderr 复核摘要。", "单侧摘要不完整。")
  "real-model-evidence" = @("如果记录 real-model-runtime，补齐 model/hash/license/input 元数据；发布 proof 仍需 package-consumer-runtime。", "真实模型样例 proof 不等于 NuGet package consumer proof。")
  "native-assets" = @("确认 packaged native assets 已复制到 clean consumer 输出目录。", "restore 成功不等于 native asset 就绪。")
  "smoke-passed" = @("在兼容主机上实际运行 package consumer smoke 并取得 smokeStatus=passed。", "blocked-by-cuda-driver、pending、failed 都不是 passed。")
  "not-dependency-probe-only" = @("在 dependency probe 之外运行真实 TensorRT package consumer smoke。", "dependency-probe-only 只能证明依赖探测，不证明 runtime 执行。")
  "not-driver-blocked" = @("换到兼容 CUDA driver / TensorRT runtime 主机，或保留 blocked 状态等待 owner 处理。", "blocked-by-cuda-driver 不能作为通过证明。")
  "declared-runtime-proof" = @("只有所有 proof-required 检查通过后，真实 record 才能声明 isRuntimeExecutionEvidence=true 和 canPromoteRuntimeProof=true。", "手动提前置 true 会被 validationItems 拦截。")
}

foreach ($item in $validationItems) {
  if ($validationOwnerGuidance.ContainsKey($item.id)) {
    $item.ownerAction = $validationOwnerGuidance[$item.id][0]
    $item.boundary = $validationOwnerGuidance[$item.id][1]
  }
}

$failedBlockers = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedProofItems = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "proof-required" })
$isTemplateOnly = $templateOnly -or $recordKind -in @("external-runtime-proof-record-template", "external-runtime-proof-record-input-template") -or $proofState -in @("template-only", "input-template")
$canPromoteRuntimeProof = $failedBlockers.Count -eq 0 -and $failedProofItems.Count -eq 0 -and -not $isTemplateOnly -and -not $isExampleOnly
$isRuntimeExecutionEvidence = $canPromoteRuntimeProof -and $isRuntimeExecutionEvidenceDeclared

if ($isRuntimeExecutionEvidence) {
  $validationState = "real-runtime-proof"
}
elseif ($isExampleOnly) {
  $validationState = "example-not-for-publication"
}
elseif ($isDraftRecord -and $driverBlocked) {
  $validationState = "draft-blocked-by-cuda-driver"
}
elseif ($isDraftRecord) {
  $validationState = "draft-rich-but-not-proof"
}
elseif ($isTemplateOnly) {
  $validationState = "template-only"
}
elseif ($driverBlocked) {
  $validationState = "blocked-by-cuda-driver"
}
elseif ($dependencyProbeOnlyBlocked) {
  $validationState = "dependency-probe-only"
}
elseif ($failedBlockers.Count -gt 0) {
  $validationState = "invalid-record"
}
else {
  $validationState = "incomplete-runtime-proof"
}

$classificationDiagnostic = switch ($proofClassification) {
  "build-only" { "build-only evidence cannot be promoted as runtime proof."; break }
  "dependency-probe-only" { "dependency-probe-only evidence cannot be promoted as runtime proof."; break }
  "synthetic-input-runtime" { "synthetic-input-runtime evidence is pipeline evidence, not real model or package-consumer proof."; break }
  "real-model-runtime" {
    if ($realModelEvidenceReady) {
      "real-model-runtime evidence includes model metadata, but external promotion still requires package-consumer-runtime."
    }
    else {
      "real-model-runtime evidence must include model/hash/license/input metadata and still cannot replace package-consumer-runtime proof."
    }
    break
  }
  "package-consumer-runtime" { "package-consumer-runtime is the only promotable external runtime proof classification when all other proof checks pass."; break }
  "template-only" { "template-only records are not runtime proof."; break }
  default { "proofClassification is missing or unknown."; break }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = "artifacts\final-release"
}

if ([IO.Path]::IsPathRooted($OutputRoot)) {
  $outputRoot = $OutputRoot
}
else {
  $outputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-validation.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-validation.md"

$requiredEvidenceSummary = @(
  "recordKind=external-runtime-proof-record",
  "runtimePackageKey matches the release target runtime package key",
  "runtime package key exists in package-consumer-runtime-proof-preflight-matrix.json",
  "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound align with RuntimeProofPreflight",
  "templateOnly=false and exampleOnly=false",
  "owner/machine/OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
  "managed/runtime package sources with no ProjectReference and clean consumer project identity",
  "restore/build/smoke commands, smokeCommand runtime package key, exitCode=0, logPath, and logSha256",
  "nativeAssetsCopied=true",
  "smokeStatus=passed",
  "isDependencyProbeOnly=false",
  "proofClassification=package-consumer-runtime",
  "stdoutSummary reviewed",
  "stderrSummary reviewed, or an explicit no-stderr-emitted note when stderr is empty",
  "isRuntimeExecutionEvidence=true and canPromoteRuntimeProof=true declared by the real record"
)

if ($RequireExistingLog.IsPresent) {
  $requiredEvidenceSummary += "logPath must point to an existing file and logSha256 must match the file"
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "external-runtime-proof-validation"
  inputPath = $resolvedInputPath
  inputRecordKind = $recordKind
  isDraftRecord = $isDraftRecord
  runtimePackageKey = $recordRuntimePackageKey
  expectedRuntimePackageKey = $RuntimePackageKey
  runtimePackageKeyMatches = $runtimePackageKeyMatches
  packageSourceRuntimePackageKeyMatches = $packageSourceRuntimeKeyMatches
  runtimeProofPreflight = [pscustomobject]@{
    matrixFound = $runtimeProofPreflightMatrixFound
    matrixSchemaVersion = if ($runtimeProofPreflightMatrixFound) { [string]$packageConsumerRuntimeProofPreflightMatrix.schemaVersion } else { "not-found" }
    entryFound = $runtimeProofPreflightEntryFound
    runtimePackageKeyMatches = $preflightRuntimePackageKeyMatches
    runtimePackageId = $preflightRuntimePackageId
    runtimePackageIdMatches = $runtimePackageIdMatchesPreflight
    restoreSourceMode = $preflightRestoreSourceMode
    restoreSourceModeMatches = $restoreSourceModeMatchesPreflight
    nativeAssetCopyExpected = $preflightNativeAssetCopyExpected
    nativeAssetsExpected = $nativeAssetsExpectedValue
    nativeAssetsExpectedMatches = $nativeAssetsExpectedMatchesPreflight
    nativeAssetsFound = $nativeAssetsFoundValue
    nativeAssetsFoundMatches = $nativeAssetsFoundMatchesPreflight
    canPromotePackageConsumerRuntimeProof = $preflightCanPromotePackageConsumerRuntimeProof
    ownerActionRequired = $preflightOwnerActionRequired
    aligned = $runtimeProofPreflightAligned
  }
  packageSourceRuntimePackageId = $packageSourceRuntimePackageId
  packageSourceRestoreSourceMode = $packageSourceRestoreSourceMode
  managedNupkgSha256Ready = $managedNupkgSha256Ready
  runtimeNupkgSha256Ready = $runtimeNupkgSha256Ready
  consumerProjectName = $consumerProjectName
  consumerProjectPath = $consumerProjectPath
  consumerProjectIdentityReady = $consumerProjectIdentityReady
  validationState = $validationState
  proofClassification = $proofClassification
  proofClassificationKnown = $proofClassificationKnown
  proofClassificationPromotable = $proofClassificationPromotable
  classificationDiagnostic = $classificationDiagnostic
  isTemplateOnly = $isTemplateOnly
  isExampleOnly = $isExampleOnly
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  isDependencyProbeOnly = $isDependencyProbeOnly
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  stdoutSummaryReady = $stdoutSummaryReady
  stderrSummaryReady = $stderrSummaryReady
  stdoutStderrSummariesReady = $stdoutStderrSummariesReady
  realModelEvidenceReady = $realModelEvidenceReady
  stdoutSummary = $stdoutSummary
  stderrSummary = $stderrSummary
  failedBlockerCount = $failedBlockers.Count
  failedProofItemCount = $failedProofItems.Count
  smokeStatus = $smokeStatus
  dependencyProbeStatus = $dependencyProbeStatus
  driverBlocked = $driverBlocked
  requireExistingLog = $RequireExistingLog.IsPresent
  logPath = $logPath
  resolvedLogPath = $resolvedLogPath
  logSha256 = $logSha256
  computedLogSha256 = $computedLogSha256
  logSha256FormatReady = $logSha256FormatReady
  logSha256Matches = $logSha256Matches
  noProjectReference = $noProjectReference
  packageSourceReady = $packageSourceReady
  commandsReady = $commandsReady
  smokeCommandRuntimeKeyReady = $smokeCommandRuntimeKeyReady
  hostReady = $hostReady
  requiredEvidenceSummary = @($requiredEvidenceSummary)
  validationItems = @($validationItems)
  evidenceClassifications = @(
    "build-only",
    "dependency-probe-only",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
  )
  classificationRules = @(
    "build-only records prove restore/build/conversion only and cannot be promoted as runtime proof.",
    "dependency-probe-only records prove bridge/load diagnostics only and cannot be promoted as runtime proof.",
    "synthetic-input-runtime records prove pipeline execution with synthetic input only; they are not real model proof.",
    "real-model-runtime records require model/hash/license/input/log evidence but are still separate from clean package-consumer proof.",
    "package-consumer-runtime is the required classification for external runtime proof promotion."
  )
  ownerActionSummary = @(
    "Owner must run the clean package consumer on a compatible CUDA/TensorRT host.",
    "Owner must fill a real external-runtime-proof-record.json, not a template/draft/example.",
    "Owner must run this validator with -RequireExistingLog -FailOnNotProof before promotion.",
    "If validationState is blocked-by-cuda-driver, dependency-probe-only, template-only, example-not-for-publication, draft-rich-but-not-proof, or incomplete-runtime-proof, the release issue must remain open."
  )
  nonSubstituteProofKinds = @(
    "collection package",
    "runbook",
    "template",
    "input-template",
    "input package",
    "draft",
    "example",
    "local inventory",
    "local feed",
    "ProjectReference",
    "build-only",
    "parse-only",
    "sidecar-only",
    "bridge-only package consumer log",
    "bridge-only wrapper surface",
    "Skipped=True",
    "dependency-probe-only",
    "blocked-by-cuda-driver",
    "WrapperSurfaceEvidenceKind=compile-surface-proof",
    "IsRuntimeExecutionProof=False",
    "mismatched log SHA256",
    "Parser/ParserRefitter diagnostic snapshots",
    "copied managed diagnostic snapshot",
    "managed-readiness",
    "managed-readiness-only",
    "callback-allocator-readiness-snapshot",
    "CallbackAllocatorReadinessSnapshot",
    "TensorRtCallbackAllocatorReadinessSnapshot",
    "precheck-only",
    "dry-run-only",
    "schema-only",
    "Windows handoff for Linux proof"
  )
  promotionRules = @(
    "Template-only records are not runtime proof.",
    "Example records are not runtime proof.",
    "runtimePackageKey must match the release target runtime package key.",
    "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound must align with RuntimeProofPreflight.",
    "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source.",
    "logSha256 must be a 64-character SHA256 hash and must match the log when -RequireExistingLog is used.",
    "proofClassification must be package-consumer-runtime before this record can be promoted.",
    "DependencyProbe-only evidence is not runtime execution proof.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Bridge-only package consumer logs, bridge-only wrapper surface, Skipped=True, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, and copied Parser/ParserRefitter diagnostic snapshots are not runtime proof.",
    "TensorRtCallbackAllocatorReadinessSnapshot and RuntimeEvidenceKind=managed-readiness only prove managed wrapper readiness; they cannot promote runtime proof.",
    "precheck-only, dry-run-only, schema-only, and managed-readiness-only records cannot be promoted to package-consumer-runtime.",
    "Real runtime proof requires compatible host smoke, exitCode=0, native assets copied, and no ProjectReference."
  )
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- runtime package key: ``$recordRuntimePackageKey``")
$lines.Add("- expected runtime package key: ``$RuntimePackageKey``")
$lines.Add("- runtime package key matches: ``$runtimePackageKeyMatches``")
$lines.Add("- preflight matrix found: ``$runtimeProofPreflightMatrixFound``")
$lines.Add("- preflight entry found: ``$runtimeProofPreflightEntryFound``")
$lines.Add("- preflight aligned: ``$runtimeProofPreflightAligned``")
$lines.Add("- proof classification: ``$proofClassification``")
$lines.Add("- classification diagnostic: $classificationDiagnostic")
$lines.Add("- runtime execution evidence: ``$isRuntimeExecutionEvidence``")
$lines.Add("- dependency probe only: ``$isDependencyProbeOnly``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- stdout summary ready: ``$stdoutSummaryReady``")
$lines.Add("- stderr summary ready: ``$stderrSummaryReady``")
$lines.Add("- stdout/stderr summaries ready: ``$stdoutStderrSummariesReady``")
$lines.Add("- real model evidence ready: ``$realModelEvidenceReady``")
$lines.Add("- log SHA256 format ready: ``$logSha256FormatReady``")
$lines.Add("- log SHA256 matches: ``$logSha256Matches``")
$lines.Add("- failed blockers: $($failedBlockers.Count)")
$lines.Add("- failed proof items: $($failedProofItems.Count)")
$lines.Add("- input path: ``$resolvedInputPath``")
$lines.Add("")
$lines.Add("## Required Evidence Summary")
$lines.Add("")
foreach ($item in $requiredEvidenceSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Evidence Classifications")
$lines.Add("")
foreach ($item in $summary.evidenceClassifications) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Classification Rules")
$lines.Add("")
foreach ($rule in $summary.classificationRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $summary.ownerActionSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Runtime Proof Preflight")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| matrix found | ``$runtimeProofPreflightMatrixFound`` |")
$lines.Add("| matrix schema | ``$($summary.runtimeProofPreflight.matrixSchemaVersion)`` |")
$lines.Add("| entry found | ``$runtimeProofPreflightEntryFound`` |")
$lines.Add("| preflight package id | ``$preflightRuntimePackageId`` |")
$lines.Add("| record package id | ``$packageSourceRuntimePackageId`` |")
$lines.Add("| package id matches | ``$runtimePackageIdMatchesPreflight`` |")
$lines.Add("| preflight restore source mode | ``$preflightRestoreSourceMode`` |")
$lines.Add("| record restore source mode | ``$packageSourceRestoreSourceMode`` |")
$lines.Add("| restore source mode matches | ``$restoreSourceModeMatchesPreflight`` |")
$lines.Add("| preflight native assets expected | ``$preflightNativeAssetCopyExpected`` |")
$lines.Add("| record native assets expected | ``$nativeAssetsExpectedValue`` |")
$lines.Add("| record native assets found | ``$nativeAssetsFoundValue`` |")
$lines.Add("| native assets expected matches | ``$nativeAssetsExpectedMatchesPreflight`` |")
$lines.Add("| native assets found matches | ``$nativeAssetsFoundMatchesPreflight`` |")
$lines.Add("| preflight can promote | ``$preflightCanPromotePackageConsumerRuntimeProof`` |")
$lines.Add("| preflight owner action required | ``$preflightOwnerActionRequired`` |")
$lines.Add("")
$lines.Add("## Non-Substitute Proof Kinds")
$lines.Add("")
foreach ($item in $summary.nonSubstituteProofKinds) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Owner Actions By Validation Item")
$lines.Add("")
$lines.Add("| ID | Owner action | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | $(ConvertTo-MarkdownCell $item.ownerAction) | $(ConvertTo-MarkdownCell $item.boundary) |")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $summary.promotionRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof validation written to $jsonPath"
Write-Host "External runtime proof validation written to $markdownPath"
Write-Host "ValidationState=$validationState ProofClassification=$proofClassification IsRuntimeExecutionEvidence=$isRuntimeExecutionEvidence CanPromoteRuntimeProof=$canPromoteRuntimeProof FailedBlockers=$($failedBlockers.Count) FailedProofItems=$($failedProofItems.Count)"

if ($FailOnNotProof.IsPresent -and -not $isRuntimeExecutionEvidence) {
  Write-Error "External runtime proof is not real runtime proof. ValidationState=$validationState"
  exit 1
}
