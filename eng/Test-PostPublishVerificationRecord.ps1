[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-verification-record-template.json",
  [string]$RepositoryRoot,
  [switch]$RequireExistingLog,
  [switch]$FailOnNotProof,
  [string]$OutputRoot
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

  return [bool]::Parse([string]$Value)
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

function Get-Sha256OrEmpty {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return ""
  }

  $stream = [System.IO.File]::OpenRead($Path)
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      return -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
      $sha.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }
}

function Test-ExistingSha256 {
  param(
    [string]$Path,
    [string]$DeclaredSha256,
    [bool]$DeclaredSha256Ready,
    [bool]$RequireExisting
  )

  $resolvedPath = ""
  if (-not [string]::IsNullOrWhiteSpace($Path)) {
    $resolvedPath = Resolve-InputPath -Path $Path
  }

  $exists = -not [string]::IsNullOrWhiteSpace($resolvedPath) -and (Test-Path -LiteralPath $resolvedPath -PathType Leaf)
  $computedSha256 = ""
  $matches = $false

  if ($RequireExisting) {
    if ($exists -and $DeclaredSha256Ready) {
      $computedSha256 = Get-Sha256OrEmpty -Path $resolvedPath
      $matches = [string]::Equals($computedSha256, $DeclaredSha256, [System.StringComparison]::OrdinalIgnoreCase)
    }
  }
  else {
    $matches = $DeclaredSha256Ready
  }

  [pscustomobject]@{
    path = $Path
    resolvedPath = $resolvedPath
    exists = $exists
    declaredSha256 = $DeclaredSha256
    declaredSha256Ready = $DeclaredSha256Ready
    computedSha256 = $computedSha256
    matches = $matches
  }
}

$postPublishRequiredEvidence = @(
  "selectedChannel",
  "channelSourceUri",
  "publishedPackageUrl",
  "managedPackageUrl",
  "runtimePackageUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "cleanConsumerRootOutsideRepository",
  "consumerProjectPath",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "restoreLogPath",
  "nativeAssetListingSha256",
  "dependencyProbeLogPath",
  "dependencyProbeLogSha256",
  "runtimeSmokeLogPath",
  "runtimeSmokeLogSha256",
  "runtimeSmokePassed",
  "runtimeSmokeExitCode",
  "stdoutSummary",
  "stderrSummary",
  "hostMetadata"
)

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Post-publish verification record '$InputPath' was not found."
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordKind = [string](Get-PropertyOrNull -Object $record -Name "recordKind")
$templateOnly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "templateOnly")
$verificationStateInput = [string](Get-PropertyOrNull -Object $record -Name "verificationState")
$postPublishProofClassification = [string](Get-PropertyOrNull -Object $record -Name "postPublishProofClassification")
$selectedChannel = [string](Get-PropertyOrNull -Object $record -Name "selectedChannel")
$channelSourceUri = [string](Get-PropertyOrNull -Object $record -Name "channelSourceUri")
$publishedPackageUrl = [string](Get-PropertyOrNull -Object $record -Name "publishedPackageUrl")
$declaredPostPublishRequiredEvidence = @(Get-PropertyOrNull -Object $record -Name "postPublishRequiredEvidence")
$missingDeclaredPostPublishRequiredEvidence = @($postPublishRequiredEvidence | Where-Object { $declaredPostPublishRequiredEvidence -notcontains $_ })
$packageIdentity = Get-PropertyOrNull -Object $record -Name "packageIdentity"
$managedPackageId = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedPackageId")
$managedPackageVersion = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedPackageVersion")
$runtimePackageId = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimePackageId")
$runtimePackageVersion = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimePackageVersion")
$managedPackageUrl = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedPackageUrl")
$runtimePackageUrl = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimePackageUrl")
$managedNupkgSha256 = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedNupkgSha256")
$runtimeNupkgSha256 = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimeNupkgSha256")
$managedPackageSha256Source = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedPackageSha256Source")
$runtimePackageSha256Source = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimePackageSha256Source")
$managedPackageDownloadTimestampUtc = [string](Get-PropertyOrNull -Object $packageIdentity -Name "managedPackageDownloadTimestampUtc")
$runtimePackageDownloadTimestampUtc = [string](Get-PropertyOrNull -Object $packageIdentity -Name "runtimePackageDownloadTimestampUtc")
$cleanConsumerRoot = [string](Get-PropertyOrNull -Object $record -Name "cleanConsumerRoot")
$cleanConsumerRootOutsideRepository = Test-Truthy (Get-PropertyOrNull -Object $record -Name "cleanConsumerRootOutsideRepository")
$consumerProjectName = [string](Get-PropertyOrNull -Object $record -Name "consumerProjectName")
$consumerProjectPath = [string](Get-PropertyOrNull -Object $record -Name "consumerProjectPath")
$hostMetadata = Get-PropertyOrNull -Object $record -Name "host"
if ($null -eq $hostMetadata) {
  $hostMetadata = Get-PropertyOrNull -Object $record -Name "hostMetadata"
}
$hostOwnerName = [string](Get-PropertyOrNull -Object $hostMetadata -Name "ownerName")
$hostMachineName = [string](Get-PropertyOrNull -Object $hostMetadata -Name "machineName")
$hostOsDescription = [string](Get-PropertyOrNull -Object $hostMetadata -Name "osDescription")
$hostGpuName = [string](Get-PropertyOrNull -Object $hostMetadata -Name "gpuName")
$hostDriverVersion = [string](Get-PropertyOrNull -Object $hostMetadata -Name "driverVersion")
$hostCudaDriverSupportedRuntime = [string](Get-PropertyOrNull -Object $hostMetadata -Name "cudaDriverSupportedRuntime")
$hostCudaRuntimeVersion = [string](Get-PropertyOrNull -Object $hostMetadata -Name "cudaRuntimeVersion")
$hostTensorRtRuntimeVersion = [string](Get-PropertyOrNull -Object $hostMetadata -Name "tensorRtRuntimeVersion")
$hostTensorRtLine = [string](Get-PropertyOrNull -Object $hostMetadata -Name "tensorRtLine")
$hostCudnnVersion = [string](Get-PropertyOrNull -Object $hostMetadata -Name "cudnnVersion")
$restoreCommand = [string](Get-PropertyOrNull -Object $record -Name "restoreCommand")
$buildCommand = [string](Get-PropertyOrNull -Object $record -Name "buildCommand")
$smokeCommand = [string](Get-PropertyOrNull -Object $record -Name "smokeCommand")
$stdoutSummary = [string](Get-PropertyOrNull -Object $record -Name "stdoutSummary")
$stderrSummary = [string](Get-PropertyOrNull -Object $record -Name "stderrSummary")
$restoreLogPath = [string](Get-PropertyOrNull -Object $record -Name "restoreLogPath")
$restoreLogSha256 = [string](Get-PropertyOrNull -Object $record -Name "restoreLogSha256")
$nativeAssetListingPath = [string](Get-PropertyOrNull -Object $record -Name "nativeAssetListingPath")
$nativeAssetListingSha256 = [string](Get-PropertyOrNull -Object $record -Name "nativeAssetListingSha256")
$dependencyProbeLogPath = [string](Get-PropertyOrNull -Object $record -Name "dependencyProbeLogPath")
$dependencyProbeLogSha256 = [string](Get-PropertyOrNull -Object $record -Name "dependencyProbeLogSha256")
$runtimeSmokeLogPath = [string](Get-PropertyOrNull -Object $record -Name "runtimeSmokeLogPath")
$runtimeSmokeLogSha256 = [string](Get-PropertyOrNull -Object $record -Name "runtimeSmokeLogSha256")
$smokeLogPath = [string](Get-PropertyOrNull -Object $record -Name "smokeLogPath")
$smokeLogSha256 = [string](Get-PropertyOrNull -Object $record -Name "smokeLogSha256")
if ([string]::IsNullOrWhiteSpace($runtimeSmokeLogPath)) { $runtimeSmokeLogPath = $smokeLogPath }
if ([string]::IsNullOrWhiteSpace($runtimeSmokeLogSha256)) { $runtimeSmokeLogSha256 = $smokeLogSha256 }
if ([string]::IsNullOrWhiteSpace($smokeLogPath)) { $smokeLogPath = $runtimeSmokeLogPath }
if ([string]::IsNullOrWhiteSpace($smokeLogSha256)) { $smokeLogSha256 = $runtimeSmokeLogSha256 }
$managedPackageSource = [string](Get-PropertyOrNull -Object $record -Name "managedPackageSource")
$runtimePackageSource = [string](Get-PropertyOrNull -Object $record -Name "runtimePackageSource")
$expectedRuntimePackageKey = [string](Get-PropertyOrNull -Object $record -Name "expectedRuntimePackageKey")
$noProjectReference = Test-Truthy (Get-PropertyOrNull -Object $record -Name "noProjectReference")
$noLocalPackageSource = Test-Truthy (Get-PropertyOrNull -Object $record -Name "noLocalPackageSource")
$noLocalNupkgPackageReference = Test-Truthy (Get-PropertyOrNull -Object $record -Name "noLocalNupkgPackageReference")
$cleanConsumerProjectScanPassed = Test-Truthy (Get-PropertyOrNull -Object $record -Name "cleanConsumerProjectScanPassed")
$nativeAssetsCopied = Test-Truthy (Get-PropertyOrNull -Object $record -Name "nativeAssetsCopied")
$dependencyProbePassed = Test-Truthy (Get-PropertyOrNull -Object $record -Name "dependencyProbePassed")
$runtimeSmokePassed = Test-Truthy (Get-PropertyOrNull -Object $record -Name "runtimeSmokePassed")
$runtimeSmokeExitCodeValue = Get-PropertyOrNull -Object $record -Name "runtimeSmokeExitCode"
$smokeStatus = [string](Get-PropertyOrNull -Object $record -Name "smokeStatus")
$performsPublish = Test-Truthy (Get-PropertyOrNull -Object $record -Name "performsPublish")
$declaredProof = Test-Truthy (Get-PropertyOrNull -Object $record -Name "isPostPublishVerificationProof")
$declaredCanClose = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canCloseReleaseIssue")
$ownerName = [string](Get-PropertyOrNull -Object $record -Name "ownerName")
$reviewerName = [string](Get-PropertyOrNull -Object $record -Name "reviewerName")
$publishedVersion = [string](Get-PropertyOrNull -Object $record -Name "publishedVersion")
$verificationItems = @(Get-PropertyOrNull -Object $record -Name "verificationItems")
$executionSteps = @(Get-PropertyOrNull -Object $record -Name "executionSteps")
$stdoutSummaryFieldPresent = $record.PSObject.Properties.Name -contains "stdoutSummary"
$stderrSummaryFieldPresent = $record.PSObject.Properties.Name -contains "stderrSummary"

$requiredItemIds = @(
  "published-package-version",
  "clean-directory",
  "clean-consumer-project-identity",
  "no-project-reference",
  "managed-package-source",
  "runtime-package-source",
  "host-runtime-metadata",
  "native-assets-copied",
  "dependency-probe",
  "runtime-key-smoke-command",
  "stdout-summary",
  "stderr-summary",
  "stdout-stderr-summary",
  "compatible-host-smoke"
)

$requiredExecutionStepIds = @(
  "publish-channel-confirmed",
  "download-packages",
  "create-clean-consumer",
  "capture-host-metadata",
  "restore-from-channel",
  "verify-native-assets",
  "run-dependency-probe",
  "run-compatible-host-smoke",
  "validate-record"
)

$itemById = @{}
foreach ($item in $verificationItems) {
  $id = [string](Get-PropertyOrNull -Object $item -Name "id")
  if (-not [string]::IsNullOrWhiteSpace($id)) {
    $itemById[$id] = $item
  }
}

$missingItemIds = @($requiredItemIds | Where-Object { -not $itemById.ContainsKey($_) })
$executionStepById = @{}
foreach ($step in $executionSteps) {
  $id = [string](Get-PropertyOrNull -Object $step -Name "id")
  if (-not [string]::IsNullOrWhiteSpace($id)) {
    $executionStepById[$id] = $step
  }
}

$missingExecutionStepIds = @($requiredExecutionStepIds | Where-Object { -not $executionStepById.ContainsKey($_) })
$notPassedIds = New-Object System.Collections.Generic.List[string]
$missingEvidenceIds = New-Object System.Collections.Generic.List[string]

foreach ($id in $requiredItemIds) {
  if (-not $itemById.ContainsKey($id)) {
    continue
  }

  $item = $itemById[$id]
  $status = [string](Get-PropertyOrNull -Object $item -Name "status")
  $evidence = [string](Get-PropertyOrNull -Object $item -Name "evidence")
  if ([string]::IsNullOrWhiteSpace($evidence)) {
    $evidence = [string](Get-PropertyOrNull -Object $item -Name "requiredEvidence")
  }

  if ($status -notin @("passed", "ready", "succeeded")) {
    $notPassedIds.Add($id) | Out-Null
  }

  if ([string]::IsNullOrWhiteSpace($evidence)) {
    $missingEvidenceIds.Add($id) | Out-Null
  }
}

$isTemplateOnly = $templateOnly -or $recordKind -eq "post-publish-verification-record-template" -or $verificationStateInput -eq "template-only"
if ([string]::IsNullOrWhiteSpace($postPublishProofClassification)) {
  $postPublishProofClassification = if ($isTemplateOnly) { "template-only" } else { "owner-action-required" }
}

$allowedPostPublishProofClassifications = @(
  "template-only",
  "owner-action-required",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "post-publish-package-consumer-runtime"
)
$postPublishProofClassificationKnown = $postPublishProofClassification -in $allowedPostPublishProofClassifications
$postPublishProofClassificationPromotable = [string]::Equals($postPublishProofClassification, "post-publish-package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)

$metadataReady = -not [string]::IsNullOrWhiteSpace($selectedChannel) -and
  -not [string]::IsNullOrWhiteSpace($channelSourceUri) -and
  -not [string]::IsNullOrWhiteSpace($ownerName) -and
  -not [string]::IsNullOrWhiteSpace($reviewerName) -and
  -not [string]::IsNullOrWhiteSpace($publishedVersion)

$sha256Pattern = "^[0-9a-fA-F]{64}$"
$managedNupkgSha256Ready = -not [string]::IsNullOrWhiteSpace($managedNupkgSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($managedNupkgSha256, $sha256Pattern)
$runtimeNupkgSha256Ready = -not [string]::IsNullOrWhiteSpace($runtimeNupkgSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($runtimeNupkgSha256, $sha256Pattern)
$restoreLogSha256Ready = -not [string]::IsNullOrWhiteSpace($restoreLogSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($restoreLogSha256, $sha256Pattern)
$nativeAssetListingSha256Ready = -not [string]::IsNullOrWhiteSpace($nativeAssetListingSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($nativeAssetListingSha256, $sha256Pattern)
$dependencyProbeLogSha256Ready = -not [string]::IsNullOrWhiteSpace($dependencyProbeLogSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($dependencyProbeLogSha256, $sha256Pattern)
$smokeLogSha256Ready = -not [string]::IsNullOrWhiteSpace($smokeLogSha256) -and [System.Text.RegularExpressions.Regex]::IsMatch($smokeLogSha256, $sha256Pattern)

$managedPackageUrlReady = -not [string]::IsNullOrWhiteSpace($managedPackageUrl) -and
  (
    $managedPackageUrl.IndexOf("://", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $managedPackageUrl.StartsWith("file:", [System.StringComparison]::OrdinalIgnoreCase)
  )
$runtimePackageUrlReady = -not [string]::IsNullOrWhiteSpace($runtimePackageUrl) -and
  (
    $runtimePackageUrl.IndexOf("://", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $runtimePackageUrl.StartsWith("file:", [System.StringComparison]::OrdinalIgnoreCase)
  )
$publishedPackageUrlReady = -not [string]::IsNullOrWhiteSpace($publishedPackageUrl)
$packageSha256SourceReady = -not [string]::IsNullOrWhiteSpace($managedPackageSha256Source) -and
  -not [string]::IsNullOrWhiteSpace($runtimePackageSha256Source) -and
  -not [string]::IsNullOrWhiteSpace($managedPackageDownloadTimestampUtc) -and
  -not [string]::IsNullOrWhiteSpace($runtimePackageDownloadTimestampUtc)

$channelSourceReady = -not [string]::IsNullOrWhiteSpace($selectedChannel) -and
  -not [string]::IsNullOrWhiteSpace($channelSourceUri) -and
  -not [string]::IsNullOrWhiteSpace($managedPackageSource) -and
  -not [string]::IsNullOrWhiteSpace($runtimePackageSource)

$runtimePackageKeyReady = -not [string]::IsNullOrWhiteSpace($expectedRuntimePackageKey) -and
  (
    ($runtimePackageId.IndexOf($expectedRuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -or
    ($runtimePackageSource.IndexOf($expectedRuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -or
    ($runtimePackageUrl.IndexOf($expectedRuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
  )

$packageIdentityReady = -not [string]::IsNullOrWhiteSpace($managedPackageId) -and
  -not [string]::IsNullOrWhiteSpace($managedPackageVersion) -and
  -not [string]::IsNullOrWhiteSpace($runtimePackageId) -and
  -not [string]::IsNullOrWhiteSpace($runtimePackageVersion) -and
  $managedPackageUrlReady -and
  $runtimePackageUrlReady -and
  $managedNupkgSha256Ready -and
  $runtimeNupkgSha256Ready -and
  $packageSha256SourceReady

$cleanConsumerReady = -not [string]::IsNullOrWhiteSpace($cleanConsumerRoot) -and
  -not [string]::Equals($cleanConsumerRoot.TrimEnd('\', '/'), $RepositoryRoot.TrimEnd('\', '/'), [System.StringComparison]::OrdinalIgnoreCase)
$cleanConsumerOutsideRepository = $false
if ($cleanConsumerReady) {
  try {
    $repositoryRootFull = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
    $cleanConsumerRootFull = [System.IO.Path]::GetFullPath($cleanConsumerRoot).TrimEnd('\', '/')
    $cleanConsumerOutsideRepository = -not [string]::Equals($cleanConsumerRootFull, $repositoryRootFull, [System.StringComparison]::OrdinalIgnoreCase) -and
      -not $cleanConsumerRootFull.StartsWith($repositoryRootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -and
      -not $cleanConsumerRootFull.StartsWith($repositoryRootFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    $cleanConsumerOutsideRepository = $false
  }
}

$consumerProjectIdentityReady = -not [string]::IsNullOrWhiteSpace($consumerProjectName) -and
  -not [string]::IsNullOrWhiteSpace($consumerProjectPath) -and
  $consumerProjectPath.EndsWith(".csproj", [System.StringComparison]::OrdinalIgnoreCase)

$hostReady = -not [string]::IsNullOrWhiteSpace($hostOwnerName) -and
  -not [string]::IsNullOrWhiteSpace($hostMachineName) -and
  -not [string]::IsNullOrWhiteSpace($hostOsDescription) -and
  -not [string]::IsNullOrWhiteSpace($hostGpuName) -and
  -not [string]::IsNullOrWhiteSpace($hostDriverVersion) -and
  -not [string]::IsNullOrWhiteSpace($hostCudaDriverSupportedRuntime) -and
  -not [string]::IsNullOrWhiteSpace($hostCudaRuntimeVersion) -and
  -not [string]::IsNullOrWhiteSpace($hostTensorRtRuntimeVersion) -and
  -not [string]::IsNullOrWhiteSpace($hostTensorRtLine) -and
  -not [string]::IsNullOrWhiteSpace($hostCudnnVersion)

$commandsReady = -not [string]::IsNullOrWhiteSpace($restoreCommand) -and
  -not [string]::IsNullOrWhiteSpace($buildCommand) -and
  -not [string]::IsNullOrWhiteSpace($smokeCommand)

$smokeCommandRuntimeKeyReady = -not [string]::IsNullOrWhiteSpace($expectedRuntimePackageKey) -and
  ($smokeCommand.IndexOf("--runtime-package-key", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -and
  ($smokeCommand.IndexOf($expectedRuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)

$stdoutSummaryReady = $stdoutSummaryFieldPresent -and -not [string]::IsNullOrWhiteSpace($stdoutSummary)
$stderrSummaryReady = $stderrSummaryFieldPresent -and -not [string]::IsNullOrWhiteSpace($stderrSummary)
$stdoutStderrSummaryReady = $stdoutSummaryReady -and $stderrSummaryReady

$restoreLogCheck = Test-ExistingSha256 -Path $restoreLogPath -DeclaredSha256 $restoreLogSha256 -DeclaredSha256Ready $restoreLogSha256Ready -RequireExisting $RequireExistingLog.IsPresent
$nativeAssetListingCheck = Test-ExistingSha256 -Path $nativeAssetListingPath -DeclaredSha256 $nativeAssetListingSha256 -DeclaredSha256Ready $nativeAssetListingSha256Ready -RequireExisting $RequireExistingLog.IsPresent
$dependencyProbeLogCheck = Test-ExistingSha256 -Path $dependencyProbeLogPath -DeclaredSha256 $dependencyProbeLogSha256 -DeclaredSha256Ready $dependencyProbeLogSha256Ready -RequireExisting $RequireExistingLog.IsPresent
$smokeLogCheck = Test-ExistingSha256 -Path $smokeLogPath -DeclaredSha256 $smokeLogSha256 -DeclaredSha256Ready $smokeLogSha256Ready -RequireExisting $RequireExistingLog.IsPresent

$restoreLogReady = -not [string]::IsNullOrWhiteSpace($restoreLogPath) -and $restoreLogCheck.matches
$nativeAssetListingReady = -not [string]::IsNullOrWhiteSpace($nativeAssetListingPath) -and $nativeAssetListingCheck.matches
$dependencyProbeLogReady = -not [string]::IsNullOrWhiteSpace($dependencyProbeLogPath) -and $dependencyProbeLogCheck.matches
$smokeLogReady = -not [string]::IsNullOrWhiteSpace($smokeLogPath) -and $smokeLogCheck.matches

$runtimeSmokeExitCodeIsZero = $false
if ($null -ne $runtimeSmokeExitCodeValue) {
  $runtimeSmokeExitCodeIsZero = [int]$runtimeSmokeExitCodeValue -eq 0
}

$smokePassedByStatus = [string]::Equals($smokeStatus, "passed", [System.StringComparison]::OrdinalIgnoreCase)
$driverBlocked = $smokeStatus -like "*blocked-by-cuda-driver*"

$allItemsPassed = $missingItemIds.Count -eq 0 -and $notPassedIds.Count -eq 0 -and $missingEvidenceIds.Count -eq 0

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ($recordKind -in @("post-publish-verification-record", "post-publish-verification-record-template")) -Severity "blocker" -Detail "Record kind must identify a post-publish verification record or template."
  New-ValidationItem -Id "post-publish-proof-classification-known" -Passed $postPublishProofClassificationKnown -Severity "blocker" -Detail "postPublishProofClassification must be template-only, owner-action-required, dependency-probe-only, blocked-by-cuda-driver, or post-publish-package-consumer-runtime."
  New-ValidationItem -Id "real-record-kind" -Passed ($recordKind -eq "post-publish-verification-record" -and -not $templateOnly) -Severity "proof-required" -Detail "Real proof requires recordKind=post-publish-verification-record and templateOnly=false."
  New-ValidationItem -Id "post-publish-proof-classification-promotable" -Passed $postPublishProofClassificationPromotable -Severity "proof-required" -Detail "Post-publish proof requires postPublishProofClassification=post-publish-package-consumer-runtime."
  New-ValidationItem -Id "post-publish-required-evidence-declared" -Passed ($missingDeclaredPostPublishRequiredEvidence.Count -eq 0) -Severity "blocker" -Detail "postPublishRequiredEvidence must declare every owner command plan required field; missing=$($missingDeclaredPostPublishRequiredEvidence -join ', ')."
  New-ValidationItem -Id "metadata" -Passed $metadataReady -Severity "proof-required" -Detail "selectedChannel, channelSourceUri, ownerName, reviewerName, and publishedVersion are required."
  New-ValidationItem -Id "performs-publish" -Passed (-not $performsPublish) -Severity "blocker" -Detail "Verification records must not perform publish."
  New-ValidationItem -Id "channel-source" -Passed $channelSourceReady -Severity "proof-required" -Detail "selectedChannel, channelSourceUri, managedPackageSource, and runtimePackageSource must be filled from the published channel."
  New-ValidationItem -Id "runtime-package-key" -Passed $runtimePackageKeyReady -Severity "proof-required" -Detail "expectedRuntimePackageKey must be filled and match the runtime package id, source, or URL from the selected channel."
  New-ValidationItem -Id "published-package-version" -Passed ($packageIdentityReady -and $publishedPackageUrlReady) -Severity "proof-required" -Detail "Published package id/version, publishedPackageUrl, package URL, and downloaded nupkg SHA256 are required."
  New-ValidationItem -Id "published-package-url" -Passed ($managedPackageUrlReady -and $runtimePackageUrlReady) -Severity "proof-required" -Detail "Managed/runtime package URLs must be absolute selected-channel URLs or file: URIs."
  New-ValidationItem -Id "published-package-sha256-source" -Passed ($managedNupkgSha256Ready -and $runtimeNupkgSha256Ready -and $packageSha256SourceReady) -Severity "proof-required" -Detail "Managed/runtime package SHA256 values, SHA256 source notes, and download timestamps are required."
  New-ValidationItem -Id "clean-consumer-root" -Passed ($cleanConsumerReady -and $cleanConsumerRootOutsideRepository) -Severity "proof-required" -Detail "cleanConsumerRoot must point to a fresh consumer directory outside the source repository and cleanConsumerRootOutsideRepository=true is required."
  New-ValidationItem -Id "clean-consumer-outside-repository" -Passed $cleanConsumerOutsideRepository -Severity "proof-required" -Detail "cleanConsumerRoot must resolve outside the source repository tree."
  New-ValidationItem -Id "clean-consumer-project-identity" -Passed $consumerProjectIdentityReady -Severity "proof-required" -Detail "consumerProjectName and consumerProjectPath ending in .csproj are required for the clean consumer."
  New-ValidationItem -Id "host-runtime-metadata" -Passed $hostReady -Severity "proof-required" -Detail "Host owner, machine, OS, GPU, driver, CUDA driver/runtime, TensorRT runtime/line, and cuDNN metadata are required."
  New-ValidationItem -Id "command-lines" -Passed $commandsReady -Severity "proof-required" -Detail "restoreCommand, buildCommand, and smokeCommand must be recorded."
  New-ValidationItem -Id "runtime-key-smoke-command" -Passed $smokeCommandRuntimeKeyReady -Severity "proof-required" -Detail "smokeCommand must include --runtime-package-key and the expected runtime package key."
  New-ValidationItem -Id "stdout-summary" -Passed $stdoutSummaryReady -Severity "proof-required" -Detail "A reviewed stdoutSummary is required; a log path alone is not enough."
  New-ValidationItem -Id "stderr-summary" -Passed $stderrSummaryReady -Severity "proof-required" -Detail "A reviewed stderrSummary is required. Use an explicit no-stderr-emitted note when the process produced no stderr."
  New-ValidationItem -Id "stdout-stderr-summary" -Passed $stdoutStderrSummaryReady -Severity "proof-required" -Detail "Both stdoutSummary and stderrSummary must be reviewed for real post-publish package-consumer proof."
  New-ValidationItem -Id "no-project-reference" -Passed $noProjectReference -Severity "proof-required" -Detail "Post-publish verification requires no ProjectReference."
  New-ValidationItem -Id "no-local-package-source" -Passed $noLocalPackageSource -Severity "proof-required" -Detail "Post-publish verification requires no local NuGet package source, repository artifact feed, or fallback folder."
  New-ValidationItem -Id "no-local-nupkg-reference" -Passed $noLocalNupkgPackageReference -Severity "proof-required" -Detail "Post-publish verification requires versioned PackageReference from the selected channel, not a direct local .nupkg reference."
  New-ValidationItem -Id "clean-consumer-scan-passed" -Passed $cleanConsumerProjectScanPassed -Severity "proof-required" -Detail "The post-publish clean consumer scan must pass before a verification record can close the release issue."
  New-ValidationItem -Id "restore-log" -Passed $restoreLogReady -Severity "proof-required" -Detail "restoreLogPath and a matching 64-character restoreLogSha256 are required when -RequireExistingLog is used."
  New-ValidationItem -Id "native-assets-copied" -Passed ($nativeAssetsCopied -and $nativeAssetListingReady) -Severity "proof-required" -Detail "nativeAssetsCopied=true plus nativeAssetListingPath and matching nativeAssetListingSha256 are required."
  New-ValidationItem -Id "dependency-probe-log" -Passed ($dependencyProbePassed -and $dependencyProbeLogReady) -Severity "proof-required" -Detail "dependencyProbePassed=true plus dependencyProbeLogPath and matching dependencyProbeLogSha256 are required, but DependencyProbe is not runtime proof."
  New-ValidationItem -Id "runtime-smoke-log" -Passed ($runtimeSmokePassed -and $runtimeSmokeExitCodeIsZero -and $smokePassedByStatus -and $smokeLogReady) -Severity "proof-required" -Detail "runtimeSmokePassed=true, runtimeSmokeExitCode=0, smokeStatus=passed, runtimeSmokeLogPath, and a matching 64-character runtimeSmokeLogSha256 are required."
  New-ValidationItem -Id "restore-log-sha256-match" -Passed $restoreLogCheck.matches -Severity "proof-required" -Detail "restoreLogSha256 must match restoreLogPath when -RequireExistingLog is used."
  New-ValidationItem -Id "native-asset-listing-sha256-match" -Passed $nativeAssetListingCheck.matches -Severity "proof-required" -Detail "nativeAssetListingSha256 must match nativeAssetListingPath when -RequireExistingLog is used."
  New-ValidationItem -Id "dependency-probe-log-sha256-match" -Passed $dependencyProbeLogCheck.matches -Severity "proof-required" -Detail "dependencyProbeLogSha256 must match dependencyProbeLogPath when -RequireExistingLog is used."
  New-ValidationItem -Id "smoke-log-sha256-match" -Passed $smokeLogCheck.matches -Severity "proof-required" -Detail "smokeLogSha256 must match smokeLogPath when -RequireExistingLog is used."
  New-ValidationItem -Id "not-driver-blocked" -Passed (-not $driverBlocked) -Severity "proof-required" -Detail "blocked-by-cuda-driver is not smoke passed."
  New-ValidationItem -Id "required-items-present" -Passed ($missingItemIds.Count -eq 0) -Severity "blocker" -Detail "All required post-publish verification items must be present."
  New-ValidationItem -Id "execution-steps-present" -Passed ($missingExecutionStepIds.Count -eq 0) -Severity "blocker" -Detail "Post-publish records must retain the publish, download, clean consumer, restore, native-copy, dependency probe, runtime smoke, and validator execution steps."
  New-ValidationItem -Id "required-items-passed" -Passed ($notPassedIds.Count -eq 0) -Severity "proof-required" -Detail "All required post-publish verification items must pass."
  New-ValidationItem -Id "item-evidence" -Passed ($missingEvidenceIds.Count -eq 0) -Severity "proof-required" -Detail "All required items must carry evidence or evidence references."
  New-ValidationItem -Id "declared-proof" -Passed ($declaredProof -and $declaredCanClose) -Severity "proof-required" -Detail "Record must explicitly declare post-publish proof and issue-close readiness."
)

$validationOwnerGuidance = @{
  "record-kind" = @("真实发布后将 template 复制为 post-publish-verification-record。", "template 不能关闭 release issue。")
  "post-publish-proof-classification-known" = @("真实发布后的 clean consumer proof 必须使用 post-publish-package-consumer-runtime。", "未知 proof classification 不能进入 release close gate。")
  "real-record-kind" = @("设置 recordKind=post-publish-verification-record 且 templateOnly=false。", "post-publish template / draft 不是 proof。")
  "post-publish-proof-classification-promotable" = @("确认 proof 来自真实发布 channel 的 clean consumer restore/build/smoke。", "owner-action-required、dependency-probe-only、blocked-by-cuda-driver 不能关闭 release issue。")
  "post-publish-required-evidence-declared" = @("保留 postPublishRequiredEvidence 中所有 owner command plan required field。", "缺失或删除 required evidence 字段不能制造 post-publish proof。")
  "metadata" = @("回填真实发布 channel、owner/reviewer 和已发布版本。", "发布计划或 dry-run metadata 不能替代真实发布后 metadata。")
  "performs-publish" = @("保持 performsPublish=false；发布动作只能由 owner 明确执行。", "验证器不能承担发布动作。")
  "channel-source" = @("填写 NuGet/GitHub Packages 等真实 channel URL 和 package source。", "local feed 或本地 bin 输出不能作为 post-publish proof。")
  "runtime-package-key" = @("确认 post-publish consumer 使用目标 runtime package key。", "其它 CUDA/TensorRT 组合不能替代当前 release proof。")
  "published-package-version" = @("从真实 channel 下载包并记录 id/version/URL/SHA256。", "本地构建产物 hash 不能证明公开 channel 已发布。")
  "published-package-url" = @("记录 managed/runtime 包在真实 channel 上的绝对 URL 或 file: URI。", "本地相对路径或 bin 输出不能证明 package 已发布。")
  "published-package-sha256-source" = @("记录 SHA256 的来源和下载时间戳，并确认 hash 来自 channel 下载包。", "只写 hash 而没有来源和下载时间不可审计。")
  "clean-consumer-root" = @("在源码仓库外创建全新 clean consumer 目录。", "源码工作区不是 clean post-publish consumer。")
  "clean-consumer-outside-repository" = @("确认 clean consumer 解析后的完整路径不在源码仓库树下。", "仓库内 consumer 会污染 post-publish proof。")
  "clean-consumer-project-identity" = @("记录 clean consumer 项目名和 .csproj 路径。", "未标识 consumer project 的日志不可复核。")
  "host-runtime-metadata" = @("填写运行 post-publish smoke 的 OS/GPU/driver/CUDA/TensorRT/cuDNN 信息。", "没有 host metadata 的运行结果不能进入 release close gate。")
  "command-lines" = @("保留 restore/build/smoke 原始命令。", "只记录结果不记录命令不可重放。")
  "runtime-key-smoke-command" = @("smokeCommand 必须显式包含 --runtime-package-key 和目标 key。", "未绑定 runtime key 的 smoke 不证明目标包。")
  "stdout-summary" = @("人工复核 restore/build/probe/smoke stdout 并填写摘要。", "只保留 log 路径不代表 reviewer 已复核输出。")
  "stderr-summary" = @("人工复核 stderr；为空时填写 no-stderr-emitted。", "遗漏 stderrSummary 不能关闭 release issue。")
  "stdout-stderr-summary" = @("同时保留 stdout/stderr 复核摘要。", "摘要不完整时 release issue 必须保持打开。")
  "no-project-reference" = @("确认 clean consumer 只通过真实 channel PackageReference 消费包。", "ProjectReference 会绕过发布包验证。")
  "no-local-package-source" = @("确认 clean consumer 没有 RestoreSources、fallback folders、NuGet.config local feed 或 repository artifact feed。", "本地 package source 会绕过真实发布 channel。")
  "no-local-nupkg-reference" = @("确认 PackageReference 使用发布版本号，而不是本地 .nupkg 路径。", "直接 .nupkg 引用不能证明 post-publish channel package。")
  "clean-consumer-scan-passed" = @("先运行 Test-PostPublishCleanConsumerProject.ps1 并只接受 scanPassed=true 的结果。", "失败的 clean consumer 扫描不能被 post-publish record 手动绕过。")
  "restore-log" = @("保存 restore log 并计算 SHA256。", "restore 成功口头说明不能替代可审计 log。")
  "native-assets-copied" = @("保存 native asset listing 并计算 SHA256。", "restore 成功不等于 native assets 已复制。")
  "dependency-probe-log" = @("运行 dependency probe 并保存 log/hash，作为诊断链的一部分。", "dependency probe 仍不能替代 runtime smoke。")
  "runtime-smoke-log" = @("在兼容主机运行 runtime smoke，保存 log/hash，并确认 exitCode=0、smokeStatus=passed。", "blocked-by-cuda-driver、pending、failed 都不是 post-publish proof。")
  "restore-log-sha256-match" = @("用 -RequireExistingLog 校验 restore log hash。", "hash 不匹配的 restore log 不可采信。")
  "native-asset-listing-sha256-match" = @("用 -RequireExistingLog 校验 native asset listing hash。", "hash 不匹配的 native listing 不可采信。")
  "dependency-probe-log-sha256-match" = @("用 -RequireExistingLog 校验 dependency probe log hash。", "hash 不匹配的 probe log 不可采信。")
  "smoke-log-sha256-match" = @("用 -RequireExistingLog 校验 smoke log hash。", "hash 不匹配的 smoke log 不能关闭 release issue。")
  "not-driver-blocked" = @("换到兼容 CUDA driver / TensorRT runtime 主机重新运行 post-publish smoke。", "blocked-by-cuda-driver 永远不是 smoke passed。")
  "required-items-present" = @("保留模板中的所有 required verification item。", "删除 required item 不能制造完成度。")
  "execution-steps-present" = @("保留模板中的所有 execution step。", "删除 execution step 不能关闭 release issue。")
  "required-items-passed" = @("每个 required item 都必须有真实执行结果并标记 passed。", "pending/blocked/failed item 不能进入 close gate。")
  "item-evidence" = @("为每个 required item 填写 evidence 或 evidencePath。", "status=passed 但无 evidence 不可审计。")
  "declared-proof" = @("只有所有 checks 通过后才把 isPostPublishVerificationProof 与 canCloseReleaseIssue 两个字段置为 true。", "提前声明 true 会被 validationItems 拦截。")
}

foreach ($item in $validationItems) {
  if ($validationOwnerGuidance.ContainsKey($item.id)) {
    $item.ownerAction = $validationOwnerGuidance[$item.id][0]
    $item.boundary = $validationOwnerGuidance[$item.id][1]
  }
}

$failedBlockers = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedProofItems = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "proof-required" })
$notPassedCount = $notPassedIds.Count
$missingEvidenceCount = $missingEvidenceIds.Count
$failedBlockerCount = $failedBlockers.Count
$failedProofItemCount = $failedProofItems.Count
$canCloseReleaseIssue = $failedBlockers.Count -eq 0 -and $failedProofItems.Count -eq 0 -and -not $isTemplateOnly
$isPostPublishVerificationProof = $canCloseReleaseIssue -and $declaredProof

if ($isPostPublishVerificationProof) {
  $validationState = "real-post-publish-verification-proof"
}
elseif ($isTemplateOnly) {
  $validationState = "template-only"
}
elseif ($failedBlockers.Count -gt 0) {
  $validationState = "invalid-record"
}
else {
  $validationState = "incomplete-post-publish-verification"
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
$jsonPath = Join-Path $outputRoot "post-publish-verification-validation.json"
$markdownPath = Join-Path $outputRoot "post-publish-verification-validation.md"

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "post-publish-verification-validation"
  inputPath = $resolvedInputPath
  inputRecordKind = $recordKind
  validationState = $validationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationKnown = $postPublishProofClassificationKnown
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  isTemplateOnly = $isTemplateOnly
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  canCloseReleaseIssue = $canCloseReleaseIssue
  failedBlockerCount = $failedBlockerCount
  failedProofItemCount = $failedProofItemCount
  notPassedCount = $notPassedCount
  missingEvidenceCount = $missingEvidenceCount
  selectedChannel = $selectedChannel
  channelSourceUri = $channelSourceUri
  publishedPackageUrl = $publishedPackageUrl
  managedPackageId = $managedPackageId
  managedPackageVersion = $managedPackageVersion
  runtimePackageId = $runtimePackageId
  runtimePackageVersion = $runtimePackageVersion
  expectedRuntimePackageKey = $expectedRuntimePackageKey
  runtimePackageKeyReady = $runtimePackageKeyReady
  managedPackageUrl = $managedPackageUrl
  runtimePackageUrl = $runtimePackageUrl
  managedPackageUrlReady = $managedPackageUrlReady
  runtimePackageUrlReady = $runtimePackageUrlReady
  managedNupkgSha256Ready = $managedNupkgSha256Ready
  runtimeNupkgSha256Ready = $runtimeNupkgSha256Ready
  managedPackageSha256Source = $managedPackageSha256Source
  runtimePackageSha256Source = $runtimePackageSha256Source
  managedPackageDownloadTimestampUtc = $managedPackageDownloadTimestampUtc
  runtimePackageDownloadTimestampUtc = $runtimePackageDownloadTimestampUtc
  packageSha256SourceReady = $packageSha256SourceReady
  cleanConsumerRoot = $cleanConsumerRoot
  cleanConsumerRootOutsideRepository = $cleanConsumerRootOutsideRepository
  cleanConsumerOutsideRepository = $cleanConsumerOutsideRepository
  consumerProjectName = $consumerProjectName
  consumerProjectPath = $consumerProjectPath
  consumerProjectIdentityReady = $consumerProjectIdentityReady
  hostReady = $hostReady
  host = [pscustomobject]@{
    ownerName = $hostOwnerName
    machineName = $hostMachineName
    osDescription = $hostOsDescription
    gpuName = $hostGpuName
    driverVersion = $hostDriverVersion
    cudaDriverSupportedRuntime = $hostCudaDriverSupportedRuntime
    cudaRuntimeVersion = $hostCudaRuntimeVersion
    tensorRtRuntimeVersion = $hostTensorRtRuntimeVersion
    tensorRtLine = $hostTensorRtLine
    cudnnVersion = $hostCudnnVersion
  }
  commandsReady = $commandsReady
  smokeCommandRuntimeKeyReady = $smokeCommandRuntimeKeyReady
  stdoutSummaryReady = $stdoutSummaryReady
  stderrSummaryReady = $stderrSummaryReady
  stdoutStderrSummaryReady = $stdoutStderrSummaryReady
  restoreCommand = $restoreCommand
  buildCommand = $buildCommand
  smokeCommand = $smokeCommand
  stdoutSummary = $stdoutSummary
  stderrSummary = $stderrSummary
  managedPackageSource = $managedPackageSource
  runtimePackageSource = $runtimePackageSource
  noProjectReference = $noProjectReference
  noLocalPackageSource = $noLocalPackageSource
  noLocalNupkgPackageReference = $noLocalNupkgPackageReference
  cleanConsumerProjectScanPassed = $cleanConsumerProjectScanPassed
  nativeAssetsCopied = $nativeAssetsCopied
  dependencyProbePassed = $dependencyProbePassed
  runtimeSmokePassed = $runtimeSmokePassed
  runtimeSmokeExitCodeIsZero = $runtimeSmokeExitCodeIsZero
  smokeStatus = $smokeStatus
  requireExistingLog = $RequireExistingLog.IsPresent
  restoreLogSha256Ready = $restoreLogSha256Ready
  nativeAssetListingSha256Ready = $nativeAssetListingSha256Ready
  dependencyProbeLogSha256Ready = $dependencyProbeLogSha256Ready
  smokeLogSha256Ready = $smokeLogSha256Ready
  runtimeSmokeLogPath = $runtimeSmokeLogPath
  runtimeSmokeLogSha256 = $runtimeSmokeLogSha256
  runtimeSmokeLogSha256Ready = $smokeLogSha256Ready
  restoreLogSha256Matches = $restoreLogCheck.matches
  nativeAssetListingSha256Matches = $nativeAssetListingCheck.matches
  dependencyProbeLogSha256Matches = $dependencyProbeLogCheck.matches
  smokeLogSha256Matches = $smokeLogCheck.matches
  restoreLog = $restoreLogCheck
  nativeAssetListing = $nativeAssetListingCheck
  dependencyProbeLog = $dependencyProbeLogCheck
  smokeLog = $smokeLogCheck
  missingItemIds = @($missingItemIds)
  missingExecutionStepIds = @($missingExecutionStepIds)
  notPassedIds = @($notPassedIds.ToArray())
  missingEvidenceIds = @($missingEvidenceIds.ToArray())
  validationItems = @($validationItems)
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
  missingPostPublishRequiredEvidence = @($missingDeclaredPostPublishRequiredEvidence)
  promotionRules = @(
    "Template-only records are not post-publish proof.",
    "Post-publish verification does not publish packages.",
    "Package id/version, absolute package URL, runtime package key, downloaded nupkg SHA256, SHA256 source notes, and download timestamps must come from the selected channel.",
    "Clean consumer project identity outside the repository tree, no ProjectReference, package source, native-copy, dependency probe, compatible-host smoke evidence, host runtime metadata, --runtime-package-key smoke command, reviewed stdout/stderr summaries, and matching log SHA256 values are required before closing the release issue.",
    "Clean consumer scans must reject local package sources, repository artifact feeds, fallback folders, direct .nupkg references, and ProjectReference entries.",
    "When -RequireExistingLog is used, restore/native asset listing/dependency probe/smoke SHA256 values must match the referenced files.",
    "DependencyProbe evidence is diagnostics only and cannot promote runtime execution proof.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Bridge-only package consumer logs, bridge-only wrapper surface, Skipped=True, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, and copied Parser/ParserRefitter diagnostic snapshots are not post-publish proof."
  )
  ownerActionSummary = @(
    "Owner must first perform the real publish on the selected channel outside this validator.",
    "Owner must record managed/runtime package URLs, SHA256 values, SHA256 source notes, and download timestamps from the selected channel.",
    "Owner must create a clean consumer outside the repository and restore from the real published channel.",
    "Owner must verify PackageReference consumption with no ProjectReference in the clean consumer.",
    "Owner must preserve restore/native asset listing/dependency probe/smoke logs and matching SHA256 values.",
    "Owner must run this validator with -RequireExistingLog -FailOnNotProof before closing the release issue.",
    "If validationState is template-only, incomplete-post-publish-verification, dependency-probe-only, blocked-by-cuda-driver, or invalid-record, canCloseReleaseIssue must remain false."
  )
  nonSubstituteProofKinds = @(
    "collection package",
    "backfill plan",
    "runbook",
    "input package",
    "template",
    "draft",
    "example",
    "local feed",
    "local package source",
    "local nupkg",
    "repository artifact package source",
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
    "pre-publish dry run",
    "Windows handoff for Linux proof"
  )
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Verification Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- proof classification: ``$postPublishProofClassification``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- failed blocker count: ``$failedBlockerCount``")
$lines.Add("- failed proof item count: ``$failedProofItemCount``")
$lines.Add("- not passed item count: ``$notPassedCount``")
$lines.Add("- missing evidence count: ``$missingEvidenceCount``")
$lines.Add("- missing execution step count: ``$($missingExecutionStepIds.Count)``")
$lines.Add("- selected channel: ``$selectedChannel``")
$lines.Add("- channel source URI: ``$channelSourceUri``")
$lines.Add("- managed package: ``$managedPackageId`` ``$managedPackageVersion``")
$lines.Add("- runtime package: ``$runtimePackageId`` ``$runtimePackageVersion``")
$lines.Add("- expected runtime package key: ``$expectedRuntimePackageKey``")
$lines.Add("- runtime package key ready: ``$runtimePackageKeyReady``")
$lines.Add("- clean consumer root: ``$cleanConsumerRoot``")
$lines.Add("- consumer project identity ready: ``$consumerProjectIdentityReady``")
$lines.Add("- host metadata ready: ``$hostReady``")
$lines.Add("- commands ready: ``$commandsReady``")
$lines.Add("- smoke command runtime key ready: ``$smokeCommandRuntimeKeyReady``")
$lines.Add("- stdout summary ready: ``$stdoutSummaryReady``")
$lines.Add("- stderr summary ready: ``$stderrSummaryReady``")
$lines.Add("- stdout/stderr summary ready: ``$stdoutStderrSummaryReady``")
$lines.Add("- require existing log: ``$($RequireExistingLog.IsPresent)``")
$lines.Add("- restore log SHA256 ready: ``$restoreLogSha256Ready``")
$lines.Add("- native asset listing SHA256 ready: ``$nativeAssetListingSha256Ready``")
$lines.Add("- dependency probe log SHA256 ready: ``$dependencyProbeLogSha256Ready``")
$lines.Add("- smoke log SHA256 ready: ``$smokeLogSha256Ready``")
$lines.Add("- postPublishRequiredEvidence count: ``$($postPublishRequiredEvidence.Count)``")
$lines.Add("- missing postPublishRequiredEvidence declarations: ``$($missingDeclaredPostPublishRequiredEvidence.Count)``")
$lines.Add("- restore log SHA256 matches: ``$($restoreLogCheck.matches)``")
$lines.Add("- native asset listing SHA256 matches: ``$($nativeAssetListingCheck.matches)``")
$lines.Add("- dependency probe log SHA256 matches: ``$($dependencyProbeLogCheck.matches)``")
$lines.Add("- smoke log SHA256 matches: ``$($smokeLogCheck.matches)``")
$lines.Add("- runtime smoke passed: ``$runtimeSmokePassed``")
$lines.Add("- input path: ``$resolvedInputPath``")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Execution Steps")
$lines.Add("")
$lines.Add("| ID | Present |")
$lines.Add("| --- | --- |")
foreach ($id in $requiredExecutionStepIds) {
  $lines.Add("| ``$id`` | ``$($executionStepById.ContainsKey($id))`` |")
}
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $summary.promotionRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $summary.ownerActionSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Non-Substitute Proof Kinds")
$lines.Add("")
foreach ($item in $summary.nonSubstituteProofKinds) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Owner Actions By Validation Item")
$lines.Add("")
$lines.Add("| ID | Owner action | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | $(ConvertTo-MarkdownCell $item.ownerAction) | $(ConvertTo-MarkdownCell $item.boundary) |")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification validation written to $jsonPath"
Write-Host "Post-publish verification validation written to $markdownPath"
Write-Host "ValidationState=$validationState IsPostPublishVerificationProof=$isPostPublishVerificationProof CanCloseReleaseIssue=$canCloseReleaseIssue FailedBlockers=$failedBlockerCount FailedProofItems=$failedProofItemCount NotPassedItems=$notPassedCount MissingEvidenceItems=$missingEvidenceCount"

if ($FailOnNotProof.IsPresent -and -not $isPostPublishVerificationProof) {
  Write-Error "Post-publish verification is not proof. ValidationState=$validationState"
  exit 1
}

