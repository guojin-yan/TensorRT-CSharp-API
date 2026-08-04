[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot,
  [switch]$AllowRuntimeSmokeBlocked,
  [switch]$WarnOnly
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Test-File {
  param([string]$RelativePath)
  return Test-Path -LiteralPath (Join-Path $RepositoryRoot $RelativePath) -PathType Leaf
}

function Test-Directory {
  param([string]$RelativePath)
  return Test-Path -LiteralPath (Join-Path $RepositoryRoot $RelativePath) -PathType Container
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
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

function New-Check {
  param(
    [string]$Category,
    [string]$Name,
    [string]$Status,
    [string]$Severity,
    [string]$Detail,
    [string]$EvidencePath
  )

  [pscustomobject]@{
    category = $Category
    name = $Name
    status = $Status
    severity = $Severity
    detail = $Detail
    evidencePath = $EvidencePath
    isBlocking = [string]::Equals($Severity, "blocker", [System.StringComparison]::OrdinalIgnoreCase) -and -not [string]::Equals($Status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
    isWarning = [string]::Equals($Severity, "warning", [System.StringComparison]::OrdinalIgnoreCase) -and -not [string]::Equals($Status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
  }
}

function New-SampleStatus {
  param(
    [string]$Name,
    [string]$RelativePath,
    [string]$Kind
  )

  $exists = (Test-Directory $RelativePath) -or (Test-File $RelativePath)
  if ($exists) {
    $status = "cataloged-not-run"
    $diagnostic = "sample or smoke entry is present; execution is tracked separately by environment-specific smoke gates."
  }
  else {
    $status = "missing"
    $diagnostic = "sample or smoke entry was not found."
  }

  [pscustomobject]@{
    name = $Name
    kind = $Kind
    path = $RelativePath
    status = $status
    diagnostic = $diagnostic
  }
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackage = @($manifest.packages | Where-Object { [string]::Equals([string]$_.key, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) }) | Select-Object -First 1
if (-not $runtimePackage) {
  throw "Runtime package key '$RuntimePackageKey' was not found in '$manifestPath'."
}

$runtimeReadiness = Read-JsonOrNull "artifacts\package-readiness\runtime-package-readiness-summary.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$bridgeConsumer = Read-JsonOrNull "artifacts\package-consumer\bridge-package-consumer-validation-summary.json"
$compatibleBridgeRuntimeProofPath = "artifacts\package-consumer\bridge-runtime\win-x64-trt10.11-cuda12.9-cudnn9.22\bridge-package-runtime-consumer-proof.json"
$compatibleBridgeRuntimeProof = Read-JsonOrNull $compatibleBridgeRuntimeProofPath
$localFeedConsumer = Read-JsonOrNull "artifacts\local-feed-consumer\local-nuget-feed-consumer-summary.json"
$releaseChecklist = Read-JsonOrNull "artifacts\release-candidate\release-candidate-checklist.json"
$compatibleHostRuntimeProofCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$runtimeLimitationPolicyPath = "pack\runtime-validation-disclosure-policy.json"
$runtimeLimitationPolicy = Read-JsonOrNull $runtimeLimitationPolicyPath
$runtimeLimitationEntry = @(
  if ($runtimeLimitationPolicy) {
    $runtimeLimitationPolicy.runtimeKeys | Where-Object {
      [string]::Equals([string]$_.runtimePackageKey, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)
    }
  }
) | Select-Object -First 1
$runtimeLimitationNoticePath = if ($runtimeLimitationPolicy) { [string]$runtimeLimitationPolicy.releaseNoticePath } else { "" }
$runtimeLimitationNotice = if (-not [string]::IsNullOrWhiteSpace($runtimeLimitationNoticePath)) {
  $noticePath = Join-Path $RepositoryRoot $runtimeLimitationNoticePath
  if (Test-Path -LiteralPath $noticePath -PathType Leaf) {
    Get-Content -LiteralPath $noticePath -Raw -Encoding utf8
  }
  else {
    ""
  }
}
else {
  ""
}
$runtimeLimitationMissingNoticeMarkers = @(
  if ($runtimeLimitationEntry) {
    foreach ($marker in @($runtimeLimitationEntry.requiredNoticeMarkers)) {
      if (-not $runtimeLimitationNotice.Contains([string]$marker, [System.StringComparison]::Ordinal)) {
        [string]$marker
      }
    }
  }
)

$checks = New-Object System.Collections.Generic.List[object]

if ($runtimeReadiness -and $runtimeReadiness.managedPackage.status -eq "ready") { $managedPackageStatus = "ready" } else { $managedPackageStatus = "missing" }
$fullRuntimePackageStatus = "retired-not-required"
if ($runtimeReadiness -and $runtimeReadiness.bridgePackage.status -eq "ready") { $bridgePackageStatus = "ready" } else { $bridgePackageStatus = "missing" }
if ($bridgeConsumer -and $bridgeConsumer.NativeDependencyStatus -eq "ready") { $bridgeConsumerStatus = "ready" } else { $bridgeConsumerStatus = "missing-or-not-ready" }
if ($packageConsumer -and $packageConsumer.NativeAssetsFound -eq $packageConsumer.NativeAssetsExpected) { $packageConsumerStatus = "historical-record-present" } else { $packageConsumerStatus = "historical-record-missing" }
$compatibleBridgeRuntimeProofReady = $compatibleBridgeRuntimeProof -and
  [string]$compatibleBridgeRuntimeProof.proofClassification -eq "compatible-host-bridge-package-runtime" -and
  [string]$compatibleBridgeRuntimeProof.smokeStatus -eq "passed" -and
  [int]$compatibleBridgeRuntimeProof.exitCode -eq 0 -and
  [bool]$compatibleBridgeRuntimeProof.isRuntimeExecutionProof -and
  -not [bool]$compatibleBridgeRuntimeProof.isPackageConsumerRuntimeProof -and
  [bool]$compatibleBridgeRuntimeProof.canPromoteCompatibleHostRuntimeProof -and
  -not [bool]$compatibleBridgeRuntimeProof.canPublishPublicly -and
  -not [bool]$compatibleBridgeRuntimeProof.canCloseReleaseIssue
$compatibleBridgeRuntimeProofStatus = if ($compatibleBridgeRuntimeProofReady) { "ready" } elseif ($compatibleBridgeRuntimeProof) { "not-ready" } else { "missing" }

$checks.Add((New-Check -Category "managed-package" -Name "Managed package exists" -Status $managedPackageStatus -Severity "blocker" -Detail "managed package status from runtime readiness." -EvidencePath "artifacts/package-readiness/runtime-package-readiness-summary.json")) | Out-Null
$checks.Add((New-Check -Category "package-policy" -Name "Vendor package route retired" -Status "ready" -Severity "blocker" -Detail "Only managed and bridge packages are publishable; full/vendor package absence is required and not a readiness blocker." -EvidencePath "pack/external-vendor-runtime-policy.json")) | Out-Null
$checks.Add((New-Check -Category "runtime-package" -Name "Bridge package exists" -Status $bridgePackageStatus -Severity "blocker" -Detail "bridge split package status from runtime readiness." -EvidencePath "artifacts/package-readiness/runtime-package-readiness-summary.json")) | Out-Null
$checks.Add((New-Check -Category "consumer" -Name "Bridge package consumer" -Status $bridgeConsumerStatus -Severity "blocker" -Detail "bridge consumer validates wrapper surface and native dependency probe without claiming full runtime callback proof." -EvidencePath "artifacts/package-consumer/bridge-package-consumer-validation-summary.json")) | Out-Null
$checks.Add((New-Check -Category "runtime-proof" -Name "TRT10 compatible bridge package runtime proof" -Status $compatibleBridgeRuntimeProofStatus -Severity "warning" -Detail "External PackageReference consumer executes identity build/serialize/deserialize/enqueue/output-compare with system TensorRT/CUDA dependencies. This is compatible-host runtime execution evidence only; it remains isPackageConsumerRuntimeProof=false and does not make TRT11/CUDA13.2 runtime validated." -EvidencePath $compatibleBridgeRuntimeProofPath.Replace('\', '/'))) | Out-Null
$checks.Add((New-Check -Category "consumer" -Name "Historical vendor-package consumer" -Status $packageConsumerStatus -Severity "warning" -Detail "Historical vendor-package consumer records remain diagnostic inputs only and are not required by bridge-only publication." -EvidencePath "artifacts/package-consumer/package-consumer-validation-summary.json")) | Out-Null

if ($packageConsumer) { $smokeStatus = [string]$packageConsumer.SmokeResult } else { $smokeStatus = "missing" }
$fallbackPackageConsumerEvidenceKind = switch ($smokeStatus) {
  "passed" { "historical-vendor-package-consumer-smoke"; break }
  "not-requested" { "package-consumer-native-copy"; break }
  "blocked-by-application-control" { "historical-vendor-package-consumer-smoke-application-control-blocked"; break }
  "blocked-by-cuda-driver" { "historical-vendor-package-consumer-smoke-driver-blocked"; break }
  "failed" { "historical-vendor-package-consumer-smoke-failed"; break }
  "missing" { "missing"; break }
  default { "historical-vendor-package-consumer-diagnostic"; break }
}
$fallbackRuntimeSmokeClassification = switch ($smokeStatus) {
  "passed" { "runtime-smoke-passed"; break }
  "not-requested" { "not-requested"; break }
  "blocked-by-application-control" { "runtime-smoke-application-control-blocked"; break }
  "blocked-by-cuda-driver" { "runtime-smoke-driver-blocked"; break }
  "failed" { "runtime-smoke-failed"; break }
  "missing" { "missing"; break }
  default { "runtime-smoke-diagnostic"; break }
}
$packageConsumerEvidenceKind = if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "EvidenceKind" -and -not [string]::IsNullOrWhiteSpace([string]$packageConsumer.EvidenceKind)) { [string]$packageConsumer.EvidenceKind } else { $fallbackPackageConsumerEvidenceKind }
$runtimeSmokeClassification = if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "RuntimeSmokeClassification" -and -not [string]::IsNullOrWhiteSpace([string]$packageConsumer.RuntimeSmokeClassification)) { [string]$packageConsumer.RuntimeSmokeClassification } else { $fallbackRuntimeSmokeClassification }
$isRuntimeExecutionEvidence = if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsRuntimeExecutionEvidence") { [bool]$packageConsumer.IsRuntimeExecutionEvidence } else { $smokeStatus -eq "passed" }
$isDependencyProbeOnly = if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsDependencyProbeOnly") { [bool]$packageConsumer.IsDependencyProbeOnly } else { -not $isRuntimeExecutionEvidence }
$isPackageConsumerRealCallbackRuntimeProof = if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsRealCallbackRuntimeProof") { [bool]$packageConsumer.IsRealCallbackRuntimeProof } else { $false }
$runtimeProofStatus = if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofStatus" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeReadiness.runtimeProofStatus)) {
  [string]$runtimeReadiness.runtimeProofStatus
}
elseif ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeExecution" -and $runtimeReadiness.runtimeExecution -and $runtimeReadiness.runtimeExecution.PSObject.Properties.Name -contains "status" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeReadiness.runtimeExecution.status)) {
  [string]$runtimeReadiness.runtimeExecution.status
}
elseif ($isRuntimeExecutionEvidence) {
  "ready"
}
else {
  $smokeStatus
}
$runtimeProofDiagnostic = if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofDiagnostic" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeReadiness.runtimeProofDiagnostic)) {
  [string]$runtimeReadiness.runtimeProofDiagnostic
}
elseif ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeExecution" -and $runtimeReadiness.runtimeExecution -and $runtimeReadiness.runtimeExecution.PSObject.Properties.Name -contains "diagnostic" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeReadiness.runtimeExecution.diagnostic)) {
  [string]$runtimeReadiness.runtimeExecution.diagnostic
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "SmokeDiagnostic" -and -not [string]::IsNullOrWhiteSpace([string]$packageConsumer.SmokeDiagnostic)) {
  [string]$packageConsumer.SmokeDiagnostic
}
else {
  "runtime proof evidence was not found."
}
$runtimeProofRequiredForRelease = if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") {
  [bool]$runtimeReadiness.runtimeProofRequiredForRelease
}
else {
  -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)
}
$runtimeLimitationStatusEligible = $runtimeLimitationEntry -and @($runtimeLimitationEntry.eligibleRuntimeProofStatuses) -contains $runtimeProofStatus
$runtimeLimitationDisclosureReady = $runtimeLimitationPolicy -and
  [bool]$runtimeLimitationPolicy.allowReleaseWithDocumentedUnverifiedRuntime -and
  $runtimeLimitationEntry -and
  [bool]$runtimeLimitationEntry.mayDowngradeToWarning -and
  $runtimeLimitationStatusEligible -and
  -not [string]::IsNullOrWhiteSpace($runtimeLimitationNotice) -and
  $runtimeLimitationMissingNoticeMarkers.Count -eq 0
$runtimeProofUsesDocumentedLimitation = $runtimeProofRequiredForRelease -and
  -not $isRuntimeExecutionEvidence -and
  $AllowRuntimeSmokeBlocked.IsPresent -and
  $runtimeLimitationDisclosureReady
$runtimeProofReleaseDisposition = if ($isRuntimeExecutionEvidence) {
  "runtime-validated"
}
elseif ($runtimeProofUsesDocumentedLimitation) {
  "documented-unverified-runtime"
}
else {
  "blocking-runtime-proof-required"
}
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)
$compatibleHostRuntimeProofCollectionBundleCommands = Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "commands" -DefaultValue $null
$compatibleHostRuntimeProofCollectionBundleStateFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")
$compatibleHostRuntimeProofCollectionBundleStateFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "state" -DefaultValue $compatibleHostRuntimeProofCollectionBundleStateFallback)
$compatibleHostRuntimeProofCollectionBundleState = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "collectionState" -DefaultValue $compatibleHostRuntimeProofCollectionBundleStateFallback)
$compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostRuntimeProofCollectionBundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundlePerformsPublish" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason" -DefaultValue "blocked-by-cuda-driver is not smoke passed")))
$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommandFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RunSmoke -RequirePackageReference")
$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommandFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "runPackageConsumerSmokeCommand" -DefaultValue $compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommandFallback)
$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundleCommands -Name "runPackageConsumerSmoke" -DefaultValue $compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommandFallback)
$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommandFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof")
$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommandFallback = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "validateFilledRecordCommand" -DefaultValue $compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommandFallback)
$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundleCommands -Name "validateFilledRecord" -DefaultValue $compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommandFallback)
$smokeSeverity = "warning"
$runtimeProofSeverity = if ($runtimeProofRequiredForRelease -and -not $isRuntimeExecutionEvidence -and -not $runtimeProofUsesDocumentedLimitation) { "blocker" } else { "warning" }
if ($packageConsumer) { $smokeDetail = [string]$packageConsumer.SmokeDiagnostic } else { $smokeDetail = "package consumer smoke report was not found." }
$smokeDetail = "$smokeDetail EvidenceKind=$packageConsumerEvidenceKind; RuntimeSmokeClassification=$runtimeSmokeClassification; IsRuntimeExecutionEvidence=$isRuntimeExecutionEvidence; IsDependencyProbeOnly=$isDependencyProbeOnly; IsRealCallbackRuntimeProof=$isPackageConsumerRealCallbackRuntimeProof; RuntimeProofStatus=$runtimeProofStatus; RuntimeProofRequiredForRelease=$runtimeProofRequiredForRelease."
$checks.Add((New-Check -Category "runtime-smoke" -Name "Historical vendor-package consumer smoke" -Status $smokeStatus -Severity $smokeSeverity -Detail "$smokeDetail Historical vendor-package smoke cannot satisfy current package-consumer or post-publish proof." -EvidencePath "artifacts/package-consumer/package-consumer-validation-summary.json")) | Out-Null
$runtimeProofDetail = "runtime proof status is separate from package/readiness overall status; release-required=$runtimeProofRequiredForRelease; release-disposition=$runtimeProofReleaseDisposition; runtime-execution-evidence=$isRuntimeExecutionEvidence; externalRuntimeProofConsumerProjectIdentityReady=$externalRuntimeProofConsumerProjectIdentityReady; externalRuntimeProofSmokeCommandRuntimeKeyReady=$externalRuntimeProofSmokeCommandRuntimeKeyReady; externalRuntimeProofHostReady=$externalRuntimeProofHostReady; externalRuntimeProofCommandsReady=$externalRuntimeProofCommandsReady; $runtimeProofDiagnostic Missing real external-runtime-proof-record.json blocks a runtime-validated claim. With explicit Owner opt-in and a validated limitation notice, an environment-limited runtime key may remain an unverified warning; blocked-by-cuda-driver is never smoke passed."
$checks.Add((New-Check -Category "runtime-proof" -Name "External bridge runtime proof" -Status $runtimeProofStatus -Severity $runtimeProofSeverity -Detail "$runtimeProofDetail Current proof must use managed plus bridge-only packages with host-installed NVIDIA dependencies." -EvidencePath "artifacts/package-readiness/runtime-package-readiness-summary.json; artifacts/final-release/external-runtime-proof-record.json")) | Out-Null
$runtimeLimitationDisclosureStatus = if ($isRuntimeExecutionEvidence) {
  "not-required-runtime-validated"
}
elseif ($runtimeLimitationDisclosureReady) {
  "ready"
}
else {
  "missing-or-invalid"
}
$runtimeLimitationDisclosureSeverity = if ($AllowRuntimeSmokeBlocked.IsPresent -and -not $isRuntimeExecutionEvidence -and -not $runtimeLimitationDisclosureReady) { "blocker" } else { "warning" }
$runtimeLimitationDisclosureDetail = "Owner opt-in=$($AllowRuntimeSmokeBlocked.IsPresent); policy=$runtimeLimitationPolicyPath; releaseNotice=$runtimeLimitationNoticePath; missingMarkers=$($runtimeLimitationMissingNoticeMarkers -join ','); disposition=$runtimeProofReleaseDisposition. A ready disclosure does not create runtime execution evidence."
$checks.Add((New-Check -Category "runtime-proof" -Name "Environment-limited runtime release disclosure" -Status $runtimeLimitationDisclosureStatus -Severity $runtimeLimitationDisclosureSeverity -Detail $runtimeLimitationDisclosureDetail -EvidencePath "$runtimeLimitationPolicyPath; $runtimeLimitationNoticePath")) | Out-Null
$compatibleHostRuntimeProofCollectionBundleReady = $compatibleHostRuntimeProofCollectionBundle -and
  -not $compatibleHostRuntimeProofCollectionBundlePerformsPublish -and
  -not $compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease -and
  -not $compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof -and
  -not $compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence
if ($compatibleHostRuntimeProofCollectionBundleReady) { $compatibleHostRuntimeProofCollectionBundleStatus = "ready" } else { $compatibleHostRuntimeProofCollectionBundleStatus = "missing-or-unsafe" }
$compatibleHostRuntimeProofCollectionBundleDetail = "compatible-host-runtime-proof-collection-bundle is executable guidance, not proof; state=$compatibleHostRuntimeProofCollectionBundleState; compatibleHostRequired=$compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired; performsPublish=$compatibleHostRuntimeProofCollectionBundlePerformsPublish; approvesPublicRelease=$compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease; canPromoteRuntimeProof=$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof; runtimeExecutionEvidence=$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence; promotionBlockedReason=$compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason; runPackageConsumerSmokeCommand=$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand; validateFilledRecordCommand=$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand."
$checks.Add((New-Check -Category "runtime-proof" -Name "Compatible host runtime proof collection bundle" -Status $compatibleHostRuntimeProofCollectionBundleStatus -Severity "warning" -Detail $compatibleHostRuntimeProofCollectionBundleDetail -EvidencePath "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json")) | Out-Null

if ($localFeedConsumer) { $localFeedStatus = [string]$localFeedConsumer.RunStatus } else { $localFeedStatus = "missing" }
$localFeedReady = $localFeedConsumer -and -not [bool]$localFeedConsumer.UsesProjectReference -and $localFeedConsumer.NativeAssetsFound -eq $localFeedConsumer.NativeAssetsExpected -and $localFeedStatus -in @("dependency-probe-passed", "runtime-probe-passed", "blocked-by-cuda-driver")
if ($localFeedReady) { $localFeedCheckStatus = "ready" } else { $localFeedCheckStatus = $localFeedStatus }
if ($localFeedConsumer) { $localFeedDetail = [string]$localFeedConsumer.RunDiagnostic } else { $localFeedDetail = "local feed consumer summary was not found; run eng/Test-LocalNuGetFeedConsumer.ps1." }
$checks.Add((New-Check -Category "consumer" -Name "Local NuGet feed consumer" -Status $localFeedCheckStatus -Severity "blocker" -Detail $localFeedDetail -EvidencePath "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json")) | Out-Null

$docsReady = (Test-File "docs\_site\index.html") -and (Test-File "docs\articles\zh-cn\release-candidate-gate.md") -and (Test-File "docs\articles\zh-cn\local-nuget-feed-consumer.md") -and (Test-File "docs\articles\zh-cn\runtime-package-matrix.md")
if ($docsReady) { $docsStatus = "ready" } else { $docsStatus = "missing" }
$checks.Add((New-Check -Category "docs" -Name "Docs site and release articles" -Status $docsStatus -Severity "blocker" -Detail "DocFX site output plus release candidate, local feed, and runtime matrix articles." -EvidencePath "docs/_site/index.html")) | Out-Null

$coverageReady = (Test-File "artifacts\interface-coverage\tensorrt-interface-comparison.csv") -and (Test-File "artifacts\interface-coverage\project-completion-review.md")
if ($coverageReady) { $coverageStatus = "ready" } else { $coverageStatus = "missing" }
$checks.Add((New-Check -Category "coverage" -Name "Interface coverage and deferred boundary reports" -Status $coverageStatus -Severity "blocker" -Detail "coverage matrix and completion review reports must remain available for release audit." -EvidencePath "artifacts/interface-coverage")) | Out-Null

$realProof = $false
if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "debugListenerRealCallbackRuntimeProof") {
  $realProof = [bool]$runtimeReadiness.debugListenerRealCallbackRuntimeProof.isRealCallbackRuntimeProof
}
if ($realProof) { $realProofStatus = "ready" } else { $realProofStatus = "not-proved" }
$checks.Add((New-Check -Category "callback-proof" -Name "DebugListener real callback runtime proof" -Status $realProofStatus -Severity "warning" -Detail "overall package readiness does not imply real callback runtime proof; InvocationCount must be greater than zero in full package consumer evidence." -EvidencePath "artifacts/package-readiness/runtime-package-readiness-summary.json")) | Out-Null

$signed = $false
if ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "ConsumerOutputSigned") {
  $signed = [bool]$packageConsumer.ConsumerOutputSigned
}
if ($signed) { $signingStatus = "signed" } else { $signingStatus = "unsigned-or-not-requested" }
if ($signed) { $signingCheckStatus = "ready" } else { $signingCheckStatus = "unsigned-or-not-requested" }
$checks.Add((New-Check -Category "signing" -Name "Consumer output signing and trust" -Status $signingCheckStatus -Severity "warning" -Detail "local package consumer signing is tracked for WDAC/trust diagnostics; unsigned local output is a release warning, not API proof." -EvidencePath "artifacts/package-consumer/package-consumer-validation-summary.json")) | Out-Null

$sampleCatalog = @(
  New-SampleStatus -Name "PluginRegistryInventorySmokeRunner" -RelativePath "smoke\PluginRegistryInventorySmokeRunner" -Kind "smoke"
  New-SampleStatus -Name "CallbackAllocatorSafeControlsSmokeRunner" -RelativePath "smoke\CallbackAllocatorSafeControlsSmokeRunner" -Kind "smoke"
  New-SampleStatus -Name "TensorRtSmokeRunner" -RelativePath "smoke\TensorRtSmokeRunner" -Kind "smoke"
  New-SampleStatus -Name "CudaSmokeRunner" -RelativePath "smoke\CudaSmokeRunner" -Kind "smoke"
  New-SampleStatus -Name "OnnxToEngineSmokeRunner" -RelativePath "smoke\OnnxToEngineSmokeRunner" -Kind "smoke"
  New-SampleStatus -Name "DynamicShape" -RelativePath "samples\DynamicShape" -Kind "sample"
  New-SampleStatus -Name "MultiStream" -RelativePath "samples\MultiStream" -Kind "sample"
)

$sampleMissing = @($sampleCatalog | Where-Object { $_.status -eq "missing" })
if ($sampleMissing.Count -eq 0) {
  $sampleCatalogStatus = "cataloged"
  $sampleCatalogSeverity = "warning"
}
else {
  $sampleCatalogStatus = "missing"
  $sampleCatalogSeverity = "blocker"
}
$checks.Add((New-Check -Category "samples" -Name "Sample and smoke catalog" -Status $sampleCatalogStatus -Severity $sampleCatalogSeverity -Detail "sample/smoke entries are cataloged here; execution remains environment-specific and is not inferred from catalog presence." -EvidencePath "samples;smoke")) | Out-Null

if ($releaseChecklist) {
  $pendingChecklistItems = @($releaseChecklist | Where-Object { -not [bool]$_.ready })
  if ($pendingChecklistItems.Count -eq 0) { $releaseChecklistStatus = "ready" } else { $releaseChecklistStatus = "pending-items" }
  $checks.Add((New-Check -Category "release-checklist" -Name "Existing release candidate checklist" -Status $releaseChecklistStatus -Severity "warning" -Detail "$($pendingChecklistItems.Count) pending checklist item(s) in existing checklist." -EvidencePath "artifacts/release-candidate/release-candidate-checklist.json")) | Out-Null
}
else {
  $checks.Add((New-Check -Category "release-checklist" -Name "Existing release candidate checklist" -Status "missing" -Severity "warning" -Detail "Export-ReleaseCandidateChecklist.ps1 has not produced the legacy checklist in this workspace." -EvidencePath "artifacts/release-candidate/release-candidate-checklist.json")) | Out-Null
}

$matrix = @(
  foreach ($package in @($manifest.packages)) {
    $isCurrent = [string]::Equals([string]$package.key, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)
    if ($isCurrent -and $packageConsumer) {
      $consumerStatus = [string]$packageConsumer.SmokeResult
    }
    elseif ($package.validationState -eq "local-validated") {
      $consumerStatus = "previously-local-validated"
    }
    elseif ($package.validationState -eq "dry-run-only") {
      $consumerStatus = "dry-run-only"
    }
    else {
      $consumerStatus = [string]$package.validationState
    }

    if ($isCurrent -and $runtimeReadiness -and $runtimeReadiness.overallStatus -eq "ready") {
      $materialization = "materialized-current"
    }
    elseif ($package.validationState -eq "local-validated") {
      $materialization = "materialized-historical"
    }
    elseif ($package.validationState -eq "dry-run-only") {
      $materialization = "planned-dry-run"
    }
    else {
      $materialization = "pending-local-validation"
    }

    if ($isCurrent) {
      $matrixRuntimeProofStatus = $runtimeProofStatus
    }
    elseif ($package.validationState -eq "local-validated") {
      $matrixRuntimeProofStatus = "historical-local-validation"
    }
    elseif ($package.validationState -eq "dry-run-only") {
      $matrixRuntimeProofStatus = "not-proved"
    }
    else {
      $matrixRuntimeProofStatus = "pending-local-validation"
    }

    if ($isCurrent -and $smokeStatus -eq "blocked-by-cuda-driver") {
      $knownLimitation = "Current machine blocks runtime smoke with CUDA error 35; package layout and native-copy evidence remain valid."
    }
    else {
      $knownLimitation = [string]$package.distributionNotes
    }

    [pscustomobject]@{
      key = [string]$package.key
      packageId = [string]$package.packageId
      platform = [string]$package.platform
      rid = [string]$package.rid
      tensorRtLine = [string]$package.tensorRtLine
      tensorRtVersion = [string]$package.tensorRtVersion
      cudaLine = [string]$package.cudaLine
      cudaVersion = [string]$package.cudaVersion
      cudnnVersion = [string]$package.cudnnVersion
      distributionTier = [string]$package.distributionTier
      validationState = [string]$package.validationState
      buildPreset = [string]$package.buildPreset
      materializationStatus = $materialization
      consumerStatus = $consumerStatus
      runtimeProofStatus = $matrixRuntimeProofStatus
      knownLimitation = $knownLimitation
    }
  }
)

$outputRoot = Join-Path $RepositoryRoot "artifacts\release-candidate"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$matrixJsonPath = Join-Path $outputRoot "runtime-package-matrix.json"
$matrixMarkdownPath = Join-Path $outputRoot "runtime-package-matrix.md"
$matrix | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $matrixJsonPath -Encoding utf8

$matrixLines = New-Object System.Collections.Generic.List[string]
$matrixLines.Add("# Runtime Package Matrix")
$matrixLines.Add("")
$matrixLines.Add("| Runtime key | Platform | TensorRT | CUDA | cuDNN | Validation | Materialization | Consumer | Runtime proof |")
$matrixLines.Add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
foreach ($entry in $matrix) {
  $matrixLines.Add("| $($entry.key) | $($entry.platform) | $($entry.tensorRtVersion) | $($entry.cudaVersion) | $($entry.cudnnVersion) | $($entry.validationState) | $($entry.materializationStatus) | $($entry.consumerStatus) | $($entry.runtimeProofStatus) |")
}
$matrixLines | Set-Content -LiteralPath $matrixMarkdownPath -Encoding utf8

$blocking = @($checks | Where-Object { $_.isBlocking })
$warnings = @($checks | Where-Object { $_.isWarning })
if ($blocking.Count -gt 0) {
  $overallStatus = "blocked"
}
elseif ($warnings.Count -gt 0) {
  $overallStatus = "ready-with-warnings"
}
else {
  $overallStatus = "ready"
}

if ($runtimeReadiness) { $runtimeReadinessStatus = [string]$runtimeReadiness.overallStatus } else { $runtimeReadinessStatus = "missing" }

$summary = [pscustomobject]@{
  runtimePackageKey = $RuntimePackageKey
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  overallStatus = $overallStatus
  blockingIssueCount = $blocking.Count
  warningCount = $warnings.Count
  allowRuntimeSmokeBlocked = $AllowRuntimeSmokeBlocked.IsPresent
  runtimeReadinessStatus = $runtimeReadinessStatus
  packageConsumerSmokeStatus = $smokeStatus
  packageConsumerEvidenceKind = $packageConsumerEvidenceKind
  runtimeSmokeClassification = $runtimeSmokeClassification
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofDiagnostic = $runtimeProofDiagnostic
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  runtimeProofReleaseDisposition = $runtimeProofReleaseDisposition
  runtimeProofUsesDocumentedLimitation = [bool]$runtimeProofUsesDocumentedLimitation
  runtimeLimitationDisclosureReady = [bool]$runtimeLimitationDisclosureReady
  runtimeLimitationPolicyPath = $runtimeLimitationPolicyPath
  runtimeLimitationNoticePath = $runtimeLimitationNoticePath
  runtimeLimitationMissingNoticeMarkers = @($runtimeLimitationMissingNoticeMarkers)
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  isDependencyProbeOnly = $isDependencyProbeOnly
  isRealCallbackRuntimeProof = $isPackageConsumerRealCallbackRuntimeProof
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostRuntimeProofCollectionBundleState
  compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = $compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofCollectionBundlePerformsPublish = $compatibleHostRuntimeProofCollectionBundlePerformsPublish
  compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = $compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof
  compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = $compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence
  compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = $compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = $compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = $compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand
  compatibleBridgeRuntimeProofStatus = $compatibleBridgeRuntimeProofStatus
  compatibleBridgeRuntimeProofReady = [bool]$compatibleBridgeRuntimeProofReady
  compatibleBridgeRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $compatibleBridgeRuntimeProof -Name "proofClassification" -DefaultValue "missing")
  compatibleBridgeRuntimeProofIsRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $compatibleBridgeRuntimeProof -Name "isRuntimeExecutionProof" -DefaultValue $false)
  compatibleBridgeRuntimeProofIsPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleBridgeRuntimeProof -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
  localFeedConsumerStatus = $localFeedStatus
  realCallbackRuntimeProof = $realProof
  signingStatus = $signingStatus
  checks = @($checks.ToArray())
  sampleSmokeCatalog = @($sampleCatalog)
  runtimePackageMatrixPath = $matrixJsonPath
}

$jsonPath = Join-Path $outputRoot "release-candidate-readiness-summary.json"
$markdownPath = Join-Path $outputRoot "release-candidate-readiness-summary.md"
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Candidate Readiness Summary")
$lines.Add("")
$lines.Add("- runtime key: ``$RuntimePackageKey``")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- blocking issues: $($blocking.Count)")
$lines.Add("- warnings: $($warnings.Count)")
$lines.Add("- package consumer smoke: ``$smokeStatus``")
$lines.Add("- package consumer evidence kind: ``$packageConsumerEvidenceKind``")
$lines.Add("- runtime smoke classification: ``$runtimeSmokeClassification``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- runtime proof release disposition: ``$runtimeProofReleaseDisposition``")
$lines.Add("- documented runtime limitation used: ``$runtimeProofUsesDocumentedLimitation``")
$lines.Add("- runtime limitation disclosure ready: ``$runtimeLimitationDisclosureReady``")
$lines.Add("- runtime limitation release notice: ``$runtimeLimitationNoticePath``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- runtime proof diagnostic: $runtimeProofDiagnostic")
$lines.Add("- required external runtime proof record: ``artifacts/final-release/external-runtime-proof-record.json``")
$lines.Add("- runtime execution evidence: ``$isRuntimeExecutionEvidence``")
$lines.Add("- dependency probe only: ``$isDependencyProbeOnly``")
$lines.Add("- package consumer real callback proof: ``$isPackageConsumerRealCallbackRuntimeProof``")
$lines.Add("- compatible host collection bundle state: ``$compatibleHostRuntimeProofCollectionBundleState``")
$lines.Add("- compatible host collection bundle can promote runtime proof: ``$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof``")
$lines.Add("- compatible host collection bundle runtime execution evidence: ``$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle smoke command: ``$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host collection bundle validation command: ``$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand``")
$lines.Add("- TRT10 compatible bridge runtime proof status: ``$compatibleBridgeRuntimeProofStatus``")
$lines.Add("- TRT10 compatible bridge runtime execution proof: ``$([bool](Get-PropertyOrDefault -Object $compatibleBridgeRuntimeProof -Name "isRuntimeExecutionProof" -DefaultValue $false))``")
$lines.Add("- TRT10 compatible bridge package-consumer proof: ``$([bool](Get-PropertyOrDefault -Object $compatibleBridgeRuntimeProof -Name "isPackageConsumerRuntimeProof" -DefaultValue $false))``")
$lines.Add("- local feed consumer: ``$localFeedStatus``")
$lines.Add("- real callback runtime proof: ``$realProof``")
$lines.Add("- signing status: ``$signingStatus``")
$lines.Add("")
$lines.Add("| Category | Check | Status | Severity | Detail | Evidence |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($check in $checks) {
  $lines.Add("| $($check.category) | $($check.name) | $($check.status) | $($check.severity) | $(ConvertTo-MarkdownCell $check.detail) | ``$($check.evidencePath)`` |")
}
$lines.Add("")
$lines.Add("## Sample And Smoke Catalog")
$lines.Add("")
$lines.Add("| Name | Kind | Status | Path |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($sample in $sampleCatalog) {
  $lines.Add("| $($sample.name) | $($sample.kind) | $($sample.status) | ``$($sample.path)`` |")
}
$lines.Add("")
$lines.Add("`overallStatus=$overallStatus` is still not `IsRealCallbackRuntimeProof=True`. Callback proof remains controlled by the full package consumer `InvocationCount>0` evidence contract.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate readiness summary written to $jsonPath"
Write-Host "Release candidate readiness summary written to $markdownPath"
Write-Host "Runtime package matrix written to $matrixJsonPath"
Write-Host "Runtime package matrix written to $matrixMarkdownPath"

if ($blocking.Count -gt 0) {
  $message = "Release candidate readiness has $($blocking.Count) blocking issue(s)."
  if ($runtimeProofRequiredForRelease -and -not $isRuntimeExecutionEvidence) {
    $message = "$message Missing real external-runtime-proof-record.json; blocked-by-cuda-driver is not smoke passed; compatible-host-runtime-proof-collection-bundle is guidance only. Run smoke: $compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand Validate proof: $compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand"
  }

  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
