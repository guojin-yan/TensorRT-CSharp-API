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

function Get-FirstFile {
  param(
    [string]$Directory,
    [string]$Filter
  )

  $root = Join-Path $RepositoryRoot $Directory
  if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    return $null
  }

  return Get-ChildItem -LiteralPath $root -Filter $Filter -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

function New-Gate {
  param(
    [string]$Name,
    [string]$Status,
    [string]$Severity,
    [string]$Evidence,
    [string]$Detail
  )

  [pscustomobject]@{
    name = $Name
    status = $Status
    severity = $Severity
    evidence = $Evidence
    detail = $Detail
    isBlocking = [string]::Equals($Severity, "blocker", [System.StringComparison]::OrdinalIgnoreCase) -and
      -not [string]::Equals($Status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
    isManualApproval = [string]::Equals($Severity, "manual-approval", [System.StringComparison]::OrdinalIgnoreCase) -and
      -not [string]::Equals($Status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
    isWarning = [string]::Equals($Severity, "warning", [System.StringComparison]::OrdinalIgnoreCase) -and
      -not [string]::Equals($Status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-Sha256Ready {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return -not [string]::IsNullOrWhiteSpace($text) -and [System.Text.RegularExpressions.Regex]::IsMatch($text, "^[0-9a-fA-F]{64}$")
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackage = @($manifest.packages | Where-Object { [string]::Equals([string]$_.key, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) }) | Select-Object -First 1
if (-not $runtimePackage) {
  throw "Runtime package key '$RuntimePackageKey' was not found in '$manifestPath'."
}

$runtimeReadiness = Read-JsonOrNull "artifacts\package-readiness\runtime-package-readiness-summary.json"
$releaseReadiness = Read-JsonOrNull "artifacts\release-candidate\release-candidate-readiness-summary.json"
$releaseChecklist = Read-JsonOrNull "artifacts\release-candidate\release-candidate-checklist.json"
$localFeedConsumer = Read-JsonOrNull "artifacts\local-feed-consumer\local-nuget-feed-consumer-summary.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$bridgeConsumer = Read-JsonOrNull "artifacts\package-consumer\bridge-package-consumer-validation-summary.json"
$bilingualDocumentationAudit = Read-JsonOrNull "artifacts\api-doc-audit\public-api-bilingual-documentation-audit.json"
$bilingualDocumentationBacklog = Read-JsonOrNull "artifacts\api-doc-audit\public-api-bilingual-documentation-backlog.json"
$userAcceptanceCatalog = Read-JsonOrNull "artifacts\user-acceptance\sample-smoke-catalog.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$packageConsumerRuntimeProofOwnerInputSchema = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input.schema.json"
$packageConsumerRuntimeProofForbiddenSubstituteScan = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-forbidden-substitute-scan.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofDraft = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.draft.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$compatibleHostRuntimeProofCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$postPublish = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$managedPackage = Get-FirstFile -Directory "artifacts\managed" -Filter "JYPPX.TensorRT.CSharp.API.*.nupkg"
$runtimePackageFileName = "$($runtimePackage.packageId).*nupkg"
$runtimePackageArtifact = Get-FirstFile -Directory "artifacts\runtime-nupkg" -Filter $runtimePackageFileName

$gates = New-Object System.Collections.Generic.List[object]

if ($managedPackage) { $managedStatus = "ready" } else { $managedStatus = "missing" }
$gates.Add((New-Gate -Name "Managed package artifact" -Status $managedStatus -Severity "blocker" -Evidence "artifacts/managed" -Detail "Managed package must exist before any final release dry run can be trusted.")) | Out-Null

if ($runtimePackageArtifact) { $runtimeArtifactStatus = "ready" } else { $runtimeArtifactStatus = "missing" }
$gates.Add((New-Gate -Name "Runtime package artifact" -Status $runtimeArtifactStatus -Severity "blocker" -Evidence "artifacts/runtime-nupkg" -Detail "Runtime package for the selected runtime key must exist.")) | Out-Null

if ($runtimeReadiness) {
  $runtimeReadinessOverallStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "overallStatus" -DefaultValue "missing")
  $runtimeReadinessRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "runtimeProofStatus" -DefaultValue "")
  $runtimeReadinessReleaseRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $releaseReadiness -Name "runtimeProofStatus" -DefaultValue "")
  $runtimeReadinessIsRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $runtimeReadiness -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
  $runtimeReadinessIsDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $runtimeReadiness -Name "isDependencyProbeOnly" -DefaultValue $false)
  $runtimeReadinessOwnerAction = Get-PropertyOrDefault -Object $runtimeReadiness -Name "runtimeProofBlockerOwnerAction" -DefaultValue $null
  $runtimeReadinessOwnerActionStatus = [string](Get-PropertyOrDefault -Object $runtimeReadinessOwnerAction -Name "status" -DefaultValue "")
  $runtimeReadinessOwnerActionCategory = [string](Get-PropertyOrDefault -Object $runtimeReadinessOwnerAction -Name "blockerCategory" -DefaultValue "")
  $runtimeReadinessProofIsOnlyNotRequested = [string]::Equals($runtimeReadinessRuntimeProofStatus, "not-requested", [System.StringComparison]::OrdinalIgnoreCase) -or
    [string]::IsNullOrWhiteSpace($runtimeReadinessRuntimeProofStatus)
  $runtimeReadinessReleaseHasDriverOwnerAction = [string]::Equals($runtimeReadinessReleaseRuntimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)
  $runtimeReadinessKnownRuntimeProofBlocker = $AllowRuntimeSmokeBlocked.IsPresent -and
    ([string]::Equals($runtimeReadinessRuntimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -or
      ($runtimeReadinessReleaseHasDriverOwnerAction -and $runtimeReadinessProofIsOnlyNotRequested)) -and
    (-not $runtimeReadinessIsRuntimeExecutionEvidence) -and
    ($runtimeReadinessIsDependencyProbeOnly -or
      [string]::Equals($runtimeReadinessOwnerActionStatus, "owner-action-required", [System.StringComparison]::OrdinalIgnoreCase) -or
      [string]::Equals($runtimeReadinessOwnerActionCategory, "cuda-driver-runtime-compatibility", [System.StringComparison]::OrdinalIgnoreCase))
  $runtimeReadinessKnownNotRequestedOwnerAction = $AllowRuntimeSmokeBlocked.IsPresent -and
    $runtimeReadinessProofIsOnlyNotRequested -and
    (-not $runtimeReadinessIsRuntimeExecutionEvidence) -and
    ($runtimeReadinessIsDependencyProbeOnly -or
      [string]::Equals($runtimeReadinessOwnerActionStatus, "owner-action-required", [System.StringComparison]::OrdinalIgnoreCase) -or
      [string]::Equals($runtimeReadinessOwnerActionCategory, "runtime-smoke-not-requested", [System.StringComparison]::OrdinalIgnoreCase))
}
else {
  $runtimeReadinessOverallStatus = "missing"
  $runtimeReadinessRuntimeProofStatus = "missing"
  $runtimeReadinessReleaseRuntimeProofStatus = "missing"
  $runtimeReadinessIsRuntimeExecutionEvidence = $false
  $runtimeReadinessIsDependencyProbeOnly = $false
  $runtimeReadinessOwnerActionStatus = "missing"
  $runtimeReadinessOwnerActionCategory = "missing"
  $runtimeReadinessKnownRuntimeProofBlocker = $false
  $runtimeReadinessKnownNotRequestedOwnerAction = $false
}

if ($runtimeReadiness -and $runtimeReadinessOverallStatus -in @("ready", "ready-with-warnings")) {
  $runtimeReadinessStatus = "ready"
  $runtimeReadinessSeverity = "blocker"
  $runtimeReadinessDetail = "Runtime readiness aggregates package, native dependency, and wrapper surface evidence."
}
elseif ($runtimeReadinessKnownRuntimeProofBlocker) {
  $runtimeReadinessStatus = "blocked-by-cuda-driver-owner-action"
  $runtimeReadinessSeverity = "manual-approval"
  $runtimeReadinessDetail = "Runtime readiness is blocked by CUDA driver/runtime compatibility owner action. AllowRuntimeSmokeBlocked keeps this visible as manual approval, not runtime proof. RuntimeReadinessProofStatus=$runtimeReadinessRuntimeProofStatus; ReleaseReadinessProofStatus=$runtimeReadinessReleaseRuntimeProofStatus; OwnerActionStatus=$runtimeReadinessOwnerActionStatus; OwnerActionCategory=$runtimeReadinessOwnerActionCategory; IsRuntimeExecutionEvidence=$runtimeReadinessIsRuntimeExecutionEvidence; IsDependencyProbeOnly=$runtimeReadinessIsDependencyProbeOnly."
}
elseif ($runtimeReadinessKnownNotRequestedOwnerAction) {
  $runtimeReadinessStatus = "runtime-smoke-not-requested-owner-action"
  $runtimeReadinessSeverity = "manual-approval"
  $runtimeReadinessDetail = "Runtime readiness has not requested package-consumer runtime smoke. AllowRuntimeSmokeBlocked keeps this visible as owner action, not runtime proof. RuntimeReadinessProofStatus=$runtimeReadinessRuntimeProofStatus; ReleaseReadinessProofStatus=$runtimeReadinessReleaseRuntimeProofStatus; OwnerActionStatus=$runtimeReadinessOwnerActionStatus; OwnerActionCategory=$runtimeReadinessOwnerActionCategory; IsRuntimeExecutionEvidence=$runtimeReadinessIsRuntimeExecutionEvidence; IsDependencyProbeOnly=$runtimeReadinessIsDependencyProbeOnly."
}
else {
  $runtimeReadinessStatus = "missing-or-not-ready"
  $runtimeReadinessSeverity = "blocker"
  $runtimeReadinessDetail = "Runtime readiness must aggregate package, native dependency, and wrapper surface evidence."
}
$gates.Add((New-Gate -Name "Runtime package readiness" -Status $runtimeReadinessStatus -Severity $runtimeReadinessSeverity -Evidence "artifacts/package-readiness/runtime-package-readiness-summary.json" -Detail $runtimeReadinessDetail)) | Out-Null

if ($bridgeConsumer -and $bridgeConsumer.NativeDependencyStatus -eq "ready") { $bridgeStatus = "ready" } else { $bridgeStatus = "missing-or-not-ready" }
$gates.Add((New-Gate -Name "Bridge package consumer" -Status $bridgeStatus -Severity "blocker" -Evidence "artifacts/package-consumer/bridge-package-consumer-validation-summary.json" -Detail "Bridge consumer must validate wrapper/native dependency without claiming full callback proof.")) | Out-Null

if ($packageConsumer -and $packageConsumer.NativeAssetsFound -eq $packageConsumer.NativeAssetsExpected) { $packageConsumerStatus = "ready" } else { $packageConsumerStatus = "missing-or-not-ready" }
$gates.Add((New-Gate -Name "Full package consumer native assets" -Status $packageConsumerStatus -Severity "blocker" -Evidence "artifacts/package-consumer/package-consumer-validation-summary.json" -Detail "Full package consumer must restore/build/copy expected native assets.")) | Out-Null

if ($localFeedConsumer -and -not [bool]$localFeedConsumer.UsesProjectReference -and $localFeedConsumer.NativeAssetsFound -eq $localFeedConsumer.NativeAssetsExpected) {
  $localFeedStatus = "ready"
}
else {
  $localFeedStatus = "missing-or-not-ready"
}
$gates.Add((New-Gate -Name "Local NuGet feed consumer" -Status $localFeedStatus -Severity "blocker" -Evidence "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json" -Detail "Source-free PackageReference consumer must not use ProjectReference and must copy native assets.")) | Out-Null

if ($releaseReadiness) {
  $releaseReadinessBlockingIssueCount = [int]$releaseReadiness.blockingIssueCount
  $releaseReadinessRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $releaseReadiness -Name "runtimeProofStatus" -DefaultValue "")
  $releaseReadinessSmokeStatus = [string](Get-PropertyOrDefault -Object $releaseReadiness -Name "packageConsumerSmokeStatus" -DefaultValue "")
  $releaseReadinessRuntimeProofRequired = [bool](Get-PropertyOrDefault -Object $releaseReadiness -Name "runtimeProofRequiredForRelease" -DefaultValue $true)
  $releaseReadinessIsRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $releaseReadiness -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
  $releaseReadinessIsDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $releaseReadiness -Name "isDependencyProbeOnly" -DefaultValue $false)
  $releaseReadinessKnownRuntimeProofBlocker = $AllowRuntimeSmokeBlocked.IsPresent -and
    $releaseReadinessBlockingIssueCount -ge 1 -and
    ($releaseReadinessRuntimeProofStatus -eq "blocked-by-cuda-driver" -or $releaseReadinessSmokeStatus -eq "blocked-by-cuda-driver") -and
    (-not $releaseReadinessIsRuntimeExecutionEvidence) -and
    ($releaseReadinessIsDependencyProbeOnly -or $releaseReadinessRuntimeProofRequired)
  $releaseReadinessKnownNotRequestedOwnerAction = $AllowRuntimeSmokeBlocked.IsPresent -and
    $releaseReadinessBlockingIssueCount -ge 1 -and
    ($releaseReadinessRuntimeProofStatus -eq "not-requested" -or $releaseReadinessSmokeStatus -eq "not-requested") -and
    (-not $releaseReadinessIsRuntimeExecutionEvidence) -and
    ($releaseReadinessIsDependencyProbeOnly -or $releaseReadinessRuntimeProofRequired)
}
else {
  $releaseReadinessBlockingIssueCount = -1
  $releaseReadinessRuntimeProofStatus = "missing"
  $releaseReadinessSmokeStatus = "missing"
  $releaseReadinessRuntimeProofRequired = $true
  $releaseReadinessIsRuntimeExecutionEvidence = $false
  $releaseReadinessIsDependencyProbeOnly = $false
  $releaseReadinessKnownRuntimeProofBlocker = $false
  $releaseReadinessKnownNotRequestedOwnerAction = $false
}

if ($releaseReadiness -and $releaseReadinessBlockingIssueCount -eq 0) {
  $releaseReadinessStatus = "ready"
  $releaseReadinessSeverity = "blocker"
  $releaseReadinessDetail = "Release candidate readiness has zero blockers."
}
elseif ($releaseReadinessKnownRuntimeProofBlocker) {
  $releaseReadinessStatus = "blocked-by-cuda-driver-owner-action"
  $releaseReadinessSeverity = "manual-approval"
  $releaseReadinessDetail = "Release candidate readiness is blocked by CUDA driver runtime proof owner action; AllowRuntimeSmokeBlocked keeps this as owner action, not proof. External runtime proof may still be missing and remains release-owner work. RuntimeProofStatus=$releaseReadinessRuntimeProofStatus; PackageConsumerSmokeStatus=$releaseReadinessSmokeStatus; RuntimeProofRequiredForRelease=$releaseReadinessRuntimeProofRequired; BlockingIssueCount=$releaseReadinessBlockingIssueCount; IsRuntimeExecutionEvidence=$releaseReadinessIsRuntimeExecutionEvidence; IsDependencyProbeOnly=$releaseReadinessIsDependencyProbeOnly."
}
elseif ($releaseReadinessKnownNotRequestedOwnerAction) {
  $releaseReadinessStatus = "runtime-smoke-not-requested-owner-action"
  $releaseReadinessSeverity = "manual-approval"
  $releaseReadinessDetail = "Release candidate readiness still requires owner runtime smoke. AllowRuntimeSmokeBlocked keeps this as owner action, not proof. External runtime proof may still be missing and remains release-owner work. RuntimeProofStatus=$releaseReadinessRuntimeProofStatus; PackageConsumerSmokeStatus=$releaseReadinessSmokeStatus; RuntimeProofRequiredForRelease=$releaseReadinessRuntimeProofRequired; BlockingIssueCount=$releaseReadinessBlockingIssueCount; IsRuntimeExecutionEvidence=$releaseReadinessIsRuntimeExecutionEvidence; IsDependencyProbeOnly=$releaseReadinessIsDependencyProbeOnly."
}
else {
  $releaseReadinessStatus = "blocked-or-missing"
  $releaseReadinessSeverity = "blocker"
  $releaseReadinessDetail = "Release candidate readiness must have zero blockers, unless blockers are allowed CUDA driver runtime proof owner action."
}
$gates.Add((New-Gate -Name "Release candidate readiness" -Status $releaseReadinessStatus -Severity $releaseReadinessSeverity -Evidence "artifacts/release-candidate/release-candidate-readiness-summary.json" -Detail $releaseReadinessDetail)) | Out-Null

if ($releaseChecklist) {
  $pendingChecklistItems = @($releaseChecklist | Where-Object { -not [bool]$_.ready })
  if ($pendingChecklistItems.Count -eq 0) { $checklistStatus = "ready" } else { $checklistStatus = "pending-items" }
  $checklistDetail = "$($pendingChecklistItems.Count) checklist item(s) are pending."
}
else {
  $pendingChecklistItems = @()
  $checklistStatus = "missing"
  $checklistDetail = "Release candidate checklist was not found."
}
$gates.Add((New-Gate -Name "Release candidate checklist" -Status $checklistStatus -Severity "manual-approval" -Evidence "artifacts/release-candidate/release-candidate-checklist.json" -Detail $checklistDetail)) | Out-Null

$ownerInputSchemaReady = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofOwnerInputSchema -Name "recordKind" -DefaultValue "") -eq "package-consumer-runtime-proof-owner-input-schema"
$ownerInputSchemaCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofOwnerInputSchema -Name "canPromoteRuntimeProof" -DefaultValue $false)
$forbiddenSubstituteScanState = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "scanState" -DefaultValue "missing-package-consumer-runtime-proof-forbidden-substitute-scan")
$detectedForbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "detectedForbiddenSubstituteCount" -DefaultValue -1)
$forbiddenSubstituteScanCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "canPromoteRuntimeProof" -DefaultValue $false)
$forbiddenSubstituteScanCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "canCloseReleaseIssue" -DefaultValue $false)
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputForbiddenSubstituteFree = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputForbiddenSubstituteFree" -DefaultValue $false)
$ownerInputHashFieldsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputHashFieldsReady" -DefaultValue $false)
$ownerInputPackageHashFilesMatch = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputPackageHashFilesMatch" -DefaultValue $false)
$ownerInputSmokeLogReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputSmokeLogReady" -DefaultValue $false)
$ownerInputHostMetadataReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputHostMetadataReady" -DefaultValue $false)
$ownerInputCommandEvidenceReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputCommandEvidenceReady" -DefaultValue $false)
$ownerInputCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $false)
$ownerInputBlockedReason = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerInputBlockedReason" -DefaultValue "missing-owner-input-readiness")
$ownerProofSchemaScanStatus = if ($ownerInputSchemaReady -and [string]::Equals($forbiddenSubstituteScanState, "blocked-forbidden-substitute-detected", [System.StringComparison]::OrdinalIgnoreCase) -and $detectedForbiddenSubstituteCount -ge 1 -and -not $ownerInputSchemaCanPromoteRuntimeProof -and -not $forbiddenSubstituteScanCanPromoteRuntimeProof -and -not $forbiddenSubstituteScanCanCloseReleaseIssue) {
  "blocked-forbidden-substitute-detected"
}
elseif ($ownerInputSchemaReady -and [string]::Equals($forbiddenSubstituteScanState, "clean-no-forbidden-substitutes", [System.StringComparison]::OrdinalIgnoreCase) -and -not $ownerInputSchemaCanPromoteRuntimeProof -and -not $forbiddenSubstituteScanCanPromoteRuntimeProof -and -not $forbiddenSubstituteScanCanCloseReleaseIssue) {
  "ready-non-proof-schema-scan"
}
else {
  "missing-or-invalid-ownerproof-schema-scan"
}
$gates.Add((New-Gate -Name "OwnerProof schema and forbidden substitute scan" -Status $ownerProofSchemaScanStatus -Severity "manual-approval" -Evidence "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json; artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json" -Detail "OwnerProof schema and forbidden substitute scan are publication guardrails only. OwnerInputSchemaReady=$ownerInputSchemaReady; ForbiddenSubstituteScanState=$forbiddenSubstituteScanState; DetectedForbiddenSubstituteCount=$detectedForbiddenSubstituteCount; SchemaCanPromoteRuntimeProof=$ownerInputSchemaCanPromoteRuntimeProof; ScanCanPromoteRuntimeProof=$forbiddenSubstituteScanCanPromoteRuntimeProof; ScanCanCloseReleaseIssue=$forbiddenSubstituteScanCanCloseReleaseIssue.")) | Out-Null
$ownerInputReadinessStatus = if ($cleanOwnerInputReady -and $ownerInputForbiddenSubstituteFree -and $ownerInputHashFieldsReady -and $ownerInputPackageHashFilesMatch -and $ownerInputSmokeLogReady -and $ownerInputHostMetadataReady -and $ownerInputCommandEvidenceReady -and -not $ownerInputCanPromoteRuntimeProof) {
  "clean-owner-input-ready-non-proof"
}
else {
  "blocked-owner-input-required"
}
$gates.Add((New-Gate -Name "Clean owner input readiness" -Status $ownerInputReadinessStatus -Severity "manual-approval" -Evidence "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json; artifacts/final-release/release-evidence-bundle.json" -Detail "Clean owner input readiness separates real owner-filled clean consumer metadata from runtime proof. CleanOwnerInputReady=$cleanOwnerInputReady; ForbiddenSubstituteFree=$ownerInputForbiddenSubstituteFree; HashFieldsReady=$ownerInputHashFieldsReady; PackageHashFilesMatch=$ownerInputPackageHashFilesMatch; SmokeLogReady=$ownerInputSmokeLogReady; HostMetadataReady=$ownerInputHostMetadataReady; CommandEvidenceReady=$ownerInputCommandEvidenceReady; CanPromoteRuntimeProof=$ownerInputCanPromoteRuntimeProof; BlockedReason=$ownerInputBlockedReason.")) | Out-Null

if ($bilingualDocumentationAudit) {
  $bilingualFindingCount = [int]$bilingualDocumentationAudit.findingCount
  if ($bilingualFindingCount -eq 0) { $bilingualStatus = "ready" } else { $bilingualStatus = "findings-present" }
  $bilingualDetail = "$bilingualFindingCount public documentation element(s) are not bilingual."
}
else {
  $bilingualFindingCount = -1
  $bilingualStatus = "missing"
  $bilingualDetail = "Public API bilingual documentation audit is missing."
}
$gates.Add((New-Gate -Name "Public API bilingual documentation" -Status $bilingualStatus -Severity "manual-approval" -Evidence "artifacts/api-doc-audit/public-api-bilingual-documentation-audit.json" -Detail $bilingualDetail)) | Out-Null

if ($bilingualFindingCount -eq 0) {
  $bilingualBacklogStatus = "ready"
  $bilingualBacklogDetail = "Public API bilingual documentation audit is clean; no backlog is required."
}
elseif ($bilingualDocumentationBacklog -and [int]$bilingualDocumentationBacklog.backlogFindingCount -eq $bilingualFindingCount) {
  $bilingualBacklogStatus = "ready"
  $bilingualBacklogDetail = "$([int]$bilingualDocumentationBacklog.backlogFindingCount) bilingual documentation finding(s) are grouped into a batchable backlog."
}
elseif ($bilingualDocumentationBacklog) {
  $bilingualBacklogStatus = "stale"
  $bilingualBacklogDetail = "Backlog finding count $([int]$bilingualDocumentationBacklog.backlogFindingCount) does not match audit finding count $bilingualFindingCount."
}
else {
  $bilingualBacklogStatus = "missing"
  $bilingualBacklogDetail = "Bilingual documentation findings exist, but the batchable backlog has not been generated."
}
$gates.Add((New-Gate -Name "Public API bilingual documentation backlog" -Status $bilingualBacklogStatus -Severity "manual-approval" -Evidence "artifacts/api-doc-audit/public-api-bilingual-documentation-backlog.json" -Detail $bilingualBacklogDetail)) | Out-Null

if (Test-File "docs\_site\index.html") { $docfxStatus = "ready" } else { $docfxStatus = "missing" }
$gates.Add((New-Gate -Name "DocFX site output" -Status $docfxStatus -Severity "blocker" -Evidence "docs/_site/index.html" -Detail "Documentation site output must exist for user-facing release docs.")) | Out-Null

if (Test-File "docs\articles\zh-cn\technical-article-roadmap.md") { $articleRoadmapStatus = "ready" } else { $articleRoadmapStatus = "missing" }
$gates.Add((New-Gate -Name "Technical article roadmap" -Status $articleRoadmapStatus -Severity "manual-approval" -Evidence "docs/articles/zh-cn/technical-article-roadmap.md" -Detail "Article roadmap must continue toward at least 30 high-quality technical and promotion articles.")) | Out-Null

if ((Test-File "docs\articles\zh-cn\signing-and-trust-policy.md") -and (Test-File "docs\articles\zh-cn\nuget-and-github-packages-release-guide.md")) {
  $releaseChannelDocsStatus = "ready"
}
else {
  $releaseChannelDocsStatus = "missing"
}
$gates.Add((New-Gate -Name "Signing and release channel docs" -Status $releaseChannelDocsStatus -Severity "manual-approval" -Evidence "docs/articles/zh-cn/signing-and-trust-policy.md;docs/articles/zh-cn/nuget-and-github-packages-release-guide.md" -Detail "Release owner guidance for signing, package sources, and channel-specific rollback must be present.")) | Out-Null

if ($userAcceptanceCatalog -and [int]$userAcceptanceCatalog.missingItemCount -eq 0) {
  $userAcceptanceStatus = "ready"
  $userAcceptanceDetail = "$([int]$userAcceptanceCatalog.itemCount) sample/smoke item(s) are cataloged for user acceptance."
}
elseif ($userAcceptanceCatalog) {
  $userAcceptanceStatus = "missing-items"
  $userAcceptanceDetail = "$([int]$userAcceptanceCatalog.missingItemCount) sample/smoke catalog item(s) are missing."
}
else {
  $userAcceptanceStatus = "missing"
  $userAcceptanceDetail = "User acceptance sample/smoke catalog is missing."
}
$gates.Add((New-Gate -Name "User acceptance sample catalog" -Status $userAcceptanceStatus -Severity "manual-approval" -Evidence "artifacts/user-acceptance/sample-smoke-catalog.json" -Detail $userAcceptanceDetail)) | Out-Null

if (Test-Directory "artifacts\linux-dry-run") { $linuxStatus = "handoff-present" } else { $linuxStatus = "missing" }
$gates.Add((New-Gate -Name "Linux dry-run handoff" -Status $linuxStatus -Severity "manual-approval" -Evidence "artifacts/linux-dry-run" -Detail "Linux evidence is a handoff/dry-run state unless validated by matching Linux runners.")) | Out-Null

if ($releaseReadiness) { $callbackProof = [bool]$releaseReadiness.realCallbackRuntimeProof } else { $callbackProof = $false }
if ($callbackProof) { $callbackStatus = "ready" } else { $callbackStatus = "not-proved" }
$gates.Add((New-Gate -Name "DebugListener real callback runtime proof" -Status $callbackStatus -Severity "warning" -Evidence "artifacts/release-candidate/release-candidate-readiness-summary.json" -Detail "Final release dry run must not equate package readiness with InvocationCount>0 callback proof.")) | Out-Null

if ($releaseReadiness) { $signingStatus = [string]$releaseReadiness.signingStatus } else { $signingStatus = "missing" }
if ($signingStatus -eq "signed") { $signingGateStatus = "ready" } else { $signingGateStatus = $signingStatus }
$gates.Add((New-Gate -Name "Signing and trust" -Status $signingGateStatus -Severity "manual-approval" -Evidence "artifacts/release-candidate/release-candidate-readiness-summary.json" -Detail "Unsigned local output can be acceptable for RC dry run but needs release-owner policy before public publication.")) | Out-Null

if ($releaseReadiness) { $smokeStatus = [string]$releaseReadiness.packageConsumerSmokeStatus } else { $smokeStatus = "missing" }
$packageConsumerEvidenceKind = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "packageConsumerEvidenceKind" -and -not [string]::IsNullOrWhiteSpace([string]$releaseReadiness.packageConsumerEvidenceKind)) {
  [string]$releaseReadiness.packageConsumerEvidenceKind
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "EvidenceKind" -and -not [string]::IsNullOrWhiteSpace([string]$packageConsumer.EvidenceKind)) {
  [string]$packageConsumer.EvidenceKind
}
else {
  switch ($smokeStatus) {
    "passed" { "full-runtime-package-consumer-smoke"; break }
    "not-requested" { "package-consumer-native-copy"; break }
    "blocked-by-application-control" { "full-runtime-package-consumer-smoke-application-control-blocked"; break }
    "blocked-by-cuda-driver" { "full-runtime-package-consumer-smoke-driver-blocked"; break }
    "failed" { "full-runtime-package-consumer-smoke-failed"; break }
    "missing" { "missing"; break }
    default { "full-runtime-package-consumer-diagnostic"; break }
  }
}
$runtimeSmokeClassification = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "runtimeSmokeClassification" -and -not [string]::IsNullOrWhiteSpace([string]$releaseReadiness.runtimeSmokeClassification)) {
  [string]$releaseReadiness.runtimeSmokeClassification
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "RuntimeSmokeClassification" -and -not [string]::IsNullOrWhiteSpace([string]$packageConsumer.RuntimeSmokeClassification)) {
  [string]$packageConsumer.RuntimeSmokeClassification
}
else {
  switch ($smokeStatus) {
    "passed" { "runtime-smoke-passed"; break }
    "not-requested" { "not-requested"; break }
    "blocked-by-application-control" { "runtime-smoke-application-control-blocked"; break }
    "blocked-by-cuda-driver" { "runtime-smoke-driver-blocked"; break }
    "failed" { "runtime-smoke-failed"; break }
    "missing" { "missing"; break }
    default { "runtime-smoke-diagnostic"; break }
  }
}
$isRuntimeExecutionEvidence = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "isRuntimeExecutionEvidence") {
  [bool]$releaseReadiness.isRuntimeExecutionEvidence
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsRuntimeExecutionEvidence") {
  [bool]$packageConsumer.IsRuntimeExecutionEvidence
}
else {
  $smokeStatus -eq "passed"
}
$isDependencyProbeOnly = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "isDependencyProbeOnly") {
  [bool]$releaseReadiness.isDependencyProbeOnly
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsDependencyProbeOnly") {
  [bool]$packageConsumer.IsDependencyProbeOnly
}
else {
  -not $isRuntimeExecutionEvidence
}
$isPackageConsumerRealCallbackRuntimeProof = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "isRealCallbackRuntimeProof") {
  [bool]$releaseReadiness.isRealCallbackRuntimeProof
}
elseif ($packageConsumer -and $packageConsumer.PSObject.Properties.Name -contains "IsRealCallbackRuntimeProof") {
  [bool]$packageConsumer.IsRealCallbackRuntimeProof
}
else {
  $false
}
$runtimeReadinessProofStatus = if ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofStatus" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeReadiness.runtimeProofStatus)) {
  [string]$runtimeReadiness.runtimeProofStatus
}
else {
  ""
}
$releaseReadinessProofStatus = if ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "runtimeProofStatus" -and -not [string]::IsNullOrWhiteSpace([string]$releaseReadiness.runtimeProofStatus)) {
  [string]$releaseReadiness.runtimeProofStatus
}
else {
  ""
}
$runtimeReadinessProofIsOnlyNotRequested = [string]::Equals($runtimeReadinessProofStatus, "not-requested", [System.StringComparison]::OrdinalIgnoreCase) -or
  [string]::IsNullOrWhiteSpace($runtimeReadinessProofStatus)
$releaseReadinessHasDriverOwnerAction = [string]::Equals($releaseReadinessProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -and
  $AllowRuntimeSmokeBlocked.IsPresent
$runtimeProofStatus = if ($releaseReadinessHasDriverOwnerAction -and $runtimeReadinessProofIsOnlyNotRequested) {
  $releaseReadinessProofStatus
}
elseif (-not [string]::IsNullOrWhiteSpace($runtimeReadinessProofStatus)) {
  $runtimeReadinessProofStatus
}
elseif (-not [string]::IsNullOrWhiteSpace($releaseReadinessProofStatus)) {
  $releaseReadinessProofStatus
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
elseif ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "runtimeProofDiagnostic" -and -not [string]::IsNullOrWhiteSpace([string]$releaseReadiness.runtimeProofDiagnostic)) {
  [string]$releaseReadiness.runtimeProofDiagnostic
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
elseif ($releaseReadiness -and $releaseReadiness.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") {
  [bool]$releaseReadiness.runtimeProofRequiredForRelease
}
else {
  -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)
}
$runtimeProofBlockerOwnerActionStatus = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerActionStatus") {
  [string]$releaseEvidenceBundle.runtimeProofBlockerOwnerActionStatus
}
elseif ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerAction" -and $runtimeReadiness.runtimeProofBlockerOwnerAction -and $runtimeReadiness.runtimeProofBlockerOwnerAction.PSObject.Properties.Name -contains "status") {
  [string]$runtimeReadiness.runtimeProofBlockerOwnerAction.status
}
elseif ([string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)) {
  "resolved"
}
else {
  "owner-action-required"
}
$runtimeProofBlockerCategory = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerActionCategory") {
  [string]$releaseEvidenceBundle.runtimeProofBlockerOwnerActionCategory
}
elseif ($runtimeReadiness -and $runtimeReadiness.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerAction" -and $runtimeReadiness.runtimeProofBlockerOwnerAction -and $runtimeReadiness.runtimeProofBlockerOwnerAction.PSObject.Properties.Name -contains "blockerCategory") {
  [string]$runtimeReadiness.runtimeProofBlockerOwnerAction.blockerCategory
}
else {
  switch ($runtimeProofStatus) {
    "ready" { "none"; break }
    "blocked-by-cuda-driver" { "cuda-driver-runtime-compatibility"; break }
    "blocked-by-application-control" { "application-control-policy"; break }
    "not-requested" { "runtime-smoke-not-requested"; break }
    default { "runtime-proof-incomplete"; break }
  }
}
if ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -and
  [string]::Equals($runtimeProofBlockerCategory, "runtime-smoke-not-requested", [System.StringComparison]::OrdinalIgnoreCase)) {
  $runtimeProofBlockerCategory = "cuda-driver-runtime-compatibility"
}
$externalRuntimeProofStateFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofState" -DefaultValue "missing-external-runtime-proof-record-template") } else { "missing-external-runtime-proof-validation" }
$externalRuntimeProofClassificationFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofClassification" -DefaultValue "missing-proof-classification") } else { "missing-proof-classification" }
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue $externalRuntimeProofStateFallback)
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofClassification" -DefaultValue $externalRuntimeProofClassificationFallback)
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofPackageSourceRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)))
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)))
$externalRuntimeProofManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofLogSha256FormatReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256FormatReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)))
$externalRuntimeProofLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256Matches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$externalRuntimeProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofOwnerActionStatus = if ($externalRuntimeProofCanPromoteRuntimeProof) { "resolved" } else { "owner-action-required" }
$externalRuntimeProofDraftPackageSource = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "packageSource" -DefaultValue $null
$externalRuntimeProofDraftCommand = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "command" -DefaultValue $null
$externalRuntimeProofDraftResults = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "results" -DefaultValue $null
$externalRuntimeProofDraftState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftState" -DefaultValue ([string](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "proofState" -DefaultValue "missing-external-runtime-proof-draft")))
$externalRuntimeProofDraftClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "proofClassification" -DefaultValue "missing-proof-classification")))
$externalRuntimeProofDraftManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftManagedNupkgSha256Ready" -DefaultValue (Test-Sha256Ready (Get-PropertyOrDefault -Object $externalRuntimeProofDraftPackageSource -Name "managedNupkgSha256" -DefaultValue "")))
$externalRuntimeProofDraftRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftRuntimeNupkgSha256Ready" -DefaultValue (Test-Sha256Ready (Get-PropertyOrDefault -Object $externalRuntimeProofDraftPackageSource -Name "runtimeNupkgSha256" -DefaultValue "")))
$externalRuntimeProofDraftLogSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftLogSha256Ready" -DefaultValue (Test-Sha256Ready (Get-PropertyOrDefault -Object $externalRuntimeProofDraftCommand -Name "logSha256" -DefaultValue "")))
$externalRuntimeProofDraftNoProjectReference = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftNoProjectReference" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofDraftPackageSource -Name "noProjectReference" -DefaultValue $false)))
$externalRuntimeProofDraftSmokeStatus = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftSmokeStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $externalRuntimeProofDraftResults -Name "smokeStatus" -DefaultValue "missing-smoke-status")))
$compatibleHostRuntimeProofCollectionBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleState" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "state" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")))
$compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "compatibleHostRequired" -DefaultValue $true)))
$compatibleHostRuntimeProofCollectionBundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "performsPublish" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "approvesPublicRelease" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "isRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "promotionBlockedReason" -DefaultValue "blocked-by-cuda-driver is not smoke passed")))
$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "runPackageConsumerSmokeCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RunSmoke -RequirePackageReference")))
$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostRuntimeProofCollectionBundle -Name "validateFilledRecordCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof")))
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-validation")))
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")))
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassificationPromotable" -DefaultValue $false)))
$postPublishManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$postPublishRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "hostReady" -DefaultValue $false)))
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "commandsReady" -DefaultValue $false)))
$postPublishStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutSummaryReady" -DefaultValue $false)))
$postPublishStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "stderrSummaryReady" -DefaultValue $false)))
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutStderrSummaryReady" -DefaultValue $false)))
$postPublishAllLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishAllLogSha256Matches" -DefaultValue $false)
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "canCloseReleaseIssue" -DefaultValue $false)))
if ($smokeStatus -eq "passed") { $smokeGateStatus = "ready" } else { $smokeGateStatus = $smokeStatus }
$runtimeSmokeDetail = "CUDA error 35 or other environment blockers must remain visible and must not be treated as API proof. AllowRuntimeSmokeBlocked records dry-run intent only; it does not promote blocked smoke to ready. Missing real external-runtime-proof-record.json remains release-owner action. compatible-host-runtime-proof-collection-bundle is executable owner guidance, not runtime proof. EvidenceKind=$packageConsumerEvidenceKind; RuntimeSmokeClassification=$runtimeSmokeClassification; IsRuntimeExecutionEvidence=$isRuntimeExecutionEvidence; IsDependencyProbeOnly=$isDependencyProbeOnly; IsRealCallbackRuntimeProof=$isPackageConsumerRealCallbackRuntimeProof; RuntimeProofStatus=$runtimeProofStatus; RuntimeProofRequiredForRelease=$runtimeProofRequiredForRelease; RuntimeProofBlockerOwnerActionStatus=$runtimeProofBlockerOwnerActionStatus; RuntimeProofBlockerCategory=$runtimeProofBlockerCategory; ExternalRuntimeProofState=$externalRuntimeProofState; ExternalRuntimeProofClassification=$externalRuntimeProofClassification; ExternalRuntimeProofRuntimePackageKeyMatches=$externalRuntimeProofRuntimePackageKeyMatches; ExternalRuntimeProofPackageSourceRuntimePackageKeyMatches=$externalRuntimeProofPackageSourceRuntimePackageKeyMatches; ExternalRuntimeProofConsumerProjectIdentityReady=$externalRuntimeProofConsumerProjectIdentityReady; ExternalRuntimeProofSmokeCommandRuntimeKeyReady=$externalRuntimeProofSmokeCommandRuntimeKeyReady; ExternalRuntimeProofHostReady=$externalRuntimeProofHostReady; ExternalRuntimeProofCommandsReady=$externalRuntimeProofCommandsReady; ExternalRuntimeProofManagedNupkgSha256Ready=$externalRuntimeProofManagedNupkgSha256Ready; ExternalRuntimeProofRuntimeNupkgSha256Ready=$externalRuntimeProofRuntimeNupkgSha256Ready; ExternalRuntimeProofLogSha256FormatReady=$externalRuntimeProofLogSha256FormatReady; ExternalRuntimeProofLogSha256Matches=$externalRuntimeProofLogSha256Matches; ExternalRuntimeProofFailedProofItemCount=$externalRuntimeProofFailedProofItemCount; CompatibleHostRuntimeProofCollectionBundleState=$compatibleHostRuntimeProofCollectionBundleState; CompatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof=$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof; CompatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence=$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence."
$gates.Add((New-Gate -Name "Full package runtime smoke" -Status $smokeGateStatus -Severity "manual-approval" -Evidence "artifacts/package-consumer/package-consumer-validation-summary.json" -Detail $runtimeSmokeDetail)) | Out-Null
$gates.Add((New-Gate -Name "External bridge runtime proof" -Status $runtimeProofStatus -Severity "manual-approval" -Evidence "artifacts/package-readiness/runtime-package-readiness-summary.json; artifacts/release-candidate/release-candidate-readiness-summary.json; artifacts/final-release/external-runtime-proof-record.json" -Detail "Runtime proof is separate from final dry-run overall status and must use managed plus bridge-only packages with host-installed NVIDIA dependencies; release-required=$runtimeProofRequiredForRelease; $runtimeProofDiagnostic Missing real external-runtime-proof-record.json remains a release blocker; blocked-by-cuda-driver is not smoke passed.")) | Out-Null
$gates.Add((New-Gate -Name "External runtime proof record" -Status $externalRuntimeProofState -Severity "manual-approval" -Evidence "artifacts/final-release/external-runtime-proof-validation.json; artifacts/final-release/external-runtime-proof-record.draft.json" -Detail "External runtime proof must be package-consumer-runtime from the target runtime package key, matching packageSource.runtimePackageKey, clean consumer project identity, complete CUDA/TensorRT/cuDNN host metadata, smokeCommand with --runtime-package-key, downloaded managed/runtime nupkg SHA256, and verified smoke log SHA256. Classification=$externalRuntimeProofClassification; RuntimePackageKeyMatches=$externalRuntimeProofRuntimePackageKeyMatches; PackageSourceRuntimePackageKeyMatches=$externalRuntimeProofPackageSourceRuntimePackageKeyMatches; ConsumerProjectIdentityReady=$externalRuntimeProofConsumerProjectIdentityReady; SmokeCommandRuntimeKeyReady=$externalRuntimeProofSmokeCommandRuntimeKeyReady; HostReady=$externalRuntimeProofHostReady; CommandsReady=$externalRuntimeProofCommandsReady; ManagedNupkgSha256Ready=$externalRuntimeProofManagedNupkgSha256Ready; RuntimeNupkgSha256Ready=$externalRuntimeProofRuntimeNupkgSha256Ready; LogSha256FormatReady=$externalRuntimeProofLogSha256FormatReady; LogSha256Matches=$externalRuntimeProofLogSha256Matches; FailedProofItemCount=$externalRuntimeProofFailedProofItemCount; DraftState=$externalRuntimeProofDraftState; DraftClassification=$externalRuntimeProofDraftClassification; DraftManagedNupkgSha256Ready=$externalRuntimeProofDraftManagedNupkgSha256Ready; DraftRuntimeNupkgSha256Ready=$externalRuntimeProofDraftRuntimeNupkgSha256Ready; DraftLogSha256Ready=$externalRuntimeProofDraftLogSha256Ready; DraftNoProjectReference=$externalRuntimeProofDraftNoProjectReference; DraftSmokeStatus=$externalRuntimeProofDraftSmokeStatus; OwnerAction=$externalRuntimeProofOwnerActionStatus; CanPromoteRuntimeProof=$externalRuntimeProofCanPromoteRuntimeProof.")) | Out-Null
$gates.Add((New-Gate -Name "Compatible host runtime proof collection bundle" -Status $compatibleHostRuntimeProofCollectionBundleState -Severity "manual-approval" -Evidence "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json" -Detail "Collection bundle is executable owner guidance, not proof or publication approval. CompatibleHostRequired=$compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired; PerformsPublish=$compatibleHostRuntimeProofCollectionBundlePerformsPublish; ApprovesPublicRelease=$compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease; CanPromoteRuntimeProof=$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof; RuntimeExecutionEvidence=$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence; PromotionBlockedReason=$compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason; RunPackageConsumerSmokeCommand=$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand; ValidateFilledRecordCommand=$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand.")) | Out-Null
$gates.Add((New-Gate -Name "Post-publish verification record" -Status $postPublishVerificationState -Severity "manual-approval" -Evidence "artifacts/final-release/post-publish-verification-validation.json" -Detail "Post-publish verification must remain separate from pre-publish dry run. It can close the release issue only after a real channel publish plus package identity, managed/runtime nupkg hashes, clean consumer identity, compatible host metadata, explicit restore/build/smoke commands, smokeCommand with --runtime-package-key, reviewed stdout/stderr summaries, matching log SHA256 values, and validator proof. Classification=$postPublishProofClassification; Promotable=$postPublishProofClassificationPromotable; ManagedNupkgSha256Ready=$postPublishManagedNupkgSha256Ready; RuntimeNupkgSha256Ready=$postPublishRuntimeNupkgSha256Ready; ConsumerProjectIdentityReady=$postPublishConsumerProjectIdentityReady; SmokeCommandRuntimeKeyReady=$postPublishSmokeCommandRuntimeKeyReady; HostReady=$postPublishHostReady; CommandsReady=$postPublishCommandsReady; StdoutSummaryReady=$postPublishStdoutSummaryReady; StderrSummaryReady=$postPublishStderrSummaryReady; StdoutStderrSummaryReady=$postPublishStdoutStderrSummaryReady; AllLogSha256Matches=$postPublishAllLogSha256Matches; IsPostPublishVerificationProof=$isPostPublishVerificationProof; CanCloseReleaseIssue=$canCloseReleaseIssue.")) | Out-Null

$blocking = @($gates | Where-Object { $_.isBlocking })
$manualApprovals = @($gates | Where-Object { $_.isManualApproval })
$warnings = @($gates | Where-Object { $_.isWarning })

if ($blocking.Count -gt 0) {
  $overallStatus = "blocked"
}
elseif ($manualApprovals.Count -gt 0) {
  $overallStatus = "ready-needs-manual-approval"
}
elseif ($warnings.Count -gt 0) {
  $overallStatus = "ready-with-warnings"
}
else {
  $overallStatus = "ready"
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$summary = [pscustomobject]@{
  runtimePackageKey = $RuntimePackageKey
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  overallStatus = $overallStatus
  blockingIssueCount = $blocking.Count
  manualApprovalCount = $manualApprovals.Count
  warningCount = $warnings.Count
  managedPackagePath = if ($managedPackage) { $managedPackage.FullName } else { $null }
  runtimePackagePath = if ($runtimePackageArtifact) { $runtimePackageArtifact.FullName } else { $null }
  runtimePackageId = [string]$runtimePackage.packageId
  runtimePackageValidationState = [string]$runtimePackage.validationState
  releaseCandidateStatus = if ($releaseReadiness) { [string]$releaseReadiness.overallStatus } else { "missing" }
  packageConsumerSmokeStatus = $smokeStatus
  packageConsumerEvidenceKind = $packageConsumerEvidenceKind
  runtimeSmokeClassification = $runtimeSmokeClassification
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofDiagnostic = $runtimeProofDiagnostic
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  ownerInputSchemaReady = $ownerInputSchemaReady
  forbiddenSubstituteScanState = $forbiddenSubstituteScanState
  detectedForbiddenSubstituteCount = $detectedForbiddenSubstituteCount
  ownerProofSchemaScanStatus = $ownerProofSchemaScanStatus
  ownerInputSchemaCanPromoteRuntimeProof = $ownerInputSchemaCanPromoteRuntimeProof
  forbiddenSubstituteScanCanPromoteRuntimeProof = $forbiddenSubstituteScanCanPromoteRuntimeProof
  forbiddenSubstituteScanCanCloseReleaseIssue = $forbiddenSubstituteScanCanCloseReleaseIssue
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputPackageHashFilesMatch = $ownerInputPackageHashFilesMatch
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputHostMetadataReady = $ownerInputHostMetadataReady
  ownerInputCommandEvidenceReady = $ownerInputCommandEvidenceReady
  ownerInputCanPromoteRuntimeProof = $ownerInputCanPromoteRuntimeProof
  ownerInputBlockedReason = $ownerInputBlockedReason
  ownerInputReadinessStatus = $ownerInputReadinessStatus
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofRuntimePackageKeyMatches = $externalRuntimeProofRuntimePackageKeyMatches
  externalRuntimeProofPackageSourceRuntimePackageKeyMatches = $externalRuntimeProofPackageSourceRuntimePackageKeyMatches
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  externalRuntimeProofManagedNupkgSha256Ready = $externalRuntimeProofManagedNupkgSha256Ready
  externalRuntimeProofRuntimeNupkgSha256Ready = $externalRuntimeProofRuntimeNupkgSha256Ready
  externalRuntimeProofLogSha256FormatReady = $externalRuntimeProofLogSha256FormatReady
  externalRuntimeProofLogSha256Matches = $externalRuntimeProofLogSha256Matches
  externalRuntimeProofFailedProofItemCount = $externalRuntimeProofFailedProofItemCount
  externalRuntimeProofOwnerActionStatus = $externalRuntimeProofOwnerActionStatus
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromoteRuntimeProof
  externalRuntimeExecutionEvidence = $externalRuntimeProofCanPromoteRuntimeProof
  externalRuntimeProofDraftState = $externalRuntimeProofDraftState
  externalRuntimeProofDraftClassification = $externalRuntimeProofDraftClassification
  externalRuntimeProofDraftManagedNupkgSha256Ready = $externalRuntimeProofDraftManagedNupkgSha256Ready
  externalRuntimeProofDraftRuntimeNupkgSha256Ready = $externalRuntimeProofDraftRuntimeNupkgSha256Ready
  externalRuntimeProofDraftLogSha256Ready = $externalRuntimeProofDraftLogSha256Ready
  externalRuntimeProofDraftNoProjectReference = $externalRuntimeProofDraftNoProjectReference
  externalRuntimeProofDraftSmokeStatus = $externalRuntimeProofDraftSmokeStatus
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostRuntimeProofCollectionBundleState
  compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = $compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofCollectionBundlePerformsPublish = $compatibleHostRuntimeProofCollectionBundlePerformsPublish
  compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = $compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof
  compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = $compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence
  compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = $compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = $compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = $compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand
  postPublishVerificationState = $postPublishVerificationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishManagedNupkgSha256Ready = $postPublishManagedNupkgSha256Ready
  postPublishRuntimeNupkgSha256Ready = $postPublishRuntimeNupkgSha256Ready
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutSummaryReady = $postPublishStdoutSummaryReady
  postPublishStderrSummaryReady = $postPublishStderrSummaryReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  postPublishAllLogSha256Matches = $postPublishAllLogSha256Matches
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  canCloseReleaseIssue = $canCloseReleaseIssue
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  isDependencyProbeOnly = $isDependencyProbeOnly
  isRealCallbackRuntimeProof = $isPackageConsumerRealCallbackRuntimeProof
  allowRuntimeSmokeBlocked = [bool]$AllowRuntimeSmokeBlocked
  localFeedConsumerStatus = if ($localFeedConsumer) { [string]$localFeedConsumer.RunStatus } else { "missing" }
  realCallbackRuntimeProof = $callbackProof
  signingStatus = $signingStatus
  bilingualDocumentationFindingCount = $bilingualFindingCount
  bilingualDocumentationBacklogStatus = $bilingualBacklogStatus
  userAcceptanceCatalogStatus = $userAcceptanceStatus
  userAcceptanceCatalogItemCount = if ($userAcceptanceCatalog) { [int]$userAcceptanceCatalog.itemCount } else { 0 }
  pendingChecklistItems = @($pendingChecklistItems)
  gates = @($gates.ToArray())
}

$jsonPath = Join-Path $outputRoot "final-release-dry-run-summary.json"
$markdownPath = Join-Path $outputRoot "final-release-dry-run-summary.md"
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Release Dry Run Summary")
$lines.Add("")
$lines.Add("- runtime key: ``$RuntimePackageKey``")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- blocking issues: $($blocking.Count)")
$lines.Add("- manual approvals: $($manualApprovals.Count)")
$lines.Add("- warnings: $($warnings.Count)")
$lines.Add("- package consumer smoke: ``$smokeStatus``")
$lines.Add("- package consumer evidence kind: ``$packageConsumerEvidenceKind``")
$lines.Add("- runtime smoke classification: ``$runtimeSmokeClassification``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- runtime proof diagnostic: $runtimeProofDiagnostic")
$lines.Add("- runtime proof blocker owner action: ``$runtimeProofBlockerOwnerActionStatus``")
$lines.Add("- runtime proof blocker category: ``$runtimeProofBlockerCategory``")
$lines.Add("- owner input schema ready: ``$ownerInputSchemaReady``")
$lines.Add("- forbidden substitute scan state: ``$forbiddenSubstituteScanState``")
$lines.Add("- detected forbidden substitute count: ``$detectedForbiddenSubstituteCount``")
$lines.Add("- owner proof schema/scan status: ``$ownerProofSchemaScanStatus``")
$lines.Add("- clean owner input ready: ``$cleanOwnerInputReady``")
$lines.Add("- owner input readiness status: ``$ownerInputReadinessStatus``")
$lines.Add("- owner input smoke log ready: ``$ownerInputSmokeLogReady``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- external runtime proof runtime key matches: ``$externalRuntimeProofRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof package source runtime key matches: ``$externalRuntimeProofPackageSourceRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- external runtime proof managed nupkg SHA256 ready: ``$externalRuntimeProofManagedNupkgSha256Ready``")
$lines.Add("- external runtime proof runtime nupkg SHA256 ready: ``$externalRuntimeProofRuntimeNupkgSha256Ready``")
$lines.Add("- external runtime proof log SHA256 format ready: ``$externalRuntimeProofLogSha256FormatReady``")
$lines.Add("- external runtime proof log SHA256 matches: ``$externalRuntimeProofLogSha256Matches``")
$lines.Add("- external runtime proof failed proof item count: ``$externalRuntimeProofFailedProofItemCount``")
$lines.Add("- external runtime proof owner action: ``$externalRuntimeProofOwnerActionStatus``")
$lines.Add("- external runtime proof draft state: ``$externalRuntimeProofDraftState``")
$lines.Add("- external runtime proof draft classification: ``$externalRuntimeProofDraftClassification``")
$lines.Add("- external runtime proof draft managed nupkg SHA256 ready: ``$externalRuntimeProofDraftManagedNupkgSha256Ready``")
$lines.Add("- external runtime proof draft runtime nupkg SHA256 ready: ``$externalRuntimeProofDraftRuntimeNupkgSha256Ready``")
$lines.Add("- external runtime proof draft log SHA256 ready: ``$externalRuntimeProofDraftLogSha256Ready``")
$lines.Add("- external runtime proof draft no ProjectReference: ``$externalRuntimeProofDraftNoProjectReference``")
$lines.Add("- external runtime proof draft smoke status: ``$externalRuntimeProofDraftSmokeStatus``")
$lines.Add("- compatible host collection bundle state: ``$compatibleHostRuntimeProofCollectionBundleState``")
$lines.Add("- compatible host collection bundle can promote runtime proof: ``$compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof``")
$lines.Add("- compatible host collection bundle runtime execution evidence: ``$compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle smoke command: ``$compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host collection bundle validation command: ``$compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish proof classification: ``$postPublishProofClassification``")
$lines.Add("- post-publish proof classification promotable: ``$postPublishProofClassificationPromotable``")
$lines.Add("- post-publish managed nupkg SHA256 ready: ``$postPublishManagedNupkgSha256Ready``")
$lines.Add("- post-publish runtime nupkg SHA256 ready: ``$postPublishRuntimeNupkgSha256Ready``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout summary ready: ``$postPublishStdoutSummaryReady``")
$lines.Add("- post-publish stderr summary ready: ``$postPublishStderrSummaryReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish all log SHA256 matches: ``$postPublishAllLogSha256Matches``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- runtime execution evidence: ``$isRuntimeExecutionEvidence``")
$lines.Add("- dependency probe only: ``$isDependencyProbeOnly``")
$lines.Add("- package consumer real callback proof: ``$isPackageConsumerRealCallbackRuntimeProof``")
$lines.Add("- allow runtime smoke blocked: ``$([bool]$AllowRuntimeSmokeBlocked)``")
$lines.Add("- local feed consumer: ``$($summary.localFeedConsumerStatus)``")
$lines.Add("- real callback runtime proof: ``$callbackProof``")
$lines.Add("- signing status: ``$signingStatus``")
$lines.Add("- bilingual documentation findings: $bilingualFindingCount")
$lines.Add("- user acceptance catalog: ``$userAcceptanceStatus``")
$lines.Add("")
$lines.Add("| Gate | Status | Severity | Detail | Evidence |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($gate in $gates) {
  $lines.Add("| $($gate.name) | $($gate.status) | $($gate.severity) | $(ConvertTo-MarkdownCell $gate.detail) | ``$($gate.evidence)`` |")
}
$lines.Add("")
$lines.Add("## Manual Approval Items")
$lines.Add("")
if ($manualApprovals.Count -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($gate in $manualApprovals) {
    $lines.Add("- $($gate.name): $($gate.status) - $($gate.detail)")
  }
}
$lines.Add("")
$lines.Add("This dry run does not publish packages. It verifies package/readiness evidence and keeps external approvals, CUDA-driver runtime smoke, signing policy, Linux runner evidence, and callback proof separate from build success.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release dry run summary written to $jsonPath"
Write-Host "Final release dry run summary written to $markdownPath"

if ($blocking.Count -gt 0) {
  $message = "Final release dry run has $($blocking.Count) blocking issue(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
