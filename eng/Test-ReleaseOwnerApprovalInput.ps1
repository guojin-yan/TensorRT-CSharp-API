[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-owner-approval-input-template.json",
  [string]$RepositoryRoot,
  [switch]$FailOnBlocked
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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function New-ValidationIssue {
  param(
    [string]$Id,
    [string]$Status,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    status = $Status
    severity = $Severity
    detail = $Detail
    isBlocking = [string]::Equals($Severity, "blocker", [System.StringComparison]::OrdinalIgnoreCase) -and
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

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$externalRuntimeProofBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishVerificationBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release owner approval input '$InputPath' was not found."
}

$input = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$issues = New-Object System.Collections.Generic.List[object]

$recordKind = if ($input.PSObject.Properties.Name -contains "recordKind") { [string]$input.recordKind } else { "" }
$approvalState = if ($input.PSObject.Properties.Name -contains "approvalState") { [string]$input.approvalState } else { "" }
$templateOnly = if ($input.PSObject.Properties.Name -contains "templateOnly") { [bool]$input.templateOnly } else { $true }
$requestedCanPublishPublicly = if ($input.PSObject.Properties.Name -contains "canPublishPublicly") { [bool]$input.canPublishPublicly } else { $false }
$ownerName = if ($input.PSObject.Properties.Name -contains "ownerName") { [string]$input.ownerName } else { "" }
$decisionInputs = if ($input.PSObject.Properties.Name -contains "decisionInputs") { @($input.decisionInputs) } else { @() }

if ($templateOnly) {
  $issues.Add((New-ValidationIssue -Id "template-only" -Status "blocked" -Severity "blocker" -Detail "The input is template-only and cannot approve publication.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "template-only" -Status "ready" -Severity "blocker" -Detail "The input is not marked template-only.")) | Out-Null
}

if ($recordKind -eq "release-owner-approval-input-record") {
  $issues.Add((New-ValidationIssue -Id "record-kind" -Status "ready" -Severity "blocker" -Detail "recordKind is release-owner-approval-input-record.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "record-kind" -Status "blocked" -Severity "blocker" -Detail "recordKind must be release-owner-approval-input-record for a real approval input. Current: $recordKind")) | Out-Null
}

if ($approvalState -eq "approved-for-publication") {
  $issues.Add((New-ValidationIssue -Id "approval-state" -Status "ready" -Severity "blocker" -Detail "approvalState is approved-for-publication.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "approval-state" -Status "blocked" -Severity "blocker" -Detail "approvalState must be approved-for-publication before public promotion. Current: $approvalState")) | Out-Null
}

if ($requestedCanPublishPublicly) {
  $issues.Add((New-ValidationIssue -Id "requested-publication" -Status "ready" -Severity "blocker" -Detail "The input explicitly requests canPublishPublicly=true.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "requested-publication" -Status "blocked" -Severity "blocker" -Detail "canPublishPublicly must be explicitly true in the owner input record.")) | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($ownerName)) {
  $issues.Add((New-ValidationIssue -Id "owner-name" -Status "ready" -Severity "blocker" -Detail "ownerName is present.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "owner-name" -Status "blocked" -Severity "blocker" -Detail "ownerName is required for a real approval record.")) | Out-Null
}

if ($finalRelease -and [int]$finalRelease.blockingIssueCount -eq 0) {
  $issues.Add((New-ValidationIssue -Id "final-release-blockers" -Status "ready" -Severity "blocker" -Detail "final release dry run has zero blockers.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "final-release-blockers" -Status "blocked" -Severity "blocker" -Detail "final release dry run is missing or has blocking issues.")) | Out-Null
}

if ($staleAudit -and [int]$staleAudit.findingCount -eq 0) {
  $issues.Add((New-ValidationIssue -Id "stale-release-claims" -Status "ready" -Severity "blocker" -Detail "stale release claims audit is clean.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "stale-release-claims" -Status "blocked" -Severity "blocker" -Detail "stale release claims audit is missing or has findings.")) | Out-Null
}

$finalPackageReviewState = if ($finalPackageReview) { [string]$finalPackageReview.bundleState } else { "missing-final-package-review-bundle" }
$finalPackageReviewPackageCount = if ($finalPackageReview -and $finalPackageReview.PSObject.Properties.Name -contains "packageCount") { [int]$finalPackageReview.packageCount } else { 0 }
$finalPackageReviewNativeAssetCount = if ($finalPackageReview -and $finalPackageReview.PSObject.Properties.Name -contains "nativeAssetCount") { [int]$finalPackageReview.nativeAssetCount } else { 0 }
$finalPackageReviewCanUseAsPublicPackageProof = if ($finalPackageReview -and $finalPackageReview.PSObject.Properties.Name -contains "canUseAsPublicPackageProof") { [bool]$finalPackageReview.canUseAsPublicPackageProof } else { $false }
if ($finalPackageReview -and
  [string]::Equals($finalPackageReviewState, "owner-review-required", [System.StringComparison]::OrdinalIgnoreCase) -and
  $finalPackageReviewPackageCount -ge 1 -and
  $finalPackageReviewNativeAssetCount -gt 0 -and
  -not $finalPackageReviewCanUseAsPublicPackageProof) {
  $issues.Add((New-ValidationIssue -Id "final-package-review-bundle" -Status "ready" -Severity "blocker" -Detail "final-package-review-bundle is present as local package inventory and is not public package proof.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "final-package-review-bundle" -Status "blocked" -Severity "blocker" -Detail "final-package-review-bundle is missing or malformed. Expected bundleState=owner-review-required, packageCount>=1, nativeAssetCount>0, canUseAsPublicPackageProof=false.")) | Out-Null
}

$externalRuntimeProofBackfillPlanState = if ($externalRuntimeProofBackfillPlan -and $externalRuntimeProofBackfillPlan.PSObject.Properties.Name -contains "planState") { [string]$externalRuntimeProofBackfillPlan.planState } else { "missing-external-runtime-proof-backfill-plan" }
$externalRuntimeProofBackfillStepCount = if ($externalRuntimeProofBackfillPlan -and $externalRuntimeProofBackfillPlan.PSObject.Properties.Name -contains "backfillSteps") { @($externalRuntimeProofBackfillPlan.backfillSteps).Count } else { 0 }
$externalRuntimeProofBackfillCanPromoteRuntimeProof = if ($externalRuntimeProofBackfillPlan -and $externalRuntimeProofBackfillPlan.PSObject.Properties.Name -contains "canPromoteRuntimeProof") { [bool]$externalRuntimeProofBackfillPlan.canPromoteRuntimeProof } else { $false }
$postPublishVerificationBackfillPlanState = if ($postPublishVerificationBackfillPlan -and $postPublishVerificationBackfillPlan.PSObject.Properties.Name -contains "planState") { [string]$postPublishVerificationBackfillPlan.planState } else { "missing-post-publish-verification-backfill-plan" }
$postPublishVerificationBackfillStepCount = if ($postPublishVerificationBackfillPlan -and $postPublishVerificationBackfillPlan.PSObject.Properties.Name -contains "backfillSteps") { @($postPublishVerificationBackfillPlan.backfillSteps).Count } else { 0 }
$postPublishVerificationBackfillCanCloseReleaseIssue = if ($postPublishVerificationBackfillPlan -and $postPublishVerificationBackfillPlan.PSObject.Properties.Name -contains "canCloseReleaseIssue") { [bool]$postPublishVerificationBackfillPlan.canCloseReleaseIssue } else { $false }

if ($externalRuntimeProofBackfillPlan -and
  [string]::Equals($externalRuntimeProofBackfillPlanState, "blocked-compatible-host-proof-required", [System.StringComparison]::OrdinalIgnoreCase) -and
  $externalRuntimeProofBackfillStepCount -ge 7 -and
  -not $externalRuntimeProofBackfillCanPromoteRuntimeProof) {
  $issues.Add((New-ValidationIssue -Id "external-runtime-proof-backfill-plan" -Status "ready" -Severity "blocker" -Detail "external-runtime-proof-backfill-plan is present and blocked as guidance-only, not promotable proof.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "external-runtime-proof-backfill-plan" -Status "blocked" -Severity "blocker" -Detail "external-runtime-proof-backfill-plan is missing or malformed. Expected blocked-compatible-host-proof-required, stepCount>=7, canPromoteRuntimeProof=false.")) | Out-Null
}

if ($postPublishVerificationBackfillPlan -and
  [string]::Equals($postPublishVerificationBackfillPlanState, "blocked-real-post-publish-proof-required", [System.StringComparison]::OrdinalIgnoreCase) -and
  $postPublishVerificationBackfillStepCount -ge 9 -and
  -not $postPublishVerificationBackfillCanCloseReleaseIssue) {
  $issues.Add((New-ValidationIssue -Id "post-publish-verification-backfill-plan" -Status "ready" -Severity "blocker" -Detail "post-publish-verification-backfill-plan is present and blocked as guidance-only, not close proof.")) | Out-Null
}
else {
  $issues.Add((New-ValidationIssue -Id "post-publish-verification-backfill-plan" -Status "blocked" -Severity "blocker" -Detail "post-publish-verification-backfill-plan is missing or malformed. Expected blocked-real-post-publish-proof-required, stepCount>=9, canCloseReleaseIssue=false.")) | Out-Null
}

$requiredDecisionIds = @(
  "release-channel",
  "signing-policy",
  "nvidia-redistribution",
  "runtime-proof-disposition",
  "linux-runner-proof-disposition",
  "final-package-review-acknowledgement",
  "post-publish-verification-disposition",
  "backfill-plan-boundary-acknowledgement",
  "callback-proof-disposition"
)

$acceptableDecisionStates = @{
  "release-channel" = @("approved-public-channel", "approved-private-channel")
  "signing-policy" = @("approved-signed", "approved-unsigned-rc")
  "nvidia-redistribution" = @("approved-for-selected-channel", "approved-private-only")
  "runtime-proof-disposition" = @("approved-runtime-proof-ready", "approved-known-limitation-for-rc")
  "linux-runner-proof-disposition" = @("approved-real-linux-proof", "approved-windows-only-rc")
  "final-package-review-acknowledgement" = @("acknowledged-local-package-inventory")
  "post-publish-verification-disposition" = @("acknowledge-post-publish-required", "require-before-closing-issue")
  "backfill-plan-boundary-acknowledgement" = @("acknowledged-guidance-only", "require-proof-backfill-before-publish")
  "callback-proof-disposition" = @("approved-real-callback-proof", "approved-known-limitation-for-rc")
}

$decisionResults = @(
  foreach ($id in $requiredDecisionIds) {
    $decision = @($decisionInputs | Where-Object { [string]$_.id -eq $id }) | Select-Object -First 1
    if (-not $decision) {
      $status = "missing"
      $detail = "Required owner decision '$id' is missing."
      $rationaleReady = $false
    }
    else {
      $state = if ($decision.PSObject.Properties.Name -contains "decisionState") { [string]$decision.decisionState } else { "" }
      $rationale = if ($decision.PSObject.Properties.Name -contains "rationale") { [string]$decision.rationale } else { "" }
      $decisionOwner = if ($decision.PSObject.Properties.Name -contains "ownerName") { [string]$decision.ownerName } else { "" }
      $allowed = @($acceptableDecisionStates[$id])
      $stateReady = $state -in $allowed
      $rationaleReady = -not [string]::IsNullOrWhiteSpace($rationale)
      $ownerReady = (-not [string]::IsNullOrWhiteSpace($decisionOwner)) -or (-not [string]::IsNullOrWhiteSpace($ownerName))

      if ($stateReady -and $rationaleReady -and $ownerReady) {
        $status = "ready"
        $detail = "Decision '$id' is resolved as '$state'."
      }
      else {
        $status = "blocked"
        $allowedText = $allowed -join ", "
        $detail = "Decision '$id' is not resolved. state='$state'; allowed='$allowedText'; rationalePresent=$rationaleReady; ownerPresent=$ownerReady."
      }
    }

    [pscustomobject]@{
      id = $id
      status = $status
      detail = $detail
      isBlocking = -not [string]::Equals($status, "ready", [System.StringComparison]::OrdinalIgnoreCase)
    }
  }
)

foreach ($decisionResult in $decisionResults) {
  $severity = if ($decisionResult.isBlocking) { "blocker" } else { "blocker" }
  $issues.Add((New-ValidationIssue -Id "owner-decision:$($decisionResult.id)" -Status $decisionResult.status -Severity $severity -Detail $decisionResult.detail)) | Out-Null
}

$blocking = @($issues | Where-Object { $_.isBlocking })
$warnings = @($issues | Where-Object { $_.isWarning })
if ($blocking.Count -gt 0) {
  $overallStatus = "blocked-owner-input-required"
}
elseif ($warnings.Count -gt 0) {
  $overallStatus = "ready-with-owner-warnings"
}
else {
  $overallStatus = "ready-for-owner-approved-publication"
}

$canPublishPublicly = $overallStatus -eq "ready-for-owner-approved-publication"

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-owner-approval-input-validation"
  inputPath = $resolvedInputPath
  inputRecordKind = $recordKind
  inputApprovalState = $approvalState
  inputTemplateOnly = $templateOnly
  requestedCanPublishPublicly = $requestedCanPublishPublicly
  canPublishPublicly = $canPublishPublicly
  overallStatus = $overallStatus
  blockingIssueCount = $blocking.Count
  warningCount = $warnings.Count
  runtimeProofStatus = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofStatus") { [string]$finalRelease.runtimeProofStatus } else { "missing" }
  runtimeProofRequiredForRelease = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") { [bool]$finalRelease.runtimeProofRequiredForRelease } else { $true }
  staleReleaseClaimsFindingCount = if ($staleAudit) { [int]$staleAudit.findingCount } else { -1 }
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofBackfillCanPromoteRuntimeProof = $externalRuntimeProofBackfillCanPromoteRuntimeProof
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationBackfillCanCloseReleaseIssue = $postPublishVerificationBackfillCanCloseReleaseIssue
  issues = @($issues.ToArray())
  decisionResults = @($decisionResults)
  safetyNotes = @(
    "This validator does not publish packages.",
    "A template-only input is always blocked.",
    "canPublishPublicly=true requires an explicit non-template owner input record.",
    "Runtime proof blockers must remain visible even when accepted as an RC limitation.",
    "final-package-review-bundle is local package inventory only; it cannot be used as public package proof, runtime proof, or post-publish verification.",
    "Backfill plans are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "Linux template-only evidence and callback proof=false remain separate release limitations."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-owner-approval-input-validation.json"
$markdownPath = Join-Path $outputRoot "release-owner-approval-input-validation.md"

$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Owner Approval Input Validation")
$lines.Add("")
$lines.Add("- input path: ``$resolvedInputPath``")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- can publish publicly: ``$canPublishPublicly``")
$lines.Add("- blocking issues: $($blocking.Count)")
$lines.Add("- warnings: $($warnings.Count)")
$lines.Add("- input record kind: ``$recordKind``")
$lines.Add("- input approval state: ``$approvalState``")
$lines.Add("- input template only: ``$templateOnly``")
$lines.Add("- requested can publish publicly: ``$requestedCanPublishPublicly``")
$lines.Add("- final package review state: ``$finalPackageReviewState``")
$lines.Add("- final package review package count: ``$finalPackageReviewPackageCount``")
$lines.Add("- final package review native asset count: ``$finalPackageReviewNativeAssetCount``")
$lines.Add("- final package review can use as public package proof: ``$finalPackageReviewCanUseAsPublicPackageProof``")
$lines.Add("- external runtime proof backfill plan state: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof backfill step count: ``$externalRuntimeProofBackfillStepCount``")
$lines.Add("- external runtime proof backfill can promote runtime proof: ``$externalRuntimeProofBackfillCanPromoteRuntimeProof``")
$lines.Add("- post-publish verification backfill plan state: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification backfill step count: ``$postPublishVerificationBackfillStepCount``")
$lines.Add("- post-publish verification backfill can close release issue: ``$postPublishVerificationBackfillCanCloseReleaseIssue``")
$lines.Add("")
$lines.Add("| Issue | Status | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($issue in $issues) {
  $lines.Add("| ``$($issue.id)`` | ``$($issue.status)`` | ``$($issue.severity)`` | $(ConvertTo-MarkdownCell $issue.detail) |")
}
$lines.Add("")
$lines.Add("## Decision Results")
$lines.Add("")
$lines.Add("| Decision | Status | Detail |")
$lines.Add("| --- | --- | --- |")
foreach ($decision in $decisionResults) {
  $lines.Add("| ``$($decision.id)`` | ``$($decision.status)`` | $(ConvertTo-MarkdownCell $decision.detail) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $summary.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release owner approval input validation written to $jsonPath"
Write-Host "Release owner approval input validation written to $markdownPath"

if ($FailOnBlocked.IsPresent -and $blocking.Count -gt 0) {
  Write-Error "Release owner approval input is blocked by $($blocking.Count) issue(s)."
  exit 1
}
