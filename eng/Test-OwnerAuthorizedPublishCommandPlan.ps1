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
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Detail,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    detail = $Detail
    boundary = $Boundary
  }
}

$plan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan.json"
if ($null -eq $plan) {
  throw "Missing artifacts/final-release/owner-authorized-publish-command-plan.json. Run Export-OwnerAuthorizedPublishCommandPlan.ps1 first."
}

$recordKind = [string](Get-PropertyOrDefault -Object $plan -Name "recordKind" -DefaultValue "")
$planRuntimePackageKey = [string](Get-PropertyOrDefault -Object $plan -Name "runtimePackageKey" -DefaultValue "")
$planState = [string](Get-PropertyOrDefault -Object $plan -Name "planState" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $plan -Name "performsPublish" -DefaultValue $true)
$requiresHumanOwner = [bool](Get-PropertyOrDefault -Object $plan -Name "requiresHumanOwner" -DefaultValue $false)
$requiresExplicitOwnerAuthorization = [bool](Get-PropertyOrDefault -Object $plan -Name "requiresExplicitOwnerAuthorization" -DefaultValue $false)
$canMaterializeExecutableCommands = [bool](Get-PropertyOrDefault -Object $plan -Name "canMaterializeExecutableCommands" -DefaultValue $true)
$realExternalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $plan -Name "realExternalRuntimeProofReady" -DefaultValue $false)
$realPostPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $plan -Name "realPostPublishVerificationReady" -DefaultValue $false)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $plan -Name "runtimeProofStatus" -DefaultValue "")
$publishCommands = @(Get-PropertyOrDefault -Object $plan -Name "publishCommands" -DefaultValue @())
$postPublishCommandPlan = @(Get-PropertyOrDefault -Object $plan -Name "postPublishCommandPlan" -DefaultValue @())
$blockingReasons = @(Get-PropertyOrDefault -Object $plan -Name "blockingReasons" -DefaultValue @())
$sourceEvidence = @(Get-PropertyOrDefault -Object $plan -Name "sourceEvidence" -DefaultValue @())
$safetyNotes = @(Get-PropertyOrDefault -Object $plan -Name "safetyNotes" -DefaultValue @())
$ownerAuthorizationRequiredFields = @(Get-PropertyOrDefault -Object $plan -Name "ownerAuthorizationRequiredFields" -DefaultValue @())
$ownerAuthorizationProofGate = Get-PropertyOrDefault -Object $plan -Name "ownerAuthorizationProofGate" -DefaultValue $null
$manualMaterializationPrerequisites = @(Get-PropertyOrDefault -Object $plan -Name "manualMaterializationPrerequisites" -DefaultValue @())
$postPublishRequiredEvidence = @(Get-PropertyOrDefault -Object $plan -Name "postPublishRequiredEvidence" -DefaultValue @())
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofBackfillPlanState" -DefaultValue "")
$externalRuntimeProofBackfillStepCount = [int](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofBackfillStepCount" -DefaultValue 0)
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofBackfillCanPromoteRuntimeProof" -DefaultValue $true)
$externalRuntimeProofCollectionPackageState = [string](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofCollectionPackageState" -DefaultValue "")
$externalRuntimeProofCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofCollectionPackageStepCount" -DefaultValue 0)
$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofCollectionPackageCanPromoteRuntimeProof" -DefaultValue $true)
$externalRuntimeProofCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofCollectionPackageCanCloseReleaseIssue" -DefaultValue $true)
$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $plan -Name "externalRuntimeProofCollectionPackageRuntimeExecutionEvidence" -DefaultValue $true)
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationBackfillPlanState" -DefaultValue "")
$postPublishVerificationBackfillStepCount = [int](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationBackfillStepCount" -DefaultValue 0)
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationBackfillCanCloseReleaseIssue" -DefaultValue $true)
$postPublishVerificationCollectionPackageState = [string](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationCollectionPackageState" -DefaultValue "")
$postPublishVerificationCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationCollectionPackageStepCount" -DefaultValue 0)
$postPublishVerificationCollectionPackageProof = [bool](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationCollectionPackageProof" -DefaultValue $true)
$postPublishVerificationCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $plan -Name "postPublishVerificationCollectionPackageCanCloseReleaseIssue" -DefaultValue $true)

$allPublishCommandsSafe = $publishCommands.Count -gt 0
$allPublishCommandsPlaceholderOnly = $publishCommands.Count -gt 0
foreach ($command in $publishCommands) {
  if ([bool](Get-PropertyOrDefault -Object $command -Name "authorized" -DefaultValue $true)) {
    $allPublishCommandsSafe = $false
  }
  if ([bool](Get-PropertyOrDefault -Object $command -Name "executable" -DefaultValue $true)) {
    $allPublishCommandsSafe = $false
  }
  if ([bool](Get-PropertyOrDefault -Object $command -Name "performsPublish" -DefaultValue $true)) {
    $allPublishCommandsSafe = $false
  }

  $commandText = [string](Get-PropertyOrDefault -Object $command -Name "command" -DefaultValue "")
  $materializedExecutableCommand = [string](Get-PropertyOrDefault -Object $command -Name "materializedExecutableCommand" -DefaultValue "unexpected-materialized-command")
  $commandMaterializationState = [string](Get-PropertyOrDefault -Object $command -Name "commandMaterializationState" -DefaultValue "")
  $placeholderOnly = [bool](Get-PropertyOrDefault -Object $command -Name "placeholderOnly" -DefaultValue $false)
  $ownerExecutionOnly = [bool](Get-PropertyOrDefault -Object $command -Name "ownerExecutionOnly" -DefaultValue $false)
  $modelExecutionForbidden = [bool](Get-PropertyOrDefault -Object $command -Name "modelExecutionForbidden" -DefaultValue $false)
  $copyOnlyAfterOwnerAuthorization = [bool](Get-PropertyOrDefault -Object $command -Name "copyOnlyAfterOwnerAuthorization" -DefaultValue $false)
  $hasPlaceholders = $commandText.Contains("<", [System.StringComparison]::Ordinal) -and $commandText.Contains(">", [System.StringComparison]::Ordinal)

  if (-not $placeholderOnly -or
      -not [string]::Equals($commandMaterializationState, "placeholder-only", [System.StringComparison]::Ordinal) -or
      -not [string]::IsNullOrWhiteSpace($materializedExecutableCommand) -or
      -not $ownerExecutionOnly -or
      -not $modelExecutionForbidden -or
      -not $copyOnlyAfterOwnerAuthorization -or
      -not $hasPlaceholders) {
    $allPublishCommandsPlaceholderOnly = $false
  }
}

$hasNuGetPushPlaceholder = @($publishCommands | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "command" -DefaultValue "")).Contains("dotnet nuget push", [System.StringComparison]::Ordinal) }).Count -gt 0
$hasGitHubReleasePlaceholder = @($publishCommands | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "command" -DefaultValue "")).Contains("gh release upload", [System.StringComparison]::Ordinal) }).Count -gt 0
$hasPostPublishRuntimeSmoke = @($postPublishCommandPlan | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "run-runtime-smoke" }).Count -gt 0
$hasValidatePostPublish = @($postPublishCommandPlan | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "validate-post-publish-record" }).Count -gt 0
$ownerAuthorizationProofGateState = [string](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "gateState" -DefaultValue "")
$ownerAuthorizationProofGateRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "requiredFieldCount" -DefaultValue 0)
$ownerAuthorizationProofGateMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "missingOwnerInputCount" -DefaultValue -1)
$ownerAuthorizationProofGateCanMaterialize = [bool](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "canMaterializeExecutableCommands" -DefaultValue $true)
$ownerAuthorizationProofGateCanPublish = [bool](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "canPublishPublicly" -DefaultValue $true)
$ownerAuthorizationProofGateCanClose = [bool](Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "canCloseReleaseIssue" -DefaultValue $true)
$ownerAuthorizationProofGateValidators = @(Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "requiredValidators" -DefaultValue @())
$ownerAuthorizationProofGateArtifacts = @(Get-PropertyOrDefault -Object $ownerAuthorizationProofGate -Name "requiredArtifacts" -DefaultValue @())
$ownerRequiredFieldsComplete = $ownerAuthorizationRequiredFields.Count -ge 12 -and
  @($ownerAuthorizationRequiredFields | Where-Object { [string]$_ -eq "ownerName" }).Count -gt 0 -and
  @($ownerAuthorizationRequiredFields | Where-Object { [string]$_ -eq "approvalTimestampUtc" }).Count -gt 0 -and
  @($ownerAuthorizationRequiredFields | Where-Object { [string]$_ -eq "approvedCommandPlanSha256" }).Count -gt 0 -and
  @($ownerAuthorizationRequiredFields | Where-Object { [string]$_ -eq "approvedProofBundleSha256" }).Count -gt 0
$ownerGateBlocked = [string]::Equals($ownerAuthorizationProofGateState, "blocked-owner-authorization-required", [System.StringComparison]::Ordinal) -and
  $ownerAuthorizationProofGateRequiredFieldCount -eq $ownerAuthorizationRequiredFields.Count -and
  $ownerAuthorizationProofGateMissingOwnerInputCount -eq $ownerAuthorizationRequiredFields.Count -and
  -not $ownerAuthorizationProofGateCanMaterialize -and
  -not $ownerAuthorizationProofGateCanPublish -and
  -not $ownerAuthorizationProofGateCanClose
$ownerGateReferencesValidators = @($ownerAuthorizationProofGateValidators | Where-Object { [string]$_ -eq "Test-ReleaseOwnerApprovalInput.ps1" }).Count -gt 0 -and
  @($ownerAuthorizationProofGateValidators | Where-Object { [string]$_ -eq "Test-OwnerAuthorizedPublishCommandPlan.ps1" }).Count -gt 0
$ownerGateReferencesArtifacts = @($ownerAuthorizationProofGateArtifacts | Where-Object { [string]$_ -eq "artifacts/final-release/release-owner-approval-input-record.json" }).Count -gt 0 -and
  @($ownerAuthorizationProofGateArtifacts | Where-Object { [string]$_ -eq "artifacts/final-release/owner-authorized-publish-command-plan-validation.json" }).Count -gt 0
$manualPrerequisiteIds = @($manualMaterializationPrerequisites | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$manualPrerequisitesComplete = @("owner-authorization", "owner-decision", "freeze-summary", "package-consumer-runtime-proof", "publish-checklist", "stale-release-claims") |
  ForEach-Object { $manualPrerequisiteIds -contains $_ } |
  Where-Object { -not $_ } |
  Measure-Object |
  Select-Object -ExpandProperty Count
$manualPrerequisitesHaveBlockingState = $manualMaterializationPrerequisites.Count -ge 6 -and
  @($manualMaterializationPrerequisites | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocksMaterialization" -DefaultValue $false) }).Count -ge 3
$postPublishEvidenceComplete = @(
  "selectedChannel",
  "channelSourceUri",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "cleanConsumerRootOutsideRepository",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "runtimeSmokeLogSha256",
  "stdoutSummary",
  "stderrSummary",
  "hostMetadata"
) | ForEach-Object { $postPublishRequiredEvidence -contains $_ } | Where-Object { -not $_ } | Measure-Object | Select-Object -ExpandProperty Count
$sourceEvidenceHasFreeze = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/release/release-candidate-freeze-summary.json" }).Count -gt 0
$sourceEvidenceHasPostPublish = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/final-release/post-publish-verification-validation.json" }).Count -gt 0
$sourceEvidenceHasExternalBackfillPlan = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/final-release/external-runtime-proof-backfill-plan.json" }).Count -gt 0
$sourceEvidenceHasPostPublishBackfillPlan = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/final-release/post-publish-verification-backfill-plan.json" }).Count -gt 0
$sourceEvidenceHasExternalCollectionPackage = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/final-release/external-runtime-proof-collection-package.json" }).Count -gt 0
$sourceEvidenceHasPostPublishCollectionPackage = @($sourceEvidence | Where-Object { [string]$_ -eq "artifacts/final-release/post-publish-verification-collection-package.json" }).Count -gt 0
$safetyNotesBlockDriver = @($safetyNotes | Where-Object { ([string]$_).Contains("blocked-by-cuda-driver is not smoke passed", [System.StringComparison]::Ordinal) }).Count -gt 0
$safetyNotesBlockBackfillPromotion = @($safetyNotes | Where-Object { ([string]$_).Contains("Backfill plans and collection packages are guidance only", [System.StringComparison]::Ordinal) }).Count -gt 0

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]::Equals($recordKind, "owner-authorized-publish-command-plan", [System.StringComparison]::Ordinal)) -Detail "recordKind=$recordKind" -Boundary "The command plan must be a dedicated owner-facing record."
  New-ValidationItem -Id "runtime-package-key" -Passed ([string]::Equals($planRuntimePackageKey, $RuntimePackageKey, [System.StringComparison]::Ordinal)) -Detail "runtimePackageKey=$planRuntimePackageKey" -Boundary "Command plan must target the selected runtime package key."
  New-ValidationItem -Id "no-publish-side-effect" -Passed (-not $performsPublish) -Detail "performsPublish=$performsPublish" -Boundary "Validation must prove the plan does not execute publish."
  New-ValidationItem -Id "requires-owner" -Passed ($requiresHumanOwner -and $requiresExplicitOwnerAuthorization) -Detail "requiresHumanOwner=$requiresHumanOwner; requiresExplicitOwnerAuthorization=$requiresExplicitOwnerAuthorization" -Boundary "Owner authorization must remain explicit."
  New-ValidationItem -Id "no-materialized-executable-commands" -Passed (-not $canMaterializeExecutableCommands) -Detail "canMaterializeExecutableCommands=$canMaterializeExecutableCommands" -Boundary "Default output must stay placeholder-only."
  New-ValidationItem -Id "publish-commands-safe" -Passed $allPublishCommandsSafe -Detail "publishCommandCount=$($publishCommands.Count)" -Boundary "Every publish command must be authorized=false, executable=false, performsPublish=false."
  New-ValidationItem -Id "no-materialized-push-command" -Passed ((-not $canMaterializeExecutableCommands) -and $allPublishCommandsPlaceholderOnly) -Detail "canMaterializeExecutableCommands=$canMaterializeExecutableCommands; publishCommandCount=$($publishCommands.Count)" -Boundary "When executable command materialization is disabled, every publish command must remain placeholder-only with no materialized executable command text."
  New-ValidationItem -Id "nuget-placeholder-visible" -Passed $hasNuGetPushPlaceholder -Detail "hasNuGetPushPlaceholder=$hasNuGetPushPlaceholder" -Boundary "Owner command review must show NuGet placeholder text without executing it."
  New-ValidationItem -Id "github-release-placeholder-visible" -Passed $hasGitHubReleasePlaceholder -Detail "hasGitHubReleasePlaceholder=$hasGitHubReleasePlaceholder" -Boundary "Owner command review must show GitHub Release placeholder text without executing it."
  New-ValidationItem -Id "post-publish-smoke-plan-visible" -Passed ($hasPostPublishRuntimeSmoke -and $hasValidatePostPublish) -Detail "hasPostPublishRuntimeSmoke=$hasPostPublishRuntimeSmoke; hasValidatePostPublish=$hasValidatePostPublish" -Boundary "Post-publish proof backfill commands must remain visible."
  New-ValidationItem -Id "owner-authorization-required-fields-visible" -Passed $ownerRequiredFieldsComplete -Detail "ownerAuthorizationRequiredFieldCount=$($ownerAuthorizationRequiredFields.Count)" -Boundary "Owner authorization must list concrete owner fields instead of relying on a vague approval flag."
  New-ValidationItem -Id "owner-authorization-proof-gate-blocked" -Passed $ownerGateBlocked -Detail "gateState=$ownerAuthorizationProofGateState; requiredFieldCount=$ownerAuthorizationProofGateRequiredFieldCount; missingOwnerInputCount=$ownerAuthorizationProofGateMissingOwnerInputCount; canMaterialize=$ownerAuthorizationProofGateCanMaterialize; canPublish=$ownerAuthorizationProofGateCanPublish; canClose=$ownerAuthorizationProofGateCanClose" -Boundary "Generated owner proof gates must remain blocked until a real owner-filled record is validated."
  New-ValidationItem -Id "owner-authorization-proof-gate-sources" -Passed ($ownerGateReferencesValidators -and $ownerGateReferencesArtifacts) -Detail "requiredValidatorCount=$($ownerAuthorizationProofGateValidators.Count); requiredArtifactCount=$($ownerAuthorizationProofGateArtifacts.Count)" -Boundary "Owner proof gate must point to the real owner validators and owner input artifacts."
  New-ValidationItem -Id "manual-materialization-prerequisites-visible" -Passed ($manualPrerequisitesComplete -eq 0 -and $manualPrerequisitesHaveBlockingState) -Detail "manualMaterializationPrerequisiteCount=$($manualMaterializationPrerequisites.Count); missingRequiredPrerequisites=$manualPrerequisitesComplete" -Boundary "Publish command materialization must list every blocking proof prerequisite."
  New-ValidationItem -Id "post-publish-required-evidence-visible" -Passed ($postPublishEvidenceComplete -eq 0) -Detail "postPublishRequiredEvidenceCount=$($postPublishRequiredEvidence.Count); missingRequiredEvidence=$postPublishEvidenceComplete" -Boundary "Post-publish command plan must enumerate real channel, hash, clean consumer, smoke log, summary, and host evidence."
  New-ValidationItem -Id "source-evidence-visible" -Passed ($sourceEvidenceHasFreeze -and $sourceEvidenceHasPostPublish) -Detail "sourceEvidenceCount=$($sourceEvidence.Count)" -Boundary "Plan must cite freeze and post-publish validation artifacts."
  New-ValidationItem -Id "backfill-plan-source-evidence-visible" -Passed ($sourceEvidenceHasExternalBackfillPlan -and $sourceEvidenceHasPostPublishBackfillPlan) -Detail "sourceEvidenceHasExternalBackfillPlan=$sourceEvidenceHasExternalBackfillPlan; sourceEvidenceHasPostPublishBackfillPlan=$sourceEvidenceHasPostPublishBackfillPlan" -Boundary "Plan must cite backfill plans without treating them as proof."
  New-ValidationItem -Id "collection-package-source-evidence-visible" -Passed ($sourceEvidenceHasExternalCollectionPackage -and $sourceEvidenceHasPostPublishCollectionPackage) -Detail "sourceEvidenceHasExternalCollectionPackage=$sourceEvidenceHasExternalCollectionPackage; sourceEvidenceHasPostPublishCollectionPackage=$sourceEvidenceHasPostPublishCollectionPackage" -Boundary "Plan must cite collection packages without treating them as proof."
  New-ValidationItem -Id "external-backfill-plan-blocked" -Passed ([string]::Equals($externalRuntimeProofBackfillPlanState, "blocked-compatible-host-proof-required", [System.StringComparison]::Ordinal) -and $externalRuntimeProofBackfillStepCount -ge 7 -and -not $externalRuntimeProofBackfillCanPromoteRuntimeProof) -Detail "externalRuntimeProofBackfillPlanState=$externalRuntimeProofBackfillPlanState; externalRuntimeProofBackfillStepCount=$externalRuntimeProofBackfillStepCount; externalRuntimeProofBackfillCanPromoteRuntimeProof=$externalRuntimeProofBackfillCanPromoteRuntimeProof" -Boundary "External backfill guidance must not promote proof by itself."
  New-ValidationItem -Id "external-collection-package-blocked" -Passed ([string]::Equals($externalRuntimeProofCollectionPackageState, "owner-action-required", [System.StringComparison]::Ordinal) -and $externalRuntimeProofCollectionPackageStepCount -ge 8 -and -not $externalRuntimeProofCollectionPackageCanPromoteRuntimeProof -and -not $externalRuntimeProofCollectionPackageCanCloseReleaseIssue -and -not $externalRuntimeProofCollectionPackageRuntimeExecutionEvidence) -Detail "externalRuntimeProofCollectionPackageState=$externalRuntimeProofCollectionPackageState; externalRuntimeProofCollectionPackageStepCount=$externalRuntimeProofCollectionPackageStepCount; externalRuntimeProofCollectionPackageCanPromoteRuntimeProof=$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof; externalRuntimeProofCollectionPackageCanCloseReleaseIssue=$externalRuntimeProofCollectionPackageCanCloseReleaseIssue; externalRuntimeProofCollectionPackageRuntimeExecutionEvidence=$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence" -Boundary "External collection package guidance must not promote proof by itself."
  New-ValidationItem -Id "post-publish-backfill-plan-blocked" -Passed ([string]::Equals($postPublishVerificationBackfillPlanState, "blocked-real-post-publish-proof-required", [System.StringComparison]::Ordinal) -and $postPublishVerificationBackfillStepCount -ge 9 -and -not $postPublishVerificationBackfillCanCloseReleaseIssue) -Detail "postPublishVerificationBackfillPlanState=$postPublishVerificationBackfillPlanState; postPublishVerificationBackfillStepCount=$postPublishVerificationBackfillStepCount; postPublishVerificationBackfillCanCloseReleaseIssue=$postPublishVerificationBackfillCanCloseReleaseIssue" -Boundary "Post-publish backfill guidance must not close the release issue by itself."
  New-ValidationItem -Id "post-publish-collection-package-blocked" -Passed ([string]::Equals($postPublishVerificationCollectionPackageState, "blocked-real-publication-required", [System.StringComparison]::Ordinal) -and $postPublishVerificationCollectionPackageStepCount -ge 8 -and -not $postPublishVerificationCollectionPackageProof -and -not $postPublishVerificationCollectionPackageCanCloseReleaseIssue) -Detail "postPublishVerificationCollectionPackageState=$postPublishVerificationCollectionPackageState; postPublishVerificationCollectionPackageStepCount=$postPublishVerificationCollectionPackageStepCount; postPublishVerificationCollectionPackageProof=$postPublishVerificationCollectionPackageProof; postPublishVerificationCollectionPackageCanCloseReleaseIssue=$postPublishVerificationCollectionPackageCanCloseReleaseIssue" -Boundary "Post-publish collection package guidance must not close the release issue by itself."
  New-ValidationItem -Id "blocked-driver-not-promoted" -Passed (-not ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -and $realExternalRuntimeProofReady)) -Detail "runtimeProofStatus=$runtimeProofStatus; realExternalRuntimeProofReady=$realExternalRuntimeProofReady" -Boundary "blocked-by-cuda-driver is not smoke passed."
  New-ValidationItem -Id "post-publish-not-close-ready-by-default" -Passed (-not $realPostPublishVerificationReady) -Detail "realPostPublishVerificationReady=$realPostPublishVerificationReady" -Boundary "Default plan must not claim post-publish proof."
  New-ValidationItem -Id "safety-notes" -Passed ($safetyNotesBlockDriver -and $safetyNotesBlockBackfillPromotion) -Detail "safetyNoteCount=$($safetyNotes.Count)" -Boundary "Safety notes must preserve blocked-by-cuda-driver and backfill-plan boundary wording."
)

$failedItems = @($validationItems | Where-Object { -not $_.passed })
$validationState = if ($failedItems.Count -eq 0) { $planState } else { "invalid-owner-authorized-publish-command-plan" }

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "owner-authorized-publish-command-plan-validation"
  runtimePackageKey = $RuntimePackageKey
  validationState = $validationState
  failedValidationItemCount = $failedItems.Count
  planState = $planState
  performsPublish = $performsPublish
  requiresHumanOwner = $requiresHumanOwner
  requiresExplicitOwnerAuthorization = $requiresExplicitOwnerAuthorization
  canMaterializeExecutableCommands = $canMaterializeExecutableCommands
  publishCommandsPlaceholderOnly = $allPublishCommandsPlaceholderOnly
  realExternalRuntimeProofReady = $realExternalRuntimeProofReady
  realPostPublishVerificationReady = $realPostPublishVerificationReady
  blockingReasonCount = $blockingReasons.Count
  publishCommandCount = $publishCommands.Count
  postPublishCommandCount = $postPublishCommandPlan.Count
  ownerAuthorizationRequiredFieldCount = $ownerAuthorizationRequiredFields.Count
  ownerAuthorizationProofGateState = $ownerAuthorizationProofGateState
  ownerAuthorizationProofGateMissingOwnerInputCount = $ownerAuthorizationProofGateMissingOwnerInputCount
  manualMaterializationPrerequisiteCount = $manualMaterializationPrerequisites.Count
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofCollectionPackageState = $externalRuntimeProofCollectionPackageState
  externalRuntimeProofCollectionPackageStepCount = $externalRuntimeProofCollectionPackageStepCount
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationCollectionPackageState = $postPublishVerificationCollectionPackageState
  postPublishVerificationCollectionPackageStepCount = $postPublishVerificationCollectionPackageStepCount
  validationItems = $validationItems
  safetyNotes = @(
    "Owner command plan validation does not execute publish.",
    "blocked-by-cuda-driver is not smoke passed.",
    "No delete, delist, withdraw, NuGet push, GitHub Packages upload, or GitHub Release upload is performed."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "owner-authorized-publish-command-plan-validation.json"
$markdownPath = Join-Path $outputRoot "owner-authorized-publish-command-plan-validation.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Authorized Publish Command Plan Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- failed validation items: $($failedItems.Count)")
$lines.Add("- performs publish: ``$performsPublish``")
$lines.Add("- can materialize executable commands: ``$canMaterializeExecutableCommands``")
$lines.Add("- blocking reason count: $($blockingReasons.Count)")
$lines.Add("")
$lines.Add("| ID | Passed | Detail |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | $($item.detail.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner authorized publish command plan validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedValidationItemCount=$($failedItems.Count) PerformsPublish=$performsPublish CanMaterializeExecutableCommands=$canMaterializeExecutableCommands"

if ($failedItems.Count -gt 0) {
  throw "Owner authorized publish command plan validation failed with $($failedItems.Count) item(s)."
}
