[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$InputPath,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-OwnerPublicPublishExecutionResultCandidate.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 40
$candidateItems = @((Get-OwnerPropertyOrDefault -Object $record -Name "candidateItems" -DefaultValue @()))
$candidateItemCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "candidateItemCount" -DefaultValue 0)
$readyCandidateCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "readyCandidateCount" -DefaultValue 0)
$blockedCandidateCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "blockedCandidateCount" -DefaultValue 0)
$candidatePresent = $candidateItemCount -gt 0
$firstCandidate = if ($candidateItems.Count -gt 0) { $candidateItems[0] } else { $null }
$fieldMap = Get-OwnerPropertyOrDefault -Object $firstCandidate -Name "fieldValues" -DefaultValue (Get-OwnerPropertyOrDefault -Object $record -Name "fieldValues" -DefaultValue $null)
$summary = Get-OwnerPropertyOrDefault -Object $firstCandidate -Name "resultSummary" -DefaultValue (Get-OwnerPropertyOrDefault -Object $record -Name "resultSummary" -DefaultValue $null)
$forbiddenFindings = Get-OwnerPublicPublishForbiddenFindings -Record $record -FieldMap $fieldMap

function Add-CandidateValidationItem {
  param(
    [System.Collections.Generic.List[object]]$Items,
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  $Items.Add((New-OwnerValidationItem -Id $Id -Passed $Passed -Severity $Severity -Detail $Detail)) | Out-Null
}

function Get-SummaryValue {
  param([string]$Name)
  return [string](Get-OwnerPropertyOrDefault -Object $summary -Name $Name -DefaultValue "")
}

function Test-SummaryTrue {
  param([string]$Name)
  $value = Get-OwnerPropertyOrDefault -Object $summary -Name $Name -DefaultValue $false
  if ($value -is [bool]) { return [bool]$value }
  return ([string]$value).Equals("true", [StringComparison]::OrdinalIgnoreCase)
}

$items = New-Object System.Collections.Generic.List[object]
Add-CandidateValidationItem $items "record-kind" ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-result-candidate") "blocker" "recordKind must match."
Add-CandidateValidationItem $items "candidate-counts-consistent" ($candidateItemCount -eq $candidateItems.Count -and $readyCandidateCount -eq $candidateItemCount -and $blockedCandidateCount -ge 0) "blocker" "Candidate counts must be internally consistent."
Add-CandidateValidationItem $items "non-proof-flags" ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Candidate must remain non-proof and side-effect free."
Add-CandidateValidationItem $items "boundary" ($raw.IndexOf("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("cannot close", [StringComparison]::OrdinalIgnoreCase) -ge 0) "blocker" "Candidate must state non-proof boundary."
Add-CandidateValidationItem $items "owner-input-present" $candidatePresent "action-required" "No candidate should be ready before real Owner evidence is supplied."

if ($candidatePresent) {
  Add-CandidateValidationItem $items "source-github-actions-run-evidence-ready" (Test-SummaryTrue "sourceGitHubActionsRunEvidenceReady") "action-required" "Owner publish result must link to a ready GitHub Actions run evidence validation record."
  Add-CandidateValidationItem $items "source-github-actions-run-id-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "sourceGitHubActionsRunId"))) "action-required" "GitHub Actions run id must be linked."
  Add-CandidateValidationItem $items "source-github-actions-run-url-present" ((Get-SummaryValue "sourceGitHubActionsRunUrl") -match '^https://github.com/.+/actions/runs/[0-9]+') "action-required" "GitHub Actions run URL must be linked."
  Add-CandidateValidationItem $items "source-head-sha-format" ((Get-SummaryValue "sourceHeadSha") -match '^[0-9a-fA-F]{40}$') "action-required" "Source head SHA must link the publish result to reviewed code."
  Add-CandidateValidationItem $items "source-workflow-log-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "sourceWorkflowRunLogSha256")) "action-required" "GitHub Actions workflow run log SHA256 must be present."
  Add-CandidateValidationItem $items "source-artifact-manifest-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "sourceArtifactManifestSha256")) "action-required" "GitHub Actions artifact manifest SHA256 must be present."

  Add-CandidateValidationItem $items "public-package-id-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "publicPackageId"))) "action-required" "Public package id must be present."
  Add-CandidateValidationItem $items "public-package-version-present" ((Get-SummaryValue "publicPackageVersion") -match '^[0-9]+(\.[0-9A-Za-z][0-9A-Za-z.-]*)+$') "action-required" "Public package version must be present."
  Add-CandidateValidationItem $items "public-package-source-nuget" ((Get-SummaryValue "publicPackageSource") -match '(?i)nuget\.org|public\s+nuget') "action-required" "Public package source must be nuget.org/public NuGet, not a local feed."
  Add-CandidateValidationItem $items "public-package-url-nuget" (Test-OwnerPublicNuGetUrl -Value (Get-SummaryValue "publicPackageUrl")) "action-required" "Public package URL must be a nuget.org package page."
  Add-CandidateValidationItem $items "public-package-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "publicPackageSha256")) "action-required" "Public package SHA256 must be present."
  Add-CandidateValidationItem $items "public-package-published-at-utc" (Test-OwnerDateTimeOffset -Value (Get-SummaryValue "publicPackagePublishedAtUtc")) "action-required" "Public package publishedAtUtc must be parseable."

  Add-CandidateValidationItem $items "managed-package-id-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "managedPackageId"))) "action-required" "Managed package id must be present."
  Add-CandidateValidationItem $items "runtime-package-id-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "runtimePackageId"))) "action-required" "Runtime package id must be present."
  Add-CandidateValidationItem $items "managed-package-url-nuget" (Test-OwnerPublicNuGetUrl -Value (Get-SummaryValue "managedPackageUrl")) "action-required" "Managed package URL must be a nuget.org package page."
  Add-CandidateValidationItem $items "runtime-package-url-nuget" (Test-OwnerPublicNuGetUrl -Value (Get-SummaryValue "runtimePackageUrl")) "action-required" "Runtime package URL must be a nuget.org package page."
  Add-CandidateValidationItem $items "managed-package-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "managedPackageSha256")) "action-required" "Managed package SHA256 must be present."
  Add-CandidateValidationItem $items "runtime-package-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "runtimePackageSha256")) "action-required" "Runtime package SHA256 must be present."

  Add-CandidateValidationItem $items "github-release-url" (Test-OwnerGitHubUrl -Value (Get-SummaryValue "githubReleaseUrl")) "action-required" "GitHub release URL must be present for the managed plus bridge-only asset route."
  Add-CandidateValidationItem $items "github-release-asset-url" (Test-OwnerGitHubUrl -Value (Get-SummaryValue "githubReleaseAssetUrl")) "action-required" "GitHub release asset URL must identify a managed or bridge-only package asset."
  Add-CandidateValidationItem $items "github-release-asset-sha256" (Test-OwnerSha256Format -Value (Get-SummaryValue "githubReleaseAssetSha256")) "action-required" "GitHub release asset SHA256 must be present."
  Add-CandidateValidationItem $items "package-managed-source-channel-public" ((Get-SummaryValue "packageManagedPackageSourceChannel") -match '(?i)nuget\.org|public\s+nuget') "action-required" "Managed package source channel must identify public NuGet."
  Add-CandidateValidationItem $items "package-source-channel-public" ((Get-SummaryValue "packageSourceChannel") -match '(?i)nuget\.org|github') "action-required" "Package source channel must identify public NuGet/GitHub package routes."

  Add-CandidateValidationItem $items "owner-reviewer-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "ownerReviewer"))) "action-required" "Owner reviewer must be present."
  Add-CandidateValidationItem $items "owner-review-timestamp-utc" (Test-OwnerDateTimeOffset -Value (Get-SummaryValue "ownerReviewTimestampUtc")) "action-required" "Owner review timestamp must be parseable."
  Add-CandidateValidationItem $items "owner-authorization-id-present" (-not [string]::IsNullOrWhiteSpace((Get-SummaryValue "ownerAuthorizationId"))) "action-required" "Owner authorization id must be present."
  Add-CandidateValidationItem $items "owner-authorization-timestamp-utc" (Test-OwnerDateTimeOffset -Value (Get-SummaryValue "ownerAuthorizationTimestampUtc")) "action-required" "Owner authorization timestamp must be parseable."
}

Add-CandidateValidationItem $items "forbidden-substitutes-absent" ($forbiddenFindings.Count -eq 0) "blocker" $(if ($forbiddenFindings.Count -eq 0) { "No local feed, direct nupkg, dry-run, dashboard, manual approval, queued workflow, ProjectReference, artifact-only, or local test substitute was detected." } else { "Forbidden substitute(s): $($forbiddenFindings -join ', ')" })

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$proofCandidateReady = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-owner-public-publish-execution-result-candidate"
}
elseif ($proofCandidateReady) {
  "owner-public-publish-execution-result-candidate-ready"
}
else {
  "blocked-owner-public-publish-execution-result-input-required"
}

$validation = [ordered]@{
  recordKind = "owner-public-publish-execution-result-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  candidateState = [string](Get-OwnerPropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
  candidateItemCount = $candidateItemCount
  readyCandidateCount = $readyCandidateCount
  blockedCandidateCount = $blockedCandidateCount
  proofCandidateReady = $proofCandidateReady
  sourceGitHubActionsRunEvidenceReady = Test-SummaryTrue "sourceGitHubActionsRunEvidenceReady"
  sourceGitHubActionsRunId = Get-SummaryValue "sourceGitHubActionsRunId"
  sourceGitHubActionsRunUrl = Get-SummaryValue "sourceGitHubActionsRunUrl"
  sourceHeadSha = Get-SummaryValue "sourceHeadSha"
  publicPackageId = Get-SummaryValue "publicPackageId"
  publicPackageVersion = Get-SummaryValue "publicPackageVersion"
  publicPackageSource = Get-SummaryValue "publicPackageSource"
  publicPackageUrl = Get-SummaryValue "publicPackageUrl"
  publicPackageSha256 = Get-SummaryValue "publicPackageSha256"
  publicPackagePublishedAtUtc = Get-SummaryValue "publicPackagePublishedAtUtc"
  managedPackageUrl = Get-SummaryValue "managedPackageUrl"
  runtimePackageUrl = Get-SummaryValue "runtimePackageUrl"
  githubReleaseUrl = Get-SummaryValue "githubReleaseUrl"
  githubReleaseAssetUrl = Get-SummaryValue "githubReleaseAssetUrl"
  githubReleaseAssetSha256 = Get-SummaryValue "githubReleaseAssetSha256"
  ownerReviewer = Get-SummaryValue "ownerReviewer"
  ownerReviewTimestampUtc = Get-SummaryValue "ownerReviewTimestampUtc"
  ownerAuthorizationId = Get-SummaryValue "ownerAuthorizationId"
  forbiddenSubstituteFindings = @($forbiddenFindings)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  boundary = "Owner public publish execution result candidate validation only. It can make the owner-public-publish-result lane structurally ready after real Owner public package evidence passes, but it is not runtime proof, not post-publish proof, not release close approval, does not execute package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 40)

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-OwnerMarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-OwnerMarkdownCell $_.severity) | $(ConvertTo-OwnerMarkdownCell $_.detail) |"
}
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Candidate Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- candidateItemCount: ``$($validation.candidateItemCount)``",
  "- proofCandidateReady: ``$($validation.proofCandidateReady)``",
  "- sourceHeadSha: ``$($validation.sourceHeadSha)``",
  "- publicPackageUrl: ``$($validation.publicPackageUrl)``",
  "- managedPackageUrl: ``$($validation.managedPackageUrl)``",
  "- runtimePackageUrl: ``$($validation.runtimePackageUrl)``",
  "- githubReleaseAssetUrl: ``$($validation.githubReleaseAssetUrl)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "## Validation Items",
  "",
  "| ID | Passed | Severity | Detail |",
  "| --- | ---: | --- | --- |",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "ProofCandidateReady=$($validation.proofCandidateReady)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
Write-Host "FailedActionRequiredCount=$($validation.failedActionRequiredCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner public publish execution result candidate validation failed." }
