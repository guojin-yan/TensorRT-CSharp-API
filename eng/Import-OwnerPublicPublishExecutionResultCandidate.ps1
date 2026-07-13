[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$InputPath,
  [string]$PreflightPath,
  [string]$GitHubActionsRunEvidenceValidationPath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template.json" }
if ([string]::IsNullOrWhiteSpace($PreflightPath)) { $PreflightPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-preflight.json" }
if ([string]::IsNullOrWhiteSpace($GitHubActionsRunEvidenceValidationPath)) { $GitHubActionsRunEvidenceValidationPath = Join-Path $ctx.OutputDirectory "github-actions-run-evidence-import-validation.json" }
if (-not (Test-Path -LiteralPath $PreflightPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Test-OwnerPublicPublishExecutionResultPreflight.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory -InputPath $InputPath
}

$template = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$preflight = Get-Content -LiteralPath $PreflightPath -Raw -Encoding utf8 | ConvertFrom-Json
$githubActionsRunEvidenceValidation = if (Test-Path -LiteralPath $GitHubActionsRunEvidenceValidationPath -PathType Leaf) {
  Get-Content -LiteralPath $GitHubActionsRunEvidenceValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
} else {
  $null
}
$readyCandidateCount = [int](Get-OwnerPropertyOrDefault -Object $preflight -Name "readyCandidateCount" -DefaultValue 0)
$ownerInputFields = @((Get-OwnerPropertyOrDefault -Object $template -Name "ownerInputFields" -DefaultValue @()))
$fieldMap = ConvertTo-OwnerPublicPublishFieldMap -Fields $ownerInputFields
$resultSummary = New-OwnerPublicPublishResultSummary -FieldMap $fieldMap -GitHubActionsRunEvidence $githubActionsRunEvidenceValidation
$candidateItems = @()
if ($readyCandidateCount -gt 0) {
  $candidateItems = @([pscustomobject]@{
      candidateId = "owner-public-publish-execution-result-candidate-001"
      candidateState = "owner-public-publish-execution-result-candidate-imported"
      ownerInputFields = @($ownerInputFields)
      fieldValues = $fieldMap
      resultSummary = $resultSummary
      boundary = "Candidate only. It imports Owner-provided public publish result fields for strict validation. It is not runtime proof, not post-publish proof, not release close approval, does not execute dotnet nuget push, and cannot close the release."
    })
}

$record = [ordered]@{
  recordKind = "owner-public-publish-execution-result-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($candidateItems.Count -gt 0) { "owner-public-publish-execution-result-candidate-imported" } else { "blocked-owner-public-publish-execution-result-input-required" }
  sourceInputPath = $InputPath
  sourcePreflightPath = $PreflightPath
  sourceGitHubActionsRunEvidenceValidationPath = $GitHubActionsRunEvidenceValidationPath
  sourceGitHubActionsRunEvidenceReady = [bool](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "githubActionsRunEvidenceReady" -DefaultValue $false)
  sourceGitHubActionsRunId = [string](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "runId" -DefaultValue "")
  sourceGitHubActionsRunUrl = [string](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "runUrl" -DefaultValue "")
  sourceHeadSha = [string](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "headSha" -DefaultValue "")
  sourceWorkflowRunLogSha256 = [string](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "workflowRunLogSha256" -DefaultValue "")
  sourceArtifactManifestSha256 = [string](Get-OwnerPropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "artifactManifestSha256" -DefaultValue "")
  resultSummary = $resultSummary
  readyCandidateCount = $candidateItems.Count
  candidateItemCount = $candidateItems.Count
  blockedCandidateCount = if ($candidateItems.Count -eq 0) { 1 } else { 0 }
  candidateItems = @($candidateItems)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  whyNotProof = @(
    "Candidate import does not execute dotnet nuget push.",
    "Candidate import does not prove clean external consumer restore/build/runtime smoke.",
    "Candidate import does not approve release issue close.",
    "Candidate import cannot substitute public package download proof, post-publish proof, or release close approval."
  )
  boundary = "Owner public publish execution result candidate only. It is strict-validator input for a real Owner-supplied public publish result. It is not runtime proof, not post-publish proof, not release close approval, does not execute package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Candidate",
  "",
  "- candidateState: ``$($record.candidateState)``",
  "- candidateItemCount: ``$($record.candidateItemCount)``",
  "- readyCandidateCount: ``$($record.readyCandidateCount)``",
  "- blockedCandidateCount: ``$($record.blockedCandidateCount)``",
  "- sourceGitHubActionsRunEvidenceReady: ``$($record.sourceGitHubActionsRunEvidenceReady)``",
  "- sourceHeadSha: ``$($record.sourceHeadSha)``",
  "- publicPackageUrl: ``$($record.resultSummary.publicPackageUrl)``",
  "- managedPackageUrl: ``$($record.resultSummary.managedPackageUrl)``",
  "- runtimePackageUrl: ``$($record.resultSummary.runtimePackageUrl)``",
  "- githubReleaseAssetUrl: ``$($record.resultSummary.githubReleaseAssetUrl)``",
  "- performsPublish: ``$($record.performsPublish)``",
  "- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``",
  "",
  "> $($record.boundary)"
)

Write-Host "CandidateState=$($record.candidateState)"
Write-Host "CandidateItemCount=$($record.candidateItemCount)"
