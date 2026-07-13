[CmdletBinding()]
param(
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
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return "<owner-fill-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalEvidenceFreeze = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze.json"
$finalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseCloseCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$overlayPackValidation = Read-JsonOrNull "artifacts\final-release\real-external-proof-overlay-pack-validation.json"

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$finalEvidenceFreezePath = "artifacts/final-release/final-evidence-freeze.json"
$postPublishValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
$releaseCloseCandidateValidationPath = "artifacts/final-release/release-issue-close-record-candidate-validation.json"
$finalCloseDecisionValidationPath = "artifacts/final-release/release-issue-final-close-decision-validation.json"
$overlayPackValidationPath = "artifacts/final-release/real-external-proof-overlay-pack-validation.json"

$candidateState = "blocked-release-close-real-proof-required"
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$finalCloseDecisionState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$finalFreezeState = [string](Get-PropertyOrDefault -Object $finalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-final-evidence-freeze")
$releaseCloseCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-close-candidate-validation")
$overlayValidationState = [string](Get-PropertyOrDefault -Object $overlayPackValidation -Name "validationState" -DefaultValue "missing-real-external-proof-overlay-pack-validation")

$candidate = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-issue-close-record-overlay-candidate"
  candidateState = $candidateState
  proofLineId = "release-issue-close-record"
  releaseEvidenceBundlePath = $releaseEvidenceBundlePath
  releaseEvidenceBundleSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
  finalEvidenceFreezePath = $finalEvidenceFreezePath
  finalEvidenceFreezeSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $finalEvidenceFreezePath
  postPublishVerificationValidationPath = $postPublishValidationPath
  postPublishVerificationValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath
  releaseIssueCloseRecordCandidateValidationPath = $releaseCloseCandidateValidationPath
  releaseIssueCloseRecordCandidateValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseCloseCandidateValidationPath
  releaseIssueFinalCloseDecisionValidationPath = $finalCloseDecisionValidationPath
  releaseIssueFinalCloseDecisionValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $finalCloseDecisionValidationPath
  realExternalProofOverlayPackValidationPath = $overlayPackValidationPath
  realExternalProofOverlayPackValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayPackValidationPath
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  finalEvidenceFreezeState = $finalFreezeState
  postPublishVerificationValidationState = $postPublishState
  releaseIssueCloseRecordCandidateValidationState = $releaseCloseCandidateValidationState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionState
  realExternalProofOverlayPackValidationState = $overlayValidationState
  rollbackPlan = "<owner-fill-rollback-plan>"
  rollbackOwner = "<owner-fill-rollback-owner>"
  rollbackTrigger = "<owner-fill-rollback-trigger>"
  ownerFinalCloseDecision = "<owner-fill-approved-to-close-after-real-proof>"
  releaseIssueId = "<owner-fill-release-issue-id>"
  releaseIssueUrl = "<owner-fill-release-issue-url>"
  strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  requiredRealInputs = @(
    "real post-publish proof validation",
    "owner final close decision",
    "rollback plan/owner/trigger",
    "release issue id/url",
    "strict close validator pass"
  )
  sourceArtifacts = @(
    $releaseEvidenceBundlePath,
    $finalEvidenceFreezePath,
    $postPublishValidationPath,
    $releaseCloseCandidateValidationPath,
    $finalCloseDecisionValidationPath,
    $overlayPackValidationPath
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Overlay candidate maps close record inputs only. It cannot close release issue and must still pass Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady after real owner input."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-issue-close-record-overlay-candidate.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-record-overlay-candidate.md"
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$sourceLines = $candidate.sourceArtifacts | ForEach-Object { "- ``$_``" }
$inputLines = $candidate.requiredRealInputs | ForEach-Object { "- $_" }

$markdown = @"
# Release Issue Close Record Overlay Candidate

生成时间：$($candidate.generatedAtUtc)

该 candidate 将 final freeze、post-publish validation、final close decision validation 和 release evidence bundle 映射到 release close record 的候选输入面。它不关闭 release issue，也不替代 strict close validator。

| 项目 | 当前值 |
|---|---|
| candidateState | ``$candidateState`` |
| releaseEvidenceBundleState | ``$($candidate.releaseEvidenceBundleState)`` |
| finalEvidenceFreezeState | ``$finalFreezeState`` |
| postPublishVerificationValidationState | ``$postPublishState`` |
| releaseIssueFinalCloseDecisionValidationState | ``$finalCloseDecisionState`` |
| realExternalProofOverlayPackValidationState | ``$overlayValidationState`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Required Real Inputs

$($inputLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($candidate.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close record overlay candidate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CandidateState=$candidateState PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
