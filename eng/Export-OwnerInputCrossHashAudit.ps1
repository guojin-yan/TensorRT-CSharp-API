[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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
    return "<missing-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-AuditLine {
  param(
    [string]$Id,
    [string]$Path,
    [string]$ExpectedSha256,
    [string]$ActualSha256,
    [string]$State,
    [string]$Boundary
  )

  $matches = $ExpectedSha256 -match "^[0-9a-fA-F]{64}$" -and
    $ActualSha256 -match "^[0-9a-fA-F]{64}$" -and
    $ExpectedSha256.Equals($ActualSha256, [StringComparison]::OrdinalIgnoreCase)

  [pscustomobject]@{
    id = $Id
    path = $Path
    expectedSha256 = $ExpectedSha256
    actualSha256 = $ActualSha256
    sha256Matches = $matches
    state = $State
    proofPromotable = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalEvidenceFreeze = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseCloseCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$finalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$overlayPackValidation = Read-JsonOrNull "artifacts\final-release\real-external-proof-overlay-pack-validation.json"
$overlayCandidate = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-overlay-candidate.json"
$overlayCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-overlay-candidate-validation.json"
$backfillKitValidation = Read-JsonOrNull "artifacts\final-release\owner-external-execution-result-backfill-kit-validation.json"

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$finalEvidenceFreezePath = "artifacts/final-release/final-evidence-freeze.json"
$postPublishValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
$releaseCloseCandidateValidationPath = "artifacts/final-release/release-issue-close-record-candidate-validation.json"
$finalCloseDecisionValidationPath = "artifacts/final-release/release-issue-final-close-decision-validation.json"
$overlayPackValidationPath = "artifacts/final-release/real-external-proof-overlay-pack-validation.json"
$overlayCandidateValidationPath = "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json"
$backfillKitValidationPath = "artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json"

$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$finalEvidenceFreezeState = [string](Get-PropertyOrDefault -Object $finalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-final-evidence-freeze")
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$releaseCloseCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-candidate-validation")
$finalCloseDecisionValidationState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$overlayPackValidationState = [string](Get-PropertyOrDefault -Object $overlayPackValidation -Name "validationState" -DefaultValue "missing-real-external-proof-overlay-pack-validation")
$overlayCandidateValidationState = [string](Get-PropertyOrDefault -Object $overlayCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-overlay-candidate-validation")
$backfillKitValidationState = [string](Get-PropertyOrDefault -Object $backfillKitValidation -Name "validationState" -DefaultValue "missing-owner-external-execution-result-backfill-kit-validation")

$auditLines = @(
  (New-AuditLine -Id "release-evidence-bundle" -Path $releaseEvidenceBundlePath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath) -State $releaseEvidenceBundleState -Boundary "Bundle hash consistency is audit evidence only, not proof.")
  (New-AuditLine -Id "final-evidence-freeze" -Path $finalEvidenceFreezePath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalEvidenceFreezePath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalEvidenceFreezePath) -State $finalEvidenceFreezeState -Boundary "Final freeze hash consistency is audit evidence only, not proof.")
  (New-AuditLine -Id "post-publish-validation" -Path $postPublishValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "postPublishVerificationValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath) -State $postPublishValidationState -Boundary "Post-publish hash match cannot replace real public-channel proof.")
  (New-AuditLine -Id "release-close-candidate-validation" -Path $releaseCloseCandidateValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "releaseIssueCloseRecordCandidateValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseCloseCandidateValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseCloseCandidateValidationPath) -State $releaseCloseCandidateValidationState -Boundary "Release close candidate hash match is not release-close proof.")
  (New-AuditLine -Id "final-close-decision-validation" -Path $finalCloseDecisionValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "releaseIssueFinalCloseDecisionValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalCloseDecisionValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalCloseDecisionValidationPath) -State $finalCloseDecisionValidationState -Boundary "Final close decision validation remains owner input validation only.")
  (New-AuditLine -Id "real-external-proof-overlay-pack-validation" -Path $overlayPackValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "realExternalProofOverlayPackValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayPackValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayPackValidationPath) -State $overlayPackValidationState -Boundary "Overlay pack validation is owner guidance audit only.")
  (New-AuditLine -Id "release-issue-close-record-overlay-candidate-validation" -Path $overlayCandidateValidationPath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayCandidateValidationPath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayCandidateValidationPath) -State $overlayCandidateValidationState -Boundary "Close overlay validation is candidate input mapping only.")
  (New-AuditLine -Id "owner-external-execution-result-backfill-kit-validation" -Path $backfillKitValidationPath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $backfillKitValidationPath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $backfillKitValidationPath) -State $backfillKitValidationState -Boundary "Owner backfill kit validation is guidance only.")
)

$mismatchedLines = @($auditLines | Where-Object { -not $_.sha256Matches })
$blockedStateLines = @($auditLines | Where-Object { $_.state -like "blocked*" -or $_.state -like "missing*" -or $_.state -like "incomplete*" })

$sourceArtifacts = @(
  $releaseEvidenceBundlePath,
  $finalEvidenceFreezePath,
  $postPublishValidationPath,
  $releaseCloseCandidateValidationPath,
  $finalCloseDecisionValidationPath,
  $overlayPackValidationPath,
  $overlayCandidateValidationPath,
  $backfillKitValidationPath
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-input-cross-hash-audit"
  auditState = "blocked-owner-input-cross-hash-audit-owner-proof-required"
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  finalEvidenceFreezeState = $finalEvidenceFreezeState
  postPublishVerificationValidationState = $postPublishValidationState
  releaseIssueCloseRecordCandidateValidationState = $releaseCloseCandidateValidationState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionValidationState
  realExternalProofOverlayPackValidationState = $overlayPackValidationState
  releaseIssueCloseRecordOverlayCandidateValidationState = $overlayCandidateValidationState
  ownerExternalExecutionResultBackfillKitValidationState = $backfillKitValidationState
  auditLineCount = $auditLines.Count
  mismatchedHashCount = $mismatchedLines.Count
  blockedStateLineCount = $blockedStateLines.Count
  auditLines = $auditLines
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Cross-hash audit checks local artifact/path/hash consistency only. Hash matches cannot promote proof, publish packages, or close release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-input-cross-hash-audit.json"
$markdownPath = Join-Path $artifactRoot "owner-input-cross-hash-audit.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$rows = $auditLines | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.state)`` | ``$($_.sha256Matches)`` | $($_.boundary.Replace("|", "\|")) |"
}
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Input Cross-Hash Audit

生成时间：$($record.generatedAtUtc)

该 audit 只核对本地 artifact/path/hash 映射一致性。即使 hash 全部匹配，也不能替代真实外部执行、post-publish verification、rollback approval、final owner decision 或 strict close validation。

| 项目 | 当前值 |
|---|---|
| auditState | ``$($record.auditState)`` |
| auditLineCount | ``$($record.auditLineCount)`` |
| mismatchedHashCount | ``$($record.mismatchedHashCount)`` |
| blockedStateLineCount | ``$($record.blockedStateLineCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Audit Lines

| ID | State | SHA256 Matches | Boundary |
|---|---|---:|---|
$($rows -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner input cross-hash audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) MismatchedHashCount=$($record.mismatchedHashCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
