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
    return "<missing-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-StrictHashLine {
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
$crossHashAuditValidation = Read-JsonOrNull "artifacts\final-release\owner-input-cross-hash-audit-validation.json"

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$finalEvidenceFreezePath = "artifacts/final-release/final-evidence-freeze.json"
$postPublishValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
$releaseCloseCandidateValidationPath = "artifacts/final-release/release-issue-close-record-candidate-validation.json"
$finalCloseDecisionValidationPath = "artifacts/final-release/release-issue-final-close-decision-validation.json"
$overlayPackValidationPath = "artifacts/final-release/real-external-proof-overlay-pack-validation.json"
$overlayCandidateValidationPath = "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json"
$backfillKitValidationPath = "artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json"
$crossHashAuditValidationPath = "artifacts/final-release/owner-input-cross-hash-audit-validation.json"

$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$finalEvidenceFreezeState = [string](Get-PropertyOrDefault -Object $finalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-final-evidence-freeze")
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$releaseCloseCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-candidate-validation")
$finalCloseDecisionValidationState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$overlayPackValidationState = [string](Get-PropertyOrDefault -Object $overlayPackValidation -Name "validationState" -DefaultValue "missing-real-external-proof-overlay-pack-validation")
$overlayCandidateValidationState = [string](Get-PropertyOrDefault -Object $overlayCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-overlay-candidate-validation")
$backfillKitValidationState = [string](Get-PropertyOrDefault -Object $backfillKitValidation -Name "validationState" -DefaultValue "missing-owner-external-execution-result-backfill-kit-validation")
$crossHashAuditValidationState = [string](Get-PropertyOrDefault -Object $crossHashAuditValidation -Name "validationState" -DefaultValue "missing-owner-input-cross-hash-audit-validation")
$crossHashAuditMismatchedHashCount = [int](Get-PropertyOrDefault -Object $crossHashAuditValidation -Name "mismatchedHashCount" -DefaultValue -1)

$hashLines = @(
  (New-StrictHashLine -Id "release-evidence-bundle" -Path $releaseEvidenceBundlePath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath) -State $releaseEvidenceBundleState -Boundary "Bundle hash consistency is candidate context only, not close proof.")
  (New-StrictHashLine -Id "final-evidence-freeze" -Path $finalEvidenceFreezePath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalEvidenceFreezePath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalEvidenceFreezePath) -State $finalEvidenceFreezeState -Boundary "Final freeze hash consistency is audit context only, not close proof.")
  (New-StrictHashLine -Id "post-publish-validation" -Path $postPublishValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "postPublishVerificationValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath) -State $postPublishValidationState -Boundary "Post-publish validation must be real public-channel proof before close.")
  (New-StrictHashLine -Id "release-close-candidate-validation" -Path $releaseCloseCandidateValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "releaseIssueCloseRecordCandidateValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseCloseCandidateValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseCloseCandidateValidationPath) -State $releaseCloseCandidateValidationState -Boundary "Release close candidate validation is still candidate evidence.")
  (New-StrictHashLine -Id "final-close-decision-validation" -Path $finalCloseDecisionValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "releaseIssueFinalCloseDecisionValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalCloseDecisionValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $finalCloseDecisionValidationPath) -State $finalCloseDecisionValidationState -Boundary "Final close decision must be real owner approval before close.")
  (New-StrictHashLine -Id "real-external-proof-overlay-pack-validation" -Path $overlayPackValidationPath -ExpectedSha256 ([string](Get-PropertyOrDefault -Object $overlayCandidate -Name "realExternalProofOverlayPackValidationSha256" -DefaultValue (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayPackValidationPath))) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayPackValidationPath) -State $overlayPackValidationState -Boundary "Overlay pack validation is owner input guidance only.")
  (New-StrictHashLine -Id "release-issue-close-record-overlay-candidate-validation" -Path $overlayCandidateValidationPath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayCandidateValidationPath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $overlayCandidateValidationPath) -State $overlayCandidateValidationState -Boundary "Close overlay validation is candidate input mapping only.")
  (New-StrictHashLine -Id "owner-external-execution-result-backfill-kit-validation" -Path $backfillKitValidationPath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $backfillKitValidationPath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $backfillKitValidationPath) -State $backfillKitValidationState -Boundary "Owner external execution result kit is guidance only.")
  (New-StrictHashLine -Id "owner-input-cross-hash-audit-validation" -Path $crossHashAuditValidationPath -ExpectedSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $crossHashAuditValidationPath) -ActualSha256 (Get-RelativeFileSha256OrPlaceholder -RelativePath $crossHashAuditValidationPath) -State $crossHashAuditValidationState -Boundary "Cross-hash audit validates local consistency only, not proof.")
)

$mismatchedHashLines = @($hashLines | Where-Object { -not $_.sha256Matches })
$blockedStateLines = @($hashLines | Where-Object { $_.state -like "blocked*" -or $_.state -like "missing*" -or $_.state -like "incomplete*" -or $_.state -like "invalid*" })

$requiredOwnerFields = @(
  [pscustomobject]@{ id = "rollback-plan"; value = "<owner-fill-rollback-plan>"; requiredForClose = $true }
  [pscustomobject]@{ id = "rollback-owner"; value = "<owner-fill-rollback-owner>"; requiredForClose = $true }
  [pscustomobject]@{ id = "rollback-trigger"; value = "<owner-fill-rollback-trigger>"; requiredForClose = $true }
  [pscustomobject]@{ id = "owner-final-close-decision"; value = "<owner-fill-approved-to-close-after-real-proof>"; requiredForClose = $true }
  [pscustomobject]@{ id = "release-issue-id"; value = "<owner-fill-release-issue-id>"; requiredForClose = $true }
  [pscustomobject]@{ id = "release-issue-url"; value = "<owner-fill-release-issue-url>"; requiredForClose = $true }
  [pscustomobject]@{ id = "public-channel-package-source"; value = "<owner-fill-public-channel-source-uri>"; requiredForClose = $true }
  [pscustomobject]@{ id = "clean-consumer-runtime-smoke-log"; value = "<owner-fill-clean-consumer-runtime-smoke-log>"; requiredForClose = $true }
)

$placeholderFields = @($requiredOwnerFields | Where-Object { [string]$_.value -like "<*>" })
$missingRealProofCount = @(
  $postPublishValidationState -ne "post-publish-verification-ready"
  $finalCloseDecisionValidationState -ne "owner-final-close-decision-ready"
  $releaseCloseCandidateValidationState -ne "release-close-candidate-ready"
  $overlayCandidateValidationState -ne "release-close-overlay-ready-for-strict-close-record"
  $backfillKitValidationState -ne "owner-external-execution-results-ready"
).Where({ $_ }).Count

$releaseCloseBlockers = @(
  [pscustomobject]@{ id = "post-publish-verification"; state = $postPublishValidationState; blocker = ($postPublishValidationState -ne "post-publish-verification-ready"); requiredAction = "Provide validator-passing real public-channel post-publish proof." }
  [pscustomobject]@{ id = "final-close-decision"; state = $finalCloseDecisionValidationState; blocker = ($finalCloseDecisionValidationState -ne "owner-final-close-decision-ready"); requiredAction = "Replace owner placeholders with real final close approval and rollback review." }
  [pscustomobject]@{ id = "release-close-candidate"; state = $releaseCloseCandidateValidationState; blocker = ($releaseCloseCandidateValidationState -ne "release-close-candidate-ready"); requiredAction = "Produce validator-passing real release close record candidate." }
  [pscustomobject]@{ id = "overlay-candidate"; state = $overlayCandidateValidationState; blocker = ($overlayCandidateValidationState -ne "release-close-overlay-ready-for-strict-close-record"); requiredAction = "Replace overlay placeholders with real owner close input." }
  [pscustomobject]@{ id = "owner-external-execution-result-backfill-kit"; state = $backfillKitValidationState; blocker = ($backfillKitValidationState -ne "owner-external-execution-results-ready"); requiredAction = "Backfill real owner external execution results, logs, and hashes." }
  [pscustomobject]@{ id = "owner-input-cross-hash-audit"; state = $crossHashAuditValidationState; blocker = ($crossHashAuditValidationState -ne "blocked-owner-input-cross-hash-audit-owner-proof-required" -or $crossHashAuditMismatchedHashCount -ne 0); requiredAction = "Keep local hashes consistent; remember this is still not proof." }
)

$sourceArtifacts = @(
  $releaseEvidenceBundlePath,
  $finalEvidenceFreezePath,
  $postPublishValidationPath,
  $releaseCloseCandidateValidationPath,
  $finalCloseDecisionValidationPath,
  $overlayPackValidationPath,
  $overlayCandidateValidationPath,
  $backfillKitValidationPath,
  $crossHashAuditValidationPath
)

$candidate = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-close-strict-record-candidate"
  candidateState = "blocked-release-close-strict-record-owner-input-required"
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  finalEvidenceFreezeState = $finalEvidenceFreezeState
  postPublishVerificationValidationState = $postPublishValidationState
  releaseIssueCloseRecordCandidateValidationState = $releaseCloseCandidateValidationState
  releaseIssueFinalCloseDecisionValidationState = $finalCloseDecisionValidationState
  realExternalProofOverlayPackValidationState = $overlayPackValidationState
  releaseIssueCloseRecordOverlayCandidateValidationState = $overlayCandidateValidationState
  ownerExternalExecutionResultBackfillKitValidationState = $backfillKitValidationState
  ownerInputCrossHashAuditValidationState = $crossHashAuditValidationState
  ownerInputCrossHashAuditMismatchedHashCount = $crossHashAuditMismatchedHashCount
  releaseEvidenceBundleSha256 = ($hashLines | Where-Object id -eq "release-evidence-bundle").actualSha256
  finalEvidenceFreezeSha256 = ($hashLines | Where-Object id -eq "final-evidence-freeze").actualSha256
  postPublishVerificationValidationSha256 = ($hashLines | Where-Object id -eq "post-publish-validation").actualSha256
  releaseIssueCloseRecordCandidateValidationSha256 = ($hashLines | Where-Object id -eq "release-close-candidate-validation").actualSha256
  releaseIssueFinalCloseDecisionValidationSha256 = ($hashLines | Where-Object id -eq "final-close-decision-validation").actualSha256
  realExternalProofOverlayPackValidationSha256 = ($hashLines | Where-Object id -eq "real-external-proof-overlay-pack-validation").actualSha256
  releaseIssueCloseRecordOverlayCandidateValidationSha256 = ($hashLines | Where-Object id -eq "release-issue-close-record-overlay-candidate-validation").actualSha256
  ownerExternalExecutionResultBackfillKitValidationSha256 = ($hashLines | Where-Object id -eq "owner-external-execution-result-backfill-kit-validation").actualSha256
  ownerInputCrossHashAuditValidationSha256 = ($hashLines | Where-Object id -eq "owner-input-cross-hash-audit-validation").actualSha256
  requiredOwnerInputCount = $requiredOwnerFields.Count
  missingOwnerInputCount = $placeholderFields.Count
  missingRealProofCount = $missingRealProofCount
  placeholderFieldCount = $placeholderFields.Count
  mismatchedHashCount = $mismatchedHashLines.Count
  blockedStateLineCount = $blockedStateLines.Count
  strictValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  hashLines = $hashLines
  releaseCloseBlockers = $releaseCloseBlockers
  requiredOwnerFields = $requiredOwnerFields
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Strict close record candidate is still a candidate. Hash consistency, blocked-shape validity, and owner guidance cannot substitute real owner approval, post-publish proof, clean consumer runtime proof, rollback approval, or Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-close-strict-record-candidate.json"
$markdownPath = Join-Path $artifactRoot "release-close-strict-record-candidate.md"
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$hashRows = $hashLines | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.state)`` | ``$($_.sha256Matches)`` | ``$($_.path)`` | $($_.boundary.Replace("|", "\|")) |"
}
$blockerRows = $releaseCloseBlockers | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.state)`` | ``$($_.blocker)`` | $($_.requiredAction.Replace("|", "\|")) |"
}
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Close Strict Record Candidate

生成时间：$($candidate.generatedAtUtc)

该 candidate 将 release close 所需的 evidence bundle、final evidence freeze、post-publish validation、close candidate validation、final close decision、overlay validation、owner backfill kit 和 cross-hash audit 汇总到更严格的最终候选输入面。

它仍然不是最终 close record，不会发布包，不会批准公开发布，也不会关闭 release issue。

| 项目 | 当前值 |
|---|---|
| candidateState | ``$($candidate.candidateState)`` |
| missingOwnerInputCount | ``$($candidate.missingOwnerInputCount)`` |
| missingRealProofCount | ``$($candidate.missingRealProofCount)`` |
| mismatchedHashCount | ``$($candidate.mismatchedHashCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Hash Lines

| ID | State | SHA256 Matches | Path | Boundary |
|---|---|---:|---|---|
$($hashRows -join "`r`n")

## Release Close Blockers

| ID | State | Blocking | Required Action |
|---|---|---:|---|
$($blockerRows -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($candidate.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict record candidate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CandidateState=$($candidate.candidateState) MismatchedHashCount=$($candidate.mismatchedHashCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
