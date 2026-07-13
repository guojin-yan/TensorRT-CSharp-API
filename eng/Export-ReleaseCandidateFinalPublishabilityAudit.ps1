[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function New-PublishabilityGate {
  param(
    [string]$Id,
    [string]$Area,
    [string]$CurrentState,
    [string]$RequiredEvidence,
    [string]$Validator,
    [bool]$LocalGatePassed = $false
  )

  [pscustomobject]@{
    id = $Id
    area = $Area
    currentState = $CurrentState
    requiredEvidence = $RequiredEvidence
    validator = $Validator
    localGatePassed = $LocalGatePassed
    readyForPublish = $false
    ownerActionRequired = $true
    boundary = "Release candidate final publishability audit gate only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$finalValidator = Read-JsonOrNull "artifacts\final-release\final-release-close-record-real-validator-validation.json"
$ownerProjection = Read-JsonOrNull "artifacts\final-release\final-owner-release-close-record-projection-validation.json"
$hashGate = Read-JsonOrNull "artifacts\final-release\final-release-close-hash-consistency-gate-validation.json"
$approvalAudit = Read-JsonOrNull "artifacts\final-release\final-close-owner-approval-boundary-audit-validation.json"
$publicPublishDraft = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$cleanConsumerDraft = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$forbiddenSubstitute = Read-JsonOrNull "artifacts\final-release\public-publish-forbidden-substitute-scan-validation.json"
$runtimeMatrix = Read-JsonOrNull "artifacts\final-release\release-runtime-proof-execution-matrix.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$packageReview = Read-JsonOrNull "artifacts/final-release/final-package-review-bundle.json"
$docsReadiness = Read-JsonOrNull "artifacts/final-release/docs-publish-readiness-bundle.json"
$sampleEvidence = Read-JsonOrNull "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
$releaseIssueClose = Read-JsonOrNull "artifacts/final-release/release-issue-close-record-validation.json"

$classificationAuditState = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")
$classificationFindingCount = [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 1)

$gates = @(
  New-PublishabilityGate -Id "release-evidence-bundle-present" -Area "evidence" -CurrentState ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "release-evidence-bundle-generated")) -RequiredEvidence "Release evidence bundle generated with all current release-facing artifacts." -Validator "eng\Export-ReleaseEvidenceBundle.ps1" -LocalGatePassed ($null -ne $releaseEvidenceBundle)
  New-PublishabilityGate -Id "classification-audit-clean" -Area "evidence" -CurrentState $classificationAuditState -RequiredEvidence "Classification audit remains clean while all non-proof boundaries remain intact." -Validator "eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" -LocalGatePassed ($classificationAuditState -eq "classification-audit-passed-non-proof-boundaries-intact" -and $classificationFindingCount -eq 0)
  New-PublishabilityGate -Id "public-publish-real-result" -Area "publish" -CurrentState ([string](Get-PropertyOrDefault -Object $publicPublishDraft -Name "validationState" -DefaultValue "missing-public-publish-real-result-record-draft-validation")) -RequiredEvidence "Owner-filled real public package source, version, URL, hash, timestamp, and transcript." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-PublishabilityGate -Id "public-package-hash" -Area "publish" -CurrentState ([string](Get-PropertyOrDefault -Object $finalValidator -Name "validationState" -DefaultValue "missing-final-release-close-record-real-validator-validation")) -RequiredEvidence "Downloaded public package SHA256 recorded in final close record real fields." -Validator "eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict"
  New-PublishabilityGate -Id "clean-consumer-smoke" -Area "consumer" -CurrentState ([string](Get-PropertyOrDefault -Object $cleanConsumerDraft -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-draft-validation")) -RequiredEvidence "Repository-external clean consumer restore/build/smoke proof from the public package." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-PublishabilityGate -Id "linux-runner-proof" -Area "runtime" -CurrentState ([string](Get-PropertyOrDefault -Object $runtimeMatrix -Name "linuxRunnerProofState" -DefaultValue "blocked-linux-runner-proof-required")) -RequiredEvidence "Linux runner proof on compatible CUDA/TensorRT host." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1"
  New-PublishabilityGate -Id "real-model-runtime-proof" -Area "runtime" -CurrentState ([string](Get-PropertyOrDefault -Object $runtimeMatrix -Name "realModelRuntimeProofState" -DefaultValue "blocked-real-model-runtime-proof-required")) -RequiredEvidence "Real model runtime proof with assets, logs, hashes, and validator output." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1"
  New-PublishabilityGate -Id "post-publish-verification" -Area "post-publish" -CurrentState ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "blocked-post-publish-verification-required")) -RequiredEvidence "Post-publish verification record from public package channel." -Validator "eng\Test-PostPublishVerification.ps1"
  New-PublishabilityGate -Id "forbidden-substitute-cleared" -Area "non-substitute" -CurrentState ([string](Get-PropertyOrDefault -Object $forbiddenSubstitute -Name "validationState" -DefaultValue "missing-public-publish-forbidden-substitute-scan-validation")) -RequiredEvidence "No local feed, ProjectReference, direct nupkg, template, dry-run, runbook, dashboard, audit pack, candidate, or local-only scan used as proof." -Validator "eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-PublishabilityGate -Id "package-review-ready" -Area "package" -CurrentState ([string](Get-PropertyOrDefault -Object $packageReview -Name "bundleState" -DefaultValue "package-review-guidance-only")) -RequiredEvidence "Package metadata, files, symbols, license, native assets, and dependency review complete." -Validator "eng\Test-FinalPackageReviewBundle.ps1"
  New-PublishabilityGate -Id "docs-readiness-ready" -Area "docs" -CurrentState ([string](Get-PropertyOrDefault -Object $docsReadiness -Name "bundleState" -DefaultValue "docs-readiness-guidance-only")) -RequiredEvidence "Docs, README, samples, API articles, and publishing matrix reviewed." -Validator "eng\Test-DocsPublishReadinessBundle.ps1"
  New-PublishabilityGate -Id "sample-evidence-real" -Area "samples" -CurrentState ([string](Get-PropertyOrDefault -Object $sampleEvidence -Name "validationState" -DefaultValue "blocked-real-sample-run-evidence-required")) -RequiredEvidence "Real sample run evidence for package consumer and representative model paths." -Validator "eng\Test-SampleRunEvidenceRecord.ps1"
  New-PublishabilityGate -Id "final-close-record-fields" -Area "release-close" -CurrentState ([string](Get-PropertyOrDefault -Object $finalValidator -Name "validationState" -DefaultValue "missing-final-release-close-record-real-validator-validation")) -RequiredEvidence "All 13 final close record real fields filled and validated." -Validator "eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict"
  New-PublishabilityGate -Id "final-owner-projection" -Area "release-close" -CurrentState ([string](Get-PropertyOrDefault -Object $ownerProjection -Name "validationState" -DefaultValue "missing-final-owner-release-close-record-projection-validation")) -RequiredEvidence "All final owner release close projection lanes ready." -Validator "eng\Test-FinalOwnerReleaseCloseRecordProjection.ps1 -Strict"
  New-PublishabilityGate -Id "final-hash-consistency" -Area "release-close" -CurrentState ([string](Get-PropertyOrDefault -Object $hashGate -Name "validationState" -DefaultValue "missing-final-release-close-hash-consistency-gate-validation")) -RequiredEvidence "Final close source artifact hashes are consistent and still tied to real proof records." -Validator "eng\Test-FinalReleaseCloseHashConsistencyGate.ps1 -Strict" -LocalGatePassed ([int](Get-PropertyOrDefault -Object $hashGate -Name "mismatchedHashCount" -DefaultValue 1) -eq 0)
  New-PublishabilityGate -Id "final-owner-approval" -Area "release-close" -CurrentState ([string](Get-PropertyOrDefault -Object $approvalAudit -Name "validationState" -DefaultValue "missing-final-close-owner-approval-boundary-audit-validation")) -RequiredEvidence "Owner approval, rollback review, strict close validation, and final close decision are real and complete." -Validator "eng\Test-FinalCloseOwnerApprovalBoundaryAudit.ps1 -Strict"
  New-PublishabilityGate -Id "release-issue-close-record" -Area "release-close" -CurrentState ([string](Get-PropertyOrDefault -Object $releaseIssueClose -Name "validationState" -DefaultValue "blocked-release-issue-close-record-required")) -RequiredEvidence "Strict release issue close record passes only after real proof and owner approval." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$blockedGates = @($gates | Where-Object { -not [bool]$_.readyForPublish })
$localPassedGates = @($gates | Where-Object { [bool]$_.localGatePassed })

$record = [pscustomobject]@{
  recordKind = "release-candidate-final-publishability-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-release-candidate-final-publishability-owner-proof-required"
  publishabilityGateCount = $gates.Count
  localGatePassedCount = $localPassedGates.Count
  blockedPublishabilityGateCount = $blockedGates.Count
  publishabilityGates = @($gates)
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/final-release-close-record-real-validator-validation.json",
    "artifacts/final-release/final-owner-release-close-record-projection-validation.json",
    "artifacts/final-release/final-release-close-hash-consistency-gate-validation.json",
    "artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json",
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
    "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Release candidate final publishability audit is a blocked readiness audit only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-final-publishability-audit.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-final-publishability-audit.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $gates | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.area)`` | ``$($_.currentState)`` | ``$($_.localGatePassed)`` | ``$($_.readyForPublish)`` | $($_.requiredEvidence.Replace("|", "\|")) | ``$($_.validator)`` |"
}

$markdown = @"
# Release Candidate Final Publishability Audit

| Field | Value |
| --- | --- |
| auditState | ``$($record.auditState)`` |
| publishabilityGateCount | ``$($record.publishabilityGateCount)`` |
| localGatePassedCount | ``$($record.localGatePassedCount)`` |
| blockedPublishabilityGateCount | ``$($record.blockedPublishabilityGateCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Publishability Gates

| ID | Area | Current State | Local Gate Passed | Ready For Publish | Required Evidence | Validator |
| --- | --- | --- | ---: | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate final publishability audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Gates=$($record.publishabilityGateCount) LocalPassed=$($record.localGatePassedCount) Blocked=$($record.blockedPublishabilityGateCount)"
