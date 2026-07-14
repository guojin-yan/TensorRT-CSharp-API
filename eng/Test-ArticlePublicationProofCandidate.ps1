[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\article-publication-proof-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ArticlePublicationProofCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishLaneCandidateArtifact -Record $record -RecordKind "article-publication-proof-candidate" -LaneId "article-publication-urls" -ExpectedBlockedState "blocked-article-publication-owner-proof-required" -RequiredBoundaryText "not post-publish proof")
$items += (New-OwnerValidationItem "roadmap-not-proof" ([bool](Get-PropertyOrDefault -Object $record -Name "articleRoadmapIsNotProof" -DefaultValue $false)) "blocker" "Candidate must reject article roadmap as proof.")
$items += (New-OwnerValidationItem "multi-article-proof-records" ([bool](Get-PropertyOrDefault -Object $record -Name "supportsMultipleArticleProofRecords" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $record -Name "minimumArticleProofRecordCount" -DefaultValue 0) -ge 1 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "articleProofRecordsReady" -DefaultValue $true)) "blocker" "Candidate must support multiple article proof records and keep them blocked until real Owner publication evidence exists.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "article-publication-proof-candidate-ready-non-proof" } else { "invalid-article-publication-proof-candidate" }
$validation = [pscustomobject]@{
  recordKind = "article-publication-proof-candidate-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; candidateState = [string]$record.candidateState; requiredFieldCount = [int]$record.requiredFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; supportsMultipleArticleProofRecords = [bool](Get-PropertyOrDefault -Object $record -Name "supportsMultipleArticleProofRecords" -DefaultValue $false); minimumArticleProofRecordCount = [int](Get-PropertyOrDefault -Object $record -Name "minimumArticleProofRecordCount" -DefaultValue 0); articleProofRecordCount = [int](Get-PropertyOrDefault -Object $record -Name "articleProofRecordCount" -DefaultValue 0); articleProofReadyRecordCount = [int](Get-PropertyOrDefault -Object $record -Name "articleProofReadyRecordCount" -DefaultValue 0); articleProofRecordsReady = [bool](Get-PropertyOrDefault -Object $record -Name "articleProofRecordsReady" -DefaultValue $false); failedBlockerCount = $failedBlockers.Count; validationItems = @($items); ownerActionRequired = $true; performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Article publication proof candidate validation is non-proof; not post-publish proof, not runtime proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "article-publication-proof-candidate-validation.json"; $mdPath = Join-Path $OutputRoot "article-publication-proof-candidate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Article Publication Proof Candidate Validation", "", "- validationState: ``$state``", "- requiredFieldCount: ``$($validation.requiredFieldCount)``", "- blockedFieldCount: ``$($validation.blockedFieldCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ArticlePublicationProofCandidateValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Article publication proof candidate validation failed." }
