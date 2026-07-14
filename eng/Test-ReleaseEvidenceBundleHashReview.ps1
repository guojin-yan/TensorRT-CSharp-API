[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-evidence-bundle-hash-review.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Import-ReleaseEvidenceBundleHashReview.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-evidence-bundle-hash-review") "blocker" "recordKind must match."
  New-OwnerValidationItem "sha-field" ([string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "currentReleaseEvidenceBundleSha256" -DefaultValue "")) -or (Test-Sha256Text (Get-PropertyOrDefault -Object $record -Name "currentReleaseEvidenceBundleSha256" -DefaultValue ""))) "blocker" "Current hash must be empty or SHA256."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Review must not publish, use tokens, close, or claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("matching hash", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject hash-only proof."
)
if (-not [bool](Get-PropertyOrDefault -Object $record -Name "reviewAccepted" -DefaultValue $false)) {
  $items += New-OwnerValidationItem "blocked-has-findings" ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0) "blocker" "Blocked hash review must include findings."
}
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "release-evidence-bundle-hash-review-validation-ready-non-proof" } else { "invalid-release-evidence-bundle-hash-review" }
$validation = [pscustomobject]@{
  recordKind = "release-evidence-bundle-hash-review-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  hashMatches = [bool](Get-PropertyOrDefault -Object $record -Name "hashMatches" -DefaultValue $false)
  reviewAccepted = [bool](Get-PropertyOrDefault -Object $record -Name "reviewAccepted" -DefaultValue $false)
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Release evidence bundle hash review validation is traceability only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-evidence-bundle-hash-review-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-evidence-bundle-hash-review-validation.md") -InputObject @("# Release Evidence Bundle Hash Review Validation", "", "- validationState: ``$state``", "- hashMatches: ``$($validation.hashMatches)``", "- failedBlockerCount: ``$($failedBlockers.Count)``", "", $validation.boundary)
Write-Host "ReleaseEvidenceBundleHashReviewValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release evidence bundle hash review validation failed." }
