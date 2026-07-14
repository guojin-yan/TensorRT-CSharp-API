[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-claim-post-publish-proof-boundary-audit.json",
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
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PublicClaimPostPublishProofBoundaryAudit.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$coverage = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "ruleCoverage" -DefaultValue @()))
$coveredRules = @($coverage | Where-Object { [bool]$_.covered })
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "public-claim-post-publish-proof-boundary-audit") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "audit-state" ([string]$record.auditState -eq "blocked-owner-proof-required-claim-boundary-audit-ready") "blocker" "Audit must remain blocked on Owner proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "scan-breadth" ([int]$record.scannedFileCount -ge 20 -and [int]$record.claimRuleCount -ge 8 -and [int]$record.claimCount -gt 0) "blocker" "Audit must scan docs/README/package metadata and find claim surfaces.")) | Out-Null
$items.Add((New-OwnerValidationItem "rule-coverage" ($coveredRules.Count -ge 6) "blocker" "Audit should cover most public claim rule families.")) | Out-Null
$items.Add((New-OwnerValidationItem "no-disallowed-proof-claims" ([int]$record.disallowedPostPublishProofClaimCount -eq 0) "blocker" "No claim may be promoted as completed post-publish proof without Owner evidence.")) | Out-Null
$items.Add((New-OwnerValidationItem "owner-review-count" ([int]$record.ownerProofReviewClaimCount -ge 0 -and [int]$record.boundarySafeClaimCount -ge 0) "blocker" "Claims must be classified into boundary-safe or Owner-review buckets.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool]$record.performsPublish) -and (-not [bool]$record.usesPublishToken) -and (-not [bool]$record.canPublishPublicly) -and (-not [bool]$record.canCloseReleaseIssue) -and (-not [bool]$record.isRuntimeExecutionProof) -and (-not [bool]$record.isPostPublishProof) -and (-not [bool]$record.isReleaseCloseProof)) "blocker" "Audit must not publish, close, or promote proof.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "public-claim-post-publish-proof-boundary-audit-ready-non-proof" } else { "invalid-public-claim-post-publish-proof-boundary-audit" }

$validation = [pscustomobject]@{
  recordKind = "public-claim-post-publish-proof-boundary-audit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  auditState = [string]$record.auditState
  scannedFileCount = [int]$record.scannedFileCount
  claimCount = [int]$record.claimCount
  boundarySafeClaimCount = [int]$record.boundarySafeClaimCount
  ownerProofReviewClaimCount = [int]$record.ownerProofReviewClaimCount
  disallowedPostPublishProofClaimCount = [int]$record.disallowedPostPublishProofClaimCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public claim post-publish proof boundary audit validation is non-proof validation only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-claim-post-publish-proof-boundary-audit-validation.json"
$mdPath = Join-Path $OutputRoot "public-claim-post-publish-proof-boundary-audit-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Claim Post-Publish Proof Boundary Audit Validation",
  "",
  "- validationState: ``$state``",
  "- scannedFileCount: ``$($validation.scannedFileCount)``",
  "- claimCount: ``$($validation.claimCount)``",
  "- ownerProofReviewClaimCount: ``$($validation.ownerProofReviewClaimCount)``",
  "- disallowedPostPublishProofClaimCount: ``0``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)

Write-Host "PublicClaimPostPublishProofBoundaryAuditValidationState=$state FailedBlockers=$($failedBlockers.Count) Claims=$($validation.claimCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Public claim post-publish proof boundary audit validation failed." }
