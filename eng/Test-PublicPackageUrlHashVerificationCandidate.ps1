[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-url-hash-verification-candidate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicPackageUrlHashVerificationCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishLaneCandidateArtifact -Record $record -RecordKind "public-package-url-hash-verification-candidate" -LaneId "public-package-urls-and-hashes" -ExpectedBlockedState "blocked-public-package-url-hash-owner-proof-required" -RequiredBoundaryText "not public package proof")
$items += (New-OwnerValidationItem "channel-coverage" (@(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "expectedChannels" -DefaultValue @())).Count -ge 2) "blocker" "Candidate must cover NuGet and GitHub package routes.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "public-package-url-hash-verification-candidate-ready-non-proof" } else { "invalid-public-package-url-hash-verification-candidate" }

$validation = [pscustomobject]@{
  recordKind = "public-package-url-hash-verification-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  candidateState = [string]$record.candidateState
  requiredFieldCount = [int]$record.requiredFieldCount
  blockedFieldCount = [int]$record.blockedFieldCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items)
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public package URL/hash candidate validation is non-proof; not public package proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-package-url-hash-verification-candidate-validation.json"
$mdPath = Join-Path $OutputRoot "public-package-url-hash-verification-candidate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Public Package URL Hash Verification Candidate Validation", "", "- validationState: ``$state``", "- requiredFieldCount: ``$($validation.requiredFieldCount)``", "- blockedFieldCount: ``$($validation.blockedFieldCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "PublicPackageUrlHashVerificationCandidateValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Public package URL/hash candidate validation failed." }
