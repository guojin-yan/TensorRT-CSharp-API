[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\release-evidence-bundle-hash-review.owner-input.template.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($OwnerInputPath)) { $OwnerInputPath = Join-Path $RepositoryRoot $OwnerInputPath }

function Get-Sha256OrEmpty {
  param([string]$RelativePath)
  $path = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$bundleRelativePath = "artifacts\final-release\release-evidence-bundle.json"
$currentSha256 = Get-Sha256OrEmpty $bundleRelativePath
$templatePath = Join-Path $OutputRoot "release-evidence-bundle-hash-review.owner-input.template.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = [pscustomobject]@{
    recordKind = "release-evidence-bundle-hash-review-owner-input"
    releaseEvidenceBundlePath = $bundleRelativePath
    releaseEvidenceBundleSha256 = $currentSha256
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    ownerReviewer = "<owner-reviewer>"
    ownerReviewed = $false
  }
  Write-Utf8File -LiteralPath $templatePath -InputObject ($template | ConvertTo-Json -Depth 5)
}
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerSha256 = [string](Get-PropertyOrDefault -Object $input -Name "releaseEvidenceBundleSha256" -DefaultValue "")
$ownerReviewed = [bool](Get-PropertyOrDefault -Object $input -Name "ownerReviewed" -DefaultValue $false)
$findings = New-Object System.Collections.Generic.List[object]
if (-not (Test-Sha256Text $currentSha256)) { $findings.Add((New-OwnerFinding "current-bundle-sha256-missing" "action-required" "missing-current-hash" "Current release evidence bundle SHA256 is missing.")) | Out-Null }
if (-not (Test-Sha256Text $ownerSha256)) { $findings.Add((New-OwnerFinding "owner-bundle-sha256-invalid" "action-required" "invalid-owner-hash" "Owner release evidence bundle SHA256 must be 64 hex characters.")) | Out-Null }
if ((Test-Sha256Text $currentSha256) -and (Test-Sha256Text $ownerSha256) -and -not $ownerSha256.Equals($currentSha256, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-OwnerFinding "bundle-sha256-mismatch" "action-required" "hash-mismatch" "Owner bundle SHA256 does not match current release-evidence-bundle.json.")) | Out-Null
}
if (Test-OwnerPlaceholder (Get-PropertyOrDefault -Object $input -Name "reviewedAtUtc" -DefaultValue $null)) { $findings.Add((New-OwnerFinding "reviewedAtUtc-required" "action-required" "missing-review" "Owner reviewedAtUtc is required.")) | Out-Null }
if (Test-OwnerPlaceholder (Get-PropertyOrDefault -Object $input -Name "ownerReviewer" -DefaultValue $null)) { $findings.Add((New-OwnerFinding "ownerReviewer-required" "action-required" "missing-reviewer" "Owner reviewer is required.")) | Out-Null }
if (-not $ownerReviewed) { $findings.Add((New-OwnerFinding "ownerReviewed-required" "action-required" "missing-owner-review" "Owner must review the bundle hash.")) | Out-Null }

$failed = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" })
$accepted = $failed.Count -eq 0
$record = [pscustomobject]@{
  recordKind = "release-evidence-bundle-hash-review"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reviewState = if ($accepted) { "release-evidence-bundle-hash-review-owner-accepted-non-proof" } else { "blocked-release-evidence-bundle-hash-review-owner-input-required" }
  releaseEvidenceBundlePath = $bundleRelativePath
  currentReleaseEvidenceBundleSha256 = $currentSha256
  ownerReleaseEvidenceBundleSha256 = $ownerSha256
  hashMatches = (Test-Sha256Text $currentSha256) -and $ownerSha256.Equals($currentSha256, [StringComparison]::OrdinalIgnoreCase)
  ownerReviewed = $ownerReviewed
  reviewAccepted = $accepted
  failedActionRequiredCount = $failed.Count
  findingCount = $findings.Count
  findings = @($findings.ToArray())
  ownerActionRequired = -not $accepted
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Release evidence bundle hash review is Owner-reviewed hash traceability only; matching hash is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-evidence-bundle-hash-review.json") -InputObject ($record | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "release-evidence-bundle-hash-review.md") -InputObject @("# Release Evidence Bundle Hash Review", "", "- reviewState: ``$($record.reviewState)``", "- hashMatches: ``$($record.hashMatches)``", "- reviewAccepted: ``$accepted``", "- failedActionRequiredCount: ``$($failed.Count)``", "", $record.boundary)
Write-Host "ReleaseEvidenceBundleHashReviewState=$($record.reviewState) HashMatches=$($record.hashMatches) Accepted=$accepted"
if ($FailOnNotProof.IsPresent -and -not $accepted) { throw "Release evidence bundle hash review is not accepted." }
