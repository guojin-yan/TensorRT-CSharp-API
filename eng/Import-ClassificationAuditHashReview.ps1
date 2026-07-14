[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\classification-audit-hash-review.owner-input.template.json",
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

$auditRelativePath = "artifacts\final-release\release-evidence-classification-audit.json"
$audit = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $auditRelativePath
$currentSha256 = Get-Sha256OrEmpty $auditRelativePath
$currentAuditState = [string](Get-PropertyOrDefault -Object $audit -Name "auditState" -DefaultValue "")
$templatePath = Join-Path $OutputRoot "classification-audit-hash-review.owner-input.template.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = [pscustomobject]@{
    recordKind = "classification-audit-hash-review-owner-input"
    classificationAuditPath = $auditRelativePath
    classificationAuditSha256 = $currentSha256
    classificationAuditState = $currentAuditState
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    ownerReviewer = "<owner-reviewer>"
    ownerReviewed = $false
  }
  Write-Utf8File -LiteralPath $templatePath -InputObject ($template | ConvertTo-Json -Depth 5)
}
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerSha256 = [string](Get-PropertyOrDefault -Object $input -Name "classificationAuditSha256" -DefaultValue "")
$ownerAuditState = [string](Get-PropertyOrDefault -Object $input -Name "classificationAuditState" -DefaultValue "")
$ownerReviewed = [bool](Get-PropertyOrDefault -Object $input -Name "ownerReviewed" -DefaultValue $false)
$requiredState = "classification-audit-passed-non-proof-boundaries-intact"
$findings = New-Object System.Collections.Generic.List[object]
if (-not (Test-Sha256Text $currentSha256)) { $findings.Add((New-OwnerFinding "current-classification-audit-sha256-missing" "action-required" "missing-current-hash" "Current classification audit SHA256 is missing.")) | Out-Null }
if (-not (Test-Sha256Text $ownerSha256)) { $findings.Add((New-OwnerFinding "owner-classification-audit-sha256-invalid" "action-required" "invalid-owner-hash" "Owner classification audit SHA256 must be 64 hex characters.")) | Out-Null }
if ((Test-Sha256Text $currentSha256) -and (Test-Sha256Text $ownerSha256) -and -not $ownerSha256.Equals($currentSha256, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-OwnerFinding "classification-audit-sha256-mismatch" "action-required" "hash-mismatch" "Owner classification audit SHA256 does not match current audit.")) | Out-Null
}
if (-not $currentAuditState.Equals($requiredState, [StringComparison]::Ordinal)) {
  $findings.Add((New-OwnerFinding "current-classification-audit-state-not-passed" "action-required" "audit-not-passed" "Current classification audit state must be $requiredState.")) | Out-Null
}
if (-not $ownerAuditState.Equals($requiredState, [StringComparison]::Ordinal)) {
  $findings.Add((New-OwnerFinding "owner-classification-audit-state-not-passed" "action-required" "audit-not-passed" "Owner classification audit state must be $requiredState.")) | Out-Null
}
if (Test-OwnerPlaceholder (Get-PropertyOrDefault -Object $input -Name "reviewedAtUtc" -DefaultValue $null)) { $findings.Add((New-OwnerFinding "reviewedAtUtc-required" "action-required" "missing-review" "Owner reviewedAtUtc is required.")) | Out-Null }
if (Test-OwnerPlaceholder (Get-PropertyOrDefault -Object $input -Name "ownerReviewer" -DefaultValue $null)) { $findings.Add((New-OwnerFinding "ownerReviewer-required" "action-required" "missing-reviewer" "Owner reviewer is required.")) | Out-Null }
if (-not $ownerReviewed) { $findings.Add((New-OwnerFinding "ownerReviewed-required" "action-required" "missing-owner-review" "Owner must review the classification audit hash.")) | Out-Null }

$failed = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" })
$accepted = $failed.Count -eq 0
$record = [pscustomobject]@{
  recordKind = "classification-audit-hash-review"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reviewState = if ($accepted) { "classification-audit-hash-review-owner-accepted-non-proof" } else { "blocked-classification-audit-hash-review-owner-input-required" }
  classificationAuditPath = $auditRelativePath
  currentClassificationAuditSha256 = $currentSha256
  ownerClassificationAuditSha256 = $ownerSha256
  currentClassificationAuditState = $currentAuditState
  ownerClassificationAuditState = $ownerAuditState
  hashMatches = (Test-Sha256Text $currentSha256) -and $ownerSha256.Equals($currentSha256, [StringComparison]::OrdinalIgnoreCase)
  stateMatchesRequired = $currentAuditState.Equals($requiredState, [StringComparison]::Ordinal) -and $ownerAuditState.Equals($requiredState, [StringComparison]::Ordinal)
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
  boundary = "Classification audit hash review is Owner-reviewed audit traceability only; matching hash and passed classification audit state are not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "classification-audit-hash-review.json") -InputObject ($record | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "classification-audit-hash-review.md") -InputObject @("# Classification Audit Hash Review", "", "- reviewState: ``$($record.reviewState)``", "- hashMatches: ``$($record.hashMatches)``", "- stateMatchesRequired: ``$($record.stateMatchesRequired)``", "- reviewAccepted: ``$accepted``", "- failedActionRequiredCount: ``$($failed.Count)``", "", $record.boundary)
Write-Host "ClassificationAuditHashReviewState=$($record.reviewState) HashMatches=$($record.hashMatches) StateMatchesRequired=$($record.stateMatchesRequired) Accepted=$accepted"
if ($FailOnNotProof.IsPresent -and -not $accepted) { throw "Classification audit hash review is not accepted." }
