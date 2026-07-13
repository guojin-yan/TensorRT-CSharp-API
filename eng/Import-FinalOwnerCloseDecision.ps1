[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\final-owner-close-decision.template.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$FailOnNotReady
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($OwnerInputPath)) { $OwnerInputPath = Join-Path $RepositoryRoot $OwnerInputPath }
$templatePath = Join-Path $OutputRoot "final-owner-close-decision.template.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = [pscustomobject]@{
    recordKind = "final-owner-close-decision-owner-input"
    reviewer = "<owner-reviewer>"
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    decision = "<owner-final-close-decision>"
    acceptedRisk = "<owner-accepted-risk>"
    rollbackPlan = "<owner-rollback-plan>"
    packageVersion = "<owner-package-version>"
    evidenceBundleSha256 = "<owner-release-evidence-bundle-sha256>"
    externalCleanConsumerProofReady = $false
    postPublishProofReady = $false
    rollbackReviewReady = $false
  }
  Write-Utf8File -LiteralPath $templatePath -InputObject ($template | ConvertTo-Json -Depth 6)
}
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]
foreach ($field in @("reviewer", "reviewedAtUtc", "decision", "acceptedRisk", "rollbackPlan", "packageVersion", "evidenceBundleSha256")) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-OwnerPlaceholder $value) { $findings.Add((New-OwnerFinding "$field-placeholder" "action-required" "missing-field" "Required final close decision field is missing or placeholder: $field")) | Out-Null }
}
if (-not (Test-Sha256Text (Get-PropertyOrDefault -Object $input -Name "evidenceBundleSha256" -DefaultValue ""))) {
  $findings.Add((New-OwnerFinding "evidenceBundleSha256-format" "action-required" "missing-sha256" "Evidence bundle SHA256 must be 64 hex characters.")) | Out-Null
}
foreach ($flag in @("externalCleanConsumerProofReady", "postPublishProofReady", "rollbackReviewReady")) {
  if (-not [bool](Get-PropertyOrDefault -Object $input -Name $flag -DefaultValue $false)) {
    $findings.Add((New-OwnerFinding "$flag-required" "action-required" "missing-owner-confirmation" "Final close decision cannot bypass required proof/governance flag: $flag")) | Out-Null
  }
}
$failed = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" })
$ready = $failed.Count -eq 0
$record = [pscustomobject]@{
  recordKind = "final-owner-close-decision-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = if ($ready) { "final-owner-close-decision-ready" } else { "blocked-final-owner-close-decision-required" }
  ownerInputPath = $OwnerInputPath
  findingCount = $findings.Count
  failedActionRequiredCount = $failed.Count
  findings = @($findings.ToArray())
  finalCloseDecisionReady = $ready
  ownerActionRequired = -not $ready
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $ready
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $ready
  boundary = "Final Owner close decision import records owner release governance only. It is not runtime proof, not post-publish proof, not publish approval by itself, not package push, and cannot bypass External CleanConsumer or PostPublish proof."
}
$jsonPath = Join-Path $OutputRoot "final-owner-close-decision-import.json"
$mdPath = Join-Path $OutputRoot "final-owner-close-decision-import.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Final Owner Close Decision Import", "", "- importState: ``$($record.importState)``", "- finalCloseDecisionReady: ``$ready``", "- failedActionRequiredCount: ``$($failed.Count)``", "", "## Boundary", "", $record.boundary)
Write-Host "FinalOwnerCloseDecisionImportState=$($record.importState) Ready=$ready FailedActionRequired=$($failed.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) { throw "Final Owner close decision is not ready." }
