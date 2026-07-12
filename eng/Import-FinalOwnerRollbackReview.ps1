[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\final-owner-rollback-review.template.json",
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

$forbiddenReleaseActions = @("delete", "delist", "withdraw", "deprecate", "dotnet nuget delete", "nuget delete")
$nonProofSubstitutes = @("manual approval", "dashboard", "dry-run", "local feed", "ProjectReference", "direct .nupkg", "queued GitHub Actions run", "missing self-hosted runner")
$requiredRollbackFields = @(
  "reviewer",
  "reviewedAtUtc",
  "decision",
  "acceptedRisk",
  "rollbackPlan",
  "packageVersion",
  "rollbackTargetFeed",
  "rollbackPackageVersion",
  "rollbackExecutionApprovedBy",
  "rollbackDecisionTimestampUtc",
  "rollbackScope",
  "rollbackCommandPlanPath",
  "riskAssessment",
  "userImpactAssessment",
  "evidenceBundleSha256",
  "rollbackCommandPlanSha256",
  "withdrawCommandSha256",
  "delistCommandSha256",
  "deprecateCommandSha256"
)

$templatePath = Join-Path $OutputRoot "final-owner-rollback-review.template.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  [pscustomobject]@{
    recordKind = "final-owner-rollback-review-owner-input"
    reviewer = "<owner-reviewer>"
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    decision = "<owner-rollback-decision>"
    acceptedRisk = "<owner-accepted-risk>"
    rollbackPlan = "<owner-rollback-plan>"
    packageVersion = "<owner-package-version>"
    rollbackTargetFeed = "<owner-public-feed-or-package-source>"
    rollbackPackageIds = @("<owner-managed-package-id>", "<owner-runtime-package-id>")
    rollbackPackageVersion = "<owner-package-version>"
    rollbackExecutionApprovedBy = "<owner-rollback-approver>"
    rollbackDecisionTimestampUtc = "<owner-rollback-decision-timestamp-utc>"
    rollbackScope = "<owner-withdrawal-or-deprecation-scope>"
    rollbackCommandPlanPath = "<owner-rollback-command-plan-path>"
    rollbackCommandPlanSha256 = "<owner-rollback-command-plan-sha256>"
    withdrawCommandSha256 = "<owner-withdraw-command-sha256-or-no-execution-plan-hash>"
    delistCommandSha256 = "<owner-delist-command-sha256-or-no-execution-plan-hash>"
    deprecateCommandSha256 = "<owner-deprecate-command-sha256-or-no-execution-plan-hash>"
    riskAssessment = "<owner-risk-assessment>"
    userImpactAssessment = "<owner-user-impact-assessment>"
    evidenceBundleSha256 = "<owner-release-evidence-bundle-sha256>"
    externalCleanConsumerProofReady = $false
    postPublishProofReady = $false
    confirmsNoRollbackExecutionByAutomation = $false
    confirmsNoDeleteDelistWithdrawDeprecateExecution = $false
    confirmsRollbackPlanNotReleaseCloseProof = $false
  } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $templatePath -Encoding utf8
}
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]
foreach ($field in @("reviewer", "reviewedAtUtc", "decision", "acceptedRisk", "rollbackPlan", "packageVersion", "rollbackTargetFeed", "rollbackPackageVersion", "rollbackExecutionApprovedBy", "rollbackDecisionTimestampUtc", "rollbackScope", "rollbackCommandPlanPath", "riskAssessment", "userImpactAssessment", "evidenceBundleSha256")) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-OwnerPlaceholder $value) { $findings.Add((New-OwnerFinding "$field-placeholder" "action-required" "missing-field" "Required rollback review field is missing or placeholder: $field")) | Out-Null }
}
foreach ($field in @("evidenceBundleSha256", "rollbackCommandPlanSha256", "withdrawCommandSha256", "delistCommandSha256", "deprecateCommandSha256")) {
  if (-not (Test-Sha256Text (Get-PropertyOrDefault -Object $input -Name $field -DefaultValue ""))) {
    $findings.Add((New-OwnerFinding "$field-format" "action-required" "missing-sha256" "$field must be 64 hex characters.")) | Out-Null
  }
}
$rollbackPackageIds = @((Get-PropertyOrDefault -Object $input -Name "rollbackPackageIds" -DefaultValue @()) | ForEach-Object { [string]$_ })
if ($rollbackPackageIds.Count -lt 1 -or @($rollbackPackageIds | Where-Object { Test-OwnerPlaceholder $_ }).Count -gt 0) {
  $findings.Add((New-OwnerFinding "rollbackPackageIds-placeholder" "action-required" "missing-field" "Rollback package IDs must identify managed/runtime packages and cannot be placeholders.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "externalCleanConsumerProofReady" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "external-clean-consumer-proof-ready" "action-required" "missing-owner-confirmation" "Rollback review cannot bypass External CleanConsumer proof.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "postPublishProofReady" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "post-publish-proof-ready" "action-required" "missing-owner-confirmation" "Rollback review cannot bypass PostPublish proof.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "confirmsNoRollbackExecutionByAutomation" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "no-rollback-execution-by-automation" "action-required" "missing-owner-confirmation" "Owner must confirm automation did not execute rollback/withdraw/deprecate actions.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "confirmsNoDeleteDelistWithdrawDeprecateExecution" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "no-delete-delist-withdraw-deprecate-execution" "action-required" "missing-owner-confirmation" "Owner must confirm delete/delist/withdraw/deprecate were not executed by this import.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "confirmsRollbackPlanNotReleaseCloseProof" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "rollback-plan-not-release-close-proof" "action-required" "missing-owner-confirmation" "Owner must confirm rollback plan is governance input only, not release close proof.")) | Out-Null
}
$failed = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" })
$ready = $failed.Count -eq 0
$record = [pscustomobject]@{
  recordKind = "final-owner-rollback-review-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = if ($ready) { "final-owner-rollback-review-ready" } else { "blocked-final-owner-rollback-review-required" }
  ownerInputPath = $OwnerInputPath
  findingCount = $findings.Count
  failedActionRequiredCount = $failed.Count
  findings = @($findings.ToArray())
  rollbackReviewReady = $ready
  ownerActionRequired = -not $ready
  rollbackPlanRequired = $true
  rollbackOrWithdrawExecutionForbidden = $true
  deleteDelistWithdrawDeprecateForbidden = $true
  rollbackPlanIsReleaseCloseProof = $false
  rollbackPlanIsPublicPublishProof = $false
  requiredRollbackFields = @($requiredRollbackFields)
  forbiddenReleaseActions = @($forbiddenReleaseActions)
  nonProofSubstitutes = @($nonProofSubstitutes)
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner rollback review import records owner governance input only. It does not execute rollback, delete, delist, withdraw, or deprecate actions. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, not release close proof, and not package push. Manual approval, dashboard, dry-run, local feed, ProjectReference, direct .nupkg, queued GitHub Actions run, and missing self-hosted runner remain non-proof substitutes."
}
$jsonPath = Join-Path $OutputRoot "final-owner-rollback-review-import.json"
$mdPath = Join-Path $OutputRoot "final-owner-rollback-review-import.md"
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Final Owner Rollback Review Import", "", "- importState: ``$($record.importState)``", "- rollbackReviewReady: ``$ready``", "- failedActionRequiredCount: ``$($failed.Count)``", "", "## Boundary", "", $record.boundary)
Write-Host "FinalOwnerRollbackReviewImportState=$($record.importState) Ready=$ready FailedActionRequired=$($failed.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) { throw "Final Owner rollback review is not ready." }
