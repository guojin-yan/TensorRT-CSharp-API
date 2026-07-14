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
$requiredFields = @(
  "reviewer", "reviewedAtUtc", "ownerReviewed", "decision", "closeReason", "acceptedRisk", "rollbackPlan",
  "packageId", "packageVersion", "releaseIssueId", "releaseIssueUrl", "evidenceBundleSha256",
  "classificationAuditSha256", "publicPackageProofSha256", "externalCleanConsumerProofSha256",
  "postPublishProofSha256", "postPublishProofUrls", "externalCleanConsumerProofReady", "postPublishProofReady",
  "rollbackReviewReady", "classificationAuditPassed", "manualCloseOnlyConfirmation", "confirmsNoIssueCloseByAutomation"
)
$template = [pscustomobject]@{
  recordKind = "final-owner-close-decision-owner-input"
  reviewer = "<owner-reviewer>"
  reviewedAtUtc = "<owner-reviewed-at-utc>"
  ownerReviewed = $false
  decision = "<owner-final-close-decision>"
  closeReason = "<owner-close-reason>"
  acceptedRisk = "<owner-accepted-risk>"
  rollbackPlan = "<owner-rollback-plan>"
  packageId = "<owner-package-id>"
  packageVersion = "<owner-package-version>"
  releaseIssueId = "<owner-release-issue-id>"
  releaseIssueUrl = "<owner-release-issue-url>"
  evidenceBundleSha256 = "<owner-release-evidence-bundle-sha256>"
  classificationAuditSha256 = "<owner-classification-audit-sha256>"
  publicPackageProofSha256 = "<owner-public-package-proof-sha256>"
  externalCleanConsumerProofSha256 = "<owner-external-clean-consumer-proof-sha256>"
  postPublishProofSha256 = "<owner-post-publish-proof-sha256>"
  postPublishProofUrls = @("<owner-post-publish-proof-url>")
  externalCleanConsumerProofReady = $false
  postPublishProofReady = $false
  rollbackReviewReady = $false
  classificationAuditPassed = $false
  manualCloseOnlyConfirmation = $false
  confirmsNoIssueCloseByAutomation = $false
}
Write-Utf8File -LiteralPath $templatePath -InputObject ($template | ConvertTo-Json -Depth 8)
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]
foreach ($field in @("reviewer", "reviewedAtUtc", "decision", "closeReason", "acceptedRisk", "rollbackPlan", "packageId", "packageVersion", "releaseIssueId", "releaseIssueUrl")) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-OwnerPlaceholder $value) { $findings.Add((New-OwnerFinding "$field-placeholder" "action-required" "missing-field" "Required final close decision field is missing or placeholder: $field")) | Out-Null }
}
foreach ($field in @("evidenceBundleSha256", "classificationAuditSha256", "publicPackageProofSha256", "externalCleanConsumerProofSha256", "postPublishProofSha256")) {
  if (-not (Test-Sha256Text (Get-PropertyOrDefault -Object $input -Name $field -DefaultValue ""))) {
    $findings.Add((New-OwnerFinding "$field-format" "action-required" "missing-sha256" "$field must be 64 hex characters.")) | Out-Null
  }
}
$reviewedAtUtc = [string](Get-PropertyOrDefault -Object $input -Name "reviewedAtUtc" -DefaultValue "")
$parsedReviewedAt = [DateTimeOffset]::MinValue
if (-not [DateTimeOffset]::TryParse($reviewedAtUtc, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$parsedReviewedAt)) {
  $findings.Add((New-OwnerFinding "reviewedAtUtc-format" "action-required" "invalid-timestamp" "reviewedAtUtc must be a valid UTC timestamp.")) | Out-Null
}
$decision = [string](Get-PropertyOrDefault -Object $input -Name "decision" -DefaultValue "")
if ($decision -notin @("close", "approve-close", "approved-for-manual-close")) {
  $findings.Add((New-OwnerFinding "decision-value" "action-required" "invalid-decision" "decision must explicitly approve a manual close.")) | Out-Null
}
$releaseIssueUrl = [string](Get-PropertyOrDefault -Object $input -Name "releaseIssueUrl" -DefaultValue "")
if (-not $releaseIssueUrl.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase) -or $releaseIssueUrl -notmatch '/issues/[0-9]+') {
  $findings.Add((New-OwnerFinding "releaseIssueUrl-format" "action-required" "invalid-url" "releaseIssueUrl must be a GitHub issue URL.")) | Out-Null
}
$postPublishProofUrls = @((Get-PropertyOrDefault -Object $input -Name "postPublishProofUrls" -DefaultValue @()) | ForEach-Object { [string]$_ })
if ($postPublishProofUrls.Count -lt 1 -or @($postPublishProofUrls | Where-Object { Test-OwnerPlaceholder $_ -or -not $_.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0) {
  $findings.Add((New-OwnerFinding "postPublishProofUrls-required" "action-required" "missing-public-proof-url" "At least one public post-publish proof URL is required.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "ownerReviewed" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "ownerReviewed-required" "action-required" "missing-owner-review" "Owner must explicitly review the final close decision.")) | Out-Null
}
foreach ($flag in @("externalCleanConsumerProofReady", "postPublishProofReady", "rollbackReviewReady", "classificationAuditPassed", "manualCloseOnlyConfirmation", "confirmsNoIssueCloseByAutomation")) {
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
  closeDecisionAdmissionReady = $ready
  ownerActionRequired = -not $ready
  requiredFields = @($requiredFields)
  postPublishProofUrlCount = $postPublishProofUrls.Count
  dependenciesConfirmed = [bool](Get-PropertyOrDefault -Object $input -Name "externalCleanConsumerProofReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $input -Name "postPublishProofReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $input -Name "rollbackReviewReady" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $input -Name "classificationAuditPassed" -DefaultValue $false)
  manualCloseOnly = $true
  issueCloseExecutionForbidden = $true
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
  boundary = "Final Owner close decision import validates Owner governance fields and upstream proof references only. Even when admission-ready, this importer does not close the release issue and cannot authorize automation to close it. It is not runtime proof, not post-publish proof, not publish approval by itself, not release close proof, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "final-owner-close-decision-import.json"
$mdPath = Join-Path $OutputRoot "final-owner-close-decision-import.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Final Owner Close Decision Import", "", "- importState: ``$($record.importState)``", "- finalCloseDecisionReady: ``$ready``", "- failedActionRequiredCount: ``$($failed.Count)``", "", "## Boundary", "", $record.boundary)
Write-Host "FinalOwnerCloseDecisionImportState=$($record.importState) Ready=$ready FailedActionRequired=$($failed.Count)"
if ($FailOnNotReady.IsPresent -and -not $ready) { throw "Final Owner close decision is not ready." }
