[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$closure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure.json"
if ($null -eq $closure) {
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseStrictEvidenceClosure.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  $closure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure.json"
}

$lanes = @(Get-PropertyOrDefault -Object $closure -Name "lanes" -DefaultValue @())
$crossChecks = @(Get-PropertyOrDefault -Object $closure -Name "crossChecks" -DefaultValue @())
$groups = @(
  [pscustomobject]@{ id = "publish-result"; title = "Owner public publish result"; laneIds = @("owner-public-publish-result"); requiredArtifact = "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json"; requiredHash = "owner imported public package and command-plan SHA256 fields"; validatorCommand = "Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" },
  [pscustomobject]@{ id = "post-publish-proof"; title = "PostPublish proof"; laneIds = @("post-publish-owner-input", "post-publish-record"); requiredArtifact = "artifacts/final-release/post-publish-verification-validation.json"; requiredHash = "downloaded package/log/native asset SHA256 fields"; validatorCommand = "Test-PostPublishVerificationOwnerInput.ps1 -Strict; Test-PostPublishVerificationRecord.ps1 -Strict" },
  [pscustomobject]@{ id = "rollback-review"; title = "Rollback review"; laneIds = @("rollback-review"); requiredArtifact = "artifacts/final-release/final-owner-rollback-review-validation.json"; requiredHash = "rollback review artifact SHA256 referenced by PostPublish and close decision"; validatorCommand = "Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict" },
  [pscustomobject]@{ id = "close-decision"; title = "Final owner close decision"; laneIds = @("close-decision", "final-public-publish-acceptance-gate"); requiredArtifact = "artifacts/final-release/final-owner-close-decision-validation.json"; requiredHash = "evidence bundle SHA256 and aligned public package version"; validatorCommand = "Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict; Test-FinalPublicPublishAcceptanceGate.ps1 -Strict" },
  [pscustomobject]@{ id = "classification-audit"; title = "Classification audit"; laneIds = @("release-evidence-bundle", "classification-audit"); requiredArtifact = "artifacts/final-release/release-evidence-classification-audit.json"; requiredHash = "release evidence bundle SHA256 and non-proof boundary markers"; validatorCommand = "Export-ReleaseEvidenceBundle.ps1; Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" },
  [pscustomobject]@{ id = "forbidden-substitute-scan"; title = "Forbidden substitute scan"; laneIds = @(); requiredArtifact = "artifacts/final-release/public-publish-forbidden-substitute-scan.json"; requiredHash = "forbidden substitute scan SHA256 referenced by PostPublish"; validatorCommand = "Export-PublicPublishForbiddenSubstituteScan.ps1; Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict" }
)

$dashboardGroups = foreach ($group in $groups) {
  $groupLanes = @($lanes | Where-Object { $group.laneIds -contains [string]$_.id })
  $blocked = @($groupLanes | Where-Object { -not [bool]$_.ready })
  $relatedChecks = @($crossChecks | Where-Object {
    ([string]$_.id).Contains($group.id.Replace("-", ""), [StringComparison]::OrdinalIgnoreCase) -or
    ([string]$_.id).Contains("version", [StringComparison]::OrdinalIgnoreCase) -or
    ([string]$_.id).Contains("hash", [StringComparison]::OrdinalIgnoreCase)
  })
  [pscustomobject]@{
    id = $group.id
    title = $group.title
    dashboardGroupState = if ($blocked.Count -eq 0 -and $groupLanes.Count -gt 0) { "ready" } else { "blocked-owner-action-required" }
    laneCount = $groupLanes.Count
    blockedLaneCount = $blocked.Count
    requiredArtifact = $group.requiredArtifact
    requiredHash = $group.requiredHash
    validatorCommand = $group.validatorCommand
    ownerNextAction = if ($blocked.Count -eq 0 -and $groupLanes.Count -gt 0) { "Keep evidence current and rerun strict closure before any Owner close decision." } else { "Provide the required real Owner artifact/hash and rerun the validator command." }
    lanes = @($groupLanes)
    relatedCrossChecks = @($relatedChecks)
    canCloseReleaseIssue = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Dashboard group only; it is not proof and cannot replace owner-executed public publish, PostPublish, rollback, close decision, classification audit, or forbidden substitute scan evidence."
  }
}

$blockedGroups = @($dashboardGroups | Where-Object { [string]$_.dashboardGroupState -ne "ready" })
$state = if ($blockedGroups.Count -eq 0) { "release-close-strict-evidence-closure-dashboard-ready-non-proof" } else { "blocked-release-close-strict-evidence-closure-dashboard-owner-action-required" }
$record = [pscustomobject]@{
  recordKind = "release-close-strict-evidence-closure-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = $state
  dashboardGroupCount = @($dashboardGroups).Count
  blockedDashboardGroupCount = $blockedGroups.Count
  closureState = [string](Get-PropertyOrDefault -Object $closure -Name "closureState" -DefaultValue "missing-release-close-strict-evidence-closure")
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $closure -Name "blockedLaneCount" -DefaultValue -1)
  failedCrossCheckCount = [int](Get-PropertyOrDefault -Object $closure -Name "failedCrossCheckCount" -DefaultValue -1)
  groups = @($dashboardGroups)
  ownerActionRequired = $true
  passed = $false
  dashboardIsProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This dashboard only summarizes missing Owner materials and validator commands. It is not proof, not a release close approval, not public publish, and not a substitute for public package/PostPublish evidence."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-dashboard.json"
$mdPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-dashboard.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# ReleaseClose Strict Evidence Closure Dashboard") | Out-Null
$md.Add("") | Out-Null
$md.Add("- dashboardState: ``$state``") | Out-Null
$md.Add("- blockedDashboardGroupCount: ``$($blockedGroups.Count)``") | Out-Null
$md.Add("- dashboardIsProof: ``False``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Group | State | Required Artifact | Required Hash | Validator | Owner Next Action |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- | --- |") | Out-Null
foreach ($group in $dashboardGroups) {
  $md.Add("| $($group.id) | $($group.dashboardGroupState) | $(ConvertTo-MarkdownCell $group.requiredArtifact) | $(ConvertTo-MarkdownCell $group.requiredHash) | ``$(ConvertTo-MarkdownCell $group.validatorCommand)`` | $(ConvertTo-MarkdownCell $group.ownerNextAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "ReleaseCloseStrictEvidenceClosureDashboardState=$state BlockedGroups=$($blockedGroups.Count)"
