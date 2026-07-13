[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-strict-evidence-closure-dashboard.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseStrictEvidenceClosureDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Get-PropertyOrDefault -Object $record -Name "groups" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredGroups = @("publish-result", "post-publish-proof", "rollback-review", "close-decision", "classification-audit", "forbidden-substitute-scan")
$validationItems = New-Object System.Collections.Generic.List[object]
$recordKindOk = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-strict-evidence-closure-dashboard"
$dashboardState = [string](Get-PropertyOrDefault -Object $record -Name "dashboardState" -DefaultValue "")
$defaultBlockedNonProofOk = $dashboardState.IndexOf("blocked", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "dashboardIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$groupCountOk = $groups.Count -eq $requiredGroups.Count
$validationItems.Add((New-OwnerValidationItem "record-kind" $recordKindOk "blocker" "recordKind must match.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "default-blocked-non-proof" $defaultBlockedNonProofOk "blocker" "Dashboard must remain blocked and non-proof.")) | Out-Null
$validationItems.Add((New-OwnerValidationItem "group-count" $groupCountOk "blocker" "Dashboard must expose every required Owner material group.")) | Out-Null
foreach ($id in $requiredGroups) {
  $group = @($groups | Where-Object { [string]$_.id -eq $id })
  $valid = $group.Count -eq 1 -and -not [string]::IsNullOrWhiteSpace([string]$group[0].requiredArtifact) -and -not [string]::IsNullOrWhiteSpace([string]$group[0].requiredHash) -and -not [string]::IsNullOrWhiteSpace([string]$group[0].validatorCommand) -and -not [bool]$group[0].isReleaseCloseProof
  $validationItems.Add((New-OwnerValidationItem "dashboard-group-$id" $valid "blocker" "Dashboard group must include required artifact/hash/validator and remain non-proof: $id")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("not proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("not public publish", [StringComparison]::OrdinalIgnoreCase) -ge 0
$validationItems.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Dashboard boundary must state non-proof and non-publish status.")) | Out-Null

$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "release-close-strict-evidence-closure-dashboard-validation-ready-non-proof" } else { "blocked-release-close-strict-evidence-closure-dashboard-validation-invalid" }
$failedBlockerCount = [int]$failed.Count
$validationItemCount = [int]$validationItems.Count
$dashboardGroupCount = [int]$groups.Count
$blockedDashboardGroupCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedDashboardGroupCount" -DefaultValue -1)
$validationItemArray = @($validationItems.ToArray())
$validation = [pscustomobject]@{
  recordKind = "release-close-strict-evidence-closure-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = $validationItemCount
  dashboardGroupCount = $dashboardGroupCount
  blockedDashboardGroupCount = $blockedDashboardGroupCount
  validationItems = $validationItemArray
  ownerActionRequired = $true
  passed = $false
  dashboardIsProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseCloseProof = $false
  boundary = "Dashboard validation only; not proof and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-dashboard-validation.json"
$mdPath = Join-Path $OutputRoot "release-close-strict-evidence-closure-dashboard-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# ReleaseClose Strict Evidence Closure Dashboard Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$($failed.Count)``",
  "- blockedDashboardGroupCount: ``$($validation.blockedDashboardGroupCount)``",
  "",
  "## Boundary",
  "",
  $validation.boundary
)
Write-Host "ReleaseCloseStrictEvidenceClosureDashboardValidationState=$state FailedBlockers=$failedBlockerCount BlockedGroups=$blockedDashboardGroupCount"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "ReleaseClose strict evidence closure dashboard validation failed." }
