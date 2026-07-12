[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-publish-and-article-action-dashboard.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerPublishAndArticleActionDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$gates = @(Get-PropertyOrDefault -Object $record -Name "gates" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$requiredIds = @("final-readonly-audit", "owner-one-screen-manual", "publish-replay-checklist", "evidence-import-runbook", "owner-intake-dry-run", "post-publish-cross-check", "article-readiness-matrix", "post-publish-article-proof-gate", "release-close-strict-closure")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-publish-and-article-action-dashboard") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-by-default" (([string](Get-PropertyOrDefault -Object $record -Name "dashboardState" -DefaultValue "")).Contains("blocked") -and [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0) -eq $requiredIds.Count) "blocker" "Dashboard must remain Owner-action blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "gate-count" ($gates.Count -eq $requiredIds.Count) "blocker" "Every final Owner publish/article gate must be present.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "gate-$id" (@($gates | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and [bool]$_.blocked -and -not [bool]$_.performsPublish -and -not [bool]$_.canCloseReleaseIssue }).Count -eq 1) "blocker" "Missing or unsafe dashboard gate: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("ProjectReference") -and $text.Contains("direct nupkg") -and $text.Contains("queued workflow") -and $text.Contains("TensorRtExec report")) "blocker" "Forbidden substitutes must remain visible.")) | Out-Null
$items.Add((New-OwnerValidationItem "links-current-surfaces" ($text.Contains("final-owner-publish-execution-replay-checklist-pack-validation.json") -and $text.Contains("final-owner-publish-evidence-import-runbook-validation.json") -and $text.Contains("public-article-readiness-matrix-validation.json") -and $text.Contains("post-publish-article-proof-gate-validation.json")) "blocker" "Dashboard must link current planning and article proof surfaces.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsNuGetPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Dashboard must not publish or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-publish-and-article-action-dashboard-validation-ready-non-proof" } else { "blocked-final-owner-publish-and-article-action-dashboard-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-publish-and-article-action-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  gateCount = [int]$gates.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner publish and article action dashboard validation only; not proof, not package publication, not article publication, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-and-article-action-dashboard-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-and-article-action-dashboard-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner Publish And Article Action Dashboard Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- gateCount: ``$($validation.gateCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerPublishAndArticleActionDashboardValidationState=$state FailedBlockers=$failedBlockerCount Gates=$($validation.gateCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner publish and article action dashboard validation failed." }
