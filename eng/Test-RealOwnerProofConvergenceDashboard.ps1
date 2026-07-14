[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-owner-proof-convergence-dashboard.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInput = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInput -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Export-RealOwnerProofConvergenceDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $resolvedInput -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$gates = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "gates" -DefaultValue @()))
$categories = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "categorySummaries" -DefaultValue @()))
$categoryIds = @($categories | ForEach-Object { [string]$_.category })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-owner-proof-convergence-dashboard") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-coverage" ($lanes.Count -eq 9 -and [int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0) -eq 9) "blocker" "Dashboard must cover nine Owner proof lanes."
  New-OwnerValidationItem "gate-coverage" ($gates.Count -eq 3 -and @($gates | Where-Object { [string]$_.id -eq "all-real-owner-proof-inputs-accepted" }).Count -eq 1) "blocker" "Dashboard must include contract, preflight, and real input acceptance gates."
  New-OwnerValidationItem "category-coverage" (@("post-publish", "governance", "verification" | Where-Object { $categoryIds -notcontains $_ }).Count -eq 0) "blocker" "Dashboard must summarize all categories."
  New-OwnerValidationItem "accepted-count-consistency" ([int](Get-PropertyOrDefault -Object $record -Name "acceptedLaneCount" -DefaultValue -1) -eq @($lanes | Where-Object { [bool]$_.accepted }).Count) "blocker" "Accepted lane count must match lane details."
  New-OwnerValidationItem "blocked-or-ready-state" ([string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "") -in @("blocked-real-owner-proof-convergence-real-owner-input-required", "real-owner-proof-convergence-ready-for-final-owner-review-non-proof")) "blocker" "Convergence state must be explicit."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Dashboard must remain non-proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must reject publish/proof substitution."
)
$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "real-owner-proof-convergence-dashboard-validation-ready-non-proof" } else { "invalid-real-owner-proof-convergence-dashboard" }
$validation = [pscustomobject]@{
  recordKind = "real-owner-proof-convergence-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  convergenceState = [string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "")
  laneCount = $lanes.Count
  acceptedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "acceptedLaneCount" -DefaultValue 0)
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  gateCount = $gates.Count
  readyGateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyGateCount" -DefaultValue 0)
  blockedGateCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedGateCount" -DefaultValue 0)
  allRealOwnerProofInputsAccepted = [bool](Get-PropertyOrDefault -Object $record -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false)
  failedBlockerCount = $failed.Count
  validationItems = @($items)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof convergence dashboard validation checks aggregation only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-convergence-dashboard-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-convergence-dashboard-validation.md") -InputObject @("# Real Owner Proof Convergence Dashboard Validation", "", "- validationState: ``$state``", "- acceptedLaneCount: ``$($validation.acceptedLaneCount)/$($validation.laneCount)``", "- readyGateCount: ``$($validation.readyGateCount)/$($validation.gateCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "RealOwnerProofConvergenceDashboardValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Real Owner proof convergence dashboard validation failed." }
