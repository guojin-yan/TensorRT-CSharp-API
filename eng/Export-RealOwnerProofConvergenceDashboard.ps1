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
& (Join-Path $RepositoryRoot "eng\Export-RealOwnerProofInputContracts.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
& (Join-Path $RepositoryRoot "eng\Test-RealOwnerProofInputContracts.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict | Out-Null
& (Join-Path $RepositoryRoot "eng\Export-RealOwnerProofAdmissionPreflight.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
& (Join-Path $RepositoryRoot "eng\Test-RealOwnerProofAdmissionPreflight.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict | Out-Null
$contracts = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/real-owner-proof-input-contracts-validation.json"
$preflight = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/real-owner-proof-admission-preflight.json"
$preflightValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/real-owner-proof-admission-preflight-validation.json"
$lanes = @(Convert-ToArray (Get-PropertyOrDefault -Object $preflight -Name "lanes" -DefaultValue @()))
$contractReady = [int](Get-PropertyOrDefault -Object $contracts -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$preflightStructurallyReady = [int](Get-PropertyOrDefault -Object $preflightValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0
$allAccepted = [bool](Get-PropertyOrDefault -Object $preflight -Name "allRealOwnerProofInputsAccepted" -DefaultValue $false)
$gates = @(
  [pscustomobject]@{ id = "real-owner-proof-input-contracts"; ready = $contractReady; blockedReason = if ($contractReady) { "" } else { "input-contracts-invalid" }; artifact = "artifacts/final-release/real-owner-proof-input-contracts-validation.json" }
  [pscustomobject]@{ id = "real-owner-proof-admission-preflight"; ready = $preflightStructurallyReady; blockedReason = if ($preflightStructurallyReady) { "" } else { "admission-preflight-invalid" }; artifact = "artifacts/final-release/real-owner-proof-admission-preflight-validation.json" }
  [pscustomobject]@{ id = "all-real-owner-proof-inputs-accepted"; ready = $allAccepted; blockedReason = if ($allAccepted) { "" } else { "real-owner-proof-inputs-still-blocked" }; artifact = "artifacts/final-release/real-owner-proof-admission-preflight.json" }
)
$readyGateCount = @($gates | Where-Object { [bool]$_.ready }).Count
$blockedGateCount = $gates.Count - $readyGateCount
$acceptedLaneCount = @($lanes | Where-Object { [bool]$_.accepted }).Count
$state = if ($allAccepted -and $readyGateCount -eq $gates.Count) { "real-owner-proof-convergence-ready-for-final-owner-review-non-proof" } else { "blocked-real-owner-proof-convergence-real-owner-input-required" }
$categorySummaries = @("post-publish", "governance", "verification" | ForEach-Object {
  $category = $_
  $categoryLanes = @($lanes | Where-Object { [string]$_.category -eq $category })
  [pscustomobject]@{ category = $category; laneCount = $categoryLanes.Count; acceptedLaneCount = @($categoryLanes | Where-Object { [bool]$_.accepted }).Count; blockedLaneCount = @($categoryLanes | Where-Object { -not [bool]$_.accepted }).Count }
})
$record = [pscustomobject]@{
  recordKind = "real-owner-proof-convergence-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  convergenceState = $state
  laneCount = $lanes.Count
  acceptedLaneCount = $acceptedLaneCount
  blockedLaneCount = $lanes.Count - $acceptedLaneCount
  gateCount = $gates.Count
  readyGateCount = $readyGateCount
  blockedGateCount = $blockedGateCount
  allRealOwnerProofInputsAccepted = $allAccepted
  gates = @($gates)
  categorySummaries = @($categorySummaries)
  lanes = @($lanes)
  ownerActionRequired = -not $allAccepted
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof convergence dashboard aggregates nine admission lanes for Owner review only. It does not publish, use tokens, run external workloads, close the release issue, or promote proof. A green contract or preflight gate cannot substitute real public package, external CleanConsumer, article publication, YoloVision runtime, rollback review, close decision, CI, or hash-review evidence; this dashboard is not runtime proof, not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-convergence-dashboard.json") -InputObject ($record | ConvertTo-Json -Depth 16)
$lines = @("# Real Owner Proof Convergence Dashboard", "", "- convergenceState: ``$state``", "- acceptedLaneCount: ``$acceptedLaneCount/$($lanes.Count)``", "- readyGateCount: ``$readyGateCount/$($gates.Count)``", "", "| Category | Accepted | Blocked | Total |", "|---|---:|---:|---:|")
foreach ($summary in $categorySummaries) { $lines += "| ``$($summary.category)`` | $($summary.acceptedLaneCount) | $($summary.blockedLaneCount) | $($summary.laneCount) |" }
$lines += @("", $record.boundary)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-convergence-dashboard.md") -InputObject $lines
Write-Host "RealOwnerProofConvergenceDashboardState=$state Accepted=$acceptedLaneCount/$($lanes.Count) Gates=$readyGateCount/$($gates.Count)"
