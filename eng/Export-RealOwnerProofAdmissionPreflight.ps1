[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "RealOwnerProofContracts.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$contracts = @(Get-RealOwnerProofLaneContracts)

foreach ($contract in $contracts) {
  $refreshPath = Join-Path $RepositoryRoot "eng\$($contract.refreshScript)"
  $validationPath = Join-Path $RepositoryRoot "eng\$($contract.validationScript)"
  & $refreshPath -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
  & $validationPath -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict | Out-Null
}

$lanes = @($contracts | ForEach-Object { Get-RealOwnerProofLaneObservation -Contract $_ -RepositoryRoot $RepositoryRoot })
$acceptedLaneCount = @($lanes | Where-Object { [bool]$_.accepted }).Count
$structuralReadyLaneCount = @($lanes | Where-Object { [bool]$_.structuralReady }).Count
$blockedLaneCount = $lanes.Count - $acceptedLaneCount
$structuralBlockedLaneCount = $lanes.Count - $structuralReadyLaneCount
$failedActionRequiredCount = ($lanes | ForEach-Object { [int]$_.failedActionRequiredCount } | Measure-Object -Sum).Sum
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }
$allAccepted = $lanes.Count -gt 0 -and $acceptedLaneCount -eq $lanes.Count -and $structuralBlockedLaneCount -eq 0
$state = if ($allAccepted) { "real-owner-proof-admission-preflight-ready-for-owner-convergence-review-non-proof" } else { "blocked-real-owner-proof-admission-preflight-owner-input-required" }
$record = [pscustomobject]@{
  recordKind = "real-owner-proof-admission-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = $state
  laneCount = $lanes.Count
  structuralReadyLaneCount = $structuralReadyLaneCount
  structuralBlockedLaneCount = $structuralBlockedLaneCount
  acceptedLaneCount = $acceptedLaneCount
  blockedLaneCount = $blockedLaneCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  allRealOwnerProofInputsAccepted = $allAccepted
  ownerActionRequired = -not $allAccepted
  lanes = @($lanes)
  rejectedNonProofSubstitutes = @("template", "dashboard", "dry-run", "local-feed", "ProjectReference", "direct-nupkg", "queued-workflow", "hash-only", "validation-ready-without-owner-proof", "staging-shape-valid-only", "sample-build-only", "mock-output")
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof admission preflight refreshes and reads existing lane validators only. An accepted lane is scoped to its own evidence type; this aggregate does not publish, use tokens, run inference, run CleanConsumer, close the release issue, or promote proof. Templates, dashboards, dry-runs, local feeds, ProjectReference, direct nupkg, queued workflows, hashes alone, validation-ready records, sample builds, and mock outputs are not runtime proof, not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-admission-preflight.json") -InputObject ($record | ConvertTo-Json -Depth 16)
$lines = @("# Real Owner Proof Admission Preflight", "", "- preflightState: ``$state``", "- structuralReadyLaneCount: ``$structuralReadyLaneCount/$($lanes.Count)``", "- acceptedLaneCount: ``$acceptedLaneCount/$($lanes.Count)``", "- failedActionRequiredCount: ``$failedActionRequiredCount``", "", "| Lane | Structural | Accepted | State | Action required |", "|---|---:|---:|---|---:|")
foreach ($lane in $lanes) {
  $lines += "| ``$($lane.id)`` | ``$($lane.structuralReady)`` | ``$($lane.accepted)`` | ``$($lane.state)`` | $($lane.failedActionRequiredCount) |"
}
$lines += @("", $record.boundary)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-admission-preflight.md") -InputObject $lines
Write-Host "RealOwnerProofAdmissionPreflightState=$state Structural=$structuralReadyLaneCount/$($lanes.Count) Accepted=$acceptedLaneCount/$($lanes.Count)"
