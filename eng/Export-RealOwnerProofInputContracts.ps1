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
$requiredFieldCount = ($contracts | ForEach-Object { [int]$_.requiredFieldCount } | Measure-Object -Sum).Sum
if ($null -eq $requiredFieldCount) { $requiredFieldCount = 0 }
$record = [pscustomobject]@{
  recordKind = "real-owner-proof-input-contracts"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "real-owner-proof-input-contracts-ready-non-proof"
  laneCount = $contracts.Count
  postPublishLaneCount = @($contracts | Where-Object { [string]$_.category -eq "post-publish" }).Count
  governanceLaneCount = @($contracts | Where-Object { [string]$_.category -eq "governance" }).Count
  verificationLaneCount = @($contracts | Where-Object { [string]$_.category -eq "verification" }).Count
  requiredFieldCount = [int]$requiredFieldCount
  contracts = @($contracts)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof input contracts describe the existing nine-lane admission surface only. Contracts, templates, local validation, hashes, dashboards, and CI metadata are not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-input-contracts.json") -InputObject ($record | ConvertTo-Json -Depth 16)
$lines = @("# Real Owner Proof Input Contracts", "", "- contractState: ``$($record.contractState)``", "- laneCount: ``$($record.laneCount)``", "- requiredFieldCount: ``$($record.requiredFieldCount)``", "", "| Lane | Category | Required fields | Owner input | Validation artifact |", "|---|---|---:|---|---|")
foreach ($contract in $contracts) {
  $lines += "| ``$($contract.id)`` | ``$($contract.category)`` | $($contract.requiredFieldCount) | ``$($contract.ownerInputArtifact)`` | ``$($contract.validationArtifact)`` |"
}
$lines += @("", $record.boundary)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-input-contracts.md") -InputObject $lines
Write-Host "RealOwnerProofInputContractsState=$($record.contractState) Lanes=$($record.laneCount) RequiredFields=$($record.requiredFieldCount)"
