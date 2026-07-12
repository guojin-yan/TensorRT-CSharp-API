[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "<missing:$RelativePath>" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-HashLane {
  param([string]$Id, [string]$Path, [string]$State, [string]$Boundary)
  $sha = Get-RelativeFileSha256OrPlaceholder -RelativePath $Path
  [pscustomobject]@{
    id = $Id
    path = $Path
    state = $State
    expectedSha256 = $sha
    actualSha256 = $sha
    sha256Matches = ($sha -notlike "<missing:*")
    ready = $false
    boundary = $Boundary
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalFreeze = Read-JsonOrNull "artifacts\final-release\release-candidate-final-evidence-freeze.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$publicPublishDraftValidation = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$cleanConsumerDraftValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$realValidatorValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-record-real-validator-validation.json"
$projectionValidation = Read-JsonOrNull "artifacts\final-release\final-owner-release-close-record-projection-validation.json"
$releaseIssueCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$hashLanes = @(
  New-HashLane -Id "release-evidence-bundle" -Path "artifacts/final-release/release-evidence-bundle.json" -State ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")) -Boundary "Bundle hash lane is audit evidence only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-HashLane -Id "release-candidate-final-evidence-freeze" -Path "artifacts/final-release/release-candidate-final-evidence-freeze.json" -State ([string](Get-PropertyOrDefault -Object $finalFreeze -Name "freezeState" -DefaultValue "missing-release-candidate-final-evidence-freeze")) -Boundary "Final freeze hash lane is audit evidence only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-HashLane -Id "post-publish-validation" -Path "artifacts/final-release/post-publish-verification-validation.json" -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")) -Boundary "Post-publish hash lane cannot replace real post-publish proof."
  New-HashLane -Id "public-publish-real-result-record-draft-validation" -Path "artifacts/final-release/public-publish-real-result-record-draft-validation.json" -State ([string](Get-PropertyOrDefault -Object $publicPublishDraftValidation -Name "validationState" -DefaultValue "missing-public-publish-real-result-record-draft-validation")) -Boundary "Public publish draft hash lane cannot replace public package proof."
  New-HashLane -Id "post-publish-clean-consumer-proof-record-draft-validation" -Path "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json" -State ([string](Get-PropertyOrDefault -Object $cleanConsumerDraftValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-draft-validation")) -Boundary "Clean consumer draft hash lane cannot replace clean consumer runtime proof."
  New-HashLane -Id "final-release-close-record-real-validator-validation" -Path "artifacts/final-release/final-release-close-record-real-validator-validation.json" -State ([string](Get-PropertyOrDefault -Object $realValidatorValidation -Name "validationState" -DefaultValue "missing-final-release-close-record-real-validator-validation")) -Boundary "Real validator hash lane is contract evidence only."
  New-HashLane -Id "final-owner-release-close-record-projection-validation" -Path "artifacts/final-release/final-owner-release-close-record-projection-validation.json" -State ([string](Get-PropertyOrDefault -Object $projectionValidation -Name "validationState" -DefaultValue "missing-final-owner-release-close-record-projection-validation")) -Boundary "Projection hash lane is owner input mapping only."
  New-HashLane -Id "release-issue-close-record-validation" -Path "artifacts/final-release/release-issue-close-record-validation.json" -State ([string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -Boundary "Release issue close validation hash lane cannot close the issue unless real proof passes."
)

$mismatched = @($hashLanes | Where-Object { -not [bool]$_.sha256Matches })
$blocked = @($hashLanes | Where-Object { $_.state -like "blocked*" -or $_.state -like "missing*" -or $_.state -like "invalid*" })

$record = [pscustomobject]@{
  recordKind = "final-release-close-hash-consistency-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = "blocked-final-release-close-hash-consistency-owner-proof-required"
  hashLaneCount = $hashLanes.Count
  mismatchedHashCount = $mismatched.Count
  blockedHashLaneCount = $blocked.Count
  readyHashLaneCount = 0
  hashLanes = @($hashLanes)
  sourceArtifacts = @($hashLanes | ForEach-Object { $_.path })
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Final release close hash consistency gate checks local hashes only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-hash-consistency-gate.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-hash-consistency-gate.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $hashLanes | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.state)`` | ``$($_.sha256Matches)`` | ``$($_.path)`` | $($_.boundary.Replace("|", "\|")) |"
}

$markdown = @"
# Final Release Close Hash Consistency Gate

| Field | Value |
| --- | --- |
| gateState | ``$($record.gateState)`` |
| hashLaneCount | ``$($record.hashLaneCount)`` |
| mismatchedHashCount | ``$($record.mismatchedHashCount)`` |
| blockedHashLaneCount | ``$($record.blockedHashLaneCount)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Hash Lanes

| ID | State | SHA256 Matches | Path | Boundary |
| --- | --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release close hash consistency gate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "GateState=$($record.gateState) HashLanes=$($record.hashLaneCount) Mismatched=$($record.mismatchedHashCount) Blocked=$($record.blockedHashLaneCount)"
