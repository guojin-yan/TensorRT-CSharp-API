[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-FreezeLane {
  param(
    [string]$Id,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState,
    [string]$ReadyState
  )

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  $present = $null -ne $Record
  $ready = $present -and $state -eq $ReadyState
  $performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
  $canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
  $canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
  $canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  $isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  $isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
  $isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue 0)
  $failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue 0)

  [pscustomobject]@{
    id = $Id
    artifact = $Artifact
    present = $present
    state = $state
    readyState = $ReadyState
    ready = $ready
    blocked = -not $ready
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    performsPublish = $performsPublish
    canPublishPublicly = $canPublishPublicly
    canCloseReleaseIssue = $canCloseReleaseIssue
    canPromoteRuntimeProof = $canPromoteRuntimeProof
    isRuntimeExecutionProof = $isRuntimeExecutionProof
    isPostPublishProof = $isPostPublishProof
    isReleaseCloseProof = $isReleaseCloseProof
    boundaryOk = (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $canPromoteRuntimeProof -and -not $isRuntimeExecutionProof -and -not $isPostPublishProof -and -not $isReleaseCloseProof)
    boundary = "Prepublish quality freeze lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and not GitHub Actions proof."
  }
}

$scriptIndex = Read-JsonOrNull "artifacts\final-release\release-script-reference-index.json"
$scriptIndexValidation = Read-JsonOrNull "artifacts\final-release\release-script-reference-index-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$authorizationInputValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-authorization-input-validation.json"
$authorizationGateValidation = Read-JsonOrNull "artifacts\final-release\owner-public-publish-authorization-gate-validation.json"
$publishResultConvergenceValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-authorization-convergence-gate-validation.json"
$finalOwnerExecutionPackValidation = Read-JsonOrNull "artifacts\final-release\public-publish-final-owner-execution-pack-validation.json"
$publicPublishCommandCrossCheckValidation = Read-JsonOrNull "artifacts\final-release\public-publish-command-cross-check-validation.json"
$finalEvidenceFreezeNonProofAuditValidation = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze-non-proof-audit-validation.json"
$releaseCandidatePackageInventory = Read-JsonOrNull "artifacts\final-release\release-candidate-package-inventory.json"
$finalQualityFreezeValidation = Read-JsonOrNull "artifacts\final-release\final-quality-freeze-dashboard-validation.json"

$lanes = @(
  New-FreezeLane -Id "release-script-reference-index" -Artifact "artifacts/final-release/release-script-reference-index.json" -Record $scriptIndex -StateProperty "indexState" -DefaultState "missing-release-script-reference-index" -ReadyState "release-script-reference-index-ready-non-proof"
  New-FreezeLane -Id "release-script-reference-index-validation" -Artifact "artifacts/final-release/release-script-reference-index-validation.json" -Record $scriptIndexValidation -StateProperty "validationState" -DefaultState "missing-release-script-reference-index-validation" -ReadyState "release-script-reference-index-validation-ready-non-proof"
  New-FreezeLane -Id "release-evidence-bundle" -Artifact "artifacts/final-release/release-evidence-bundle.json" -Record $releaseEvidenceBundle -StateProperty "bundleState" -DefaultState "missing-release-evidence-bundle" -ReadyState "blocked-real-proof-required"
  New-FreezeLane -Id "release-evidence-classification-audit" -Artifact "artifacts/final-release/release-evidence-classification-audit.json" -Record $classificationAudit -StateProperty "auditState" -DefaultState "missing-release-evidence-classification-audit" -ReadyState "classification-audit-passed-non-proof-boundaries-intact"
  New-FreezeLane -Id "owner-public-publish-authorization-input" -Artifact "artifacts/final-release/owner-public-publish-authorization-input-validation.json" -Record $authorizationInputValidation -StateProperty "validationState" -DefaultState "missing-owner-public-publish-authorization-input-validation" -ReadyState "owner-public-publish-authorization-input-ready"
  New-FreezeLane -Id "owner-public-publish-authorization-gate" -Artifact "artifacts/final-release/owner-public-publish-authorization-gate-validation.json" -Record $authorizationGateValidation -StateProperty "validationState" -DefaultState "missing-owner-public-publish-authorization-gate-validation" -ReadyState "owner-public-publish-authorized-for-manual-execution-review"
  New-FreezeLane -Id "public-publish-result-authorization-convergence-gate" -Artifact "artifacts/final-release/public-publish-result-authorization-convergence-gate-validation.json" -Record $publishResultConvergenceValidation -StateProperty "validationState" -DefaultState "missing-public-publish-result-authorization-convergence-gate-validation" -ReadyState "public-publish-result-authorization-convergence-ready"
  New-FreezeLane -Id "public-publish-final-owner-execution-pack" -Artifact "artifacts/final-release/public-publish-final-owner-execution-pack-validation.json" -Record $finalOwnerExecutionPackValidation -StateProperty "validationState" -DefaultState "missing-public-publish-final-owner-execution-pack-validation" -ReadyState "public-publish-final-owner-execution-pack-ready"
  New-FreezeLane -Id "public-publish-command-cross-check" -Artifact "artifacts/final-release/public-publish-command-cross-check-validation.json" -Record $publicPublishCommandCrossCheckValidation -StateProperty "validationState" -DefaultState "missing-public-publish-command-cross-check-validation" -ReadyState "public-publish-command-cross-check-ready"
  New-FreezeLane -Id "final-evidence-freeze-non-proof-audit" -Artifact "artifacts/final-release/final-evidence-freeze-non-proof-audit-validation.json" -Record $finalEvidenceFreezeNonProofAuditValidation -StateProperty "validationState" -DefaultState "missing-final-evidence-freeze-non-proof-audit-validation" -ReadyState "blocked-final-evidence-freeze-non-proof-audit"
  New-FreezeLane -Id "release-candidate-package-inventory" -Artifact "artifacts/final-release/release-candidate-package-inventory.json" -Record $releaseCandidatePackageInventory -StateProperty "inventoryState" -DefaultState "missing-release-candidate-package-inventory" -ReadyState "release-candidate-package-inventory-ready"
  New-FreezeLane -Id "final-quality-freeze-dashboard" -Artifact "artifacts/final-release/final-quality-freeze-dashboard-validation.json" -Record $finalQualityFreezeValidation -StateProperty "validationState" -DefaultState "missing-final-quality-freeze-dashboard-validation" -ReadyState "blocked-final-quality-freeze-real-proof-required"
)

$blockedLanes = @($lanes | Where-Object { -not $_.ready })
$boundaryFailures = @($lanes | Where-Object { -not $_.boundaryOk })
$failedBlockers = @($lanes | Where-Object { $_.failedBlockerCount -gt 0 })
$failedActionRequiredCount = ($lanes | Measure-Object -Property failedActionRequiredCount -Sum).Sum
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }
$ownerAuthorizationReady = [string](Get-PropertyOrDefault -Object $authorizationInputValidation -Name "validationState" -DefaultValue "") -eq "owner-public-publish-authorization-input-ready"
$ownerGateReady = [string](Get-PropertyOrDefault -Object $authorizationGateValidation -Name "validationState" -DefaultValue "") -eq "owner-public-publish-authorized-for-manual-execution-review"
$allLocalNonPublishGatesOk = $boundaryFailures.Count -eq 0 -and $failedBlockers.Count -eq 0
$readyForOwnerPublicPublishExecution = $ownerAuthorizationReady -and $ownerGateReady -and $allLocalNonPublishGatesOk -and $blockedLanes.Count -eq 0
$freezeState = if ($readyForOwnerPublicPublishExecution) { "final-prepublish-quality-freeze-ready-for-owner-manual-public-publish-review" } else { "blocked-final-prepublish-quality-freeze-owner-action-required" }

$record = [pscustomobject]@{
  recordKind = "final-prepublish-quality-freeze-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  freezeState = $freezeState
  laneCount = $lanes.Count
  readyLaneCount = @($lanes | Where-Object { $_.ready }).Count
  blockedLaneCount = $blockedLanes.Count
  boundaryFailureCount = $boundaryFailures.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int]$failedActionRequiredCount
  ownerAuthorizationReady = $ownerAuthorizationReady
  ownerAuthorizationGateReady = $ownerGateReady
  readyForOwnerPublicPublishExecution = $readyForOwnerPublicPublishExecution
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  ownerExecutionOnly = $true
  notExecutedByAutomation = $true
  lanes = @($lanes)
  sourceArtifacts = @($lanes | ForEach-Object { $_.artifact })
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dry-run", "dashboard", "manual approval", "queued workflow", "missing runner", "sidecar-only", "TensorRtExec report", "local dotnet test", "local package consumer")
  safetyBoundary = "Final prepublish quality freeze dashboard is a local read-only status aggregator only; it does not run dotnet nuget push, does not trigger GitHub Actions, does not publish packages, does not close release issues, and is not runtime proof, post-publish proof, release close approval, or GitHub Actions proof."
}

$jsonPath = Join-Path $OutputRoot "final-prepublish-quality-freeze-dashboard.json"
$markdownPath = Join-Path $OutputRoot "final-prepublish-quality-freeze-dashboard.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $lanes | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.state) | ``$($_.boundaryOk)`` |" }
$markdown = @"
# Final Prepublish Quality Freeze Dashboard

| Item | Value |
|---|---|
| freezeState | ``$($record.freezeState)`` |
| laneCount | ``$($record.laneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| boundaryFailureCount | ``$($record.boundaryFailureCount)`` |
| ownerAuthorizationReady | ``$($record.ownerAuthorizationReady)`` |
| readyForOwnerPublicPublishExecution | ``$($record.readyForOwnerPublicPublishExecution)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Lanes

| ID | Ready | State | Boundary OK |
|---|---:|---|---:|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final prepublish quality freeze dashboard written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "FreezeState=$($record.freezeState) Ready=$($record.readyLaneCount) Blocked=$($record.blockedLaneCount)"
