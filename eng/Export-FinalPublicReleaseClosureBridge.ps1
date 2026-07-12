[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
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

function New-ClosureLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$RequiredState,
    [string]$OwnerAction,
    [string]$Boundary,
    [string[]]$RequiredBeforeClose
  )

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue "missing-$Id")
  $ready = $state -eq $RequiredState
  $exists = $null -ne $Record
  return [pscustomobject]@{
    laneId = $Id
    title = $Title
    artifact = $Artifact
    artifactExists = $exists
    state = $state
    requiredState = $RequiredState
    ready = $ready
    ownerAction = $OwnerAction
    requiredBeforeClose = @($RequiredBeforeClose)
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = $false
    boundary = $Boundary
  }
}

$ownerAuthorization = Read-JsonOrNull "artifacts\final-release\owner-publish-authorization-input-validation.json"
$publicDownload = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-input-validation.json"
$cleanConsumerSmoke = Read-JsonOrNull "artifacts\final-release\clean-external-consumer-smoke-input-validation.json"
$postPublishProof = Read-JsonOrNull "artifacts\final-release\post-publish-proof-input-validation.json"
$releaseCloseDecision = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$strictCloseDashboard = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"

$lanes = @(
  New-ClosureLane `
    -Id "owner-publish-authorization" `
    -Title "Owner publish authorization input" `
    -Artifact "artifacts/final-release/owner-publish-authorization-input-validation.json" `
    -Record $ownerAuthorization `
    -StateProperty "validationState" `
    -RequiredState "owner-publish-authorization-input-ready-for-owner-run" `
    -OwnerAction "Owner reviews publish commands, hashes, release notes, rollback plan, and explicitly approves only an owner-run publish." `
    -RequiredBeforeClose @("owner identity", "package hashes reviewed", "publish command reviewed", "post-publish proof still required") `
    -Boundary "Authorization validation never publishes, stores tokens, proves public download, proves runtime smoke, or closes the release issue."
  New-ClosureLane `
    -Id "public-package-download-proof" `
    -Title "Public package download proof input" `
    -Artifact "artifacts/final-release/public-package-download-proof-input-validation.json" `
    -Record $publicDownload `
    -StateProperty "validationState" `
    -RequiredState "public-package-download-proof-input-ready" `
    -OwnerAction "Download managed/runtime packages from public package sources and backfill URLs, paths, and SHA256 values." `
    -RequiredBeforeClose @("public managed package URL", "public runtime package URL", "downloaded managed SHA256", "downloaded runtime SHA256") `
    -Boundary "Public download proof is not a local feed, direct nupkg, dry-run artifact, or GitHub Actions artifact substitute."
  New-ClosureLane `
    -Id "clean-external-consumer-smoke" `
    -Title "Clean external consumer smoke input" `
    -Artifact "artifacts/final-release/clean-external-consumer-smoke-input-validation.json" `
    -Record $cleanConsumerSmoke `
    -StateProperty "validationState" `
    -RequiredState "clean-external-consumer-smoke-input-ready" `
    -OwnerAction "Run a repository-external clean consumer using package references and capture stdout/stderr/runtime probe hashes." `
    -RequiredBeforeClose @("external consumer root", "no ProjectReference", "runtime-package-key smoke", "native assets copied", "exit code zero") `
    -Boundary "A sample, ProjectReference, local RestoreSources, direct nupkg, build-only run, or dependency-probe-only run cannot replace smoke proof."
  New-ClosureLane `
    -Id "post-publish-proof" `
    -Title "Post-publish proof input" `
    -Artifact "artifacts/final-release/post-publish-proof-input-validation.json" `
    -Record $postPublishProof `
    -StateProperty "validationState" `
    -RequiredState "post-publish-proof-input-ready" `
    -OwnerAction "After public publish, repeat public package download and clean external consumer smoke and validate all hashes and host metadata." `
    -RequiredBeforeClose @("HTTPS public package metadata", "downloaded public package hashes", "external consumer smoke logs", "host/runtime metadata") `
    -Boundary "Post-publish proof may become proof when ready, but it still does not automatically close the release issue."
  New-ClosureLane `
    -Id "release-issue-close-owner-decision" `
    -Title "Release issue close owner decision input" `
    -Artifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" `
    -Record $releaseCloseDecision `
    -StateProperty "validationState" `
    -RequiredState "release-issue-close-owner-decision-input-ready" `
    -OwnerAction "Owner signs final close decision only after all proof lanes and rollback review are ready." `
    -RequiredBeforeClose @("release issue URL", "owner final close decision", "rollback plan", "approved proof hashes") `
    -Boundary "Owner close decision validation is separate from publishing and cannot close the issue by itself."
  New-ClosureLane `
    -Id "strict-close-ready-convergence-dashboard" `
    -Title "Strict close ready convergence dashboard" `
    -Artifact "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json" `
    -Record $strictCloseDashboard `
    -StateProperty "validationState" `
    -RequiredState "strict-close-ready-convergence-dashboard-ready" `
    -OwnerAction "Refresh the strict close dashboard after every real owner input and proof validator passes." `
    -RequiredBeforeClose @("all strict close lanes ready", "classification audit clean", "no proof substitutes") `
    -Boundary "The dashboard summarizes readiness only; it is not package push, publish approval, runtime proof, or issue closure."
)

$blockedLanes = @($lanes | Where-Object { -not $_.ready })
$readyLanes = @($lanes | Where-Object { $_.ready })
$allSafe = $true
foreach ($lane in $lanes) {
  $allSafe = $allSafe -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "usesPublishToken" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true)
}

$bridgeState = if ($blockedLanes.Count -eq 0) { "final-public-release-closure-bridge-ready-for-owner-close-review" } else { "blocked-final-public-release-closure-real-owner-proof-required" }

$record = [pscustomobject]@{
  recordKind = "final-public-release-closure-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = $bridgeState
  laneCount = $lanes.Count
  readyLaneCount = $readyLanes.Count
  blockedLaneCount = $blockedLanes.Count
  missingArtifactCount = @($lanes | Where-Object { -not $_.artifactExists }).Count
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  allLanesSideEffectSafe = $allSafe
  closureLanes = @($lanes)
  sourceArtifacts = @($lanes | ForEach-Object { $_.artifact })
  nextOwnerActions = @($blockedLanes | ForEach-Object { [pscustomobject]@{ laneId = $_.laneId; state = $_.state; ownerAction = $_.ownerAction; requiredBeforeClose = $_.requiredBeforeClose } })
  safetyBoundary = "Final public release closure bridge only joins owner authorization, public package download proof, clean external consumer smoke, post-publish proof, release issue close owner decision, and strict close dashboard. It does not publish packages, use tokens, claim runtime proof, claim post-publish proof, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "final-public-release-closure-bridge.json"
$markdownPath = Join-Path $OutputRoot "final-public-release-closure-bridge.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = $record.closureLanes | ForEach-Object {
  "| ``$($_.laneId)`` | ``$($_.state)`` | ``$($_.ready)`` | $($_.ownerAction.Replace("|", "\|")) |"
}

$markdown = @"
# Final Public Release Closure Bridge

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| bridgeState | ``$($record.bridgeState)`` |
| laneCount | ``$($record.laneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| missingArtifactCount | ``$($record.missingArtifactCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| usesPublishToken | ``$($record.usesPublishToken)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Closure Lanes

| Lane | State | Ready | Owner Action |
|---|---:|---:|---|
$($laneRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final public release closure bridge written to $jsonPath"
Write-Host "BridgeState=$($record.bridgeState) Ready=$($record.readyLaneCount) Blocked=$($record.blockedLaneCount)"
