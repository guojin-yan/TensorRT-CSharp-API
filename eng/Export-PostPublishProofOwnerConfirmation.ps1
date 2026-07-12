[CmdletBinding()]
param(
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ConfirmationGate {
  param([string]$Id, [string]$State, [string]$RequiredState, [string]$Action, [string]$SourceArtifact)
  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    gateId = $Id
    state = $State
    requiredState = $RequiredState
    gateState = if ($ready) { "ready" } else { "blocked-post-publish-proof-owner-confirmation-required" }
    ready = $ready
    requiredAction = $Action
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$publicPackageValidation = Read-JsonOrNull "artifacts\final-release\public-package-proof-owner-input-validation.json"
$postPublishOwnerValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$ownerResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$releaseCloseOwnerBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-owner-input-bridge-validation.json"

$gates = @(
  New-ConfirmationGate -Id "public-package-proof-owner-input" -State ([string](Get-PropertyOrDefault -Object $publicPackageValidation -Name "validationState" -DefaultValue "missing-public-package-proof-owner-input-validation")) -RequiredState "public-package-proof-owner-input-ready" -Action "Owner must confirm public package source, URLs, nupkg hashes, registry and review metadata." -SourceArtifact "artifacts/final-release/public-package-proof-owner-input-validation.json"
  New-ConfirmationGate -Id "post-publish-verification-owner-input" -State ([string](Get-PropertyOrDefault -Object $postPublishOwnerValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")) -RequiredState "owner-input-ready-for-record-projection" -Action "Owner must fill clean consumer, commands, logs, hashes and host metadata." -SourceArtifact "artifacts/final-release/post-publish-verification-owner-input-validation.json"
  New-ConfirmationGate -Id "post-publish-verification-record" -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")) -RequiredState "post-publish-verification-ready" -Action "Strict post-publish record validator must promote real public-channel proof." -SourceArtifact "artifacts/final-release/post-publish-verification-validation.json"
  New-ConfirmationGate -Id "owner-external-proof-result-import" -State ([string](Get-PropertyOrDefault -Object $ownerResultImportValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-result-import-validation")) -RequiredState "owner-external-proof-execution-result-ready" -Action "Owner external proof result import must contain real files, hashes, metadata and reviewer fields." -SourceArtifact "artifacts/final-release/owner-external-proof-execution-result-import-validation.json"
  New-ConfirmationGate -Id "release-close-owner-input-bridge" -State ([string](Get-PropertyOrDefault -Object $releaseCloseOwnerBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-owner-input-bridge-validation")) -RequiredState "release-close-owner-input-ready" -Action "Release close owner gate bridge must be ready after real proof and final owner decisions." -SourceArtifact "artifacts/final-release/release-close-owner-input-bridge-validation.json"
)

$blocked = @($gates | Where-Object { -not [bool]$_.ready })
$ready = @($gates | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "post-publish-proof-owner-confirmation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  confirmationState = if ($blocked.Count -eq 0) { "post-publish-proof-owner-confirmation-ready" } else { "blocked-post-publish-proof-owner-confirmation-required" }
  confirmationGateCount = $gates.Count
  blockedConfirmationGateCount = $blocked.Count
  readyConfirmationGateCount = $ready.Count
  confirmationGates = $gates
  sourceArtifacts = @(
    "artifacts/final-release/public-package-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/release-close-owner-input-bridge-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Post-publish proof owner confirmation is blocked owner gate aggregation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "post-publish-proof-owner-confirmation.json"
$markdownPath = Join-Path $artifactRoot "post-publish-proof-owner-confirmation.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.confirmationGates | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.gateId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.requiredAction) |"
}

$markdown = @"
# Post-Publish Proof Owner Confirmation

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| confirmationState | ``$($record.confirmationState)`` |
| confirmationGateCount | ``$($record.confirmationGateCount)`` |
| blockedConfirmationGateCount | ``$($record.blockedConfirmationGateCount)`` |
| readyConfirmationGateCount | ``$($record.readyConfirmationGateCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Gates

| Gate | State | Required State | Ready | Owner Action |
|---|---|---|---:|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish proof owner confirmation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ConfirmationState=$($record.confirmationState) Gates=$($record.confirmationGateCount) Blocked=$($record.blockedConfirmationGateCount) Ready=$($record.readyConfirmationGateCount)"
