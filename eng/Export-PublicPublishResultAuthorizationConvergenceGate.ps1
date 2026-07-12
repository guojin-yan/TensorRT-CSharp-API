[CmdletBinding()]
param(
  [string]$AuthorizationValidationPath = "artifacts\final-release\owner-public-publish-authorization-input-validation.json",
  [string]$AuthorizationImportPath = "artifacts\final-release\owner-public-publish-authorization-input-import.json",
  [string]$AuthorizationGateValidationPath = "artifacts\final-release\owner-public-publish-authorization-gate-validation.json",
  [string]$PublicPublishResultValidationPath = "artifacts\final-release\public-publish-result-import-validation.json",
  [string]$PublicPublishResultImportPath = "artifacts\final-release\public-publish-result-import.json",
  [string]$PostPublishCleanConsumerValidationPath = "artifacts\final-release\post-publish-clean-consumer-real-proof-from-owner-result-validation.json",
  [string]$ReleaseCloseStrictBridgeValidationPath = "artifacts\final-release\release-close-strict-validation-bridge-validation.json",
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

function New-GateItem {
  param([string]$Id, [bool]$Ready, [string]$State, [string]$Detail)
  [pscustomobject]@{ id = $Id; ready = $Ready; state = $State; detail = $Detail }
}

$authorizationValidation = Read-JsonOrNull $AuthorizationValidationPath
$authorizationImport = Read-JsonOrNull $AuthorizationImportPath
$authorizationGateValidation = Read-JsonOrNull $AuthorizationGateValidationPath
$publicPublishResultValidation = Read-JsonOrNull $PublicPublishResultValidationPath
$publicPublishResultImport = Read-JsonOrNull $PublicPublishResultImportPath
$postPublishCleanConsumerValidation = Read-JsonOrNull $PostPublishCleanConsumerValidationPath
$releaseCloseStrictBridgeValidation = Read-JsonOrNull $ReleaseCloseStrictBridgeValidationPath

$authorizationInputReady = [string](Get-PropertyOrDefault -Object $authorizationValidation -Name "validationState" -DefaultValue "") -eq "owner-public-publish-authorization-input-ready"
$authorizationGateReady = [string](Get-PropertyOrDefault -Object $authorizationGateValidation -Name "validationState" -DefaultValue "") -eq "owner-public-publish-authorized-for-manual-execution-review"
$publicPublishResultReady = [string](Get-PropertyOrDefault -Object $publicPublishResultValidation -Name "validationState" -DefaultValue "") -eq "public-publish-result-import-ready"
$postPublishReady = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerValidation -Name "validationState" -DefaultValue "") -eq "post-publish-clean-consumer-real-owner-proof-ready"
$strictCloseReady = [string](Get-PropertyOrDefault -Object $releaseCloseStrictBridgeValidation -Name "validationState" -DefaultValue "") -eq "release-close-strict-validation-ready"

$authorizedManagedId = [string](Get-PropertyOrDefault -Object $authorizationImport -Name "managedPackageId" -DefaultValue "")
$publishedManagedId = [string](Get-PropertyOrDefault -Object $publicPublishResultImport -Name "packageId" -DefaultValue "")
$authorizedManagedHash = [string](Get-PropertyOrDefault -Object $authorizationImport -Name "managedPackageSha256" -DefaultValue "")
$publishedManagedHash = [string](Get-PropertyOrDefault -Object $publicPublishResultImport -Name "managedNupkgSha256" -DefaultValue "")
$authorizedRuntimeHash = [string](Get-PropertyOrDefault -Object $authorizationImport -Name "runtimePackageSha256" -DefaultValue "")
$publishedRuntimeHashes = @((Get-PropertyOrDefault -Object $publicPublishResultImport -Name "runtimePackageSha256" -DefaultValue @()))
$hashesMatch = -not [string]::IsNullOrWhiteSpace($authorizedManagedHash) -and $authorizedManagedHash -eq $publishedManagedHash -and ($publishedRuntimeHashes -contains $authorizedRuntimeHash)
$packageIdsMatch = -not [string]::IsNullOrWhiteSpace($authorizedManagedId) -and $authorizedManagedId -eq $publishedManagedId
$packageIdConsistencyState = if ($packageIdsMatch) { "package-id-consistency-ready" } else { "blocked-package-id-consistency-required" }
$packageHashConsistencyState = if ($hashesMatch) { "package-hash-consistency-ready" } else { "blocked-package-hash-consistency-required" }

$items = @(
  New-GateItem -Id "owner-authorization-input" -Ready $authorizationInputReady -State ([string](Get-PropertyOrDefault -Object $authorizationValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-authorization-input-validation")) -Detail "Owner authorization input must be complete before publish result can converge."
  New-GateItem -Id "owner-authorization-gate" -Ready $authorizationGateReady -State ([string](Get-PropertyOrDefault -Object $authorizationGateValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-authorization-gate-validation")) -Detail "Authorization gate must explicitly authorize manual execution review."
  New-GateItem -Id "public-publish-result-import" -Ready $publicPublishResultReady -State ([string](Get-PropertyOrDefault -Object $publicPublishResultValidation -Name "validationState" -DefaultValue "missing-public-publish-result-import-validation")) -Detail "Public publish result import must have real URL/source/hash/transcript evidence."
  New-GateItem -Id "package-id-consistency" -Ready $packageIdsMatch -State $packageIdConsistencyState -Detail "Authorized managedPackageId must match imported public publish packageId."
  New-GateItem -Id "package-hash-consistency" -Ready $hashesMatch -State $packageHashConsistencyState -Detail "Authorized package SHA256 values must match imported public publish result hashes."
  New-GateItem -Id "post-publish-clean-consumer-proof" -Ready $postPublishReady -State ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumerValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-real-proof-validation")) -Detail "PostPublish clean consumer proof must be real external proof, not local substitute."
  New-GateItem -Id "release-close-strict-bridge" -Ready $strictCloseReady -State ([string](Get-PropertyOrDefault -Object $releaseCloseStrictBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-strict-validation-bridge-validation")) -Detail "Strict bridge must be ready before release close can be eligible."
)

$blockedItems = @($items | Where-Object { -not $_.ready })
$gateState = if ($blockedItems.Count -eq 0) { "public-publish-result-authorization-convergence-ready" } else { "blocked-public-publish-result-authorization-convergence-required" }

$record = [pscustomobject]@{
  recordKind = "public-publish-result-authorization-convergence-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = $gateState
  gateItemCount = $items.Count
  readyGateItemCount = @($items | Where-Object { $_.ready }).Count
  blockedGateItemCount = $blockedItems.Count
  authorizedManagedPackageId = $authorizedManagedId
  publishedManagedPackageId = $publishedManagedId
  packageIdsMatch = $packageIdsMatch
  authorizedManagedPackageSha256 = $authorizedManagedHash
  publishedManagedPackageSha256 = $publishedManagedHash
  authorizedRuntimePackageSha256 = $authorizedRuntimeHash
  packageHashesMatch = $hashesMatch
  gateItems = @($items)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  eligibleForOwnerCloseInstruction = $false
  safetyBoundary = "Convergence gate is a strict status aggregator only. It does not publish, dispatch workflows, close releases, or turn local/dry-run/dashboard evidence into proof."
}

$jsonPath = Join-Path $OutputRoot "public-publish-result-authorization-convergence-gate.json"
$markdownPath = Join-Path $OutputRoot "public-publish-result-authorization-convergence-gate.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# Public Publish Result Authorization Convergence Gate

| Item | Value |
|---|---|
| gateState | ``$($record.gateState)`` |
| readyGateItemCount | ``$($record.readyGateItemCount)`` |
| blockedGateItemCount | ``$($record.blockedGateItemCount)`` |
| packageIdsMatch | ``$($record.packageIdsMatch)`` |
| packageHashesMatch | ``$($record.packageHashesMatch)`` |
| eligibleForOwnerCloseInstruction | ``$($record.eligibleForOwnerCloseInstruction)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Gate Items

| ID | Ready | State | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish result authorization convergence gate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "GateState=$($record.gateState) Ready=$($record.readyGateItemCount) Blocked=$($record.blockedGateItemCount)"
