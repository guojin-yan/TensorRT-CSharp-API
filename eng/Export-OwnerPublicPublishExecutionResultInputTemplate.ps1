[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$ContractPath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($ContractPath)) { $ContractPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-contract.json" }
if (-not (Test-Path -LiteralPath $ContractPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-OwnerPublicPublishExecutionResultInputContract.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$contract = Get-Content -LiteralPath $ContractPath -Raw -Encoding utf8 | ConvertFrom-Json
$requiredFields = @((Get-OwnerPropertyOrDefault -Object $contract -Name "requiredFields" -DefaultValue @()))
$dualPackageRouteFields = @($requiredFields | Where-Object {
    @("nugetSmallBridgeCoreRoute", "githubPackagesFullRuntimeRoute") -contains [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
  })
$ownerInputFields = @($requiredFields | ForEach-Object {
    [pscustomobject]@{
      group = [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
      name = [string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")
      value = "<owner-fill-$([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "field"))>"
      valueState = "owner-input-required"
      ready = $false
      forbiddenSubstitutes = @((Get-OwnerPropertyOrDefault -Object $_ -Name "forbiddenSubstitutes" -DefaultValue @()))
    }
  })

$record = [ordered]@{
  recordKind = "owner-public-publish-execution-result-input-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  templateState = "blocked-owner-public-publish-execution-result-input-required"
  sourceContractPath = $ContractPath
  requiredFieldCount = $requiredFields.Count
  placeholderFieldCount = $ownerInputFields.Count
  dualPackageRouteCount = [int](Get-OwnerPropertyOrDefault -Object $contract -Name "dualPackageRouteCount" -DefaultValue 2)
  dualPackageRouteRequiredFieldCount = $dualPackageRouteFields.Count
  dualPackageRoutePlaceholderFieldCount = $dualPackageRouteFields.Count
  ownerInputFields = @($ownerInputFields)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner public publish execution result input template only. It contains placeholders and is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Input Template",
  "",
  "- templateState: ``$($record.templateState)``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- placeholderFieldCount: ``$($record.placeholderFieldCount)``",
  "- dualPackageRouteCount: ``$($record.dualPackageRouteCount)``",
  "- dualPackageRouteRequiredFieldCount: ``$($record.dualPackageRouteRequiredFieldCount)``",
  "- performsPublish: ``$($record.performsPublish)``",
  "- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``",
  "",
  "> $($record.boundary)"
)

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
Write-Host "PlaceholderFieldCount=$($record.placeholderFieldCount)"
