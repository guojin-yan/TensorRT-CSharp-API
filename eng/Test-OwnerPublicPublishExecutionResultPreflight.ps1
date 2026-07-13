[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$InputPath,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$template = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$fields = @((Get-OwnerPropertyOrDefault -Object $template -Name "ownerInputFields" -DefaultValue @()))
$readyFields = @($fields | Where-Object { Test-OwnerInputValueReady -Name ([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) -Value (Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue "") })
$blockedFields = @($fields | Where-Object { -not (Test-OwnerInputValueReady -Name ([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) -Value (Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue "")) })
$dualPackageRouteFields = @($fields | Where-Object {
    @("nugetSmallBridgeCoreRoute", "githubPackagesFullRuntimeRoute") -contains [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
  })
$dualPackageRouteReadyFields = @($dualPackageRouteFields | Where-Object { Test-OwnerInputValueReady -Name ([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) -Value (Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue "") })
$readyCandidateCount = if ($fields.Count -ge 100 -and $blockedFields.Count -eq 0) { 1 } else { 0 }
$failedBlockerCount = if ($fields.Count -ge 100) { 0 } else { 1 }

$record = [ordered]@{
  recordKind = "owner-public-publish-execution-result-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = if ($failedBlockerCount -eq 0) { "blocked-owner-public-publish-execution-result-input-required" } else { "failed-owner-public-publish-execution-result-preflight" }
  sourceInputPath = $InputPath
  requiredFieldCount = $fields.Count
  readyFieldCount = $readyFields.Count
  blockedRequiredFieldCount = $blockedFields.Count
  dualPackageRouteCount = [int](Get-OwnerPropertyOrDefault -Object $template -Name "dualPackageRouteCount" -DefaultValue 0)
  dualPackageRouteRequiredFieldCount = $dualPackageRouteFields.Count
  dualPackageRouteReadyFieldCount = $dualPackageRouteReadyFields.Count
  dualPackageRouteBlockedFieldCount = $dualPackageRouteFields.Count - $dualPackageRouteReadyFields.Count
  readyCandidateCount = $readyCandidateCount
  blockedCandidateCount = if ($readyCandidateCount -eq 0) { 1 } else { 0 }
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = if ($readyCandidateCount -eq 0) { [Math]::Max(1, $blockedFields.Count) } else { 0 }
  preflightResults = @($fields | ForEach-Object {
      $name = [string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")
      $value = Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue ""
      [pscustomobject]@{
        group = [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
        name = $name
        ready = Test-OwnerInputValueReady -Name $name -Value $value
        valueState = if (Test-OwnerInputValueReady -Name $name -Value $value) { "owner-input-ready-for-candidate" } else { "owner-input-required" }
      }
    })
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner public publish execution result preflight only. It screens Owner input and is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-preflight.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-preflight.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Preflight",
  "",
  "- preflightState: ``$($record.preflightState)``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- dualPackageRouteRequiredFieldCount: ``$($record.dualPackageRouteRequiredFieldCount)``",
  "- dualPackageRouteReadyFieldCount: ``$($record.dualPackageRouteReadyFieldCount)``",
  "- dualPackageRouteBlockedFieldCount: ``$($record.dualPackageRouteBlockedFieldCount)``",
  "- readyFieldCount: ``$($record.readyFieldCount)``",
  "- blockedRequiredFieldCount: ``$($record.blockedRequiredFieldCount)``",
  "- readyCandidateCount: ``$($record.readyCandidateCount)``",
  "- failedBlockerCount: ``$($record.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($record.failedActionRequiredCount)``",
  "",
  "> $($record.boundary)"
)

Write-Host "PreflightState=$($record.preflightState)"
Write-Host "ReadyCandidateCount=$($record.readyCandidateCount)"
Write-Host "FailedBlockerCount=$($record.failedBlockerCount)"
if ($Strict -and $failedBlockerCount -gt 0) { throw "Owner public publish execution result preflight failed." }
