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

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$fields = @((Get-OwnerPropertyOrDefault -Object $record -Name "ownerInputFields" -DefaultValue @()))
$placeholderFields = @($fields | Where-Object { -not (Test-OwnerInputValueReady -Name ([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) -Value (Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue "")) })
$dualPackageRouteFields = @($fields | Where-Object {
    @("nugetSmallBridgeCoreRoute", "githubPackagesFullRuntimeRoute") -contains [string](Get-OwnerPropertyOrDefault -Object $_ -Name "group" -DefaultValue "")
  })
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem -Id "record-kind" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-result-input-template") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "template-blocked" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "templateState" -DefaultValue "") -eq "blocked-owner-public-publish-execution-result-input-required") -Severity "blocker" -Detail "Template must remain blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "field-count" -Passed ($fields.Count -ge 100) -Severity "blocker" -Detail "Template must mirror the 100+ field contract.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "dual-package-route-fields" -Passed ([int](Get-OwnerPropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0) -eq 2 -and $dualPackageRouteFields.Count -ge 18) -Severity "blocker" -Detail "Template must mirror dual-package route fields.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "placeholders" -Passed ($placeholderFields.Count -eq $fields.Count) -Severity "action-required" -Detail "Owner must replace placeholders with real public publish evidence.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true))) -Severity "blocker" -Detail "Template must not publish or close.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validation = [ordered]@{
  recordKind = "owner-public-publish-execution-result-input-template-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { "blocked-owner-public-publish-execution-result-input-required" } else { "failed-owner-public-publish-execution-result-input-template" }
  requiredFieldCount = $fields.Count
  placeholderFieldCount = $placeholderFields.Count
  dualPackageRouteCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0)
  dualPackageRouteRequiredFieldCount = $dualPackageRouteFields.Count
  dualPackageRoutePlaceholderFieldCount = @($dualPackageRouteFields | Where-Object { -not (Test-OwnerInputValueReady -Name ([string](Get-OwnerPropertyOrDefault -Object $_ -Name "name" -DefaultValue "")) -Value (Get-OwnerPropertyOrDefault -Object $_ -Name "value" -DefaultValue "")) }).Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = 1
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Owner public publish execution result input template validation only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-input-template-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Input Template Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- requiredFieldCount: ``$($validation.requiredFieldCount)``",
  "- placeholderFieldCount: ``$($validation.placeholderFieldCount)``",
  "- dualPackageRouteRequiredFieldCount: ``$($validation.dualPackageRouteRequiredFieldCount)``",
  "- dualPackageRoutePlaceholderFieldCount: ``$($validation.dualPackageRoutePlaceholderFieldCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner public publish execution result input template validation failed." }
