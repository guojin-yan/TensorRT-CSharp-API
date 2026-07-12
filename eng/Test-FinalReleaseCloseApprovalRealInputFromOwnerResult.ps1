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
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "final-release-close-approval-real-input-from-owner-result.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 32
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem -Id "record-kind" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-approval-real-input-from-owner-result") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "blocked-default" -Passed ([int](Get-OwnerPropertyOrDefault -Object $record -Name "readyCloseApprovalCount" -DefaultValue 999) -eq 0) -Severity "action-required" -Detail "No release close approval is ready without real Owner approval.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "closesReleaseIssue" -DefaultValue $true))) -Severity "blocker" -Detail "Close approval import must remain blocked and non-proof by default.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "boundary" -Passed ($raw.IndexOf("not release close approval", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("cannot close", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Boundary must be explicit.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validation = [ordered]@{
  recordKind = "final-release-close-approval-real-input-from-owner-result-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-release-close-owner-approval-real-input-required" } else { "failed-final-release-close-approval-real-input-from-owner-result" }
  importState = [string](Get-OwnerPropertyOrDefault -Object $record -Name "importState" -DefaultValue "")
  readyCloseApprovalCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "readyCloseApprovalCount" -DefaultValue 0)
  blockedApprovalFieldCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "blockedApprovalFieldCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 1)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  closesReleaseIssue = $false
  validationItems = @($items.ToArray())
  boundary = "Final release close approval real input from Owner result validation only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "final-release-close-approval-real-input-from-owner-result-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "final-release-close-approval-real-input-from-owner-result-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Release Close Approval Real Input From Owner Result Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- readyCloseApprovalCount: ``$($validation.readyCloseApprovalCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Final release close approval real input from Owner result validation failed." }
