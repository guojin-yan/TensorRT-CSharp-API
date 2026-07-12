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
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 32
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem -Id "record-kind" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-clean-consumer-real-proof-from-owner-result") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "blocked-default" -Passed ([int](Get-OwnerPropertyOrDefault -Object $record -Name "readyProofCount" -DefaultValue 999) -eq 0) -Severity "action-required" -Detail "No post-publish proof can be ready without real Owner evidence.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true))) -Severity "blocker" -Detail "Import must remain non-proof by default.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "boundary" -Passed ($raw.IndexOf("clean consumer", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Boundary must be explicit.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validation = [ordered]@{
  recordKind = "post-publish-clean-consumer-real-proof-from-owner-result-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { "blocked-post-publish-clean-consumer-real-owner-proof-required" } else { "failed-post-publish-clean-consumer-real-proof-from-owner-result" }
  importState = [string](Get-OwnerPropertyOrDefault -Object $record -Name "importState" -DefaultValue "")
  readyProofCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "readyProofCount" -DefaultValue 0)
  blockedProofLaneCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "blockedProofLaneCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 1)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Post-publish clean consumer real proof from Owner result validation only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Post-Publish Clean Consumer Real Proof From Owner Result Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- readyProofCount: ``$($validation.readyProofCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Post-publish clean consumer real proof from Owner result validation failed." }
