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
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-OwnerPublicPublishExecutionResultCandidate.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 32
$candidateItemCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "candidateItemCount" -DefaultValue 0)
$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem -Id "record-kind" -Passed ([string](Get-OwnerPropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-result-candidate") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseReady" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-OwnerPropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Candidate must remain non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "boundary" -Passed ($raw.IndexOf("not proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("cannot close", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Candidate must state non-proof boundary.")) | Out-Null
$items.Add((New-OwnerValidationItem -Id "candidate-blocked-by-default" -Passed ($candidateItemCount -eq 0) -Severity "action-required" -Detail "No candidate should be ready before real Owner evidence is supplied.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validation = [ordered]@{
  recordKind = "owner-public-publish-execution-result-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { [string](Get-OwnerPropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "blocked-owner-public-publish-execution-result-input-required") } else { "failed-owner-public-publish-execution-result-candidate" }
  candidateState = [string](Get-OwnerPropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
  candidateItemCount = $candidateItemCount
  readyCandidateCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "readyCandidateCount" -DefaultValue 0)
  blockedCandidateCount = [int](Get-OwnerPropertyOrDefault -Object $record -Name "blockedCandidateCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = if ($candidateItemCount -eq 0) { 1 } else { 0 }
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "Owner public publish execution result candidate validation only. It is not proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate-validation.json"
$markdownPath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate-validation.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Public Publish Execution Result Candidate Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- candidateItemCount: ``$($validation.candidateItemCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner public publish execution result candidate validation failed." }

