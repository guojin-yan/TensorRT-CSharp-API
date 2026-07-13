[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\github-actions-runner-non-proof-guard-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-GitHubActionsRunnerNonProofGuardPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$signals = @(Get-PropertyOrDefault -Object $record -Name "nonProofSignals" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "github-actions-runner-non-proof-guard-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "signal-count" ($signals.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "signalCount" -DefaultValue 0) -eq $signals.Count) "blocker" "Guard must include key GitHub Actions and runner non-proof signals.")) | Out-Null
$items.Add((New-OwnerValidationItem "signal-scope" ($text.Contains("queued workflow") -and $text.Contains("manual approval") -and $text.Contains("missing self-hosted runner") -and $text.Contains("TensorRtExec report") -and $text.Contains("local feed")) "blocker" "Guard must cover queued workflows, manual approval, missing runners, TensorRtExec, and local feed substitutes.")) | Out-Null
$items.Add((New-OwnerValidationItem "proof-boundary" ($text.Contains("Public package source evidence") -and $text.Contains("external clean consumer") -and $text.Contains("runtime smoke")) "blocker" "Guard must point to real proof requirements.")) | Out-Null
$items.Add((New-OwnerValidationItem "destructive-actions-forbidden" ($text.Contains("deletes") -and $text.Contains("delists") -and $text.Contains("withdraws") -and $text.Contains("deprecates")) "blocker" "Boundary must forbid destructive package actions.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Guard must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "github-actions-runner-non-proof-guard-pack-validation-ready-non-proof" } else { "blocked-github-actions-runner-non-proof-guard-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "github-actions-runner-non-proof-guard-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  signalCount = [int](Get-PropertyOrDefault -Object $record -Name "signalCount" -DefaultValue 0)
  detectedArtifactFileCount = [int](Get-PropertyOrDefault -Object $record -Name "detectedArtifactFileCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "GitHub Actions runner non-proof guard validation only; not workflow dispatch, not package publication, not article publication, not destructive package action, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "github-actions-runner-non-proof-guard-pack-validation.json"
$mdPath = Join-Path $OutputRoot "github-actions-runner-non-proof-guard-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# GitHub Actions Runner Non-Proof Guard Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- signalCount: ``$($validation.signalCount)``",
  "- detectedArtifactFileCount: ``$($validation.detectedArtifactFileCount)``",
  "",
  $validation.boundary
)
Write-Host "GitHubActionsRunnerNonProofGuardPackValidationState=$state FailedBlockers=$failedBlockerCount Signals=$($validation.signalCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "GitHub Actions runner non-proof guard validation failed." }
