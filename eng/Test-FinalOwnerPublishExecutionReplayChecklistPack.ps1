[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-publish-execution-replay-checklist-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$requiredIds = @("readonly-freeze", "owner-authorization-capture", "publish-command-capture", "public-package-identity", "post-publish-clean-consumer", "rollback-and-close-readiness")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-publish-execution-replay-checklist-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-by-default" (([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "")).Contains("blocked") -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerActionRequired" -DefaultValue $false)) "blocker" "Checklist must remain Owner-action blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "step-count" ($steps.Count -eq $requiredIds.Count -and [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0) -eq $requiredIds.Count) "blocker" "Every replay step must be present and blocked.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "step-$id" (@($steps | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and -not [bool]$_.performsPublish -and -not [bool]$_.isProof }).Count -eq 1) "blocker" "Missing or unsafe replay checklist step: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsNuGetPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Replay checklist must not publish or close release.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("ProjectReference") -and $text.Contains("direct nupkg") -and $text.Contains("TensorRtExec report") -and $text.Contains("queued workflow")) "blocker" "Forbidden substitutes must remain visible.")) | Out-Null
$items.Add((New-OwnerValidationItem "validator-links" ($text.Contains("Test-OwnerPublicPublishExecutionResultCandidate.ps1") -and $text.Contains("Test-PostPublishVerificationRecord.ps1") -and $text.Contains("Test-ReleaseCloseStrictEvidenceClosure.ps1")) "blocker" "Replay checklist must link publish result, PostPublish, and close validators.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-publish-execution-replay-checklist-pack-validation-ready-non-proof" } else { "blocked-final-owner-publish-execution-replay-checklist-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-publish-execution-replay-checklist-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  stepCount = [int]$steps.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Replay checklist validation only; not proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-publish-execution-replay-checklist-pack-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-publish-execution-replay-checklist-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner Publish Execution Replay Checklist Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- stepCount: ``$($validation.stepCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerPublishExecutionReplayChecklistPackValidationState=$state FailedBlockers=$failedBlockerCount Steps=$($validation.stepCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner publish execution replay checklist pack validation failed." }
