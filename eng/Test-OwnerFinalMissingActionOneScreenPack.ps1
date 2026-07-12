[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-final-missing-action-one-screen-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerFinalMissingActionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$actions = @(Get-PropertyOrDefault -Object $record -Name "missingActions" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-final-missing-action-one-screen-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "missing-actions" ($actions.Count -ge 7 -and [int](Get-PropertyOrDefault -Object $record -Name "missingActionCount" -DefaultValue 0) -eq $actions.Count) "blocker" "One-screen pack must include all final Owner action lanes.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-lanes" ($text.Contains("owner-authorize-real-publish") -and $text.Contains("run-real-public-publish") -and $text.Contains("run-external-clean-consumer") -and $text.Contains("clear-article-proof-gate") -and $text.Contains("close-release-strictly")) "blocker" "One-screen pack must cover publish, clean consumer, article gate, and release close.")) | Out-Null
$items.Add((New-OwnerValidationItem "validator-bindings" ($text.Contains("Test-OwnerPublicPublishExecutionResultPreflight.ps1") -and $text.Contains("Test-PostPublishStrictCrossCheckPack.ps1") -and $text.Contains("Test-ReleaseCloseStrictEvidenceClosure.ps1")) "blocker" "One-screen pack must include concrete validator commands.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("local feed") -and $text.Contains("ProjectReference") -and $text.Contains("direct nupkg") -and $text.Contains("TensorRtExec")) "blocker" "One-screen pack must preserve forbidden substitute warnings.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "One-screen pack must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "owner-final-missing-action-one-screen-pack-validation-ready-non-proof" } else { "blocked-owner-final-missing-action-one-screen-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-final-missing-action-one-screen-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  missingActionCount = [int](Get-PropertyOrDefault -Object $record -Name "missingActionCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner final missing-action one-screen pack validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-final-missing-action-one-screen-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-final-missing-action-one-screen-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Final Missing Action One-Screen Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- missingActionCount: ``$($validation.missingActionCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerFinalMissingActionOneScreenPackValidationState=$state FailedBlockers=$failedBlockerCount MissingActions=$($validation.missingActionCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Owner final missing-action one-screen pack validation failed." }
