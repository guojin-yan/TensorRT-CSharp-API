[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-one-screen-execution-manual.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerOneScreenExecutionManual.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$forbidden = @(Get-PropertyOrDefault -Object $record -Name "forbiddenNonProofSubstitutes" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredSteps = @("preflight-freeze", "owner-authorization", "public-publish-command", "package-page-and-download", "post-publish-clean-consumer", "rollback-review", "close-decision", "strict-final-verification")
$requiredSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-one-screen-execution-manual") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-non-proof" (([string](Get-PropertyOrDefault -Object $record -Name "manualState" -DefaultValue "")).Contains("blocked") -and -not [bool](Get-PropertyOrDefault -Object $record -Name "manualIsProof" -DefaultValue $true)) "blocker" "Manual must remain blocked/non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsNuGetPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Manual must never execute publish or close issue.")) | Out-Null
$items.Add((New-OwnerValidationItem "step-count" ($steps.Count -eq $requiredSteps.Count) "blocker" "Manual must include the full Owner execution order.")) | Out-Null
foreach ($id in $requiredSteps) {
  $items.Add((New-OwnerValidationItem "step-$id" (@($steps | Where-Object { [string]$_.id -eq $id -and [bool]$_.notExecutedByAutomation -and -not [bool]$_.performsPublish -and -not [bool]$_.canCloseReleaseIssue -and -not [string]::IsNullOrWhiteSpace([string]$_.validatorCommand) }).Count -eq 1) "blocker" "Missing or unsafe manual step: $id")) | Out-Null
}
foreach ($substitute in $requiredSubstitutes) {
  $items.Add((New-OwnerValidationItem "substitute-$substitute" (@($forbidden | Where-Object { [string]$_ -eq $substitute }).Count -eq 1) "blocker" "Missing forbidden substitute: $substitute")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("may show dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("never executes publish", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -ge 0
$items.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Boundary must allow only Owner-only placeholders and forbid automation publish.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-owner-one-screen-execution-manual-validation-ready-non-proof" } else { "blocked-final-owner-one-screen-execution-manual-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-owner-one-screen-execution-manual-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  stepCount = [int]$steps.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  manualIsProof = $false
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseCloseProof = $false
  boundary = "Manual validation only; not proof, not publish approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual-validation.json"
$mdPath = Join-Path $OutputRoot "final-owner-one-screen-execution-manual-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Owner One-Screen Execution Manual Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- stepCount: ``$($validation.stepCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalOwnerOneScreenExecutionManualValidationState=$state FailedBlockers=$failedBlockerCount Steps=$($validation.stepCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final Owner one-screen execution manual validation failed." }
