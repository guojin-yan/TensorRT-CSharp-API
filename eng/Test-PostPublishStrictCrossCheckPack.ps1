[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-strict-cross-check-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PostPublishStrictCrossCheckPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$checks = @(Get-PropertyOrDefault -Object $record -Name "crossChecks" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredChecks = @("package-page-url-owner-input-to-record", "managed-download-url-owner-input-to-validation", "runtime-download-url-owner-input-to-validation", "managed-sha256-owner-input-to-validation", "runtime-sha256-owner-input-to-validation", "clean-consumer-root-outside-repository", "no-project-reference-local-feed-direct-nupkg")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-strict-cross-check-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-by-default" (([string](Get-PropertyOrDefault -Object $record -Name "crossCheckState" -DefaultValue "")).Contains("blocked") -and [int](Get-PropertyOrDefault -Object $record -Name "failedCrossCheckCount" -DefaultValue 0) -ge 1) "blocker" "Cross-check pack must remain blocked while placeholders are present.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Cross-check pack must not publish or close release.")) | Out-Null
$items.Add((New-OwnerValidationItem "check-count" ($checks.Count -eq $requiredChecks.Count) "blocker" "Every PostPublish cross-check must be present.")) | Out-Null
foreach ($id in $requiredChecks) {
  $items.Add((New-OwnerValidationItem "check-$id" (@($checks | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and -not [bool]$_.isProof }).Count -eq 1) "blocker" "Missing or unsafe cross-check: $id")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("direct nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("never publishes", [StringComparison]::OrdinalIgnoreCase) -ge 0
$items.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Boundary must forbid fake PostPublish substitutes and publish side effects.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "post-publish-strict-cross-check-pack-validation-ready-non-proof" } else { "blocked-post-publish-strict-cross-check-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "post-publish-strict-cross-check-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  crossCheckCount = [int]$checks.Count
  failedCrossCheckCount = [int](Get-PropertyOrDefault -Object $record -Name "failedCrossCheckCount" -DefaultValue -1)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "PostPublish cross-check validation only; not proof, not publish approval, not release close approval."
}

$jsonPath = Join-Path $OutputRoot "post-publish-strict-cross-check-pack-validation.json"
$mdPath = Join-Path $OutputRoot "post-publish-strict-cross-check-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# PostPublish Strict Cross-Check Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- crossCheckCount: ``$($validation.crossCheckCount)``",
  "- failedCrossCheckCount: ``$($validation.failedCrossCheckCount)``",
  "",
  $validation.boundary
)
Write-Host "PostPublishStrictCrossCheckPackValidationState=$state FailedBlockers=$failedBlockerCount CrossChecks=$($validation.crossCheckCount) FailedCrossChecks=$($validation.failedCrossCheckCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "PostPublish strict cross-check pack validation failed." }
