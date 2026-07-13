[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-draft-boundary-scan.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleDraftBoundaryScan.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$patterns = @(Get-PropertyOrDefault -Object $record -Name "patterns" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-draft-boundary-scan") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "scan-scope" ([int](Get-PropertyOrDefault -Object $record -Name "scannedFileCount" -DefaultValue 0) -ge 1 -and [string](Get-PropertyOrDefault -Object $record -Name "articlesRoot" -DefaultValue "") -eq "docs/articles/zh-cn") "blocker" "Scan must cover docs/articles/zh-cn markdown files.")) | Out-Null
$items.Add((New-OwnerValidationItem "pattern-count" ($patterns.Count -ge 8) "blocker" "Scan must include key public publish/proof claim patterns.")) | Out-Null
$items.Add((New-OwnerValidationItem "pattern-scope" ($text.Contains("nuget-published") -and $text.Contains("public-install-verified") -and $text.Contains("clean-consumer-passed") -and $text.Contains("release-closed") -and $text.Contains("runtime-proof-complete") -and $text.Contains("local-feed-proof")) "blocker" "Scan patterns must cover NuGet, public install, clean consumer, release close, runtime proof, and local substitute claims.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Scan must not publish articles/packages or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-draft-boundary-scan-validation-ready-non-proof" } else { "blocked-public-article-draft-boundary-scan-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-draft-boundary-scan-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  scanPatternCount = [int]$patterns.Count
  scannedFileCount = [int](Get-PropertyOrDefault -Object $record -Name "scannedFileCount" -DefaultValue 0)
  blockedClaimMatchCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedClaimMatchCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article draft boundary scan validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-draft-boundary-scan-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-draft-boundary-scan-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Draft Boundary Scan Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- scanPatternCount: ``$($validation.scanPatternCount)``",
  "- scannedFileCount: ``$($validation.scannedFileCount)``",
  "- blockedClaimMatchCount: ``$($validation.blockedClaimMatchCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleDraftBoundaryScanValidationState=$state FailedBlockers=$failedBlockerCount Patterns=$($validation.scanPatternCount) BlockedClaims=$($validation.blockedClaimMatchCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article draft boundary scan validation failed." }
