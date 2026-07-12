[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-article-proof-gate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PostPublishArticleProofGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$gates = @(Get-PropertyOrDefault -Object $record -Name "claimGates" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$requiredIds = @("nuget-published", "github-packages-published", "clean-consumer-verified", "release-closed", "runtime-proof")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-article-proof-gate") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-by-default" (([string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "")).Contains("blocked") -and [int](Get-PropertyOrDefault -Object $record -Name "blockedClaimGateCount" -DefaultValue 0) -eq $requiredIds.Count) "blocker" "Every article proof claim must remain blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "claim-count" ($gates.Count -eq $requiredIds.Count) "blocker" "Every protected public article claim gate must be present.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "claim-$id" (@($gates | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerActionRequired -and -not [bool]$_.passed -and -not [bool]$_.isProof }).Count -eq 1) "blocker" "Missing or unsafe article proof claim gate: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("ProjectReference") -and $text.Contains("direct .nupkg") -and $text.Contains("dry-run") -and $text.Contains("TensorRtExec report")) "blocker" "Forbidden substitutes must remain blocked.")) | Out-Null
$items.Add((New-OwnerValidationItem "post-publish-claims" ($text.Contains("NuGet") -and $text.Contains("GitHub Packages") -and $text.Contains("clean consumer") -and $text.Contains("release") -and $text.Contains("runtime proof")) "blocker" "Gate must cover NuGet, GitHub Packages, clean consumer, release close, and runtime proof claims.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Article proof gate must not publish or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "post-publish-article-proof-gate-validation-ready-non-proof" } else { "blocked-post-publish-article-proof-gate-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "post-publish-article-proof-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  claimGateCount = [int]$gates.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "PostPublish article proof gate validation only; not article publication, not package publication, not runtime proof, not post-publish proof, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "post-publish-article-proof-gate-validation.json"
$mdPath = Join-Path $OutputRoot "post-publish-article-proof-gate-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# PostPublish Article Proof Gate Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- claimGateCount: ``$($validation.claimGateCount)``",
  "",
  $validation.boundary
)
Write-Host "PostPublishArticleProofGateValidationState=$state FailedBlockers=$failedBlockerCount Claims=$($validation.claimGateCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "PostPublish article proof gate validation failed." }
