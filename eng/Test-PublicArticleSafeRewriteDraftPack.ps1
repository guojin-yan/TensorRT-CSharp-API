[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-safe-rewrite-draft-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSafeRewriteDraftPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$draftItems = @(Get-PropertyOrDefault -Object $record -Name "draftItems" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-safe-rewrite-draft-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "draft-items" ($draftItems.Count -eq [int](Get-PropertyOrDefault -Object $record -Name "draftItemCount" -DefaultValue -1) -and $draftItems.Count -ge 1) "blocker" "Draft pack must include rewrite suggestions for review items.")) | Out-Null
$items.Add((New-OwnerValidationItem "artifact-only" (-not [bool](Get-PropertyOrDefault -Object $record -Name "writesSourceArticles" -DefaultValue $true) -and $text.Contains("artifact-only-no-source-overwrite")) "blocker" "Draft pack must be artifact-only and must not overwrite article sources.")) | Out-Null
$items.Add((New-OwnerValidationItem "safe-language" ($text.Contains("不得写成已通过") -or $text.Contains("不得宣称已关闭") -or $text.Contains("不能作为 proof")) "blocker" "Draft suggestions must use conservative non-proof language.")) | Out-Null
$items.Add((New-OwnerValidationItem "source-gate-binding" ($text.Contains("post-publish-owner-input") -and $text.Contains("release-close-strict-closure")) "blocker" "Draft suggestions must retain proof gate bindings for currently observed blocked claims.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Draft pack must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-safe-rewrite-draft-pack-validation-ready-non-proof" } else { "blocked-public-article-safe-rewrite-draft-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-safe-rewrite-draft-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  draftItemCount = [int](Get-PropertyOrDefault -Object $record -Name "draftItemCount" -DefaultValue 0)
  affectedFileCount = [int](Get-PropertyOrDefault -Object $record -Name "affectedFileCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article safe rewrite draft pack validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-safe-rewrite-draft-pack-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-safe-rewrite-draft-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Safe Rewrite Draft Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- draftItemCount: ``$($validation.draftItemCount)``",
  "- affectedFileCount: ``$($validation.affectedFileCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleSafeRewriteDraftPackValidationState=$state FailedBlockers=$failedBlockerCount DraftItems=$($validation.draftItemCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article safe rewrite draft pack validation failed." }
