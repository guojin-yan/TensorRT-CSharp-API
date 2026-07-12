[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-source-patch-proposal-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSourcePatchProposalPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$proposals = @(Get-PropertyOrDefault -Object $record -Name "proposals" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-source-patch-proposal-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "proposal-count" ($proposals.Count -ge 64 -and [int](Get-PropertyOrDefault -Object $record -Name "proposalCount" -DefaultValue 0) -eq $proposals.Count) "blocker" "Patch proposal pack must include the blocked claim proposals.")) | Out-Null
$items.Add((New-OwnerValidationItem "artifact-only" (-not [bool](Get-PropertyOrDefault -Object $record -Name "writesSourceArticles" -DefaultValue $true) -and $text.Contains("artifact-only-no-source-overwrite") -and $text.Contains("artifact-only-proposal")) "blocker" "Patch proposal pack must be artifact-only and must not overwrite article sources.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-fields" ($text.Contains("originalMatchedText") -and $text.Contains("safeRewriteZh") -and $text.Contains("requiredProofFieldOrGate") -and $text.Contains("patchRisk")) "blocker" "Patch proposals must include required review fields.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-language" ($text.Contains("不能作为 proof") -and $text.Contains("不得宣称已关闭") -and $text.Contains("证据导入前")) "blocker" "Patch proposals must retain conservative non-proof language.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitute-boundary" ($text.Contains("local feed") -and $text.Contains("ProjectReference") -and $text.Contains("direct nupkg") -and $text.Contains("TensorRtExec")) "blocker" "Patch proposals must retain forbidden substitute boundaries.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Patch proposal pack must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-source-patch-proposal-pack-validation-ready-non-proof" } else { "blocked-public-article-source-patch-proposal-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-source-patch-proposal-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  proposalCount = [int](Get-PropertyOrDefault -Object $record -Name "proposalCount" -DefaultValue 0)
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
  boundary = "Public article source patch proposal pack validation only; not article publication, not package publication, not proof promotion, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-source-patch-proposal-pack-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-source-patch-proposal-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Source Patch Proposal Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- proposalCount: ``$($validation.proposalCount)``",
  "- affectedFileCount: ``$($validation.affectedFileCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleSourcePatchProposalPackValidationState=$state FailedBlockers=$failedBlockerCount Proposals=$($validation.proposalCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article source patch proposal pack validation failed." }
