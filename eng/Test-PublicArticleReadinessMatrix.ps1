[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-readiness-matrix.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleReadinessMatrix.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 14
$requiredIds = @("project-overview", "api-interface-guide", "source-build-cpp", "dual-package-strategy", "yolovision-series", "onnx-to-engine-trtexec", "tensorrtexec-application", "post-publish-clean-consumer", "faq-troubleshooting")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-article-readiness-matrix") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "minimum-roadmap" ([int](Get-PropertyOrDefault -Object $record -Name "roadmapArticleCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "minimumArticleCount" -DefaultValue 0) -eq 30) "blocker" "Article roadmap must retain at least 30 planned articles.")) | Out-Null
$items.Add((New-OwnerValidationItem "lane-count" ($lanes.Count -eq $requiredIds.Count -and [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -eq $requiredIds.Count) "blocker" "Every public article lane must be blocked until proof/boundary review.")) | Out-Null
foreach ($id in $requiredIds) {
  $items.Add((New-OwnerValidationItem "lane-$id" (@($lanes | Where-Object { [string]$_.id -eq $id -and [bool]$_.ownerProofRequired -and -not [bool]$_.canPublishNow -and [int]$_.mediaRequirementCount -ge 1 -and [int]$_.codePathCount -ge 1 }).Count -eq 1) "blocker" "Missing or unsafe article readiness lane: $id")) | Out-Null
}
$items.Add((New-OwnerValidationItem "required-scope" ($text.Contains("YoloVision") -and $text.Contains("OnnxToEngine") -and $text.Contains("TensorRtExec") -and $text.Contains("NuGet") -and $text.Contains("clean consumer") -and $text.Contains("源码编译")) "blocker" "Matrix must cover corrected sample/app/package/source-build article scope.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-claims" ($text.Contains("已发布到 NuGet") -and $text.Contains("公开包已经可安装") -and $text.Contains("release closed") -and $text.Contains("dry-run 证明真实发布")) "blocker" "Matrix must block unproven public publish claims.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Matrix must not publish or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "public-article-readiness-matrix-validation-ready-non-proof" } else { "blocked-public-article-readiness-matrix-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "public-article-readiness-matrix-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  laneCount = [int]$lanes.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Article readiness validation only; not article publication, not package publication, not runtime proof, not post-publish proof, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "public-article-readiness-matrix-validation.json"
$mdPath = Join-Path $OutputRoot "public-article-readiness-matrix-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Public Article Readiness Matrix Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- laneCount: ``$($validation.laneCount)``",
  "",
  $validation.boundary
)
Write-Host "PublicArticleReadinessMatrixValidationState=$state FailedBlockers=$failedBlockerCount Lanes=$($validation.laneCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Public article readiness matrix validation failed." }
