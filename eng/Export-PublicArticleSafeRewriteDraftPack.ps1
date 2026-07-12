[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-blocked-claim-owner-review-list.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleBlockedClaimOwnerReviewList.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

function Get-SafeRewrite {
  param([string]$PatternId, [string]$MatchedText, [string]$RequiredProofFieldOrGate)
  switch ($PatternId) {
    "nuget-published" { return "发布状态仍需 Owner 提供真实公开包页面、下载地址、哈希和发布转录证据；在证据导入前仅能描述为待发布/待验证。" }
    "public-install-verified" { return "公开渠道安装验证仍待真实公开源 clean consumer 记录确认；在证据导入前不得写成已通过。" }
    "clean-consumer-passed" { return "clean consumer 验证必须来自仓库外部且只使用公开源；当前只能记录为待 Owner 执行与导入证据。" }
    "release-closed" { return "release close 仍受 strict closure gate 阻断；在 Owner close decision 和真实证据通过前不得宣称已关闭。" }
    "runtime-proof-complete" { return "runtime proof 仍需真实运行日志和严格校验记录；当前仅能描述为 proof 收集项。" }
    "dry-run-as-proof" { return "dry-run、dashboard、manual approval 或 queued workflow 不能作为 proof；请删除该证明语气并改为真实证据要求。" }
    "local-feed-proof" { return "local feed、ProjectReference 或 direct nupkg 不能作为公开包 consumer proof；请改为公开源 clean consumer 证据要求。" }
    "tensorrtexec-proof" { return "TensorRtExec 报告只能作为 sidecar 参考，不能替代 runtime smoke proof；请改为等待真实 runtime smoke 证据。" }
    default { return "该宣称需要绑定真实 proof gate：$RequiredProofFieldOrGate；证据导入前不得写成已完成。" }
  }
}

$review = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$reviewItems = @(Get-PropertyOrDefault -Object $review -Name "reviewItems" -DefaultValue @())
$draftItems = @($reviewItems | ForEach-Object {
  $patternId = [string](Get-PropertyOrDefault -Object $_ -Name "patternId" -DefaultValue "unknown")
  $requiredGate = [string](Get-PropertyOrDefault -Object $_ -Name "requiredProofFieldOrGate" -DefaultValue "owner-real-proof-required")
  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $_ -Name "order" -DefaultValue 0)
    path = [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "")
    line = [int](Get-PropertyOrDefault -Object $_ -Name "line" -DefaultValue 0)
    patternId = $patternId
    originalMatchedText = [string](Get-PropertyOrDefault -Object $_ -Name "matchedText" -DefaultValue "")
    recommendedAction = [string](Get-PropertyOrDefault -Object $_ -Name "recommendedAction" -DefaultValue "wait-for-real-proof")
    requiredProofFieldOrGate = $requiredGate
    safeRewriteZh = Get-SafeRewrite -PatternId $patternId -MatchedText ([string](Get-PropertyOrDefault -Object $_ -Name "matchedText" -DefaultValue "")) -RequiredProofFieldOrGate $requiredGate
    draftOnly = $true
    modifiesSourceArticle = $false
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Safe rewrite draft item only. It proposes conservative text and does not modify the markdown source file."
  }
})

$files = @($draftItems | Group-Object path | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{
    path = $_.Name
    draftItemCount = $_.Count
    outputPolicy = "artifact-only-no-source-overwrite"
  }
})

$record = [pscustomobject]@{
  recordKind = "public-article-safe-rewrite-draft-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  draftState = if ($draftItems.Count -gt 0) { "blocked-public-article-safe-rewrite-draft-owner-review-required" } else { "public-article-safe-rewrite-draft-clean-non-proof" }
  sourceReviewListPath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $InputPath).Replace("\", "/")
  draftItemCount = $draftItems.Count
  affectedFileCount = $files.Count
  affectedFiles = @($files)
  draftItems = @($draftItems)
  ownerActionRequired = $draftItems.Count -gt 0
  writesSourceArticles = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article safe rewrite draft pack only. It writes artifact-only conservative rewrite suggestions and never overwrites source articles, publishes articles/packages, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "public-article-safe-rewrite-draft-pack.json"
$mdPath = Join-Path $OutputRoot "public-article-safe-rewrite-draft-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Safe Rewrite Draft Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- draftState: ``$($record.draftState)``") | Out-Null
$md.Add("- draftItemCount: ``$($record.draftItemCount)``") | Out-Null
$md.Add("- affectedFileCount: ``$($record.affectedFileCount)``") | Out-Null
$md.Add("- writesSourceArticles: ``$($record.writesSourceArticles)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Pattern | File | Line | Action | Safe Rewrite Suggestion |") | Out-Null
$md.Add("| ---: | --- | --- | ---: | --- | --- |") | Out-Null
foreach ($item in @($draftItems | Select-Object -First 120)) {
  $md.Add("| $($item.order) | ``$($item.patternId)`` | ``$(ConvertTo-MarkdownCell $item.path)`` | $($item.line) | ``$($item.recommendedAction)`` | $(ConvertTo-MarkdownCell $item.safeRewriteZh) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleSafeRewriteDraftPackState=$($record.draftState) DraftItems=$($record.draftItemCount) Files=$($record.affectedFileCount)"
