[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-safe-rewrite-draft-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSafeRewriteDraftPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

function Get-PatchRisk {
  param([string]$RecommendedAction)
  switch ($RecommendedAction) {
    "remove" { return "medium-remove-or-reword-proof-claim" }
    "downgrade-to-roadmap" { return "low-roadmap-rewording-only" }
    "wait-for-real-proof" { return "medium-wait-for-owner-proof" }
    "bind-to-evidence-field" { return "medium-bind-to-real-proof-gate" }
    default { return "medium-owner-review-required" }
  }
}

function Get-ProposedReplacement {
  param([string]$RecommendedAction, [string]$SafeRewriteZh)
  switch ($RecommendedAction) {
    "remove" { return "删除或改写该证明语气；建议替换为：$SafeRewriteZh" }
    "downgrade-to-roadmap" { return "降级为路线图/待验证描述；建议替换为：$SafeRewriteZh" }
    "wait-for-real-proof" { return "等待真实 Owner proof 后再恢复宣称；建议替换为：$SafeRewriteZh" }
    "bind-to-evidence-field" { return "绑定真实 evidence field 后再发布；建议替换为：$SafeRewriteZh" }
    default { return $SafeRewriteZh }
  }
}

$draft = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$draftItems = @(Get-PropertyOrDefault -Object $draft -Name "draftItems" -DefaultValue @())
$proposals = @($draftItems | ForEach-Object {
  $action = [string](Get-PropertyOrDefault -Object $_ -Name "recommendedAction" -DefaultValue "wait-for-real-proof")
  $safeRewrite = [string](Get-PropertyOrDefault -Object $_ -Name "safeRewriteZh" -DefaultValue "")
  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $_ -Name "order" -DefaultValue 0)
    path = [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "")
    line = [int](Get-PropertyOrDefault -Object $_ -Name "line" -DefaultValue 0)
    patternId = [string](Get-PropertyOrDefault -Object $_ -Name "patternId" -DefaultValue "unknown")
    originalMatchedText = [string](Get-PropertyOrDefault -Object $_ -Name "originalMatchedText" -DefaultValue "")
    safeRewriteZh = $safeRewrite
    proposedReplacementZh = Get-ProposedReplacement -RecommendedAction $action -SafeRewriteZh $safeRewrite
    recommendedAction = $action
    requiredProofFieldOrGate = [string](Get-PropertyOrDefault -Object $_ -Name "requiredProofFieldOrGate" -DefaultValue "owner-real-proof-required")
    proofBoundary = "证据导入前，该 proposal 只能作为 Owner 审阅草稿，不能声明已发布、已通过、已关闭或 proof complete。"
    patchRisk = Get-PatchRisk -RecommendedAction $action
    patchMode = "artifact-only-proposal"
    appliesToSourceFile = $false
    modifiesSourceArticle = $false
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Patch proposal item only. It does not change source markdown and must be reviewed before any source edit."
  }
})

$riskGroups = @($proposals | Group-Object patchRisk | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{
    patchRisk = $_.Name
    count = $_.Count
    outputPolicy = "artifact-only-no-source-overwrite"
  }
})

$record = [pscustomobject]@{
  recordKind = "public-article-source-patch-proposal-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  proposalState = if ($proposals.Count -gt 0) { "blocked-public-article-source-patch-proposal-owner-review-required" } else { "public-article-source-patch-proposal-clean-non-proof" }
  sourceDraftPackPath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $InputPath).Replace("\", "/")
  proposalCount = $proposals.Count
  affectedFileCount = @($proposals | Group-Object path).Count
  riskGroupCount = $riskGroups.Count
  riskGroups = @($riskGroups)
  proposals = @($proposals)
  ownerActionRequired = $proposals.Count -gt 0
  writesSourceArticles = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article source patch proposal pack only. It writes artifact-only patch proposals and never overwrites source articles, publishes articles/packages, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "public-article-source-patch-proposal-pack.json"
$mdPath = Join-Path $OutputRoot "public-article-source-patch-proposal-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Source Patch Proposal Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- proposalState: ``$($record.proposalState)``") | Out-Null
$md.Add("- proposalCount: ``$($record.proposalCount)``") | Out-Null
$md.Add("- affectedFileCount: ``$($record.affectedFileCount)``") | Out-Null
$md.Add("- writesSourceArticles: ``$($record.writesSourceArticles)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Risk | Count |") | Out-Null
$md.Add("| --- | ---: |") | Out-Null
foreach ($group in $riskGroups) {
  $md.Add("| ``$($group.patchRisk)`` | $($group.count) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("| # | Pattern | File | Line | Action | Risk | Required Proof Field / Gate |") | Out-Null
$md.Add("| ---: | --- | --- | ---: | --- | --- | --- |") | Out-Null
foreach ($proposal in @($proposals | Select-Object -First 120)) {
  $md.Add("| $($proposal.order) | ``$($proposal.patternId)`` | ``$(ConvertTo-MarkdownCell $proposal.path)`` | $($proposal.line) | ``$($proposal.recommendedAction)`` | ``$($proposal.patchRisk)`` | $(ConvertTo-MarkdownCell $proposal.requiredProofFieldOrGate) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleSourcePatchProposalPackState=$($record.proposalState) Proposals=$($record.proposalCount) Files=$($record.affectedFileCount)"
