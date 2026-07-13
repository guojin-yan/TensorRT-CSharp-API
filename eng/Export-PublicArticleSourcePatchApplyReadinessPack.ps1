[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-source-patch-proposal-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSourcePatchProposalPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

function Test-LineContains {
  param([string]$LineText, [string]$Needle)
  if ([string]::IsNullOrWhiteSpace($Needle)) { return $false }
  return $LineText.Contains($Needle, [System.StringComparison]::Ordinal)
}

function Get-NearbyMatch {
  param([string[]]$Lines, [int]$LineNumber, [string]$Needle)
  if ([string]::IsNullOrWhiteSpace($Needle) -or $Lines.Count -eq 0) {
    return [pscustomobject]@{ found = $false; line = 0 }
  }

  $start = [Math]::Max(1, $LineNumber - 3)
  $end = [Math]::Min($Lines.Count, $LineNumber + 3)
  for ($i = $start; $i -le $end; $i++) {
    if ($Lines[$i - 1].Contains($Needle, [System.StringComparison]::Ordinal)) {
      return [pscustomobject]@{ found = $true; line = $i }
    }
  }

  return [pscustomobject]@{ found = $false; line = 0 }
}

$proposalPack = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$proposals = @(Get-PropertyOrDefault -Object $proposalPack -Name "proposals" -DefaultValue @())
$readinessItems = @($proposals | ForEach-Object {
  $relativePath = [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "")
  $lineNumber = [int](Get-PropertyOrDefault -Object $_ -Name "line" -DefaultValue 0)
  $matchedText = [string](Get-PropertyOrDefault -Object $_ -Name "originalMatchedText" -DefaultValue "")
  $fullPath = Join-Path $RepositoryRoot $relativePath
  $fileExists = Test-Path -LiteralPath $fullPath -PathType Leaf
  $lineExists = $false
  $matchedTextStillPresent = $false
  $nearbyMatchFound = $false
  $nearbyMatchLine = 0
  $currentLinePreview = ""

  if ($fileExists) {
    $lines = @(Get-Content -LiteralPath $fullPath -Encoding utf8)
    $lineExists = $lineNumber -ge 1 -and $lineNumber -le $lines.Count
    if ($lineExists) {
      $currentLinePreview = $lines[$lineNumber - 1]
      $matchedTextStillPresent = Test-LineContains -LineText $currentLinePreview -Needle $matchedText
    }
    $nearby = Get-NearbyMatch -Lines $lines -LineNumber $lineNumber -Needle $matchedText
    $nearbyMatchFound = [bool]$nearby.found
    $nearbyMatchLine = [int]$nearby.line
  }

  $canMapProposal = $fileExists -and $lineExists -and ($matchedTextStillPresent -or $nearbyMatchFound)
  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $_ -Name "order" -DefaultValue 0)
    path = $relativePath
    line = $lineNumber
    patternId = [string](Get-PropertyOrDefault -Object $_ -Name "patternId" -DefaultValue "unknown")
    lineExists = $lineExists
    fileExists = $fileExists
    matchedTextStillPresent = $matchedTextStillPresent
    nearbyMatchFound = $nearbyMatchFound
    nearbyMatchLine = $nearbyMatchLine
    originalMatchedText = $matchedText
    currentLinePreview = if ($currentLinePreview.Length -gt 240) { $currentLinePreview.Substring(0, 240) } else { $currentLinePreview }
    safeRewriteZh = [string](Get-PropertyOrDefault -Object $_ -Name "safeRewriteZh" -DefaultValue "")
    proposedReplacementZh = [string](Get-PropertyOrDefault -Object $_ -Name "proposedReplacementZh" -DefaultValue "")
    recommendedAction = [string](Get-PropertyOrDefault -Object $_ -Name "recommendedAction" -DefaultValue "wait-for-real-proof")
    patchRisk = [string](Get-PropertyOrDefault -Object $_ -Name "patchRisk" -DefaultValue "medium-owner-review-required")
    requiredProofFieldOrGate = [string](Get-PropertyOrDefault -Object $_ -Name "requiredProofFieldOrGate" -DefaultValue "owner-real-proof-required")
    readinessState = if ($canMapProposal) { "source-line-mapped-ready-after-owner-approval" } else { "owner-review-required-source-line-not-mapped" }
    canApplyAfterOwnerApproval = $canMapProposal
    appliesPatchNow = $false
    modifiesSourceArticle = $false
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    proofBoundary = "该 readiness 只验证 proposal 与当前 source line 的映射；证据导入前不能声明已发布、已通过、已关闭或 proof complete。"
    boundary = "Source patch apply readiness item only. It never writes source files and only reports whether a proposal can be applied after Owner approval."
  }
})

$mappedItems = @($readinessItems | Where-Object { [bool]$_.canApplyAfterOwnerApproval })
$ownerReviewItems = @($readinessItems | Where-Object { -not [bool]$_.canApplyAfterOwnerApproval })
$stateGroups = @($readinessItems | Group-Object readinessState | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{
    readinessState = $_.Name
    count = $_.Count
    outputPolicy = "artifact-only-no-source-overwrite"
  }
})

$record = [pscustomobject]@{
  recordKind = "public-article-source-patch-apply-readiness-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  readinessPackState = if ($ownerReviewItems.Count -gt 0) { "blocked-public-article-source-patch-apply-readiness-owner-review-required" } else { "public-article-source-patch-apply-readiness-ready-after-owner-approval-non-proof" }
  sourceProposalPackPath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $InputPath).Replace("\", "/")
  readinessItemCount = $readinessItems.Count
  mappedItemCount = $mappedItems.Count
  ownerReviewRequiredItemCount = $ownerReviewItems.Count
  affectedFileCount = @($readinessItems | Group-Object path).Count
  stateGroupCount = $stateGroups.Count
  stateGroups = @($stateGroups)
  readinessItems = @($readinessItems)
  ownerActionRequired = $true
  writesSourceArticles = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Public article source patch apply readiness pack only. It validates source-line mapping for patch proposals and never overwrites source articles, publishes articles/packages, promotes proof, or closes release issues."
}

$jsonPath = Join-Path $OutputRoot "public-article-source-patch-apply-readiness-pack.json"
$mdPath = Join-Path $OutputRoot "public-article-source-patch-apply-readiness-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Source Patch Apply Readiness Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- readinessPackState: ``$($record.readinessPackState)``") | Out-Null
$md.Add("- readinessItemCount: ``$($record.readinessItemCount)``") | Out-Null
$md.Add("- mappedItemCount: ``$($record.mappedItemCount)``") | Out-Null
$md.Add("- ownerReviewRequiredItemCount: ``$($record.ownerReviewRequiredItemCount)``") | Out-Null
$md.Add("- writesSourceArticles: ``$($record.writesSourceArticles)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| State | Count |") | Out-Null
$md.Add("| --- | ---: |") | Out-Null
foreach ($group in $stateGroups) {
  $md.Add("| ``$($group.readinessState)`` | $($group.count) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("| # | Pattern | File | Line | Mapped | Risk | Required Proof Field / Gate |") | Out-Null
$md.Add("| ---: | --- | --- | ---: | --- | --- | --- |") | Out-Null
foreach ($item in @($readinessItems | Select-Object -First 120)) {
  $md.Add("| $($item.order) | ``$($item.patternId)`` | ``$(ConvertTo-MarkdownCell $item.path)`` | $($item.line) | ``$($item.canApplyAfterOwnerApproval)`` | ``$($item.patchRisk)`` | $(ConvertTo-MarkdownCell $item.requiredProofFieldOrGate) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleSourcePatchApplyReadinessPackState=$($record.readinessPackState) Items=$($record.readinessItemCount) Mapped=$($record.mappedItemCount) OwnerReview=$($record.ownerReviewRequiredItemCount)"
