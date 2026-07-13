[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-article-draft-boundary-scan.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-PublicArticleDraftBoundaryScan.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

function Get-ReviewAction {
  param([string]$PatternId)
  switch ($PatternId) {
    "dry-run-as-proof" { return "remove" }
    "local-feed-proof" { return "remove" }
    "tensorrtexec-proof" { return "downgrade-to-roadmap" }
    "release-closed" { return "wait-for-real-proof" }
    "nuget-published" { return "bind-to-evidence-field" }
    "public-install-verified" { return "bind-to-evidence-field" }
    "clean-consumer-passed" { return "bind-to-evidence-field" }
    "runtime-proof-complete" { return "bind-to-evidence-field" }
    default { return "wait-for-real-proof" }
  }
}

function Get-ProofGate {
  param([string]$PatternId)
  switch ($PatternId) {
    "nuget-published" { return "owner-public-publish-contract.packagePageUrl/publicDownloadUrl" }
    "public-install-verified" { return "post-publish-owner-input.cleanConsumerRoot/runtimeSmokePassed" }
    "clean-consumer-passed" { return "post-publish-strict-cross-check.clean-consumer-root-outside-repository" }
    "release-closed" { return "release-close-strict-closure.ownerCloseDecision" }
    "runtime-proof-complete" { return "post-publish-owner-input.runtimeSmokePassed/runtimeSmokeExitCode" }
    "dry-run-as-proof" { return "forbidden-substitute-scan.noDryRunOnlyConfirmation" }
    "local-feed-proof" { return "forbidden-substitute-scan.noLocalFeedConfirmation/noProjectReferenceConfirmation/noDirectNupkgConfirmation" }
    "tensorrtexec-proof" { return "post-publish-owner-input.runtimeSmokePassed; TensorRtExec report remains sidecar-only" }
    default { return "owner-real-proof-required" }
  }
}

$scan = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$blockedMatches = @(Get-PropertyOrDefault -Object $scan -Name "matches" -DefaultValue @() | Where-Object {
  [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "blocked-claim-review-required"
})

$reviewItems = @($blockedMatches | ForEach-Object {
  $patternId = [string](Get-PropertyOrDefault -Object $_ -Name "patternId" -DefaultValue "unknown")
  $action = Get-ReviewAction -PatternId $patternId
  $proofGate = Get-ProofGate -PatternId $patternId
  [pscustomobject]@{
    order = 0
    path = [string](Get-PropertyOrDefault -Object $_ -Name "path" -DefaultValue "")
    line = [int](Get-PropertyOrDefault -Object $_ -Name "line" -DefaultValue 0)
    patternId = $patternId
    matchedText = [string](Get-PropertyOrDefault -Object $_ -Name "matchedText" -DefaultValue "")
    claim = [string](Get-PropertyOrDefault -Object $_ -Name "claim" -DefaultValue "")
    requiredProof = [string](Get-PropertyOrDefault -Object $_ -Name "requiredProof" -DefaultValue "")
    recommendedAction = $action
    requiredProofFieldOrGate = $proofGate
    ownerActionRequired = $true
    blocked = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner review item only. Do not publish, close, or promote the claim until the named real proof field/gate is satisfied."
  }
})

$index = 0
foreach ($item in $reviewItems) {
  $index++
  $item.order = $index
}

$actionGroups = @($reviewItems | Group-Object recommendedAction | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{
    action = $_.Name
    count = $_.Count
    ownerActionRequired = $true
    blocked = $true
  }
})

$record = [pscustomobject]@{
  recordKind = "public-article-blocked-claim-owner-review-list"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  reviewState = if ($reviewItems.Count -gt 0) { "blocked-public-article-claim-owner-review-required" } else { "public-article-claim-owner-review-clean-non-proof" }
  sourceScanPath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $InputPath).Replace("\", "/")
  scannedFileCount = [int](Get-PropertyOrDefault -Object $scan -Name "scannedFileCount" -DefaultValue 0)
  blockedClaimCount = $reviewItems.Count
  reviewItemCount = $reviewItems.Count
  actionGroupCount = $actionGroups.Count
  actionGroups = @($actionGroups)
  reviewItems = @($reviewItems)
  ownerActionRequired = $reviewItems.Count -gt 0
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner article claim review list only. It classifies blocked draft claims and never edits source articles, publishes packages/articles, validates public package availability, or closes the release."
}

$jsonPath = Join-Path $OutputRoot "public-article-blocked-claim-owner-review-list.json"
$mdPath = Join-Path $OutputRoot "public-article-blocked-claim-owner-review-list.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Public Article Blocked Claim Owner Review List") | Out-Null
$md.Add("") | Out-Null
$md.Add("- reviewState: ``$($record.reviewState)``") | Out-Null
$md.Add("- scannedFileCount: ``$($record.scannedFileCount)``") | Out-Null
$md.Add("- blockedClaimCount: ``$($record.blockedClaimCount)``") | Out-Null
$md.Add("- actionGroupCount: ``$($record.actionGroupCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Action | Count |") | Out-Null
$md.Add("| --- | ---: |") | Out-Null
foreach ($group in $actionGroups) {
  $md.Add("| ``$($group.action)`` | $($group.count) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("| # | Pattern | File | Line | Action | Required Proof Field / Gate |") | Out-Null
$md.Add("| ---: | --- | --- | ---: | --- | --- |") | Out-Null
foreach ($item in @($reviewItems | Select-Object -First 120)) {
  $md.Add("| $($item.order) | ``$($item.patternId)`` | ``$(ConvertTo-MarkdownCell $item.path)`` | $($item.line) | ``$($item.recommendedAction)`` | $(ConvertTo-MarkdownCell $item.requiredProofFieldOrGate) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "PublicArticleBlockedClaimOwnerReviewListState=$($record.reviewState) BlockedClaims=$($record.blockedClaimCount) Actions=$($record.actionGroupCount)"
