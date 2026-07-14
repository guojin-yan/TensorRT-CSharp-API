[CmdletBinding()]
param(
  [string]$ArticleRoadmapValidationPath = "artifacts/final-release/article-roadmap-30plus-validation.json",
  [string]$ArticlePublishingReadinessMapPath = "artifacts/final-release/article-publishing-readiness-map.json",
  [string]$PublicArticleReadinessMatrixPath = "artifacts/final-release/public-article-readiness-matrix.json",
  [string]$ReleaseDocsAndNuGetMetadataAuditValidationPath = "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json",
  [string]$FinalPublicReleaseClosureBridgePath = "artifacts/final-release/final-public-release-closure-bridge.json",
  [string]$StrictCloseReadyConvergenceDashboardPath = "artifacts/final-release/strict-close-ready-convergence-dashboard.json",
  [string]$ReleaseIssueCloseOwnerDecisionInputPath = "artifacts/final-release/release-issue-close-owner-decision-input.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) { return @($Value) }
  return @($Value)
}

function Test-StateReady {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return -not [string]::IsNullOrWhiteSpace($text) -and
    -not $text.StartsWith("missing-", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.StartsWith("failed-", [StringComparison]::OrdinalIgnoreCase)
}

function New-AssetLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$State,
    [bool]$OwnerProofRequired,
    [string[]]$SourceArtifacts,
    [string[]]$RequiredOwnerProofBeforePublish,
    [string[]]$RequiredOwnerProofBeforeReleaseIssueClose,
    [string[]]$BlockedReasons,
    [string]$NextOwnerAction
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    state = $State
    ownerProofRequired = $OwnerProofRequired
    blocked = $OwnerProofRequired
    sourceArtifacts = @($SourceArtifacts)
    requiredOwnerProofBeforePublish = @($RequiredOwnerProofBeforePublish)
    requiredOwnerProofBeforeReleaseIssueClose = @($RequiredOwnerProofBeforeReleaseIssueClose)
    blockedReasons = @($BlockedReasons)
    nextOwnerAction = $NextOwnerAction
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

function Get-LiveYoloDetMatches {
  $roots = @("README.md", "README.zh-CN.md", "docs", "samples", "applications", "src", "TensorRtSharp.sln")
  $patterns = @("\bYoloDet\b", "samples[/\\]YoloDet", "YoloDet\.csproj")
  $results = @()

  foreach ($root in $roots) {
    $resolved = Resolve-RepositoryPath -Path $root
    if (-not (Test-Path -LiteralPath $resolved)) { continue }

    $files = if (Test-Path -LiteralPath $resolved -PathType Leaf) {
      @(Get-Item -LiteralPath $resolved)
    }
    else {
      @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force | Where-Object {
          $_.FullName -notmatch "\\(bin|obj|\.git)\\"
        })
    }

    foreach ($file in $files) {
      $relativePath = [IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName).Replace("\", "/")
      foreach ($pattern in $patterns) {
        $found = @(Select-String -LiteralPath $file.FullName -Pattern $pattern -Encoding utf8 -ErrorAction SilentlyContinue)
        foreach ($item in $found) {
          $results += [pscustomobject]@{
              path = $relativePath
              line = [int]$item.LineNumber
              pattern = $pattern
              text = ([string]$item.Line).Trim()
            }
        }
      }
    }
  }

  foreach ($result in $results) {
    $result
  }
}

$articleRoadmapValidation = Read-JsonOrNull -Path $ArticleRoadmapValidationPath
$articlePublishingReadinessMap = Read-JsonOrNull -Path $ArticlePublishingReadinessMapPath
$publicArticleReadinessMatrix = Read-JsonOrNull -Path $PublicArticleReadinessMatrixPath
$releaseDocsAndNuGetMetadataAuditValidation = Read-JsonOrNull -Path $ReleaseDocsAndNuGetMetadataAuditValidationPath
$finalPublicReleaseClosureBridge = Read-JsonOrNull -Path $FinalPublicReleaseClosureBridgePath
$strictCloseReadyConvergenceDashboard = Read-JsonOrNull -Path $StrictCloseReadyConvergenceDashboardPath
$releaseIssueCloseOwnerDecisionInput = Read-JsonOrNull -Path $ReleaseIssueCloseOwnerDecisionInputPath

$liveYoloDetMatches = @(Get-LiveYoloDetMatches | Where-Object { $null -ne $_ })
$sampleRenameReady = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision") -PathType Container) -and
  (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision\YoloVision.csproj") -PathType Leaf) -and
  -not (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloDet")) -and
  $liveYoloDetMatches.Count -eq 0

$sourceReadinessSignals = @(
  (Test-StateReady (Get-PropertyOrDefault -Object $articleRoadmapValidation -Name "validationState" -DefaultValue "missing-article-roadmap-30plus-validation"))
  (Test-StateReady (Get-PropertyOrDefault -Object $articlePublishingReadinessMap -Name "readinessState" -DefaultValue "missing-article-publishing-readiness-map"))
  (Test-StateReady (Get-PropertyOrDefault -Object $publicArticleReadinessMatrix -Name "matrixState" -DefaultValue "missing-public-article-readiness-matrix"))
  (Test-StateReady (Get-PropertyOrDefault -Object $releaseDocsAndNuGetMetadataAuditValidation -Name "validationState" -DefaultValue "missing-release-docs-and-nuget-metadata-audit-validation"))
  (Test-StateReady (Get-PropertyOrDefault -Object $finalPublicReleaseClosureBridge -Name "bridgeState" -DefaultValue "missing-final-public-release-closure-bridge"))
  (Test-StateReady (Get-PropertyOrDefault -Object $strictCloseReadyConvergenceDashboard -Name "dashboardState" -DefaultValue "missing-strict-close-ready-convergence-dashboard"))
  (Test-StateReady (Get-PropertyOrDefault -Object $releaseIssueCloseOwnerDecisionInput -Name "decisionInputState" -DefaultValue "missing-release-issue-close-owner-decision-input"))
  $sampleRenameReady
)
$sourceReadinessSignalCount = @($sourceReadinessSignals | Where-Object { [bool]$_ }).Count

$publishProofs = @(
  "Owner public publish result with public NuGet/GitHub package URLs, package version, SHA256, transcript hash, and reviewer authorization.",
  "Public package download proof from public source with package identity, URL, SHA256, timestamp, and non-local source.",
  "Repository-external clean consumer restore/build/smoke evidence from public packages.",
  "YoloVision real-model-runtime assets, logs, output JSON, hashes, license notes, and owner review before article claims.",
  "Release docs and NuGet metadata audit with no live YoloDet public-facing references."
)

$closeProofs = @(
  "Final public release closure bridge has no blocked lanes.",
  "Strict close dashboard has all close readiness lanes ready.",
  "Release issue close owner decision input is approved by Owner with real evidence hashes.",
  "Post-publish user verification pack is unblocked by real public download and clean consumer proof.",
  "Release evidence classification audit still reports non-proof boundaries intact."
)

$lanes = @(
  New-AssetLane -Id "article-roadmap-30plus" -Title "30+ article roadmap" -State ([string](Get-PropertyOrDefault -Object $articleRoadmapValidation -Name "validationState" -DefaultValue "missing-article-roadmap-30plus-validation")) -OwnerProofRequired $true -SourceArtifacts @("docs/articles/zh-cn/publishing/article-roadmap-30plus.json", $ArticleRoadmapValidationPath) -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Content roadmap is planning only.", "Article claims require Owner public publish and post-publish proof.") -NextOwnerAction "After real publish, review planned article claims and replace placeholders with public package and proof evidence."
  New-AssetLane -Id "article-publishing-readiness-map" -Title "Article publishing readiness map" -State ([string](Get-PropertyOrDefault -Object $articlePublishingReadinessMap -Name "readinessState" -DefaultValue "missing-article-publishing-readiness-map")) -OwnerProofRequired $true -SourceArtifacts @($ArticlePublishingReadinessMapPath) -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Focused articles remain non-proof drafts.", "Publication claims require public package proof.") -NextOwnerAction "Use the readiness map as an article checklist only after real Owner proof is imported."
  New-AssetLane -Id "public-article-readiness-matrix" -Title "Public article readiness matrix" -State ([string](Get-PropertyOrDefault -Object $publicArticleReadinessMatrix -Name "matrixState" -DefaultValue "missing-public-article-readiness-matrix")) -OwnerProofRequired $true -SourceArtifacts @($PublicArticleReadinessMatrixPath) -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Public article lanes are still claim gates.", "Matrix cannot publish articles or packages.") -NextOwnerAction "Keep all public article lanes blocked until the matching proof lane is real and accepted."
  New-AssetLane -Id "release-docs-nuget-metadata-audit" -Title "Release docs and NuGet metadata audit" -State ([string](Get-PropertyOrDefault -Object $releaseDocsAndNuGetMetadataAuditValidation -Name "validationState" -DefaultValue "missing-release-docs-and-nuget-metadata-audit-validation")) -OwnerProofRequired $true -SourceArtifacts @($ReleaseDocsAndNuGetMetadataAuditValidationPath) -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Metadata safety is not public package proof.", "No YoloDet finding cannot substitute publish proof.") -NextOwnerAction "Re-run metadata audit after package ids, public package URLs, and documentation links are Owner-confirmed."
  New-AssetLane -Id "sample-yolovision-rename-readiness" -Title "YoloVision sample naming readiness" -State $(if ($sampleRenameReady) { "ready-yolovision-name-no-live-yolodet" } else { "blocked-legacy-yolodet-reference-review-required" }) -OwnerProofRequired $false -SourceArtifacts @("samples/YoloVision", "samples/YoloVision/YoloVision.csproj", "TensorRtSharp.sln") -RequiredOwnerProofBeforePublish @("No live public-facing YoloDet references; samples/YoloVision is the allowed sample identity.") -RequiredOwnerProofBeforeReleaseIssueClose @("Repeat stale name scan before release issue close.") -BlockedReasons $(if ($sampleRenameReady) { @() } else { @("Legacy YoloDet public-facing references still need cleanup.") }) -NextOwnerAction "Keep YoloVision as the unified YOLO-family sample name and do not reintroduce YoloDet."
  New-AssetLane -Id "owner-proof-dependent-doc-assets" -Title "Owner-proof dependent docs and assets" -State "blocked-owner-public-package-and-post-publish-proof-required" -OwnerProofRequired $true -SourceArtifacts @($FinalPublicReleaseClosureBridgePath, $StrictCloseReadyConvergenceDashboardPath, $ReleaseIssueCloseOwnerDecisionInputPath) -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Final closure bridge still has blocked owner fields.", "Strict close dashboard still blocks publish and close lanes.") -NextOwnerAction "After Owner imports real public package and post-publish evidence, refresh closure bridge and strict dashboard before public docs are published."
  New-AssetLane -Id "post-publish-case-asset-pack" -Title "Post-publish case asset pack" -State "blocked-post-publish-clean-consumer-and-public-download-proof-required" -OwnerProofRequired $true -SourceArtifacts @("artifacts/final-release/post-publish-user-verification-pack-validation.json", "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json") -RequiredOwnerProofBeforePublish $publishProofs -RequiredOwnerProofBeforeReleaseIssueClose $closeProofs -BlockedReasons @("Case assets need real public download proof.", "Clean consumer screenshots/logs must come from repository-external public package execution.") -NextOwnerAction "Collect public package download and clean consumer proof before turning case assets into publication material."
)

$blockedLanes = @($lanes | Where-Object { [bool]$_.blocked })
$record = [pscustomobject]@{
  recordKind = "post-publish-docs-article-and-sample-asset-plan"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  planState = "blocked-owner-proof-required-non-proof-planning-ready"
  assetLaneCount = @($lanes).Count
  blockedAssetLaneCount = @($blockedLanes).Count
  sourceReadinessSignalCount = $sourceReadinessSignalCount
  requiredOwnerProofBeforePublish = @($publishProofs)
  requiredOwnerProofBeforePublishCount = @($publishProofs).Count
  requiredOwnerProofBeforeReleaseIssueClose = @($closeProofs)
  requiredOwnerProofBeforeReleaseIssueCloseCount = @($closeProofs).Count
  publicFacingAssetLanes = @($lanes)
  sampleRenameReadiness = [pscustomobject]@{
    allowedSampleName = "YoloVision"
    forbiddenLegacySampleName = "YoloDet"
    sampleDirectoryExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision") -PathType Container)
    sampleProjectExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision\YoloVision.csproj") -PathType Leaf)
    legacyDirectoryExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloDet"))
    legacyYoloDetReferenceCount = $liveYoloDetMatches.Count
    liveYoloDetMatches = @($liveYoloDetMatches | ForEach-Object { "{0}:{1}: {2}" -f $_.path, $_.line, $_.text })
    ready = $sampleRenameReady
  }
  allowedSampleName = "YoloVision"
  forbiddenLegacySampleName = "YoloDet"
  legacyYoloDetReferenceCount = $liveYoloDetMatches.Count
  notExecutedByAutomation = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePackageConsumerRuntime = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    $ArticleRoadmapValidationPath,
    $ArticlePublishingReadinessMapPath,
    $PublicArticleReadinessMatrixPath,
    $ReleaseDocsAndNuGetMetadataAuditValidationPath,
    $FinalPublicReleaseClosureBridgePath,
    $StrictCloseReadyConvergenceDashboardPath,
    $ReleaseIssueCloseOwnerDecisionInputPath,
    "samples/YoloVision/YoloVision.csproj",
    "TensorRtSharp.sln"
  )
  boundary = "This post-publish docs/article/sample asset plan is non-proof planning only: not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-docs-article-and-sample-asset-plan.json"
$markdownPath = Join-Path $OutputRoot "post-publish-docs-article-and-sample-asset-plan.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | $(ConvertTo-MarkdownCell $lane.title) | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$($lane.blocked)`` | ``$($lane.canPublishPublicly)`` |"
}

$markdown = @"
# Post-Publish Docs, Article, And Sample Asset Plan

Generated at: ``$($record.generatedAtUtc)``

## Summary

- planState: ``$($record.planState)``
- assetLaneCount: ``$($record.assetLaneCount)``
- blockedAssetLaneCount: ``$($record.blockedAssetLaneCount)``
- sourceReadinessSignalCount: ``$($record.sourceReadinessSignalCount)``
- allowedSampleName: ``$($record.allowedSampleName)``
- forbiddenLegacySampleName: ``$($record.forbiddenLegacySampleName)``
- legacyYoloDetReferenceCount: ``$($record.legacyYoloDetReferenceCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- isPostPublishProof: ``False``

## Asset Lanes

| Lane | Title | State | Blocked | Can Publish |
| --- | --- | --- | ---: | ---: |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish docs/article/sample asset plan written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PlanState=$($record.planState) AssetLanes=$($record.assetLaneCount) Blocked=$($record.blockedAssetLaneCount) LegacyYoloDet=$($record.legacyYoloDetReferenceCount)"
