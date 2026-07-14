[CmdletBinding()]
param(
  [string]$OwnerRealPublishEvidenceAvailabilityLedgerPath = "artifacts/final-release/owner-real-publish-evidence-availability-ledger.json",
  [string]$OwnerRealPublishEvidenceAvailabilityLedgerValidationPath = "artifacts/final-release/owner-real-publish-evidence-availability-ledger-validation.json",
  [string]$PostPublishDocsArticleAndSampleAssetPlanPath = "artifacts/final-release/post-publish-docs-article-and-sample-asset-plan.json",
  [string]$PostPublishDocsArticleAndSampleAssetPlanValidationPath = "artifacts/final-release/post-publish-docs-article-and-sample-asset-plan-validation.json",
  [string]$StrictCloseReadyConvergenceDashboardPath = "artifacts/final-release/strict-close-ready-convergence-dashboard.json",
  [string]$FinalPublicReleaseClosureBridgePath = "artifacts/final-release/final-public-release-closure-bridge.json",
  [string]$ReleaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json",
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

function ConvertTo-RepositoryRelativePath {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  return [IO.Path]::GetRelativePath($RepositoryRoot, $resolved).Replace("\", "/")
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

function Test-RepositoryPath {
  param([string]$Path)
  return Test-Path -LiteralPath (Resolve-RepositoryPath -Path $Path)
}

function Test-AllRepositoryPaths {
  param([string[]]$Paths)
  foreach ($path in @($Paths)) {
    if (-not (Test-RepositoryPath -Path $path)) { return $false }
  }

  return $true
}

function Get-RepositoryFileState {
  param([string]$Path)
  [pscustomobject]@{
    path = (ConvertTo-RepositoryRelativePath -Path $Path)
    exists = (Test-RepositoryPath -Path $Path)
  }
}

function Get-XmlPropertyOrDefault {
  param(
    [string]$Path,
    [string]$Name,
    [string]$DefaultValue
  )

  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $DefaultValue }

  [xml]$xml = Get-Content -LiteralPath $resolved -Raw -Encoding utf8
  foreach ($group in @($xml.Project.PropertyGroup)) {
    $value = $group.$Name
    if (-not [string]::IsNullOrWhiteSpace([string]$value)) {
      return [string]$value
    }
  }

  return $DefaultValue
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

function New-LandingLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$State,
    [bool]$Ready,
    [bool]$Blocked,
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
    ready = $Ready
    blocked = $Blocked
    ownerProofRequired = $OwnerProofRequired
    proofReady = $false
    sourceArtifacts = @($SourceArtifacts)
    requiredOwnerProofBeforePublish = @($RequiredOwnerProofBeforePublish)
    requiredOwnerProofBeforeReleaseIssueClose = @($RequiredOwnerProofBeforeReleaseIssueClose)
    blockedReasons = @($BlockedReasons)
    nextOwnerAction = $NextOwnerAction
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    canPromotePackageConsumerRuntime = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$ownerLedger = Read-JsonOrNull -Path $OwnerRealPublishEvidenceAvailabilityLedgerPath
$ownerLedgerValidation = Read-JsonOrNull -Path $OwnerRealPublishEvidenceAvailabilityLedgerValidationPath
$assetPlan = Read-JsonOrNull -Path $PostPublishDocsArticleAndSampleAssetPlanPath
$assetPlanValidation = Read-JsonOrNull -Path $PostPublishDocsArticleAndSampleAssetPlanValidationPath
$strictCloseDashboard = Read-JsonOrNull -Path $StrictCloseReadyConvergenceDashboardPath
$closureBridge = Read-JsonOrNull -Path $FinalPublicReleaseClosureBridgePath

$liveYoloDetMatches = @(Get-LiveYoloDetMatches | Where-Object { $null -ne $_ })
$yoloVisionSampleReady = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision") -PathType Container) -and
  (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision\YoloVision.csproj") -PathType Leaf) -and
  -not (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloDet")) -and
  $liveYoloDetMatches.Count -eq 0

$docsSiteFiles = @(
  "docs/index.md",
  "docs/toc.yml",
  "docs/docfx.json",
  "docs/articles/zh-cn/release-evidence-bundle.md",
  "docs/articles/zh-cn/docs-publish-readiness-bundle.md"
)

$readmeFrontDoorFiles = @(
  "README.md",
  "README.zh-CN.md",
  "samples/README.md",
  "samples/YoloVision/README.md"
)

$nugetMetadataFiles = @(
  "Directory.Build.props",
  "pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj",
  "pack/JYPPX.TensorRT.CSharp.API/README.md",
  "src/JYPPX.TensorRtSharp/JYPPX.TensorRtSharp.csproj"
)

$releaseNotesFiles = @(
  "README.md",
  "docs/articles/zh-cn/release-candidate-gate.md",
  "docs/articles/zh-cn/release-evidence-closure-index.md",
  "docs/articles/zh-cn/release-evidence-non-substitute-guide.md",
  "artifacts/final-release/release-evidence-bundle.md"
)

$yoloQuickstartFiles = @(
  "samples/YoloVision/README.md",
  "samples/YoloVision/YoloVision.csproj",
  "samples/YoloVision/yolo-model-matrix.md",
  "samples/YoloVision/yolovision-task-output-contract.json"
)

$yoloRealAssetFiles = @(
  "samples/assets/yolovision-article-case-pack.json",
  "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
  "samples/assets/yolovision-family-task-real-asset-roadmap.json",
  "docs/articles/zh-cn/yolovision-real-asset-walkthrough.md",
  "docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md"
)

$externalCleanInstallFiles = @(
  "docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md",
  "docs/articles/zh-cn/external-clean-consumer-proof-kit.md",
  "artifacts/final-release/external-clean-consumer-execution-workspace-contract.json",
  "artifacts/final-release/external-clean-consumer-owner-command-pack.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json"
)

$articleCaseFiles = @(
  "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
  "docs/articles/zh-cn/publishing/yolovision-overview-public-article.md",
  "docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md",
  "docs/articles/zh-cn/publishing/package-consumer-proof-public-article.md",
  "samples/assets/yolovision-article-case-pack.json",
  "artifacts/final-release/article-publishing-readiness-map.json",
  "artifacts/final-release/public-article-readiness-matrix.json"
)

$ownerProofDependencyFiles = @(
  $OwnerRealPublishEvidenceAvailabilityLedgerPath,
  $OwnerRealPublishEvidenceAvailabilityLedgerValidationPath,
  $PostPublishDocsArticleAndSampleAssetPlanPath,
  $PostPublishDocsArticleAndSampleAssetPlanValidationPath,
  $ReleaseEvidenceBundlePath,
  $StrictCloseReadyConvergenceDashboardPath,
  $FinalPublicReleaseClosureBridgePath
)

$packPackageId = Get-XmlPropertyOrDefault -Path "pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj" -Name "PackageId" -DefaultValue ""
$srcPackageId = Get-XmlPropertyOrDefault -Path "src/JYPPX.TensorRtSharp/JYPPX.TensorRtSharp.csproj" -Name "PackageId" -DefaultValue ""
$packDescription = Get-XmlPropertyOrDefault -Path "pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj" -Name "Description" -DefaultValue ""
$repositoryUrl = Get-XmlPropertyOrDefault -Path "Directory.Build.props" -Name "RepositoryUrl" -DefaultValue ""
$packageMetadataReady = $packPackageId -eq "JYPPX.TensorRT.CSharp.API" -and
  $srcPackageId -eq "JYPPX.TensorRT.CSharp.API" -and
  -not [string]::IsNullOrWhiteSpace($packDescription) -and
  $repositoryUrl -eq "https://github.com/guojin-yan/TensorRT-CSharp-API" -and
  (Test-AllRepositoryPaths -Paths $nugetMetadataFiles)

$requiredOwnerProofBeforePublish = @(
  "Owner public publish result with package ids, versions, public package URLs, SHA256 values, transcript hashes, and selected channels.",
  "GitHub Actions run proof for release-quality-gate and any owner-triggered publish workflows.",
  "Public package download proof from non-local sources for managed and runtime package routes.",
  "Repository-external clean consumer restore/build/smoke proof using public package sources only.",
  "Owner-reviewed YoloVision real asset proof with model, labels, image/tensor, output JSON, logs, SHA256 values, and license notes."
)

$requiredOwnerProofBeforeReleaseIssueClose = @(
  "Accepted post-publish verification record with public package download and clean consumer evidence.",
  "Strict close dashboard with no blocked lanes after real Owner evidence import.",
  "Final public release closure bridge unblocked by real public publish and post-publish proof.",
  "Release issue close Owner decision with evidence bundle hash and rollback decision.",
  "Release issue close record validation accepted without template, local-feed, ProjectReference, direct nupkg, or dry-run substitutes."
)

$docsSiteReady = Test-AllRepositoryPaths -Paths $docsSiteFiles
$readmeReady = Test-AllRepositoryPaths -Paths $readmeFrontDoorFiles
$releaseNotesReady = Test-AllRepositoryPaths -Paths $releaseNotesFiles
$yoloQuickstartReady = Test-AllRepositoryPaths -Paths $yoloQuickstartFiles
$yoloRealAssetIndexReady = Test-AllRepositoryPaths -Paths $yoloRealAssetFiles
$externalCleanInstallReady = Test-AllRepositoryPaths -Paths $externalCleanInstallFiles
$articleCaseIndexReady = Test-AllRepositoryPaths -Paths $articleCaseFiles
$ownerProofDependencyFilesReady = Test-AllRepositoryPaths -Paths $ownerProofDependencyFiles

$ownerLedgerState = [string](Get-PropertyOrDefault -Object $ownerLedger -Name "ledgerState" -DefaultValue "missing-owner-real-publish-evidence-availability-ledger")
$ownerLedgerValidationState = [string](Get-PropertyOrDefault -Object $ownerLedgerValidation -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-availability-ledger-validation")
$assetPlanState = [string](Get-PropertyOrDefault -Object $assetPlan -Name "planState" -DefaultValue "missing-post-publish-docs-article-and-sample-asset-plan")
$assetPlanValidationState = [string](Get-PropertyOrDefault -Object $assetPlanValidation -Name "validationState" -DefaultValue "missing-post-publish-docs-article-and-sample-asset-plan-validation")
$strictCloseDashboardState = [string](Get-PropertyOrDefault -Object $strictCloseDashboard -Name "dashboardState" -DefaultValue "missing-strict-close-ready-convergence-dashboard")
$closureBridgeState = [string](Get-PropertyOrDefault -Object $closureBridge -Name "bridgeState" -DefaultValue "missing-final-public-release-closure-bridge")

$lanes = @(
  New-LandingLane -Id "docs-site-final-links" -Title "Docs site final links" -State $(if ($docsSiteReady) { "blocked-owner-proof-required-docs-site-links-ready" } else { "blocked-missing-docs-site-links" }) -Ready $docsSiteReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $docsSiteFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Docs links are local readiness only.", "Public docs publication still requires Owner public publish and post-publish evidence.") -NextOwnerAction "After real public publish, update docs links with public package URLs and evidence hashes, then rebuild DocFX."
  New-LandingLane -Id "readme-frontpage-links" -Title "README front-page links" -State $(if ($readmeReady) { "blocked-owner-proof-required-readme-frontdoor-ready" } else { "blocked-missing-readme-frontdoor" }) -Ready $readmeReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $readmeFrontDoorFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("README front door is source documentation only.", "It cannot claim public release completion without Owner proof.") -NextOwnerAction "Replace release-state placeholders only after public package URLs and post-publish clean consumer proof are real."
  New-LandingLane -Id "nuget-metadata-owner-review" -Title "NuGet metadata Owner review" -State $(if ($packageMetadataReady) { "blocked-owner-proof-required-nuget-metadata-ready" } else { "blocked-nuget-metadata-review-required" }) -Ready $packageMetadataReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $nugetMetadataFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Package metadata and README are not package publication proof.", "Owner must approve final package id, descriptions, links, and channels before publish.") -NextOwnerAction "Review PackageId, description, RepositoryUrl, package README, and runtime package guidance before public push."
  New-LandingLane -Id "release-notes-and-known-limitations" -Title "Release notes and known limitations" -State $(if ($releaseNotesReady) { "blocked-owner-proof-required-release-notes-ready" } else { "blocked-release-notes-link-review-required" }) -Ready $releaseNotesReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $releaseNotesFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Release notes remain local release-candidate material.", "Known limitation text must keep non-proof boundaries until real proof is imported.") -NextOwnerAction "After Owner evidence import, refresh release notes with real package/version/hash/proof status and keep unresolved lanes explicit."
  New-LandingLane -Id "yolovision-quickstart" -Title "YoloVision quickstart" -State $(if ($yoloQuickstartReady -and $yoloVisionSampleReady) { "blocked-owner-proof-required-yolovision-quickstart-ready" } else { "blocked-yolovision-quickstart-review-required" }) -Ready ($yoloQuickstartReady -and $yoloVisionSampleReady) -Blocked $true -OwnerProofRequired $true -SourceArtifacts $yoloQuickstartFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("YoloVision quickstart is executable sample guidance, not real-model-runtime proof.", "The legacy YoloDet name must remain absent from public-facing surfaces.") -NextOwnerAction "Keep YoloVision as the public sample identity and run real-asset proof before publishing model-quality claims."
  New-LandingLane -Id "yolovision-real-asset-proof" -Title "YoloVision real asset proof" -State $(if ($yoloRealAssetIndexReady) { "blocked-owner-real-asset-proof-required-index-ready" } else { "blocked-yolovision-real-asset-index-review-required" }) -Ready $yoloRealAssetIndexReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $yoloRealAssetFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Asset packs and article cases are templates/indexes only.", "Real model, labels, input, output JSON, run logs, hashes, license notes, and Owner review are still required.") -NextOwnerAction "Collect and validate owner-filled YoloVision real asset proof before moving any case article to public proof language."
  New-LandingLane -Id "external-clean-install-guide" -Title "External clean install guide" -State $(if ($externalCleanInstallReady) { "blocked-owner-proof-required-clean-install-guide-ready" } else { "blocked-clean-install-guide-review-required" }) -Ready $externalCleanInstallReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $externalCleanInstallFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("External clean install guide is owner-copyable guidance only.", "It must be backed by repository-external public package restore/build/smoke evidence.") -NextOwnerAction "Run clean consumer proof outside this repository after public packages are available, then import logs and hashes."
  New-LandingLane -Id "article-case-asset-index" -Title "Article case asset index" -State $(if ($articleCaseIndexReady) { "blocked-owner-proof-required-article-case-index-ready" } else { "blocked-article-case-index-review-required" }) -Ready $articleCaseIndexReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $articleCaseFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Public article case index is content planning only.", "Article screenshots and claims need real public package and real model/runtime evidence.") -NextOwnerAction "After proof import, choose the first public article batch and attach package URLs, evidence hashes, and screenshots."
  New-LandingLane -Id "owner-proof-dependency" -Title "Owner proof dependency" -State $(if ($ownerProofDependencyFilesReady) { "blocked-owner-real-publish-evidence-required-dependencies-linked" } else { "blocked-owner-proof-dependency-artifacts-missing" }) -Ready $ownerProofDependencyFilesReady -Blocked $true -OwnerProofRequired $true -SourceArtifacts $ownerProofDependencyFiles -RequiredOwnerProofBeforePublish $requiredOwnerProofBeforePublish -RequiredOwnerProofBeforeReleaseIssueClose $requiredOwnerProofBeforeReleaseIssueClose -BlockedReasons @("Owner ledger, release evidence bundle, strict close dashboard, and closure bridge are dependency surfaces only.", "They do not publish packages, approve close, or prove post-publish execution by themselves.") -NextOwnerAction "Import real Owner public publish, public download, clean consumer, and close decision evidence, then regenerate the dependency chain."
)

$readyLanes = @($lanes | Where-Object { [bool]$_.ready })
$blockedLanes = @($lanes | Where-Object { [bool]$_.blocked })
$proofReadyLanes = @($lanes | Where-Object { [bool]$_.proofReady })

$allSourceArtifacts = @(
  $docsSiteFiles +
  $readmeFrontDoorFiles +
  $nugetMetadataFiles +
  $releaseNotesFiles +
  $yoloQuickstartFiles +
  $yoloRealAssetFiles +
  $externalCleanInstallFiles +
  $articleCaseFiles +
  $ownerProofDependencyFiles
) | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

$record = [pscustomobject]@{
  recordKind = "post-publish-docs-and-samples-final-landing-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  landingPackState = "blocked-owner-real-publish-evidence-required-final-landing-ready"
  landingLaneCount = @($lanes).Count
  readyLandingLaneCount = @($readyLanes).Count
  blockedLandingLaneCount = @($blockedLanes).Count
  proofReadyLandingLaneCount = @($proofReadyLanes).Count
  ownerProofRequiredLaneCount = @($lanes | Where-Object { [bool]$_.ownerProofRequired }).Count
  requiredOwnerProofBeforePublish = @($requiredOwnerProofBeforePublish)
  requiredOwnerProofBeforePublishCount = @($requiredOwnerProofBeforePublish).Count
  requiredOwnerProofBeforeReleaseIssueClose = @($requiredOwnerProofBeforeReleaseIssueClose)
  requiredOwnerProofBeforeReleaseIssueCloseCount = @($requiredOwnerProofBeforeReleaseIssueClose).Count
  lanes = @($lanes)
  sampleStatus = [pscustomobject]@{
    allowedSampleName = "YoloVision"
    forbiddenLegacySampleName = "YoloDet"
    yoloVisionSampleReady = $yoloVisionSampleReady
    sampleDirectoryExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision") -PathType Container)
    sampleProjectExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloVision\YoloVision.csproj") -PathType Leaf)
    legacyDirectoryExists = (Test-Path -LiteralPath (Resolve-RepositoryPath -Path "samples\YoloDet"))
    legacyYoloDetReferenceCount = $liveYoloDetMatches.Count
    liveYoloDetMatches = @($liveYoloDetMatches | ForEach-Object { "{0}:{1}: {2}" -f $_.path, $_.line, $_.text })
  }
  allowedSampleName = "YoloVision"
  forbiddenLegacySampleName = "YoloDet"
  yoloVisionSampleReady = $yoloVisionSampleReady
  legacyYoloDetReferenceCount = $liveYoloDetMatches.Count
  packageMetadata = [pscustomobject]@{
    packPackageId = $packPackageId
    sourcePackageId = $srcPackageId
    repositoryUrl = $repositoryUrl
    descriptionReady = (-not [string]::IsNullOrWhiteSpace($packDescription))
    packageMetadataReady = $packageMetadataReady
  }
  ownerProofDependencies = [pscustomobject]@{
    ownerLedgerState = $ownerLedgerState
    ownerLedgerValidationState = $ownerLedgerValidationState
    postPublishDocsArticleAndSampleAssetPlanState = $assetPlanState
    postPublishDocsArticleAndSampleAssetPlanValidationState = $assetPlanValidationState
    strictCloseReadyConvergenceDashboardState = $strictCloseDashboardState
    finalPublicReleaseClosureBridgeState = $closureBridgeState
    sourceArtifactCount = @($ownerProofDependencyFiles).Count
    sourceArtifacts = @($ownerProofDependencyFiles)
  }
  sourceArtifacts = @($allSourceArtifacts)
  sourceArtifactStates = @($allSourceArtifacts | ForEach-Object { Get-RepositoryFileState -Path $_ })
  notExecutedByAutomation = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePackageConsumerRuntime = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This post-publish docs and samples final landing pack is non-proof planning and publication readiness only: not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-docs-and-samples-final-landing-pack.json"
$markdownPath = Join-Path $OutputRoot "post-publish-docs-and-samples-final-landing-pack.md"
$record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | $(ConvertTo-MarkdownCell $lane.title) | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$($lane.ready)`` | ``$($lane.blocked)`` | ``$($lane.proofReady)`` |"
}

$markdown = @"
# Post-Publish Docs And Samples Final Landing Pack

Generated at: ``$($record.generatedAtUtc)``

## Summary

- landingPackState: ``$($record.landingPackState)``
- landingLaneCount: ``$($record.landingLaneCount)``
- readyLandingLaneCount: ``$($record.readyLandingLaneCount)``
- blockedLandingLaneCount: ``$($record.blockedLandingLaneCount)``
- proofReadyLandingLaneCount: ``$($record.proofReadyLandingLaneCount)``
- yoloVisionSampleReady: ``$($record.yoloVisionSampleReady)``
- legacyYoloDetReferenceCount: ``$($record.legacyYoloDetReferenceCount)``
- packageMetadataReady: ``$($record.packageMetadata.packageMetadataReady)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- isPostPublishProof: ``False``

## Landing Lanes

| Lane | Title | State | Ready | Blocked | Proof Ready |
| --- | --- | --- | ---: | ---: | ---: |
$($rows -join "`r`n")

## Owner Proof Dependencies

- ownerLedgerState: ``$($record.ownerProofDependencies.ownerLedgerState)``
- ownerLedgerValidationState: ``$($record.ownerProofDependencies.ownerLedgerValidationState)``
- postPublishDocsArticleAndSampleAssetPlanValidationState: ``$($record.ownerProofDependencies.postPublishDocsArticleAndSampleAssetPlanValidationState)``
- strictCloseReadyConvergenceDashboardState: ``$($record.ownerProofDependencies.strictCloseReadyConvergenceDashboardState)``
- finalPublicReleaseClosureBridgeState: ``$($record.ownerProofDependencies.finalPublicReleaseClosureBridgeState)``

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish docs and samples final landing pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "LandingPackState=$($record.landingPackState) Lanes=$($record.landingLaneCount) Blocked=$($record.blockedLandingLaneCount) LegacyYoloDet=$($record.legacyYoloDetReferenceCount)"
