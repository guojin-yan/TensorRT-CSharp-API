[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Test-RepoPath {
  param([string]$Path)
  return Test-Path -LiteralPath (Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $Path)
}

function Read-RepoJson {
  param([string]$Path)
  return Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $Path
}

function Get-JsonStringArray {
  param([AllowNull()][object]$Value)
  @($Value | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function New-SampleSurface {
  param(
    [string]$Id,
    [string]$DisplayName,
    [string]$Kind,
    [string]$ProjectPath,
    [string]$ReadmePath,
    [string[]]$MatrixPaths,
    [string[]]$ArticlePaths,
    [string[]]$ProofDependencies
  )

  $allPaths = @($ProjectPath, $ReadmePath) + @($MatrixPaths) + @($ArticlePaths)
  $missing = @($allPaths | Where-Object { -not (Test-RepoPath -Path $_) })
  [pscustomobject]@{
    id = $Id
    displayName = $DisplayName
    kind = $Kind
    projectPath = $ProjectPath
    readmePath = $ReadmePath
    matrixPaths = @($MatrixPaths)
    articlePaths = @($ArticlePaths)
    sourcePathCount = @($allPaths).Count
    missingPathCount = $missing.Count
    missingPaths = @($missing)
    sourceReady = $missing.Count -eq 0
    buildStatus = if (Test-RepoPath -Path $ProjectPath) { "project-present-build-covered-by-solution" } else { "missing-project" }
    runtimeProofStatus = "blocked-owner-real-runtime-or-public-package-proof-required"
    postPublishProofStatus = "blocked-owner-public-publish-and-clean-consumer-proof-required"
    publicDocsDependency = "docs-ready-but-public-claims-require-owner-proof"
    proofDependencies = @($ProofDependencies)
    canPromoteRuntimeProof = $false
    canPromotePostPublishProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$articleRoadmap = Read-RepoJson "docs/articles/zh-cn/publishing/article-roadmap-30plus.json"
if ($null -eq $articleRoadmap) {
  $articleRoadmap = Read-RepoJson "artifacts/final-release/article-roadmap-30plus.json"
}

$yoloMatrix = Read-RepoJson "samples/YoloVision/yolo-model-matrix.json"
$onnxMatrix = Read-RepoJson "samples/OnnxToEngine/trtexec-parity-matrix.json"
$tensorRtExecMatrix = Read-RepoJson "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json"
$finalProofSweep = Read-RepoJson "artifacts/final-release/final-real-proof-input-availability-sweep-validation.json"
$closeCandidatePack = Read-RepoJson "artifacts/final-release/final-real-proof-import-and-close-candidate-pack-validation.json"
$landingPack = Read-RepoJson "artifacts/final-release/post-publish-docs-and-samples-final-landing-pack-validation.json"

$articleCount = [int](Get-PropertyOrDefault -Object $articleRoadmap -Name "articleCount" -DefaultValue 0)
$articleArray = @(Convert-ToArray (Get-PropertyOrDefault -Object $articleRoadmap -Name "articles" -DefaultValue @()))
if ($articleCount -eq 0) {
  $articleCount = $articleArray.Count
}

$yoloFamilies = @(Get-JsonStringArray (Get-PropertyOrDefault -Object $yoloMatrix -Name "families" -DefaultValue @()))
$yoloTasks = @(Get-JsonStringArray (Get-PropertyOrDefault -Object $yoloMatrix -Name "tasks" -DefaultValue @()))
$onnxEntryCount = @(Convert-ToArray (Get-PropertyOrDefault -Object $onnxMatrix -Name "entries" -DefaultValue @())).Count
$tensorRtExecFeatureCount = @(Convert-ToArray (Get-PropertyOrDefault -Object $tensorRtExecMatrix -Name "features" -DefaultValue @())).Count

$legacyYoloDetPathExists = Test-RepoPath "samples/YoloDet"
$legacyYoloDetPublicPathMatches = @()
foreach ($root in @("README.md", "docs", "samples", "applications", "TensorRtSharp.sln")) {
  $resolved = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $root
  if (-not (Test-Path -LiteralPath $resolved)) { continue }
  $files = if (Test-Path -LiteralPath $resolved -PathType Leaf) { @(Get-Item -LiteralPath $resolved) } else { @(Get-ChildItem -LiteralPath $resolved -Recurse -File | Where-Object { $_.FullName -notmatch "\\(bin|obj|_site)\\?" }) }
  foreach ($file in $files) {
    $matches = @(Select-String -LiteralPath $file.FullName -Pattern "samples[/\\]YoloDet|YoloDet\.csproj" -Encoding utf8 -ErrorAction SilentlyContinue)
    foreach ($match in $matches) {
      $legacyYoloDetPublicPathMatches += [pscustomobject]@{
        path = [IO.Path]::GetRelativePath($RepositoryRoot, $file.FullName).Replace("\", "/")
        line = [int]$match.LineNumber
        text = ([string]$match.Line).Trim()
      }
    }
  }
}

$commonProofDependencies = @(
  "final-real-proof-input-availability-sweep",
  "final-real-proof-import-and-close-candidate-pack",
  "public-package-download-proof",
  "repository-external-clean-consumer-proof",
  "post-publish-clean-consumer-proof",
  "release-issue-close-owner-decision"
)

$sampleSurfaces = @(
  New-SampleSurface -Id "onnx-to-engine" -DisplayName "OnnxToEngine" -Kind "sample" -ProjectPath "samples/OnnxToEngine/OnnxToEngine.csproj" -ReadmePath "samples/OnnxToEngine/README.md" -MatrixPaths @("samples/OnnxToEngine/trtexec-parity-matrix.json", "samples/OnnxToEngine/trtexec-parity-matrix.md") -ArticlePaths @("docs/articles/zh-cn/onnx-to-engine-quickstart.md", "docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "classification" -DisplayName "Classification" -Kind "sample" -ProjectPath "samples/Classification/Classification.csproj" -ReadmePath "samples/Classification/README.md" -MatrixPaths @("docs/articles/zh-cn/classification-model-assets.md") -ArticlePaths @("docs/articles/zh-cn/classification-real-asset-walkthrough.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "dynamic-shape" -DisplayName "DynamicShape" -Kind "sample" -ProjectPath "samples/DynamicShape/DynamicShape.csproj" -ReadmePath "samples/DynamicShape/README.md" -MatrixPaths @("docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md") -ArticlePaths @("docs/articles/zh-cn/blog-dynamic-shape-optimization-profile.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "inference-bindings" -DisplayName "InferenceBindings" -Kind "sample" -ProjectPath "samples/InferenceBindings/InferenceBindings.csproj" -ReadmePath "samples/InferenceBindings/README.md" -MatrixPaths @("docs/articles/zh-cn/inference-bindings-tutorial.md") -ArticlePaths @("docs/articles/zh-cn/blog-inference-bindings-identity-network.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "multi-stream" -DisplayName "MultiStream" -Kind "sample" -ProjectPath "samples/MultiStream/MultiStream.csproj" -ReadmePath "samples/MultiStream/README.md" -MatrixPaths @("docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md") -ArticlePaths @("docs/articles/zh-cn/blog-multistream-cuda-stream-event.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "yolovision" -DisplayName "YoloVision" -Kind "sample" -ProjectPath "samples/YoloVision/YoloVision.csproj" -ReadmePath "samples/YoloVision/README.md" -MatrixPaths @("samples/YoloVision/yolo-model-matrix.json", "samples/YoloVision/yolovision-task-output-contract.json") -ArticlePaths @("docs/articles/zh-cn/yolovision-sample-overview.md", "docs/articles/zh-cn/yolo-vision-model-matrix.md", "docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "tensorrt-exec" -DisplayName "TensorRtExec" -Kind "application" -ProjectPath "applications/TensorRtExec/TensorRtExec.csproj" -ReadmePath "applications/TensorRtExec/README.md" -MatrixPaths @("applications/TensorRtExec/tensor-rt-exec-feature-matrix.json", "applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.json") -ArticlePaths @("docs/articles/zh-cn/tensorrtexec-tool-getting-started.md", "docs/articles/zh-cn/tensorrtexec-gui-user-guide.md") -ProofDependencies $commonProofDependencies
  New-SampleSurface -Id "plugin-registry-inventory-smoke" -DisplayName "PluginRegistryInventorySmokeRunner" -Kind "smoke" -ProjectPath "smoke/PluginRegistryInventorySmokeRunner/PluginRegistryInventorySmokeRunner.csproj" -ReadmePath "docs/articles/zh-cn/plugin-inventory-readonly-api.md" -MatrixPaths @("docs/articles/zh-cn/plugin-registry-inventory-readonly-design.md") -ArticlePaths @("docs/articles/zh-cn/blog-plugin-inventory-readonly-api.md") -ProofDependencies $commonProofDependencies
)

$readySampleCount = @($sampleSurfaces | Where-Object { [bool]$_.sourceReady }).Count
$missingSamplePathCount = 0
foreach ($surface in $sampleSurfaces) { $missingSamplePathCount += [int]$surface.missingPathCount }

$ownerActionFields = @(
  "nugetManagedPackageUrl",
  "nugetManagedPackageVersion",
  "nugetManagedPackageSha256",
  "githubRuntimePackageUrl",
  "githubRuntimePackageVersion",
  "githubRuntimePackageSha256",
  "publicPackageDownloadLogSha256",
  "externalCleanConsumerRestoreLogSha256",
  "externalCleanConsumerBuildLogSha256",
  "externalCleanConsumerSmokeLogSha256",
  "yoloVisionModelSourceUrl",
  "yoloVisionModelSha256",
  "yoloVisionLabelsSha256",
  "yoloVisionOutputJsonSha256",
  "articlePublishUrl",
  "articleScreenshotAssetSha256",
  "releaseIssueCloseDecisionUrl"
)

$record = [pscustomobject]@{
  recordKind = "post-publish-public-docs-article-sample-readiness-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  readinessState = "blocked-owner-public-publish-and-post-publish-proof-required"
  articleRoadmapState = [string](Get-PropertyOrDefault -Object $articleRoadmap -Name "roadmapState" -DefaultValue "missing-article-roadmap-30plus")
  articleCount = $articleCount
  minimumArticleCount = [int](Get-PropertyOrDefault -Object $articleRoadmap -Name "minimumArticleCount" -DefaultValue 30)
  sampleSurfaceCount = @($sampleSurfaces).Count
  readySampleSurfaceCount = $readySampleCount
  missingSamplePathCount = $missingSamplePathCount
  sampleSurfaces = @($sampleSurfaces)
  yoloVision = [pscustomobject]@{
    samplePath = "samples/YoloVision"
    legacyYoloDetPathExists = $legacyYoloDetPathExists
    legacyYoloDetPublicPathMatchCount = @($legacyYoloDetPublicPathMatches).Count
    families = @($yoloFamilies)
    familyCount = @($yoloFamilies).Count
    tasks = @($yoloTasks)
    taskCount = @($yoloTasks).Count
    proofBoundary = "YoloVision matrix and docs are sample readiness only; real-model-runtime proof requires Owner assets, logs, output JSON, hashes, host/package metadata, and validators."
  }
  onnxToEngine = [pscustomobject]@{
    matrixEntryCount = $onnxEntryCount
    proofBoundary = "OnnxToEngine and trtexec parity matrices are build/report readiness only, not runtime proof and not public package proof."
  }
  tensorRtExec = [pscustomobject]@{
    featureCount = $tensorRtExecFeatureCount
    modes = @("CLI", "WinForms")
    proofBoundary = "TensorRtExec CLI/WinForms feature matrix is application readiness only, not runtime proof, not post-publish proof, and not release close approval."
  }
  ownerProofSourceStates = [pscustomobject]@{
    finalRealProofInputAvailabilitySweepValidationState = [string](Get-PropertyOrDefault -Object $finalProofSweep -Name "validationState" -DefaultValue "missing-final-real-proof-input-availability-sweep-validation")
    finalRealProofImportAndCloseCandidatePackValidationState = [string](Get-PropertyOrDefault -Object $closeCandidatePack -Name "validationState" -DefaultValue "missing-final-real-proof-import-and-close-candidate-pack-validation")
    postPublishDocsAndSamplesFinalLandingPackValidationState = [string](Get-PropertyOrDefault -Object $landingPack -Name "validationState" -DefaultValue "missing-post-publish-docs-and-samples-final-landing-pack-validation")
  }
  ownerActionFieldCount = @($ownerActionFields).Count
  ownerActionFields = @($ownerActionFields)
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This readiness pack organizes post-publish docs/articles/samples only; it is not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-public-docs-article-sample-readiness-pack.json"
$mdPath = Join-Path $OutputRoot "post-publish-public-docs-article-sample-readiness-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)

$surfaceRows = foreach ($surface in $sampleSurfaces) {
  "| ``$($surface.id)`` | ``$($surface.displayName)`` | ``$($surface.kind)`` | ``$($surface.sourceReady)`` | ``$($surface.runtimeProofStatus)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Post-Publish Public Docs Article Sample Readiness Pack",
  "",
  "- readinessState: ``$($record.readinessState)``",
  "- articleCount: ``$($record.articleCount)``",
  "- sampleSurfaceCount: ``$($record.sampleSurfaceCount)``",
  "- readySampleSurfaceCount: ``$($record.readySampleSurfaceCount)``",
  "- missingSamplePathCount: ``$($record.missingSamplePathCount)``",
  "- yoloFamilyCount: ``$($record.yoloVision.familyCount)``",
  "- yoloTaskCount: ``$($record.yoloVision.taskCount)``",
  "- onnxToEngineMatrixEntryCount: ``$($record.onnxToEngine.matrixEntryCount)``",
  "- tensorRtExecFeatureCount: ``$($record.tensorRtExec.featureCount)``",
  "- performsPublish: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Id | Surface | Kind | Source Ready | Runtime Proof Status |",
  "| --- | --- | --- | ---: | --- |",
  @($surfaceRows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "PostPublishPublicDocsArticleSampleReadinessState=$($record.readinessState) Articles=$($record.articleCount) Surfaces=$($record.sampleSurfaceCount) MissingPaths=$($record.missingSamplePathCount)"
