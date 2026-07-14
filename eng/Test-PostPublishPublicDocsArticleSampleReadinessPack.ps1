[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-public-docs-article-sample-readiness-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PostPublishPublicDocsArticleSampleReadinessPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$surfaces = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "sampleSurfaces" -DefaultValue @()))
$surfaceIds = @($surfaces | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredSurfaceIds = @("onnx-to-engine", "classification", "dynamic-shape", "inference-bindings", "multi-stream", "yolovision", "tensorrt-exec", "plugin-registry-inventory-smoke")
$requiredFamilies = @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom")
$requiredTasks = @("det", "cls", "seg", "obb", "pose", "sem")
$families = @(Get-PropertyOrDefault -Object $record.yoloVision -Name "families" -DefaultValue @())
$tasks = @(Get-PropertyOrDefault -Object $record.yoloVision -Name "tasks" -DefaultValue @())

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string]$record.recordKind -eq "post-publish-public-docs-article-sample-readiness-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "state-blocked" ([string]$record.readinessState -eq "blocked-owner-public-publish-and-post-publish-proof-required") "blocker" "Readiness pack must stay blocked until Owner public publish and post-publish proof exist.")) | Out-Null
$items.Add((New-OwnerValidationItem "article-count" ([int]$record.articleCount -ge 30 -and [int]$record.minimumArticleCount -ge 30) "blocker" "Article roadmap should cover at least 30 high-quality articles.")) | Out-Null
$items.Add((New-OwnerValidationItem "required-surfaces" (@($requiredSurfaceIds | Where-Object { $surfaceIds -notcontains $_ }).Count -eq 0) "blocker" "Required sample/application/smoke surfaces must be present.")) | Out-Null
$items.Add((New-OwnerValidationItem "sample-counts" ([int]$record.sampleSurfaceCount -ge 8 -and [int]$record.readySampleSurfaceCount -ge 8 -and [int]$record.missingSamplePathCount -eq 0) "blocker" "All required sample surfaces should have source, README/doc, and matrix/article anchors.")) | Out-Null
$items.Add((New-OwnerValidationItem "yolovision-no-yolodet" (-not [bool]$record.yoloVision.legacyYoloDetPathExists -and [int]$record.yoloVision.legacyYoloDetPublicPathMatchCount -eq 0) "blocker" "YoloVision must remain the live sample identity; YoloDet must not be a live public path.")) | Out-Null
$items.Add((New-OwnerValidationItem "yolo-families" (@($requiredFamilies | Where-Object { $families -notcontains $_ }).Count -eq 0) "blocker" "YoloVision readiness must cover YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom.")) | Out-Null
$items.Add((New-OwnerValidationItem "yolo-tasks" (@($requiredTasks | Where-Object { $tasks -notcontains $_ }).Count -eq 0) "blocker" "YoloVision readiness must cover det/cls/seg/obb/pose/sem.")) | Out-Null
$items.Add((New-OwnerValidationItem "onnx-trtexec-matrix" ([int]$record.onnxToEngine.matrixEntryCount -ge 20 -and [int]$record.tensorRtExec.featureCount -ge 15) "blocker" "OnnxToEngine and TensorRtExec readiness matrices should be broad enough for trtexec-like coverage.")) | Out-Null
$items.Add((New-OwnerValidationItem "owner-action-fields" ([int]$record.ownerActionFieldCount -ge 15) "blocker" "Owner execution field list should cover package URLs, hashes, logs, samples, articles, and close decision.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool]$record.performsPublish) -and (-not [bool]$record.usesPublishToken) -and (-not [bool]$record.canPublishPublicly) -and (-not [bool]$record.canCloseReleaseIssue) -and (-not [bool]$record.canPromoteRuntimeProof) -and (-not [bool]$record.isRuntimeExecutionProof) -and (-not [bool]$record.isPostPublishProof) -and (-not [bool]$record.isReleaseCloseProof)) "blocker" "Pack must not publish, use tokens, close, or promote proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "surface-non-proof" (@($surfaces | Where-Object { [bool]$_.canPromoteRuntimeProof -or [bool]$_.canPromotePostPublishProof -or [bool]$_.canPublishPublicly -or [bool]$_.canCloseReleaseIssue }).Count -eq 0) "blocker" "Sample surfaces must remain readiness entries only.")) | Out-Null

$failedBlockers = @($items.ToArray() | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "post-publish-public-docs-article-sample-readiness-pack-ready-non-proof" } else { "invalid-post-publish-public-docs-article-sample-readiness-pack" }

$validation = [pscustomobject]@{
  recordKind = "post-publish-public-docs-article-sample-readiness-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $state
  readinessState = [string]$record.readinessState
  articleCount = [int]$record.articleCount
  sampleSurfaceCount = [int]$record.sampleSurfaceCount
  readySampleSurfaceCount = [int]$record.readySampleSurfaceCount
  missingSamplePathCount = [int]$record.missingSamplePathCount
  yoloFamilyCount = [int]$record.yoloVision.familyCount
  yoloTaskCount = [int]$record.yoloVision.taskCount
  onnxToEngineMatrixEntryCount = [int]$record.onnxToEngine.matrixEntryCount
  tensorRtExecFeatureCount = [int]$record.tensorRtExec.featureCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Post-publish public docs/article/sample readiness validation is non-proof validation only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-public-docs-article-sample-readiness-pack-validation.json"
$mdPath = Join-Path $OutputRoot "post-publish-public-docs-article-sample-readiness-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Post-Publish Public Docs Article Sample Readiness Pack Validation",
  "",
  "- validationState: ``$state``",
  "- articleCount: ``$($validation.articleCount)``",
  "- sampleSurfaceCount: ``$($validation.sampleSurfaceCount)``",
  "- readySampleSurfaceCount: ``$($validation.readySampleSurfaceCount)``",
  "- missingSamplePathCount: ``$($validation.missingSamplePathCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)

Write-Host "PostPublishPublicDocsArticleSampleReadinessValidationState=$state FailedBlockers=$($failedBlockers.Count) Surfaces=$($validation.sampleSurfaceCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Post-publish public docs/article/sample readiness pack validation failed." }
