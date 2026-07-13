[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function New-Case {
  param(
    [string]$CaseId,
    [string]$CaseName,
    [string]$Scenario,
    [string]$SampleProject,
    [int[]]$ArticleIds,
    [string[]]$RepoPaths,
    [string[]]$RequiredAssets,
    [string[]]$RequiredCommands,
    [string[]]$ExpectedHashFields,
    [string[]]$ExpectedHostMetadata,
    [string[]]$MissingOwnerInputs,
    [string[]]$PromotionBlockers
  )

  [pscustomobject]@{
    caseId = $CaseId
    caseName = $CaseName
    scenario = $Scenario
    sampleProject = $SampleProject
    articleIds = @($ArticleIds)
    repoPaths = @($RepoPaths)
    requiredAssets = @($RequiredAssets)
    requiredCommands = @($RequiredCommands)
    expectedEvidenceRecord = "artifacts/final-release/real-case-evidence-record.json"
    expectedLogPath = "artifacts/user-acceptance/real-case/$CaseId/run.log"
    expectedHashFields = @($ExpectedHashFields)
    expectedHostMetadata = @($ExpectedHostMetadata)
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json"
    proofClassification = "blocked-owner-action-required"
    nonSubstituteProofKinds = @("template", "draft", "runbook", "article planning", "build-only", "sidecar-only", "local feed", "ProjectReference", "dependency-probe-only", "blocked-by-cuda-driver")
    ownerActionRequired = $true
    missingOwnerInputs = @($MissingOwnerInputs)
    promotionBlockers = @($PromotionBlockers)
    canPromoteRealModelRuntime = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$cases = @(
  New-Case `
    -CaseId "yolovision-detection" `
    -CaseName "YoloVision detection real model runtime" `
    -Scenario "YOLO-family detection with real ONNX, labels, preprocessed tensor, run log, hashes, and screenshot." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(7, 70, 71, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-detection-tutorial.md", "docs/articles/zh-cn/yolovision-real-asset-walkthrough.md") `
    -RequiredAssets @("detection ONNX", "labels file", "preprocessed fp32 input tensor", "optional original image", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-det.onnx --labels .\models\coco.names --input-data .\models\det-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --has-objectness auto --nms-mode class-aware") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real detection model", "license", "labels", "input tensor", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real model assets", "missing log/hash/host metadata", "sample-run evidence not validated")

  New-Case `
    -CaseId "yolovision-segmentation" `
    -CaseName "YoloVision segmentation real model runtime" `
    -Scenario "YOLO segmentation with explicit output role map and mask prototype metadata." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(72, 77, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-segmentation-tutorial.md", "docs/articles/zh-cn/yolovision-multi-output-metadata-guide.md") `
    -RequiredAssets @("segmentation ONNX", "labels file", "preprocessed fp32 input tensor", "mask prototype metadata", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-seg.onnx --labels .\models\coco.names --input-data .\models\seg-fp32.bin --input-shape 1x3x640x640 --family v8 --task seg --output-role-map boxes:det,proto:mask-prototypes --mask-coefficient-count 32") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real segmentation model", "license", "mask metadata", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real segmentation assets", "missing output metadata validation", "sample-run evidence not validated")

  New-Case `
    -CaseId "yolovision-obb" `
    -CaseName "YoloVision OBB real model runtime" `
    -Scenario "Oriented bounding box model with angle tensor metadata and real runtime evidence." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(73, 77, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-obb-tutorial.md", "docs/articles/zh-cn/yolovision-multi-output-metadata-guide.md") `
    -RequiredAssets @("OBB ONNX", "labels file", "preprocessed fp32 input tensor", "angle output metadata", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-obb.onnx --labels .\models\labels.txt --input-data .\models\obb-fp32.bin --input-shape 1x3x1024x1024 --family v8 --task obb --output-role-map boxes:det,angles:obb-angle --obb-angle-output angles") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real OBB model", "license", "angle metadata", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real OBB assets", "missing angle metadata validation", "sample-run evidence not validated")

  New-Case `
    -CaseId "yolovision-pose" `
    -CaseName "YoloVision pose real model runtime" `
    -Scenario "Pose model with keypoint tensor metadata and real runtime evidence." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(74, 77, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-pose-tutorial.md", "docs/articles/zh-cn/yolovision-multi-output-metadata-guide.md") `
    -RequiredAssets @("pose ONNX", "labels file", "preprocessed fp32 input tensor", "keypoint metadata", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-pose.onnx --labels .\models\labels.txt --input-data .\models\pose-fp32.bin --input-shape 1x3x640x640 --family v8 --task pose --output-role-map boxes:det,keypoints:pose-keypoints --pose-keypoint-count 17") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real pose model", "license", "keypoint metadata", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real pose assets", "missing keypoint metadata validation", "sample-run evidence not validated")

  New-Case `
    -CaseId "yolovision-classification" `
    -CaseName "YoloVision classification real model runtime" `
    -Scenario "YOLO-family or custom classification model with real logits/top-k output evidence." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(75, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md") `
    -RequiredAssets @("classification ONNX", "labels file", "preprocessed fp32 input tensor", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-cls.onnx --labels .\models\labels.txt --input-data .\models\cls-fp32.bin --input-shape 1x3x224x224 --family custom --task cls --classification-output logits") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real classification model", "license", "labels", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real classification assets", "sample-run evidence not validated")

  New-Case `
    -CaseId "yolovision-semantic" `
    -CaseName "YoloVision semantic segmentation real model runtime" `
    -Scenario "Semantic segmentation model with real semantic-map output evidence and class metadata." `
    -SampleProject "samples/YoloVision" `
    -ArticleIds @(75, 90) `
    -RepoPaths @("samples/YoloVision/README.md", "docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md", "docs/articles/zh-cn/yolovision-output-json-schema-guide.md") `
    -RequiredAssets @("semantic segmentation ONNX", "labels file", "preprocessed fp32 input tensor", "semantic output metadata", "sample-run evidence record") `
    -RequiredCommands @("dotnet run --project .\samples\YoloVision -- --model .\models\yolo-sem.onnx --labels .\models\labels.txt --input-data .\models\sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21") `
    -ExpectedHashFields @("modelSha256", "labelsSha256", "inputAssetSha256", "outputArtifactSha256", "sampleRunLogSha256", "screenshotSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("real semantic segmentation model", "license", "labels", "semantic output metadata", "runtime log", "hashes", "host metadata", "screenshot") `
    -PromotionBlockers @("missing real semantic segmentation assets", "missing semantic output metadata validation", "sample-run evidence not validated")

  New-Case `
    -CaseId "onnx-to-engine-build" `
    -CaseName "OnnxToEngine external ONNX build evidence" `
    -Scenario "External ONNX to serialized engine build report with explicit build-only boundary." `
    -SampleProject "samples/OnnxToEngine" `
    -ArticleIds @(8, 78, 84) `
    -RepoPaths @("samples/OnnxToEngine/README.md", "docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md", "docs/articles/zh-cn/onnxtoengine-and-tensorrtexec-boundary.md") `
    -RequiredAssets @("external ONNX", "shape profile", "build report", "optional engine output", "owner-reviewed sidecar") `
    -RequiredCommands @("dotnet run --project .\samples\OnnxToEngine -- --onnx .\models\model.onnx --saveEngine .\models\model.plan --minShapes input:1x3x640x640 --optShapes input:1x3x640x640 --maxShapes input:4x3x640x640 --exportReport .\models\model-build-report.json --buildOnly") `
    -ExpectedHashFields @("onnxSha256", "engineSha256", "buildReportSha256", "sidecarSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("external ONNX", "license", "build report", "engine hash", "host metadata", "sidecar audit result") `
    -PromotionBlockers @("build-only evidence is not runtime proof", "missing real sample runtime log")

  New-Case `
    -CaseId "tensorrtexec-build-report" `
    -CaseName "TensorRtExec CLI build/report evidence" `
    -Scenario "TensorRtExec command-line build/report workflow with normalized command and report hashes." `
    -SampleProject "applications/TensorRtExec" `
    -ArticleIds @(9, 79, 84) `
    -RepoPaths @("applications/TensorRtExec/README.md", "docs/articles/zh-cn/tensorrtexec-tool-getting-started.md", "docs/articles/zh-cn/tensorrtexec-external-onnx-build-report.md") `
    -RequiredAssets @("external ONNX", "shape profile", "TensorRtExec report", "optional evidence sidecar") `
    -RequiredCommands @("dotnet run --project .\applications\TensorRtExec -- --onnx .\models\model.onnx --saveEngine .\models\model.plan --minShapes input:1x3x640x640 --optShapes input:1x3x640x640 --maxShapes input:4x3x640x640 --buildOnly --exportReport .\models\model-build-report.json") `
    -ExpectedHashFields @("onnxSha256", "engineSha256", "buildReportSha256", "normalizedCommandSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("external ONNX", "license", "build report", "hashes", "host metadata") `
    -PromotionBlockers @("TensorRtExec report is build/report evidence only", "missing real sample runtime proof")

  New-Case `
    -CaseId "tensorrtexec-gui-workflow" `
    -CaseName "TensorRtExec WinForms GUI workflow evidence" `
    -Scenario "Windows GUI workflow screenshot and exported report, explicitly not a runtime proof substitute." `
    -SampleProject "applications/TensorRtExec" `
    -ArticleIds @(9, 80, 84) `
    -RepoPaths @("applications/TensorRtExec/README.md", "docs/articles/zh-cn/tensorrtexec-gui-user-guide.md", "docs/articles/zh-cn/tool-report-to-release-proof-record.md") `
    -RequiredAssets @("external ONNX", "GUI screenshot", "exported report", "owner-reviewed command line") `
    -RequiredCommands @("dotnet run --project .\applications\TensorRtExec -- --ui") `
    -ExpectedHashFields @("onnxSha256", "buildReportSha256", "screenshotSha256", "normalizedCommandSha256") `
    -ExpectedHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion") `
    -MissingOwnerInputs @("GUI screenshot", "external ONNX", "license", "exported report", "hashes", "host metadata") `
    -PromotionBlockers @("GUI screenshot is not runtime proof", "build/report output cannot close release blockers")
)

$blockedCaseCount = @($cases | Where-Object { $_.ownerActionRequired -or -not $_.canPromoteRealModelRuntime }).Count
$missingOwnerInputCount = @($cases | ForEach-Object { @($_.missingOwnerInputs).Count } | Measure-Object -Sum).Sum

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "real-case-proof-execution-pack"
  packState = "blocked-owner-action-required"
  performsRuntimeExecution = $false
  performsPublish = $false
  canPromoteRealModelRuntime = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  caseCount = @($cases).Count
  blockedCaseCount = $blockedCaseCount
  missingOwnerInputCount = [int]$missingOwnerInputCount
  yoloVisionCaseCount = @($cases | Where-Object { $_.sampleProject -eq "samples/YoloVision" }).Count
  onnxToEngineCaseCount = @($cases | Where-Object { $_.sampleProject -eq "samples/OnnxToEngine" }).Count
  tensorRtExecCaseCount = @($cases | Where-Object { $_.sampleProject -eq "applications/TensorRtExec" }).Count
  requiredTaskCoverage = @("det", "seg", "obb", "pose", "cls", "sem")
  requiredToolCoverage = @("OnnxToEngine", "TensorRtExec")
  expectedEvidenceRecordTemplate = "artifacts/final-release/real-case-evidence-record-template.json"
  expectedValidatedEvidenceRecord = "artifacts/final-release/real-case-evidence-record.json"
  validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json"
  releaseProofBoundary = "This pack is an owner execution plan. It does not run models, create screenshots, compute hashes, publish packages, or promote build-only/sidecar/article/template records to runtime proof."
  cases = @($cases)
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "real-case-proof-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "real-case-proof-execution-pack.md"

$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$caseRows = $cases | ForEach-Object {
  $articles = (@($_.articleIds) -join ", ")
  $commands = (@($_.requiredCommands) -join "<br>").Replace("|", "\|")
  $missing = (@($_.missingOwnerInputs) -join "<br>").Replace("|", "\|")
  "| ``$($_.caseId)`` | $($_.caseName.Replace("|", "\|")) | ``$($_.sampleProject)`` | $articles | $commands | ``$($_.proofClassification)`` | $missing |"
}

$markdown = @"
# Real Case Proof Execution Pack

生成时间：$($record.generatedAtUtc)

## Summary

| Item | Value |
|---|---|
| record kind | ``$($record.recordKind)`` |
| pack state | ``$($record.packState)`` |
| performs runtime execution | ``False`` |
| can promote real-model-runtime | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| case count | ``$($record.caseCount)`` |
| blocked case count | ``$($record.blockedCaseCount)`` |
| missing owner input count | ``$($record.missingOwnerInputCount)`` |
| YoloVision cases | ``$($record.yoloVisionCaseCount)`` |
| OnnxToEngine cases | ``$($record.onnxToEngineCaseCount)`` |
| TensorRtExec cases | ``$($record.tensorRtExecCaseCount)`` |

## Cases

| Case ID | Case name | Sample project | Article IDs | Required command | Proof classification | Missing owner input |
|---|---|---|---|---|---|---|
$($caseRows -join "`r`n")

## Boundary

$($record.releaseProofBoundary)

Build-only reports, sidecars, screenshots, article planning, templates, local feeds, ProjectReference consumers, dependency probes, and blocked-by-cuda-driver records cannot substitute owner authorization, package-consumer-runtime, Linux runner, real-model-runtime, or post-publish verification proof.
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Real case proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "CaseCount=$($record.caseCount)"
Write-Output "BlockedCaseCount=$($record.blockedCaseCount)"
Write-Output "CanPublishPublicly=False"
