[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Test-RelativePath {
  param([string]$RelativePath)
  return Test-Path -LiteralPath (Join-Path $RepositoryRoot $RelativePath)
}

function New-CatalogItem {
  param(
    [string]$Name,
    [string]$Kind,
    [string]$RelativePath,
    [string]$ProjectPath,
    [string]$Status,
    [string]$Command,
    [string]$Evidence,
    [string]$AssetRequirement,
    [string]$Purpose,
    [string]$Notes
  )

  [pscustomobject]@{
    name = $Name
    kind = $Kind
    path = $RelativePath
    projectPath = $ProjectPath
    status = $Status
    command = $Command
    evidence = $Evidence
    assetRequirement = $AssetRequirement
    purpose = $Purpose
    notes = $Notes
    pathExists = Test-RelativePath $RelativePath
    projectExists = Test-RelativePath $ProjectPath
  }
}

$sampleAssetAuditPath = Join-Path $RepositoryRoot "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$sampleAssetAudit = $null
if (Test-Path -LiteralPath $sampleAssetAuditPath -PathType Leaf) {
  $sampleAssetAudit = Get-Content -LiteralPath $sampleAssetAuditPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$sampleAssetAuditStatus = if ($sampleAssetAudit -and [int]$sampleAssetAudit.errorCount -eq 0) { "ready" } elseif ($sampleAssetAudit) { "has-errors" } else { "missing" }
$sampleAssetAuditEvidence = "artifacts/user-acceptance/sample-asset-manifest-audit.json"
$sampleAssetAcquisitionPlanPath = Join-Path $RepositoryRoot "artifacts\user-acceptance\sample-asset-acquisition-plan.json"
$sampleAssetAcquisitionPlan = $null
if (Test-Path -LiteralPath $sampleAssetAcquisitionPlanPath -PathType Leaf) {
  $sampleAssetAcquisitionPlan = Get-Content -LiteralPath $sampleAssetAcquisitionPlanPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$sampleAssetAcquisitionPlanState = if ($sampleAssetAcquisitionPlan) { [string]$sampleAssetAcquisitionPlan.planState } else { "missing" }
$sampleAssetAcquisitionPlanEvidence = "artifacts/user-acceptance/sample-asset-acquisition-plan.json"
$sampleRunEvidenceValidationPath = Join-Path $RepositoryRoot "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleRunEvidenceValidation = $null
if (Test-Path -LiteralPath $sampleRunEvidenceValidationPath -PathType Leaf) {
  $sampleRunEvidenceValidation = Get-Content -LiteralPath $sampleRunEvidenceValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$sampleRunEvidenceValidationState = if ($sampleRunEvidenceValidation) { [string]$sampleRunEvidenceValidation.validationState } else { "missing" }
$sampleRunEvidenceCanPromoteRealModelRuntime = if ($sampleRunEvidenceValidation) { [bool]$sampleRunEvidenceValidation.canPromoteRealModelRuntime } else { $false }
$sampleRunEvidenceProofClassification = if ($sampleRunEvidenceValidation) { [string]$sampleRunEvidenceValidation.proofClassification } else { "missing-proof-classification" }
$sampleRunEvidenceValidationEvidence = "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
$realModelOwnerHandoffPath = Join-Path $RepositoryRoot "artifacts\user-acceptance\real-model-owner-handoff.json"
$realModelOwnerHandoff = $null
if (Test-Path -LiteralPath $realModelOwnerHandoffPath -PathType Leaf) {
  $realModelOwnerHandoff = Get-Content -LiteralPath $realModelOwnerHandoffPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$realModelOwnerHandoffState = if ($realModelOwnerHandoff) { [string]$realModelOwnerHandoff.handoffState } else { "missing" }
$realModelOwnerHandoffCanPromoteRealModelRuntime = if ($realModelOwnerHandoff) { [bool]$realModelOwnerHandoff.canPromoteRealModelRuntime } else { $false }
$realModelOwnerHandoffProofClassification = if ($realModelOwnerHandoff) { [string]$realModelOwnerHandoff.proofClassification } else { "missing-proof-classification" }
$realModelOwnerHandoffEvidence = "artifacts/user-acceptance/real-model-owner-handoff.json"
$classificationAssetStatus = "asset-required"
$classificationProofClassification = "missing-proof-classification"
$classificationSampleRunEvidenceRecord = ""
$classificationSampleRunEvidenceValidation = ""
$classificationSampleRunEvidenceCrossCheckState = "missing"
$yoloVisionAssetStatus = "asset-required"
$yoloVisionProofClassification = "missing-proof-classification"
$yoloVisionSampleRunEvidenceRecord = ""
$yoloVisionSampleRunEvidenceValidation = ""
$yoloVisionSampleRunEvidenceCrossCheckState = "missing"
if ($sampleAssetAudit) {
  $classificationManifest = $sampleAssetAudit.items | Where-Object { $_.sampleName -eq "Classification" } | Select-Object -First 1
  $yoloManifest = $sampleAssetAudit.items | Where-Object { $_.sampleName -eq "YoloVision" } | Select-Object -First 1
  if ($classificationManifest) {
    $classificationAssetStatus = [string]$classificationManifest.status
    $classificationProofClassification = [string]$classificationManifest.proofClassification
    $classificationSampleRunEvidenceRecord = [string]$classificationManifest.sampleRunEvidenceRecord
    $classificationSampleRunEvidenceValidation = [string]$classificationManifest.sampleRunEvidenceValidation
    $classificationSampleRunEvidenceCrossCheckState = [string]$classificationManifest.sampleRunEvidenceCrossCheckState
  }
  if ($yoloManifest) {
    $yoloVisionAssetStatus = [string]$yoloManifest.status
    $yoloVisionProofClassification = [string]$yoloManifest.proofClassification
    $yoloVisionSampleRunEvidenceRecord = [string]$yoloManifest.sampleRunEvidenceRecord
    $yoloVisionSampleRunEvidenceValidation = [string]$yoloManifest.sampleRunEvidenceValidation
    $yoloVisionSampleRunEvidenceCrossCheckState = [string]$yoloManifest.sampleRunEvidenceCrossCheckState
  }
}

$items = @(
  New-CatalogItem -Name "MultiStream" -Kind "sample" -RelativePath "samples\MultiStream" -ProjectPath "samples\MultiStream\MultiStream.csproj" -Status "ready-to-run" -Command "dotnet run --project .\samples\MultiStream\MultiStream.csproj -c Debug" -Evidence "samples/MultiStream/README.md" -AssetRequirement "CUDA runtime and local bridge probing." -Purpose "CUDA stream/event and cross-stream wait tutorial." -Notes "User-facing low-asset sample; GPU/runtime compatibility is still required."
  New-CatalogItem -Name "DynamicShape" -Kind "sample" -RelativePath "samples\DynamicShape" -ProjectPath "samples\DynamicShape\DynamicShape.csproj" -Status "ready-to-run" -Command "dotnet run --project .\samples\DynamicShape\DynamicShape.csproj -c Debug" -Evidence "samples/DynamicShape/README.md" -AssetRequirement "TensorRT/CUDA runtime and local bridge probing." -Purpose "Dynamic shape, optimization profile, binding, and enqueue tutorial." -Notes "Useful first TensorRT sample after package consumer validation."
  New-CatalogItem -Name "InferenceBindings" -Kind "sample" -RelativePath "samples\InferenceBindings" -ProjectPath "samples\InferenceBindings\InferenceBindings.csproj" -Status "ready-to-run" -Command "dotnet run --project .\samples\InferenceBindings\InferenceBindings.csproj -c Debug" -Evidence "samples/InferenceBindings/README.md" -AssetRequirement "TensorRT/CUDA runtime and local bridge probing." -Purpose "High-level TensorRtInferenceBindings host/device workflow." -Notes "Shows user-facing binding shape without raw native pointers."
  New-CatalogItem -Name "OnnxToEngine" -Kind "sample" -RelativePath "samples\OnnxToEngine" -ProjectPath "samples\OnnxToEngine\OnnxToEngine.csproj" -Status "ready-to-run" -Command "dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -c Debug" -Evidence "samples/OnnxToEngine/README.md" -AssetRequirement "TensorRT/CUDA runtime and ONNX parser availability for the selected TensorRT line." -Purpose "ONNX parse, engine build, serialize, and deserialize walkthrough." -Notes "Parser support remains version-guarded, especially on TensorRT 8 Windows."
  New-CatalogItem -Name "Classification" -Kind "sample" -RelativePath "samples\Classification" -ProjectPath "samples\Classification\Classification.csproj" -Status "asset-required" -Command "dotnet run --project .\samples\Classification\Classification.csproj -c Debug -- --model .\models\classifier.onnx --labels .\models\labels.txt --input .\models\image.jpg" -Evidence "samples/Classification/README.md; samples/assets/classification-assets.template.json; artifacts/user-acceptance/sample-asset-manifest-audit.json; artifacts/user-acceptance/sample-asset-acquisition-plan.json; artifacts/user-acceptance/sample-run-evidence-record-validation.json; artifacts/user-acceptance/real-model-owner-handoff.json" -AssetRequirement "User-provided classifier ONNX, labels, and input image with compatible redistribution rights. Manifest status: $classificationAssetStatus; proof classification: $classificationProofClassification; acquisition plan: $sampleAssetAcquisitionPlanState; runner evidence state: $sampleRunEvidenceValidationState; runner can promote real model runtime: $sampleRunEvidenceCanPromoteRealModelRuntime; runner proof classification: $sampleRunEvidenceProofClassification; owner handoff state: $realModelOwnerHandoffState; owner handoff can promote real model runtime: $realModelOwnerHandoffCanPromoteRealModelRuntime; sample run evidence record: $classificationSampleRunEvidenceRecord; sample run evidence validation: $classificationSampleRunEvidenceValidation; manifest runner cross-check: $classificationSampleRunEvidenceCrossCheckState." -Purpose "Image classification deployment tutorial." -Notes "Asset manifest audit status: $sampleAssetAuditStatus. Candidate manifests, build-only records, acquisition plans, owner handoffs, and owner-action-required runner evidence are not sample smoke passes."
  New-CatalogItem -Name "YoloVision" -Kind "sample" -RelativePath "samples\YoloVision" -ProjectPath "samples\YoloVision\YoloVision.csproj" -Status "asset-required" -Command "dotnet run --project .\samples\YoloVision\YoloVision.csproj -c Debug -- --model .\models\yolo.onnx --labels .\models\coco.names --input-data .\models\yolo-preprocessed-fp32.bin --input-shape 1x3x640x640 --nms-mode class-aware" -Evidence "samples/YoloVision/README.md; samples/assets/yolovision-assets.template.json; artifacts/user-acceptance/sample-asset-manifest-audit.json; artifacts/user-acceptance/sample-asset-acquisition-plan.json; artifacts/user-acceptance/sample-run-evidence-record-validation.json; artifacts/user-acceptance/real-model-owner-handoff.json" -AssetRequirement "User-provided YOLO-family ONNX, COCO labels, source image, and preprocessed input tensor with compatible redistribution rights. Manifest status: $yoloVisionAssetStatus; proof classification: $yoloVisionProofClassification; acquisition plan: $sampleAssetAcquisitionPlanState; runner evidence state: $sampleRunEvidenceValidationState; runner can promote real model runtime: $sampleRunEvidenceCanPromoteRealModelRuntime; runner proof classification: $sampleRunEvidenceProofClassification; owner handoff state: $realModelOwnerHandoffState; owner handoff can promote real model runtime: $realModelOwnerHandoffCanPromoteRealModelRuntime; sample run evidence record: $yoloVisionSampleRunEvidenceRecord; sample run evidence validation: $yoloVisionSampleRunEvidenceValidation; manifest runner cross-check: $yoloVisionSampleRunEvidenceCrossCheckState." -Purpose "YOLO-family vision deployment tutorial." -Notes "Asset manifest audit status: $sampleAssetAuditStatus. Candidate manifests, build-only records, acquisition plans, owner handoffs, and owner-action-required runner evidence are not sample smoke passes."
  New-CatalogItem -Name "PluginRegistryInventorySmokeRunner" -Kind "smoke" -RelativePath "smoke\PluginRegistryInventorySmokeRunner" -ProjectPath "smoke\PluginRegistryInventorySmokeRunner\PluginRegistryInventorySmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\PluginRegistryInventorySmokeRunner\PluginRegistryInventorySmokeRunner.csproj -c Debug -- --dependency-probe-only" -Evidence "tests/JYPPX.ProjectQuality.Tests/PluginRegistryInventoryTests.cs" -AssetRequirement "TensorRT runtime with plugin registry available." -Purpose "Plugin creator inventory and read-only registry validation." -Notes "Cataloged for user acceptance; execution remains environment-specific."
  New-CatalogItem -Name "CallbackAllocatorSafeControlsSmokeRunner" -Kind "smoke" -RelativePath "smoke\CallbackAllocatorSafeControlsSmokeRunner" -ProjectPath "smoke\CallbackAllocatorSafeControlsSmokeRunner\CallbackAllocatorSafeControlsSmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\CallbackAllocatorSafeControlsSmokeRunner\CallbackAllocatorSafeControlsSmokeRunner.csproj -c Debug -- --dependency-probe-only" -Evidence "smoke/README.md" -AssetRequirement "TensorRT/CUDA runtime for dependency probe; real callback proof requires stricter full package evidence." -Purpose "Callback/allocator safety gates and proof-boundary diagnostics." -Notes "Does not prove real callback runtime; InvocationCount=0 remains non-proof."
  New-CatalogItem -Name "CudaSmokeRunner" -Kind "smoke" -RelativePath "smoke\CudaSmokeRunner" -ProjectPath "smoke\CudaSmokeRunner\CudaSmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\CudaSmokeRunner\CudaSmokeRunner.csproj -c Debug" -Evidence "smoke/README.md" -AssetRequirement "Compatible CUDA runtime and driver." -Purpose "CUDA device, stream, event, memory, and runtime diagnostics." -Notes "May be blocked by CUDA driver/runtime mismatch."
  New-CatalogItem -Name "TensorRtSmokeRunner" -Kind "smoke" -RelativePath "smoke\TensorRtSmokeRunner" -ProjectPath "smoke\TensorRtSmokeRunner\TensorRtSmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\TensorRtSmokeRunner\TensorRtSmokeRunner.csproj -c Debug -- --tensor-rt-line 11" -Evidence "smoke/README.md" -AssetRequirement "TensorRT/CUDA runtime for selected line." -Purpose "TensorRT builder/runtime/engine/context diagnostics." -Notes "Runtime line and CUDA compatibility must match."
  New-CatalogItem -Name "OnnxToEngineSmokeRunner" -Kind "smoke" -RelativePath "smoke\OnnxToEngineSmokeRunner" -ProjectPath "smoke\OnnxToEngineSmokeRunner\OnnxToEngineSmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\OnnxToEngineSmokeRunner\OnnxToEngineSmokeRunner.csproj -c Debug -- --tensor-rt-line 11" -Evidence "smoke/README.md" -AssetRequirement "TensorRT/CUDA runtime and ONNX parser for selected line." -Purpose "ONNX parser to engine smoke path." -Notes "Parser support is version-guarded."
  New-CatalogItem -Name "CudaGraphSmokeRunner" -Kind "smoke" -RelativePath "smoke\CudaGraphSmokeRunner" -ProjectPath "smoke\CudaGraphSmokeRunner\CudaGraphSmokeRunner.csproj" -Status "cataloged-not-run" -Command "dotnet run --project .\smoke\CudaGraphSmokeRunner\CudaGraphSmokeRunner.csproj -c Debug" -Evidence "smoke/README.md" -AssetRequirement "Compatible CUDA graph-capable runtime." -Purpose "CUDA graph construction, launch, and diagnostics." -Notes "May be skipped or blocked on older CUDA/driver stacks."
)

$missingItems = @($items | Where-Object { -not $_.pathExists -or -not $_.projectExists })
$outputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  itemCount = $items.Count
  missingItemCount = $missingItems.Count
  sampleAssetManifestAuditStatus = $sampleAssetAuditStatus
  sampleAssetManifestAuditEvidence = $sampleAssetAuditEvidence
  sampleAssetManifestCount = if ($sampleAssetAudit) { [int]$sampleAssetAudit.manifestCount } else { 0 }
  sampleAssetManifestErrorCount = if ($sampleAssetAudit) { [int]$sampleAssetAudit.errorCount } else { -1 }
  sampleAssetAcquisitionPlanState = $sampleAssetAcquisitionPlanState
  sampleAssetAcquisitionPlanEvidence = $sampleAssetAcquisitionPlanEvidence
  sampleAssetAcquisitionPlanItemCount = if ($sampleAssetAcquisitionPlan) { [int]$sampleAssetAcquisitionPlan.itemCount } else { 0 }
  sampleRunEvidenceValidationState = $sampleRunEvidenceValidationState
  sampleRunEvidenceValidationEvidence = $sampleRunEvidenceValidationEvidence
  sampleRunEvidenceCanPromoteRealModelRuntime = $sampleRunEvidenceCanPromoteRealModelRuntime
  sampleRunEvidenceProofClassification = $sampleRunEvidenceProofClassification
  realModelOwnerHandoffState = $realModelOwnerHandoffState
  realModelOwnerHandoffEvidence = $realModelOwnerHandoffEvidence
  realModelOwnerHandoffCanPromoteRealModelRuntime = $realModelOwnerHandoffCanPromoteRealModelRuntime
  realModelOwnerHandoffProofClassification = $realModelOwnerHandoffProofClassification
  statusCounts = @($items | Group-Object status | Sort-Object Name | ForEach-Object {
      [pscustomobject]@{
        status = $_.Name
        count = $_.Count
      }
    })
  items = @($items)
}

$jsonPath = Join-Path $outputRoot "sample-smoke-catalog.json"
$markdownPath = Join-Path $outputRoot "sample-smoke-catalog.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# User Acceptance Sample And Smoke Catalog")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$lines.Add("- item count: $($summary.itemCount)")
$lines.Add("- missing item count: $($summary.missingItemCount)")
$lines.Add("- sample asset manifest audit: ``$($summary.sampleAssetManifestAuditStatus)``")
$lines.Add("- sample asset manifest errors: $($summary.sampleAssetManifestErrorCount)")
$lines.Add("- sample asset acquisition plan: ``$($summary.sampleAssetAcquisitionPlanState)``")
$lines.Add("- sample run evidence validation: ``$($summary.sampleRunEvidenceValidationState)``")
$lines.Add("- sample run evidence can promote real model runtime: ``$($summary.sampleRunEvidenceCanPromoteRealModelRuntime)``")
$lines.Add("- sample run evidence proof classification: ``$($summary.sampleRunEvidenceProofClassification)``")
$lines.Add("- real model owner handoff: ``$($summary.realModelOwnerHandoffState)``")
$lines.Add("- real model owner handoff can promote real model runtime: ``$($summary.realModelOwnerHandoffCanPromoteRealModelRuntime)``")
$lines.Add("")
$lines.Add("| Name | Kind | Status | Command | Assets | Notes |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($item in $items) {
  $command = ([string]$item.command).Replace("|", "\|")
  $assets = ([string]$item.assetRequirement).Replace("|", "\|")
  $notes = ([string]$item.notes).Replace("|", "\|")
  $lines.Add("| $($item.name) | $($item.kind) | $($item.status) | ``$command`` | $assets | $notes |")
}
$lines.Add("")
$lines.Add("`cataloged-not-run` means the project exists and is part of user acceptance planning, not that it passed on the current machine. CUDA error 35, missing model assets, and real callback proof must remain separate evidence categories.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "User acceptance catalog written to $jsonPath"
Write-Host "User acceptance catalog written to $markdownPath"

if ($missingItems.Count -gt 0) {
  Write-Warning "User acceptance catalog has $($missingItems.Count) missing item(s)."
  exit 1
}
