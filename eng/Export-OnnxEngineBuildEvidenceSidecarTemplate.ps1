[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [switch]$SkipSampleSpecific
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = "artifacts\user-acceptance"
}

if ([IO.Path]::IsPathRooted($OutputRoot)) {
  $outputRoot = $OutputRoot
}
else {
  $outputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$evidenceClassifications = @(
  "template-only",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$requiredEvidenceSummary = @(
  "proofClassification records the evidence boundary; templates start as template-only.",
  "modelEvidence.modelSource identifies the ONNX file or source URL used by TensorRtExec/OnnxToEngine.",
  "modelEvidence.modelSha256 must be a 64-character SHA256 before real-model-runtime is promotable.",
  "modelEvidence.modelLicense records the reviewed model/license terms.",
  "modelEvidence.inputAssetName and modelEvidence.inputAssetSha256 identify the real source input asset.",
  "modelEvidence.preprocessedInputTensorName and modelEvidence.preprocessedInputTensorSha256 identify the tensor passed to --input-data when the sample uses an external preprocessed tensor.",
  "stdoutSummary or stderrSummary must summarize the reviewed build/sample run output.",
  "The sidecar enriches reports but does not mark Classification/YoloVision smoke passed."
)

$classificationRules = @(
  "template-only sidecars are owner templates and are not proof.",
  "build-only sidecars can enrich parser/builder reports but are not runtime proof.",
  "dependency-probe-only sidecars are diagnostics only.",
  "synthetic-input-runtime sidecars are pipeline evidence, not real model proof.",
  "real-model-runtime sidecars require model/input hashes, license, input asset, and stdout/stderr summary.",
  "package-consumer-runtime belongs to release proof records; a sidecar can record it but cannot promote a build report."
)

$promotionRules = @(
  "Do not use a sidecar to promote TensorRtExec or OnnxToEngine build reports to package-consumer-runtime.",
  "package-consumer-runtime cannot be claimed by TensorRtExec/OnnxToEngine build reports.",
  "Do not mark a sample manifest smoke-passed until the sample runner uses real assets and records log SHA256.",
  "Do not fabricate modelSha256, inputAssetSha256, stdoutSummary, or stderrSummary.",
  "Run eng/Test-OnnxEngineBuildEvidenceSidecar.ps1 and eng/Test-SampleAssetManifest.ps1 after backfill."
)

function New-SidecarTemplate {
  param(
    [string]$TemplateName,
    [string]$SampleName,
    [string]$ModelSource,
    [string]$InputAssetName,
    [string]$PreprocessedInputTensorName,
    [string]$RecommendedBuildCommand,
    [string]$RecommendedSampleRunCommand,
    [string]$ManifestPath
  )

  [pscustomobject]@{
    schemaVersion = 1
    recordKind = "onnx-engine-build-evidence-sidecar-template"
    templateName = $TemplateName
    templateOnly = $true
    sampleName = $SampleName
    proofClassification = "template-only"
    evidenceClassifications = $evidenceClassifications
    stdoutSummary = ""
    stderrSummary = ""
    modelEvidence = [pscustomobject]@{
      modelSource = $ModelSource
      modelSha256 = ""
      modelLicense = ""
      inputAssetName = $InputAssetName
      inputAssetSha256 = ""
      preprocessedInputTensorName = $PreprocessedInputTensorName
      preprocessedInputTensorSha256 = ""
    }
    recommendedBuildCommand = $RecommendedBuildCommand
    recommendedSampleRunCommand = $RecommendedSampleRunCommand
    manifestPath = $ManifestPath
    requiredEvidenceSummary = $requiredEvidenceSummary
    classificationRules = $classificationRules
    promotionRules = $promotionRules
    notes = @(
      "Fill this sidecar after model, labels, image, license, and command output are known.",
      "Keep proofClassification=template-only until there is a real build or sample run record.",
      "package-consumer-runtime cannot be claimed by TensorRtExec/OnnxToEngine build reports."
    )
  }
}

function Write-SidecarTemplate {
  param(
    [object]$Template,
    [string]$JsonPath,
    [string]$MarkdownPath
  )

  $Template | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $JsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# ONNX Engine Build Evidence Sidecar Template")
  $lines.Add("")
  $lines.Add("- template name: ``$($Template.templateName)``")
  $lines.Add("- sample name: ``$($Template.sampleName)``")
  $lines.Add("- proof classification: ``$($Template.proofClassification)``")
  $lines.Add("- template only: ``$($Template.templateOnly)``")
  $lines.Add("- model source: ``$($Template.modelEvidence.modelSource)``")
  $lines.Add("- input asset: ``$($Template.modelEvidence.inputAssetName)``")
  if (-not [string]::IsNullOrWhiteSpace($Template.modelEvidence.preprocessedInputTensorName)) {
    $lines.Add("- preprocessed input tensor: ``$($Template.modelEvidence.preprocessedInputTensorName)``")
  }
  $lines.Add("")
  $lines.Add("## Required Evidence")
  $lines.Add("")
  foreach ($item in $Template.requiredEvidenceSummary) {
    $lines.Add("- $item")
  }
  $lines.Add("")
  $lines.Add("## Recommended Commands")
  $lines.Add("")
  if (-not [string]::IsNullOrWhiteSpace($Template.recommendedBuildCommand)) {
    $lines.Add("Build/report:")
    $lines.Add("")
    $lines.Add('```powershell')
    $lines.Add($Template.recommendedBuildCommand)
    $lines.Add('```')
    $lines.Add("")
  }
  if (-not [string]::IsNullOrWhiteSpace($Template.recommendedSampleRunCommand)) {
    $lines.Add("Sample run:")
    $lines.Add("")
    $lines.Add('```powershell')
    $lines.Add($Template.recommendedSampleRunCommand)
    $lines.Add('```')
    $lines.Add("")
  }
  $lines.Add("## Classification Rules")
  $lines.Add("")
  foreach ($rule in $Template.classificationRules) {
    $lines.Add("- $rule")
  }
  $lines.Add("")
  $lines.Add("## Promotion Rules")
  $lines.Add("")
  foreach ($rule in $Template.promotionRules) {
    $lines.Add("- $rule")
  }

  $lines | Set-Content -LiteralPath $MarkdownPath -Encoding utf8
}

$generic = New-SidecarTemplate `
  -TemplateName "generic" `
  -SampleName "" `
  -ModelSource ".\models\model.onnx" `
  -InputAssetName ".\models\input.asset" `
  -PreprocessedInputTensorName "" `
  -RecommendedBuildCommand "dotnet run --project .\applications\TensorRtExec -- --onnx .\models\model.onnx --saveEngine .\models\model.plan --buildOnly --exportReport .\models\model-build-report.json --evidenceSidecar .\models\model-evidence.sidecar.json" `
  -RecommendedSampleRunCommand "" `
  -ManifestPath ""

$genericJsonPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar.template.json"
$genericMarkdownPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar.template.md"
Write-SidecarTemplate -Template $generic -JsonPath $genericJsonPath -MarkdownPath $genericMarkdownPath

$sampleSpecificTemplates = New-Object System.Collections.Generic.List[object]

if (-not $SkipSampleSpecific.IsPresent) {
  $definitions = @(
    [pscustomobject]@{
      name = "classification"
      jsonFileName = "onnx-engine-build-evidence-sidecar.classification.template.json"
      markdownFileName = "onnx-engine-build-evidence-sidecar.classification.template.md"
      sampleName = "Classification"
      modelSource = ".\models\classifier.onnx"
      inputAssetName = ".\models\image.jpg"
      preprocessedInputTensorName = ""
      buildCommand = "dotnet run --project .\applications\TensorRtExec -- --onnx .\models\classifier.onnx --saveEngine .\models\classifier.plan --minShapes input:1x3x224x224 --optShapes input:1x3x224x224 --maxShapes input:1x3x224x224 --buildOnly --exportReport .\models\classifier-build-report.json --evidenceSidecar .\models\classifier-evidence.sidecar.json"
      runCommand = "dotnet run --project .\samples\ComputerVision\01.Classification -- --model .\models\classifier.onnx --labels .\models\labels.txt --input .\models\image.jpg --input-shape 1x3x224x224 --tensor-rt-line 10 --top-k 5"
      manifestPath = "samples/assets/classification-assets.template.json"
    },
    [pscustomobject]@{
      name = "yolovision"
      jsonFileName = "onnx-engine-build-evidence-sidecar.yolovision.template.json"
      markdownFileName = "onnx-engine-build-evidence-sidecar.yolovision.template.md"
      sampleName = "YoloVision"
      modelSource = ".\models\yolo.onnx"
      inputAssetName = ".\models\image.jpg"
      preprocessedInputTensorName = ".\models\yolo-preprocessed-fp32.bin"
      buildCommand = "dotnet run --project .\applications\TensorRtExec -- --onnx .\models\yolo.onnx --saveEngine .\models\yolo.plan --minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:1x3x640x640 --buildOnly --exportReport .\models\yolo-build-report.json --evidenceSidecar .\models\yolo-evidence.sidecar.json"
      runCommand = "dotnet run --project .\applications\YoloVision -- --model .\models\yolo.onnx --labels .\models\coco.names --input-data .\models\yolo-preprocessed-fp32.bin --input-shape 1x3x640x640 --tensor-rt-line 10 --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25"
      manifestPath = "samples/assets/yolovision-assets.template.json"
    },
    [pscustomobject]@{
      name = "yolox-s"
      jsonFileName = "onnx-engine-build-evidence-sidecar.yolox-s.template.json"
      markdownFileName = "onnx-engine-build-evidence-sidecar.yolox-s.template.md"
      sampleName = "YoloVision"
      modelSource = ".\models\yolox_s.onnx"
      inputAssetName = ".\models\yolox-test.jpg"
      preprocessedInputTensorName = ".\models\yolox_s-preprocessed-fp32.bin"
      buildCommand = "dotnet run --project .\applications\TensorRtExec -- --onnx .\models\yolox_s.onnx --saveEngine .\models\yolox_s.plan --minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:1x3x640x640 --fp16 --workspace 512 --buildOnly --exportReport .\models\yolox_s-build-report.json --evidenceSidecar .\models\yolox_s-evidence.sidecar.json"
      runCommand = "dotnet run --project .\applications\YoloVision -- --model .\models\yolox_s.onnx --labels .\models\coco.names --input-data .\models\yolox_s-preprocessed-fp32.bin --input-shape 1x3x640x640 --tensor-rt-line 10 --family custom --task det --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45"
      manifestPath = "samples/assets/yolovision-yolox-s-example.json"
    }
  )

  foreach ($definition in $definitions) {
    $template = New-SidecarTemplate `
      -TemplateName $definition.name `
      -SampleName $definition.sampleName `
      -ModelSource $definition.modelSource `
      -InputAssetName $definition.inputAssetName `
      -PreprocessedInputTensorName $definition.preprocessedInputTensorName `
      -RecommendedBuildCommand $definition.buildCommand `
      -RecommendedSampleRunCommand $definition.runCommand `
      -ManifestPath $definition.manifestPath
    $jsonPath = Join-Path $outputRoot $definition.jsonFileName
    $markdownPath = Join-Path $outputRoot $definition.markdownFileName
    Write-SidecarTemplate -Template $template -JsonPath $jsonPath -MarkdownPath $markdownPath

    $sampleSpecificTemplates.Add([pscustomobject]@{
        templateName = $definition.name
        sampleName = $definition.sampleName
        jsonPath = $jsonPath
        markdownPath = $markdownPath
        manifestPath = $definition.manifestPath
      })
  }
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  exportKind = "onnx-engine-build-evidence-sidecar-template-export"
  genericTemplateJson = $genericJsonPath
  genericTemplateMarkdown = $genericMarkdownPath
  sampleSpecificTemplateCount = $sampleSpecificTemplates.Count
  sampleSpecificTemplates = @($sampleSpecificTemplates.ToArray())
  proofClassification = "template-only"
  evidenceClassifications = $evidenceClassifications
  requiredEvidenceSummary = $requiredEvidenceSummary
  promotionRules = $promotionRules
}

$summaryJsonPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar-template-export.json"
$summaryMarkdownPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar-template-export.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryJsonPath -Encoding utf8

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("# ONNX Engine Build Evidence Sidecar Template Export")
$summaryLines.Add("")
$summaryLines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$summaryLines.Add("- proof classification: ``template-only``")
$summaryLines.Add("- generic template JSON: ``$genericJsonPath``")
$summaryLines.Add("- generic template Markdown: ``$genericMarkdownPath``")
$summaryLines.Add("- sample-specific template count: $($summary.sampleSpecificTemplateCount)")
$summaryLines.Add("")
foreach ($template in $sampleSpecificTemplates) {
  $summaryLines.Add("- ``$($template.templateName)`` -> ``$($template.jsonPath)``")
}
$summaryLines.Add("")
$summaryLines.Add("Sidecars enrich TensorRtExec/OnnxToEngine reports. They do not replace sample runner logs or package-consumer runtime proof records.")
$summaryLines | Set-Content -LiteralPath $summaryMarkdownPath -Encoding utf8

Write-Host "ONNX engine build evidence sidecar template written to $genericJsonPath"
Write-Host "ONNX engine build evidence sidecar template written to $genericMarkdownPath"
Write-Host "ONNX engine build evidence sidecar template export written to $summaryJsonPath"
