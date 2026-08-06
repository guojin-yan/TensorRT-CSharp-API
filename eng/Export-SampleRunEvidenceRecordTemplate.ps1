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
  "real-model-runtime"
)

$requiredEvidenceSummary = @(
  "recordKind identifies this as a sample-run-evidence-record template or real record.",
  "proofClassification starts as template-only and can only promote to real-model-runtime for sample evidence.",
  "modelSha256, labelsSha256, inputAssetSha256, preprocessedInputTensorSha256 when applicable, and sampleRunLogSha256 must be 64-character SHA256 values before promotion.",
  "modelLicense, labelsLicense, inputAssetLicense, and preprocessedInputTensorElementCount must be owner-filled before promotion.",
  "evidenceSidecarPath links the sample run to TensorRtExec/OnnxToEngine model evidence.",
  "buildReportPath links the sample run to parser/builder evidence.",
  "sampleRunCommand and sampleRunLogPath must identify the exact runner invocation and captured log.",
  "YoloVision real image evidence should use --input-data with a preprocessed tensor and record InputSource=external in expected evidence lines.",
  "stdoutSummary or stderrSummary must summarize reviewed runner output.",
  "isSmokePassed and canPromoteRealModelRuntime must remain false in templates."
)

$validationRules = @(
  "template-only records are owner-action-required and not proof.",
  "build-only records are conversion evidence and not sample runtime proof.",
  "dependency-probe-only records are diagnostics and not sample runtime proof.",
  "synthetic-input-runtime records are pipeline evidence and not real model proof.",
  "real-model-runtime records require model, labels, input, run log hashes, stdout/stderr summary, sampleRunLogPath, and isSmokePassed=true.",
  "package-consumer-runtime is forbidden in sample run evidence records and belongs to release proof records."
)

$promotionRules = @(
  "Only proofClassification=real-model-runtime can promote a sample run evidence record.",
  "package-consumer-runtime cannot be claimed by sample run evidence records; sample run evidence is never package-consumer-runtime.",
  "Do not set isSmokePassed=true without a real sample runner log and hash.",
  "Do not fabricate modelSha256, labelsSha256, inputAssetSha256, preprocessedInputTensorSha256, sampleRunLogSha256, stdoutSummary, or stderrSummary.",
  "Do not fabricate modelLicense, labelsLicense, inputAssetLicense, preprocessedInputTensorElementCount, validatorState, or failureReasons.",
  "Run eng/Test-SampleRunEvidenceRecord.ps1, eng/Test-OnnxEngineBuildEvidenceSidecar.ps1, and eng/Test-SampleAssetManifest.ps1 after owner backfill."
)

function New-SampleRunEvidenceRecordTemplate {
  param(
    [string]$TemplateName,
    [string]$SampleName,
    [string]$ManifestPath,
    [string]$ModelPath,
    [string]$LabelsPath,
    [string]$InputAssetPath,
    [string]$PreprocessedInputTensorPath,
    [string]$EvidenceSidecarPath,
    [string]$BuildReportPath,
    [string]$SampleRunCommand,
    [string]$SampleRunLogPath,
    [string[]]$ExpectedEvidenceLines
  )

  [pscustomobject]@{
    schemaVersion = 1
    recordKind = "sample-run-evidence-record-template"
    templateName = $TemplateName
    templateOnly = $true
    sampleName = $SampleName
    manifestPath = $ManifestPath
    proofClassification = "template-only"
    evidenceClassifications = $evidenceClassifications
    modelPath = $ModelPath
    modelSha256 = ""
    modelLicense = "owner-required"
    labelsPath = $LabelsPath
    labelsSha256 = ""
    labelsLicense = "owner-required"
    inputAssetPath = $InputAssetPath
    inputAssetSha256 = ""
    inputAssetLicense = "owner-required"
    preprocessedInputTensorPath = $PreprocessedInputTensorPath
    preprocessedInputTensorSha256 = ""
    preprocessedInputTensorElementCount = ""
    evidenceSidecarPath = $EvidenceSidecarPath
    buildReportPath = $BuildReportPath
    sampleRunCommand = $SampleRunCommand
    sampleRunLogPath = $SampleRunLogPath
    sampleRunLogSha256 = ""
    stdoutSummary = ""
    stderrSummary = ""
    expectedEvidenceLines = $ExpectedEvidenceLines
    isSmokePassed = $false
    canPromoteRealModelRuntime = $false
    validatorState = "owner-action-required"
    failureReasons = @(
      "owner must fill model/labels/input licenses and hashes",
      "owner must capture real sample run log and SHA256",
      "validator must pass before canPromoteRealModelRuntime=true"
    )
    requiredEvidenceSummary = $requiredEvidenceSummary
    validationRules = $validationRules
    promotionRules = $promotionRules
    notes = @(
      "This template does not download assets or run samples.",
      "Keep proofClassification=template-only until the real sample runner log, hashes, and summaries are recorded.",
      "Sample run evidence can promote only to real-model-runtime, never package-consumer-runtime."
    )
  }
}

function Write-SampleRunEvidenceRecordTemplate {
  param(
    [object]$Template,
    [string]$JsonPath,
    [string]$MarkdownPath
  )

  $Template | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $JsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Sample Run Evidence Record Template")
  $lines.Add("")
  $lines.Add("- template name: ``$($Template.templateName)``")
  $lines.Add("- sample name: ``$($Template.sampleName)``")
  $lines.Add("- proof classification: ``$($Template.proofClassification)``")
  $lines.Add("- template only: ``$($Template.templateOnly)``")
  $lines.Add("- validator state: ``$($Template.validatorState)``")
  $lines.Add("- manifest: ``$($Template.manifestPath)``")
  $lines.Add("- model license: ``$($Template.modelLicense)``")
  $lines.Add("- labels license: ``$($Template.labelsLicense)``")
  $lines.Add("- input asset license: ``$($Template.inputAssetLicense)``")
  $lines.Add("- preprocessed tensor element count: ``$($Template.preprocessedInputTensorElementCount)``")
  $lines.Add("- evidence sidecar: ``$($Template.evidenceSidecarPath)``")
  $lines.Add("- build report: ``$($Template.buildReportPath)``")
  $lines.Add("- sample run log: ``$($Template.sampleRunLogPath)``")
  $lines.Add("")
  $lines.Add("## Sample Run Command")
  $lines.Add("")
  if ([string]::IsNullOrWhiteSpace($Template.sampleRunCommand)) {
    $lines.Add("- owner-required")
  }
  else {
    $lines.Add('```powershell')
    $lines.Add($Template.sampleRunCommand)
    $lines.Add('```')
  }
  $lines.Add("")
  $lines.Add("## Expected Evidence Lines")
  $lines.Add("")
  foreach ($line in $Template.expectedEvidenceLines) {
    $lines.Add("- ``$line``")
  }
  $lines.Add("")
  $lines.Add("## Required Evidence")
  $lines.Add("")
  foreach ($item in $Template.requiredEvidenceSummary) {
    $lines.Add("- $item")
  }
  $lines.Add("")
  $lines.Add("## Validation Rules")
  $lines.Add("")
  foreach ($rule in $Template.validationRules) {
    $lines.Add("- $rule")
  }
  $lines.Add("")
  $lines.Add("## Promotion Rules")
  $lines.Add("")
  foreach ($rule in $Template.promotionRules) {
    $lines.Add("- $rule")
  }
  $lines.Add("- Sample run evidence can promote only to real-model-runtime, never package-consumer-runtime.")
  $lines.Add("")
  $lines.Add("## Failure Reasons")
  $lines.Add("")
  foreach ($reason in $Template.failureReasons) {
    $lines.Add("- $reason")
  }

  $lines | Set-Content -LiteralPath $MarkdownPath -Encoding utf8
}

$generic = New-SampleRunEvidenceRecordTemplate `
  -TemplateName "generic" `
  -SampleName "" `
  -ManifestPath "" `
  -ModelPath ".\models\model.onnx" `
  -LabelsPath ".\models\labels.txt" `
  -InputAssetPath ".\models\input.asset" `
  -PreprocessedInputTensorPath "" `
  -EvidenceSidecarPath ".\models\model-evidence.sidecar.json" `
  -BuildReportPath ".\models\model-build-report.json" `
  -SampleRunCommand "dotnet run --project .\samples\<SampleName> -- --model .\models\model.onnx --labels .\models\labels.txt --input-data .\models\input-preprocessed-fp32.bin --input-shape owner-required --tensor-rt-line 10" `
  -SampleRunLogPath ".\models\model-sample-run.log" `
  -ExpectedEvidenceLines @("Sample TensorRtLine=...", "Sample Passed=True")

$genericJsonPath = Join-Path $outputRoot "sample-run-evidence-record.template.json"
$genericMarkdownPath = Join-Path $outputRoot "sample-run-evidence-record.template.md"
Write-SampleRunEvidenceRecordTemplate -Template $generic -JsonPath $genericJsonPath -MarkdownPath $genericMarkdownPath

$sampleSpecificTemplates = New-Object System.Collections.Generic.List[object]

if (-not $SkipSampleSpecific.IsPresent) {
  $definitions = @(
    [pscustomobject]@{
      name = "classification"
      jsonFileName = "sample-run-evidence-record.classification.template.json"
      markdownFileName = "sample-run-evidence-record.classification.template.md"
      sampleName = "Classification"
      manifestPath = "samples/assets/classification-assets.template.json"
      modelPath = ".\models\classifier.onnx"
      labelsPath = ".\models\labels.txt"
      inputAssetPath = ".\models\image.jpg"
      preprocessedInputTensorPath = ""
      evidenceSidecarPath = ".\models\classifier-evidence.sidecar.json"
      buildReportPath = ".\models\classifier-build-report.json"
      sampleRunCommand = "dotnet run --project .\samples\ComputerVision\01.Classification -- --model .\models\classifier.onnx --labels .\models\labels.txt --input .\models\image.jpg --input-shape 1x3x224x224 --tensor-rt-line 10 --top-k 5"
      sampleRunLogPath = ".\models\classifier-sample-run.log"
      expectedEvidenceLines = @("Classification TensorRtLine=...", "TopK Index=... Label=... Score=...", "Classification Passed=True")
    },
    [pscustomobject]@{
      name = "yolovision"
      jsonFileName = "sample-run-evidence-record.yolovision.template.json"
      markdownFileName = "sample-run-evidence-record.yolovision.template.md"
      sampleName = "YoloVision"
      manifestPath = "samples/assets/yolovision-assets.template.json"
      modelPath = ".\models\yolo.onnx"
      labelsPath = ".\models\coco.names"
      inputAssetPath = ".\models\image.jpg"
      preprocessedInputTensorPath = ".\models\yolo-preprocessed-fp32.bin"
      evidenceSidecarPath = ".\models\yolo-evidence.sidecar.json"
      buildReportPath = ".\models\yolo-build-report.json"
      sampleRunCommand = "dotnet run --project .\applications\YoloVision -- --model .\models\yolo.onnx --labels .\models\coco.names --input-data .\models\yolo-preprocessed-fp32.bin --input-shape 1x3x640x640 --tensor-rt-line 10 --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25"
      sampleRunLogPath = ".\models\yolo-sample-run.log"
      expectedEvidenceLines = @("YoloVision TensorRtLine=...", "Profile Family=... Task=... Layout=... Nms=... NmsMode=...", "InputSource=external InputFile=...", "Postprocess Task=...", "YoloVision Passed=True")
    },
    [pscustomobject]@{
      name = "yolox-s"
      jsonFileName = "sample-run-evidence-record.yolox-s.template.json"
      markdownFileName = "sample-run-evidence-record.yolox-s.template.md"
      sampleName = "YoloVision"
      manifestPath = "samples/assets/yolovision-yolox-s-example.json"
      modelPath = ".\models\yolox_s.onnx"
      labelsPath = ".\models\coco.names"
      inputAssetPath = ".\models\yolox-test.jpg"
      preprocessedInputTensorPath = ".\models\yolox_s-preprocessed-fp32.bin"
      evidenceSidecarPath = ".\models\yolox_s-evidence.sidecar.json"
      buildReportPath = ".\models\yolox_s-build-report.json"
      sampleRunCommand = "dotnet run --project .\applications\YoloVision -- --model .\models\yolox_s.onnx --labels .\models\coco.names --input-data .\models\yolox_s-preprocessed-fp32.bin --input-shape 1x3x640x640 --tensor-rt-line 10 --family custom --task det --layout auto --has-objectness auto --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45"
      sampleRunLogPath = ".\models\yolox_s-sample-run.log"
      expectedEvidenceLines = @("YoloVision TensorRtLine=...", "Profile Family=... Task=... Layout=... Nms=... NmsMode=...", "InputSource=external InputFile=...", "Detection Class=... Score=... BoxCxCyWh=...", "YoloVision Passed=True")
    }
  )

  foreach ($definition in $definitions) {
    $template = New-SampleRunEvidenceRecordTemplate `
      -TemplateName $definition.name `
      -SampleName $definition.sampleName `
      -ManifestPath $definition.manifestPath `
      -ModelPath $definition.modelPath `
      -LabelsPath $definition.labelsPath `
      -InputAssetPath $definition.inputAssetPath `
      -PreprocessedInputTensorPath $definition.preprocessedInputTensorPath `
      -EvidenceSidecarPath $definition.evidenceSidecarPath `
      -BuildReportPath $definition.buildReportPath `
      -SampleRunCommand $definition.sampleRunCommand `
      -SampleRunLogPath $definition.sampleRunLogPath `
      -ExpectedEvidenceLines $definition.expectedEvidenceLines

    $jsonPath = Join-Path $outputRoot $definition.jsonFileName
    $markdownPath = Join-Path $outputRoot $definition.markdownFileName
    Write-SampleRunEvidenceRecordTemplate -Template $template -JsonPath $jsonPath -MarkdownPath $markdownPath

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
  exportKind = "sample-run-evidence-record-template-export"
  genericTemplateJson = $genericJsonPath
  genericTemplateMarkdown = $genericMarkdownPath
  sampleSpecificTemplateCount = $sampleSpecificTemplates.Count
  sampleSpecificTemplates = @($sampleSpecificTemplates.ToArray())
  proofClassification = "template-only"
  evidenceClassifications = $evidenceClassifications
  requiredEvidenceSummary = $requiredEvidenceSummary
  validationRules = $validationRules
  promotionRules = $promotionRules
}

$summaryJsonPath = Join-Path $outputRoot "sample-run-evidence-record-template-export.json"
$summaryMarkdownPath = Join-Path $outputRoot "sample-run-evidence-record-template-export.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryJsonPath -Encoding utf8

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("# Sample Run Evidence Record Template Export")
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
$summaryLines.Add("Sample run evidence records connect real Classification/YoloVision runner logs to asset manifests and sidecars. They can promote only to real-model-runtime, never package-consumer-runtime.")
$summaryLines | Set-Content -LiteralPath $summaryMarkdownPath -Encoding utf8

Write-Host "Sample run evidence record template written to $genericJsonPath"
Write-Host "Sample run evidence record template written to $genericMarkdownPath"
Write-Host "Sample run evidence record template export written to $summaryJsonPath"
