[CmdletBinding()]
param(
  [string]$RepositoryRoot
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

function ConvertTo-RelativePath {
  param([string]$Path)

  return $Path.Substring($RepositoryRoot.Length).TrimStart('\')
}

function Get-StringValue {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return [string]$Value
}

function Get-PropertyOrNull {
  param([AllowNull()][object]$Object, [string]$Name)
  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
    return $null
  }
  return $Object.$Name
}

function Get-StringProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  return Get-StringValue (Get-PropertyOrNull -Object $Object -Name $Name)
}

function ConvertTo-CanonicalSampleAssetManifest {
  param([Parameter(Mandatory = $true)][object]$Manifest)

  $candidateId = Get-StringProperty -Object $Manifest -Name "candidateId"
  if ([string]::IsNullOrWhiteSpace($candidateId)) {
    return $Manifest
  }

  $modelLocalPath = Get-StringProperty -Object $Manifest.model -Name "localPath"
  $modelBaseName = [IO.Path]::GetFileNameWithoutExtension($modelLocalPath)
  return [pscustomobject]@{
    sampleName = Get-StringProperty -Object $Manifest -Name "sampleName"
    status = if ((Get-StringProperty -Object $Manifest -Name "status") -eq "owner-action-required") { "candidate-not-downloaded" } else { Get-StringProperty -Object $Manifest -Name "status" }
    proofClassification = Get-StringProperty -Object $Manifest -Name "proofClassification"
    isSmokePassed = $false
    model = [pscustomobject]@{
      name = Get-StringProperty -Object $Manifest.model -Name "name"
      sourceUrl = Get-StringProperty -Object $Manifest.model -Name "sourceUrl"
      downloadUrl = Get-StringProperty -Object $Manifest.model -Name "downloadUrl"
      localPath = $modelLocalPath
      exportCommand = Get-StringProperty -Object $Manifest.model -Name "onnxExportCommand"
    }
    labels = [pscustomobject]@{
      sourceUrl = Get-StringProperty -Object $Manifest.labels -Name "sourceUrl"
    }
    input = [pscustomobject]@{
      sourceUrl = Get-StringProperty -Object $Manifest.input -Name "imageSourceUrl"
      localPath = Get-StringProperty -Object $Manifest.input -Name "imagePath"
    }
    evidence = [pscustomobject]@{
      runCommand = Get-StringProperty -Object $Manifest.commands -Name "yoloVisionRunCommand"
      buildOnlyCommand = Get-StringProperty -Object $Manifest.commands -Name "tensorRtExecBuildCommand"
      evidenceSidecar = if ([string]::IsNullOrWhiteSpace($modelBaseName)) { "" } else { "models/$modelBaseName-evidence.sidecar.json" }
    }
  }
}

function New-PlanStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$OwnerAction,
    [string]$Evidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    ownerAction = $OwnerAction
    evidence = $Evidence
    state = "owner-action-required"
    boundary = $Boundary
  }
}

$manifestRoot = Join-Path $RepositoryRoot "samples\assets"
if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
  throw "Sample asset manifest folder was not found: $manifestRoot"
}

$manifestFiles = @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*.template.json" -File | Sort-Object Name)
$items = New-Object System.Collections.Generic.List[object]

foreach ($file in $manifestFiles) {
  $rawManifest = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  $manifest = ConvertTo-CanonicalSampleAssetManifest -Manifest $rawManifest
  $sampleName = Get-StringValue $manifest.sampleName
  $manifestRelative = ConvertTo-RelativePath -Path $file.FullName
  $modelName = Get-StringValue $manifest.model.name
  $modelSourceUrl = Get-StringValue $manifest.model.sourceUrl
  $modelDownloadUrl = Get-StringValue $manifest.model.downloadUrl
  $labelsSourceUrl = Get-StringValue $manifest.labels.sourceUrl
  $imageSourceUrl = Get-StringValue $manifest.input.sourceUrl
  $proofClassification = Get-StringValue $manifest.proofClassification
  $runCommand = Get-StringValue $manifest.evidence.runCommand
  $buildOnlyCommand = Get-StringValue $manifest.evidence.buildOnlyCommand
  $evidenceSidecar = Get-StringValue $manifest.evidence.evidenceSidecar

  $downloadAction = if ([string]::IsNullOrWhiteSpace($modelDownloadUrl)) {
    "Review $modelSourceUrl and record the owner-approved model download URL before downloading."
  }
  else {
    "Download the model from $modelDownloadUrl and save it to $($manifest.model.localPath)."
  }

  $steps = @(
    New-PlanStep `
      -Id "license-review" `
      -Title "License and redistribution review" `
      -OwnerAction "Review model, labels, and image licenses; record whether assets may be redistributed in the repository or only referenced externally." `
      -Evidence "Updated $manifestRelative with license fields and owner notes." `
      -Boundary "License review is not sample smoke proof."
    New-PlanStep `
      -Id "model-acquisition" `
      -Title "Model acquisition" `
      -OwnerAction $downloadAction `
      -Evidence "Model local path and SHA256 recorded in $manifestRelative." `
      -Boundary "A downloaded model is not TensorRT parser proof."
    New-PlanStep `
      -Id "onnx-export" `
      -Title "ONNX export or verification" `
      -OwnerAction "Run or document the export command: $($manifest.model.exportCommand)" `
      -Evidence "ONNX opset, input/output names, and SHA256 recorded in $manifestRelative." `
      -Boundary "ONNX export success is not TensorRT runtime smoke passed."
    New-PlanStep `
      -Id "labels-acquisition" `
      -Title "Labels acquisition" `
      -OwnerAction "Acquire labels from $labelsSourceUrl or another owner-approved source and verify class count." `
      -Evidence "labels.localPath, labels.license, labels.sha256, and labels.classCount recorded." `
      -Boundary "Labels alone do not prove model output correctness."
    New-PlanStep `
      -Id "image-acquisition" `
      -Title "Test image acquisition" `
      -OwnerAction "Acquire a redistributable test image from $imageSourceUrl or an owner-approved internal source." `
      -Evidence "input.localPath, input.license, input.sha256, and preprocessing notes recorded." `
      -Boundary "An image file is not classification/detection accuracy proof."
    New-PlanStep `
      -Id "hash-verification" `
      -Title "SHA256 verification" `
      -OwnerAction "Compute SHA256 for model, labels, and image and update the manifest." `
      -Evidence "Non-empty 64-character SHA256 fields in $manifestRelative." `
      -Boundary "Hash verification is not sample smoke passed."
    New-PlanStep `
      -Id "build-only-classification" `
      -Title "Build-only evidence classification" `
      -OwnerAction "Optionally run the build-only command and keep proofClassification=build-only until the sample runtime path is executed with real assets: $buildOnlyCommand" `
      -Evidence "TensorRtExec build report path, evidence sidecar $evidenceSidecar, stdout/stderr summary, and proofClassification recorded in $manifestRelative." `
      -Boundary "Build-only evidence is not model runtime or detection/classification quality proof."
    New-PlanStep `
      -Id "evidence-sidecar" `
      -Title "Evidence sidecar backfill" `
      -OwnerAction "Create or update evidence sidecar $evidenceSidecar with modelSource, modelSha256, modelLicense, inputAssetName, inputAssetSha256, stdoutSummary, stderrSummary, and proofClassification." `
      -Evidence "The sidecar can enrich TensorRtExec/OnnxToEngine reports and can be cross-checked against $manifestRelative." `
      -Boundary "An evidence sidecar cannot claim package-consumer-runtime and cannot by itself mark the sample smoke passed."
    New-PlanStep `
      -Id "sample-run" `
      -Title "Sample execution" `
      -OwnerAction "Run: $runCommand" `
      -Evidence "Command log, stdout/stderr summary, log SHA256, expected evidence lines, and manifest lastRunStatus updated." `
      -Boundary "Only real asset execution can promote the sample to real-model-runtime; synthetic input remains pipeline evidence only."
  )

  $items.Add([pscustomobject]@{
      sampleName = $sampleName
      manifest = $manifestRelative
      status = Get-StringValue $manifest.status
      modelName = $modelName
      modelSourceUrl = $modelSourceUrl
      modelDownloadUrl = $modelDownloadUrl
      labelsSourceUrl = $labelsSourceUrl
      imageSourceUrl = $imageSourceUrl
      proofClassification = $proofClassification
      runCommand = $runCommand
      buildOnlyCommand = $buildOnlyCommand
      evidenceSidecar = $evidenceSidecar
      planState = "owner-action-required"
      isSmokePassed = [bool]$manifest.isSmokePassed
      steps = $steps
    })
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  planKind = "sample-asset-acquisition-plan"
  planState = "owner-action-required"
  performsDownload = $false
  performsSampleRun = $false
  canPromoteSamples = $false
  itemCount = $items.Count
  items = @($items.ToArray())
  safetyNotes = @(
    "This plan does not download models, labels, or images.",
    "This plan does not run Classification or YoloVision.",
    "candidate-not-downloaded and asset-required are not sample smoke passed.",
    "template-only, build-only, dependency-probe-only, and synthetic-input-runtime are not real model proof.",
    "Evidence sidecars enrich reports but do not promote build reports to package-consumer-runtime.",
    "package-consumer-runtime belongs to release proof records, not sample asset manifests.",
    "Synthetic input pipeline evidence is not model accuracy or detection quality evidence."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "sample-asset-acquisition-plan.json"
$markdownPath = Join-Path $outputRoot "sample-asset-acquisition-plan.md"

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Sample Asset Acquisition Plan")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$lines.Add("- plan state: ``owner-action-required``")
$lines.Add("- performs download: ``false``")
$lines.Add("- performs sample run: ``false``")
$lines.Add("- can promote samples: ``false``")
$lines.Add("- item count: $($summary.itemCount)")
$lines.Add("")
foreach ($item in $items) {
  $lines.Add("## $($item.sampleName)")
  $lines.Add("")
  $lines.Add("- manifest: ``$($item.manifest)``")
  $lines.Add("- current status: ``$($item.status)``")
  $lines.Add("- model: $($item.modelName)")
  $lines.Add("- source: $($item.modelSourceUrl)")
  $lines.Add("- proof classification: ``$($item.proofClassification)``")
  $lines.Add("- smoke passed: ``$($item.isSmokePassed)``")
  $lines.Add("")
  $lines.Add("| Step | Owner action | Evidence | Boundary |")
  $lines.Add("| --- | --- | --- | --- |")
  foreach ($step in $item.steps) {
    $lines.Add("| ``$($step.id)`` | $($step.ownerAction.Replace("|", "\|")) | $($step.evidence.Replace("|", "\|")) | $($step.boundary.Replace("|", "\|")) |")
  }
  $lines.Add("")
}
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $summary.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Sample asset acquisition plan written to $jsonPath"
Write-Host "Sample asset acquisition plan written to $markdownPath"
