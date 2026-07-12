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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

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

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-HandoffStep {
  param(
    [string]$Id,
    [string]$Action,
    [string]$Command,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    action = $Action
    command = $Command
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
    state = "owner-action-required"
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$assetAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$assetPlan = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-acquisition-plan.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sidecarAudit = Read-JsonOrNull "artifacts\user-acceptance\onnx-engine-build-evidence-sidecar-audit.json"

$assetAuditStatus = if ($assetAudit -and [int](Get-PropertyOrDefault -Object $assetAudit -Name "errorCount" -DefaultValue -1) -eq 0) { "ready" } elseif ($assetAudit) { "has-errors" } else { "missing" }
$assetAuditErrorCount = [int](Get-PropertyOrDefault -Object $assetAudit -Name "errorCount" -DefaultValue -1)
$assetPlanState = [string](Get-PropertyOrDefault -Object $assetPlan -Name "planState" -DefaultValue "missing")
$sampleRunValidationState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing")
$sampleRunCanPromoteRealModelRuntime = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleRunProofClassification = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$sidecarAuditState = [string](Get-PropertyOrDefault -Object $sidecarAudit -Name "auditState" -DefaultValue "missing")
$sidecarAuditErrorCount = [int](Get-PropertyOrDefault -Object $sidecarAudit -Name "errorCount" -DefaultValue -1)

$manifestRoot = Join-Path $RepositoryRoot "samples\assets"
if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
  throw "Sample asset manifest folder was not found: $manifestRoot"
}

$templateManifestFiles = @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*.template.json" -File)
$exampleManifestFiles = @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*-example.json" -File)
$manifestFiles = @(
  @($templateManifestFiles + $exampleManifestFiles) |
    Sort-Object FullName -Unique
)
$items = New-Object System.Collections.Generic.List[object]

foreach ($file in $manifestFiles) {
  $manifest = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  $sampleName = Get-StringValue $manifest.sampleName
  $manifestRelative = ConvertTo-RelativePath -Path $file.FullName
  $modelName = Get-StringValue $manifest.model.name
  $modelFamily = Get-StringValue $manifest.model.family
  $modelTask = Get-StringValue $manifest.model.task
  $sourceUrl = Get-StringValue $manifest.model.sourceUrl
  $downloadUrl = Get-StringValue $manifest.model.downloadUrl
  $modelLocalPath = Get-StringValue $manifest.model.localPath
  $labelsLocalPath = Get-StringValue $manifest.labels.localPath
  $inputLocalPath = Get-StringValue $manifest.input.localPath
  $inputShape = Get-StringValue $manifest.tensor.inputShape
  $proofClassification = Get-StringValue $manifest.proofClassification
  $evidenceSidecar = Get-StringValue $manifest.evidence.evidenceSidecar
  $sampleRunEvidenceRecord = Get-StringValue $manifest.evidence.sampleRunEvidenceRecord
  $sampleRunEvidenceTemplate = switch -Wildcard ($sampleRunEvidenceRecord) {
    "*classifier*" { ".\artifacts\user-acceptance\sample-run-evidence-record.classification.template.json"; break }
    "*yolox*" { ".\artifacts\user-acceptance\sample-run-evidence-record.yolox-s.template.json"; break }
    "*yolovision*" { ".\artifacts\user-acceptance\sample-run-evidence-record.yolovision.template.json"; break }
    default { ".\artifacts\user-acceptance\sample-run-evidence-record.template.json"; break }
  }
  $buildOnlyCommand = Get-StringValue $manifest.evidence.buildOnlyCommand
  $runCommand = Get-StringValue $manifest.evidence.runCommand
  $sampleRunLogPath = Get-StringValue $manifest.evidence.lastRunLog
  $defaultRunLogPath = if ([string]::IsNullOrWhiteSpace($sampleRunLogPath)) {
    switch -Wildcard ($sampleRunEvidenceRecord) {
      "*classifier*" { ".\models\classifier-run.log"; break }
      "*yolox*" { ".\models\yolox_s-run.log"; break }
      "*yolo*" { ".\models\yolovision-run.log"; break }
      default { ".\models\$sampleName-run.log"; break }
    }
  }
  else {
    $sampleRunLogPath
  }

  $expectedEvidenceLines = @()
  if ($manifest.evidence.PSObject.Properties.Name -contains "expectedEvidenceLines") {
    $expectedEvidenceLines = @($manifest.evidence.expectedEvidenceLines)
  }

  $hashCommands = @(
    "Get-FileHash -LiteralPath `"$modelLocalPath`" -Algorithm SHA256",
    "Get-FileHash -LiteralPath `"$labelsLocalPath`" -Algorithm SHA256",
    "Get-FileHash -LiteralPath `"$inputLocalPath`" -Algorithm SHA256",
    "Get-FileHash -LiteralPath `"$defaultRunLogPath`" -Algorithm SHA256"
  )

  $steps = @(
    New-HandoffStep `
      -Id "license-review" `
      -Action "Review model, labels, and input asset licenses before downloading or redistributing anything." `
      -Command "" `
      -RequiredEvidence "model.sourceUrl, model.downloadUrl, model.license, labels.license, input.license, owner notes." `
      -Boundary "License review is required evidence, not sample runtime proof."
    New-HandoffStep `
      -Id "asset-acquisition" `
      -Action "Acquire model, labels, and input image from owner-approved sources." `
      -Command "Use owner-approved source URLs; current model source: $sourceUrl; download URL: $downloadUrl" `
      -RequiredEvidence "$modelLocalPath; $labelsLocalPath; $inputLocalPath" `
      -Boundary "Downloaded files are not smoke passed until hashes and runner output are recorded."
    New-HandoffStep `
      -Id "hash-backfill" `
      -Action "Compute SHA256 for model, labels, input image, and sample run log." `
      -Command ($hashCommands -join "; ") `
      -RequiredEvidence "64-character SHA256 values in manifest, evidence sidecar, and sample run evidence record." `
      -Boundary "Do not fabricate SHA256 values. Missing hashes keep owner-action-required."
    New-HandoffStep `
      -Id "build-only-report" `
      -Action "Optionally build an engine/report with TensorRtExec or OnnxToEngine while keeping proofClassification build-only until runtime evidence exists." `
      -Command $buildOnlyCommand `
      -RequiredEvidence "Build report JSON and $evidenceSidecar." `
      -Boundary "Build-only report is not real-model-runtime and never package-consumer-runtime."
    New-HandoffStep `
      -Id "sample-run" `
      -Action "Run the real sample with owner-approved assets and capture stdout/stderr to a log." `
      -Command "$runCommand *> $defaultRunLogPath" `
      -RequiredEvidence "$defaultRunLogPath plus expected evidence lines." `
      -Boundary "Synthetic input or dependency probe output is not real model sample proof."
    New-HandoffStep `
      -Id "sample-run-evidence" `
      -Action "Copy the generated sample-run evidence template to the record path and fill real-model-runtime fields." `
      -Command "Copy-Item $sampleRunEvidenceTemplate $sampleRunEvidenceRecord" `
      -RequiredEvidence "$sampleRunEvidenceRecord with recordKind=sample-run-evidence-record, templateOnly=false, proofClassification=real-model-runtime, isSmokePassed=true." `
      -Boundary "Sample run evidence can promote only to real-model-runtime, not package-consumer-runtime."
    New-HandoffStep `
      -Id "validators" `
      -Action "Validate sidecar, sample run evidence, manifest, catalog, and release evidence bundle." `
      -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -InputPath $sampleRunEvidenceRecord -RequireExistingLog; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1" `
      -RequiredEvidence "Validator JSON/Markdown artifacts under artifacts/user-acceptance and artifacts/final-release." `
      -Boundary "Passing sample validators does not prove package-consumer-runtime."
  )

  $items.Add([pscustomobject]@{
      sampleName = $sampleName
      manifest = $manifestRelative
      modelName = $modelName
      modelFamily = $modelFamily
      modelTask = $modelTask
      sourceUrl = $sourceUrl
      downloadUrl = $downloadUrl
      modelLocalPath = $modelLocalPath
      labelsLocalPath = $labelsLocalPath
      inputLocalPath = $inputLocalPath
      inputShape = $inputShape
      proofClassification = $proofClassification
      isSmokePassed = [bool]$manifest.isSmokePassed
      evidenceSidecar = $evidenceSidecar
      sampleRunEvidenceRecord = $sampleRunEvidenceRecord
      sampleRunEvidenceTemplate = $sampleRunEvidenceTemplate
      sampleRunLogPath = $defaultRunLogPath
      buildOnlyCommand = $buildOnlyCommand
      runCommand = $runCommand
      expectedEvidenceLines = @($expectedEvidenceLines)
      handoffState = "owner-action-required"
      canPromoteRealModelRuntime = $false
      steps = @($steps)
    })
}

$handoffState = if ($sampleRunCanPromoteRealModelRuntime) { "ready" } else { "owner-action-required" }

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "real-model-owner-handoff"
  handoffState = $handoffState
  ownerActionStatus = if ($handoffState -eq "ready") { "resolved" } else { "owner-action-required" }
  performsDownload = $false
  performsSampleRun = $false
  canPromoteRealModelRuntime = $sampleRunCanPromoteRealModelRuntime
  proofClassification = $sampleRunProofClassification
  packageConsumerRuntimeForbidden = $true
  assetAuditStatus = $assetAuditStatus
  assetAuditErrorCount = $assetAuditErrorCount
  assetAcquisitionPlanState = $assetPlanState
  sampleRunEvidenceValidationState = $sampleRunValidationState
  sidecarAuditState = $sidecarAuditState
  sidecarAuditErrorCount = $sidecarAuditErrorCount
  itemCount = $items.Count
  items = @($items.ToArray())
  sourceEvidence = @(
    "samples/assets/*.template.json",
    "samples/assets/*-example.json",
    "artifacts/user-acceptance/sample-asset-manifest-audit.json",
    "artifacts/user-acceptance/sample-asset-acquisition-plan.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
    "artifacts/user-acceptance/onnx-engine-build-evidence-sidecar-audit.json"
  )
  safetyNotes = @(
    "This handoff does not download models, labels, or images.",
    "This handoff does not run Classification or YoloVision.",
    "owner-action-required is not sample smoke passed.",
    "template-only, build-only, dependency-probe-only, and synthetic-input-runtime are not real model proof.",
    "Sample evidence can promote only to real-model-runtime.",
    "package-consumer-runtime is forbidden in sample model handoff and belongs to release proof records.",
    "Do not mark Classification Passed=True or YoloVision Passed=True without real asset logs and SHA256."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "real-model-owner-handoff.json"
$markdownPath = Join-Path $outputRoot "real-model-owner-handoff.md"

$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Model Owner Handoff")
$lines.Add("")
$lines.Add("- handoff state: ``$handoffState``")
$lines.Add("- owner action status: ``$($summary.ownerActionStatus)``")
$lines.Add("- performs download: ``false``")
$lines.Add("- performs sample run: ``false``")
$lines.Add("- can promote real model runtime: ``$sampleRunCanPromoteRealModelRuntime``")
$lines.Add("- proof classification: ``$sampleRunProofClassification``")
$lines.Add("- asset audit: ``$assetAuditStatus``")
$lines.Add("- acquisition plan: ``$assetPlanState``")
$lines.Add("- sample run evidence validation: ``$sampleRunValidationState``")
$lines.Add("- sidecar audit: ``$sidecarAuditState``")
$lines.Add("")
foreach ($item in $items) {
  $lines.Add("## $($item.sampleName): $($item.modelName)")
  $lines.Add("")
  $lines.Add("- manifest: ``$($item.manifest)``")
  $lines.Add("- proof classification: ``$($item.proofClassification)``")
  $lines.Add("- smoke passed: ``$($item.isSmokePassed)``")
  $lines.Add("- evidence sidecar: ``$($item.evidenceSidecar)``")
  $lines.Add("- sample run evidence: ``$($item.sampleRunEvidenceRecord)``")
  $lines.Add("")
  $lines.Add("| Step | Action | Command | Required evidence | Boundary |")
  $lines.Add("| --- | --- | --- | --- | --- |")
  foreach ($step in $item.steps) {
    $lines.Add("| ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.action) | ``$(ConvertTo-MarkdownCell $step.command)`` | $(ConvertTo-MarkdownCell $step.requiredEvidence) | $(ConvertTo-MarkdownCell $step.boundary) |")
  }
  if ($item.expectedEvidenceLines.Count -gt 0) {
    $lines.Add("")
    $lines.Add("Expected evidence lines:")
    foreach ($expectedLine in $item.expectedEvidenceLines) {
      $lines.Add("- ``$expectedLine``")
    }
  }
  $lines.Add("")
}
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $summary.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real model owner handoff written to $jsonPath"
Write-Host "Real model owner handoff written to $markdownPath"
Write-Host "HandoffState=$handoffState OwnerActionStatus=$($summary.ownerActionStatus) CanPromoteRealModelRuntime=$sampleRunCanPromoteRealModelRuntime ProofClassification=$sampleRunProofClassification"
