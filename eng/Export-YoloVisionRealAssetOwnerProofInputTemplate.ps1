[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OwnerBackfillPackPath = "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
  [string]$SampleRunEvidenceTemplatePath = "artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.json",
  [string]$OutputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$MarkdownPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.md"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function New-OwnerInputCase {
  param([object]$Case)

  [pscustomobject]@{
    caseId = [string]$Case.id
    task = [string]$Case.task
    article = [string]$Case.article
    ownerInputState = "owner-action-required"
    proofClassification = "template-only"
    model = [pscustomobject]@{
      sourceUrl = "owner-required"
      license = "owner-required"
      sha256 = "owner-required"
      onnxPath = [string]$Case.model.onnxPath
      onnxSha256 = "owner-required"
      exportCommand = [string]$Case.model.exportCommand
    }
    labels = [pscustomobject]@{
      path = [string]$Case.labels.path
      license = "owner-required"
      classCount = "owner-required"
      sha256 = "owner-required"
    }
    input = [pscustomobject]@{
      imagePath = [string]$Case.input.imagePath
      imageLicense = "owner-required"
      imageSha256 = "owner-required"
      preprocessedTensorPath = [string]$Case.input.preprocessedTensorPath
      preprocessedTensorSha256 = "owner-required"
      preprocessedTensorElementCount = "owner-required"
      inputShape = [string]$Case.input.inputShape
      preprocessContract = "owner-required"
    }
    tensorRtExec = [pscustomobject]@{
      buildCommand = [string]$Case.tensorRtExec.buildCommand
      reportPath = [string]$Case.tensorRtExec.reportPath
      reportSha256 = "owner-required"
      stdoutLogPath = [string]$Case.tensorRtExec.stdoutLogPath
      stdoutLogSha256 = "owner-required"
      stderrLogPath = [string]$Case.tensorRtExec.stderrLogPath
      stderrLogSha256 = "owner-required-or-no-stderr"
      enginePath = [string]$Case.tensorRtExec.enginePath
      engineSha256 = "owner-required"
      proofBoundary = [string]$Case.tensorRtExec.proofBoundary
    }
    yoloVision = [pscustomobject]@{
      runCommand = [string]$Case.yoloVision.runCommand
      runLogPath = [string]$Case.yoloVision.runLogPath
      runLogSha256 = "owner-required"
      stdoutLogPath = [string]$Case.yoloVision.stdoutLogPath
      stdoutLogSha256 = "owner-required"
      stderrLogPath = [string]$Case.yoloVision.stderrLogPath
      stderrLogSha256 = "owner-required-or-no-stderr"
      outputJsonPath = [string]$Case.yoloVision.outputJsonPath
      outputJsonSha256 = "owner-required"
      stdoutSummary = "owner-required"
      stderrSummary = "owner-required-or-no-stderr"
      expectedEvidenceLines = @($Case.yoloVision.expectedEvidenceLines)
    }
    yoloVisionPreflight = [pscustomobject]@{
      command = [string]$Case.yoloVisionPreflight.command
      reportPath = [string]$Case.yoloVisionPreflight.reportPath
      reportSha256 = "owner-required"
      schemaPath = [string]$Case.yoloVisionPreflight.schemaPath
      schemaVersion = [string]$Case.yoloVisionPreflight.schemaVersion
      expectedState = "owner-action-required"
      proofClassification = [string]$Case.yoloVisionPreflight.proofClassification
      execution = $Case.yoloVisionPreflight.execution
      boundary = $Case.yoloVisionPreflight.boundary
      proofBoundary = [string]$Case.yoloVisionPreflight.proofBoundary
    }
    outputMetadata = $Case.outputMetadata
    articleEvidence = $Case.articleEvidence
    ownerReview = [pscustomobject]@{
      reviewer = "owner-required"
      reviewedAtUtc = "owner-required"
      acceptanceDecision = "owner-required"
      acceptedForRealModelRuntimeCandidate = $false
      notes = "owner-required"
    }
    isSmokePassed = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
  }
}

$ownerBackfillResolvedPath = Resolve-RepositoryPath -Path $OwnerBackfillPackPath
$sampleRunEvidenceTemplateResolvedPath = Resolve-RepositoryPath -Path $SampleRunEvidenceTemplatePath
$outputResolvedPath = Resolve-RepositoryPath -Path $OutputPath
$markdownResolvedPath = Resolve-RepositoryPath -Path $MarkdownPath

foreach ($path in @($ownerBackfillResolvedPath, $sampleRunEvidenceTemplateResolvedPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required input was not found: $path"
  }
}

$ownerBackfillPack = Get-Content -LiteralPath $ownerBackfillResolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
$sampleRunEvidenceTemplate = Get-Content -LiteralPath $sampleRunEvidenceTemplateResolvedPath -Raw -Encoding utf8 | ConvertFrom-Json

$template = [pscustomobject]@{
  recordKind = "yolovision-real-asset-owner-proof-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerInputState = "template-owner-input-required"
  sampleName = "YoloVision"
  sourceOwnerBackfillPack = $OwnerBackfillPackPath
  sourceSampleRunEvidenceTemplate = $SampleRunEvidenceTemplatePath
  proofClassification = "template-only"
  proofBoundary = "owner proof input template only; not real-model-runtime proof; not package-consumer-runtime proof; not post-publish proof"
  performsPublish = $false
  canPublishPublicly = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  requiredGlobalEvidence = $ownerBackfillPack.requiredOwnerEvidence
  requiredPreflightEvidence = @($ownerBackfillPack.requiredPreflightEvidence)
  forbiddenSubstitutes = @($ownerBackfillPack.forbiddenSubstitutes)
  validationRules = @(
    "All hashes must be 64-character SHA256 strings before any real-model-runtime candidate can be projected.",
    "owner-required placeholders, empty strings, build-only reports, command transcript logs, dry-runs, sidecars, screenshots, GUI checklists, local feeds, ProjectReference consumers, direct .nupkg references, and TensorRtExec reports cannot promote proof.",
    "Every case must preserve the source TensorRtExec build command, YoloVision preflight command/report/schema/boundary, YoloVision run command, input shape, and expected evidence lines from the owner backfill pack.",
    "Owner must fill host metadata, package metadata, TensorRtExec stdout/stderr transcript hashes, YoloVision stdout/stderr transcript hashes, article readiness, and owner acceptance decision before validation can become a real-model-runtime candidate.",
    "YoloVision sample evidence can only become real-model-runtime after real logs and hashes pass Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog.",
    "Package-consumer-runtime belongs to release proof records and is forbidden in this owner proof input."
  )
  sampleRunEvidenceValidator = "eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
  cases = @($ownerBackfillPack.cases | ForEach-Object { New-OwnerInputCase -Case $_ })
  sampleRunEvidenceTemplateCaseCount = @($sampleRunEvidenceTemplate.cases).Count
}

New-Item -ItemType Directory -Path (Split-Path -Parent $outputResolvedPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $markdownResolvedPath) -Force | Out-Null

$template | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $outputResolvedPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# YoloVision Real Asset Owner Proof Input Template")
$lines.Add("")
$lines.Add("- record kind: ``$($template.recordKind)``")
$lines.Add("- owner input state: ``$($template.ownerInputState)``")
$lines.Add("- proof classification: ``$($template.proofClassification)``")
$lines.Add("- can promote real model runtime: ``$($template.canPromoteRealModelRuntime)``")
$lines.Add("- can promote package consumer runtime: ``$($template.canPromotePackageConsumerRuntime)``")
$lines.Add("- validator: ``$($template.sampleRunEvidenceValidator)``")
$lines.Add("")
$lines.Add("## Cases")
$lines.Add("")
$lines.Add("| Case | Task | Input Shape | Run Log | Output JSON |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($case in $template.cases) {
  $lines.Add("| ``$($case.caseId)`` | ``$($case.task)`` | ``$($case.input.inputShape)`` | ``$($case.yoloVision.runLogPath)`` | ``$($case.yoloVision.outputJsonPath)`` |")
}
$lines.Add("")
$lines.Add("## Required Global Evidence")
$lines.Add("")
foreach ($section in $template.requiredGlobalEvidence.PSObject.Properties) {
  $lines.Add("- ``$($section.Name)``")
  foreach ($property in $section.Value.PSObject.Properties) {
    $lines.Add("  - ``$($property.Name)``")
  }
}
$lines.Add("")
$lines.Add("## Required Preflight Evidence")
$lines.Add("")
foreach ($field in $template.requiredPreflightEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Validation Rules")
$lines.Add("")
foreach ($rule in $template.validationRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Proof Boundary")
$lines.Add("")
$lines.Add($template.proofBoundary)

$lines | Set-Content -LiteralPath $markdownResolvedPath -Encoding utf8

Write-Host "YoloVision real asset owner proof input template written:"
Write-Host "  Json=$outputResolvedPath"
Write-Host "  Markdown=$markdownResolvedPath"
Write-Host "Cases=$(@($template.cases).Count) ProofClassification=$($template.proofClassification) CanPromoteRealModelRuntime=$($template.canPromoteRealModelRuntime)"
