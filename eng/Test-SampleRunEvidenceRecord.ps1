[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\user-acceptance\sample-run-evidence-record.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$RequireExistingLog,
  [switch]$FailOnNotProof
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

$allowedProofClassifications = @(
  "template-only",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime"
)

$sha256Pattern = "^[a-fA-F0-9]{64}$"

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  if ($null -eq $Object) {
    return $null
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $null
}

function Get-StringProperty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrNull -Object $Object -Name $Name
  if ($null -eq $value) {
    return ""
  }

  return [string]$value
}

function Test-Truthy {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }

  if ($Value -is [bool]) {
    return [bool]$Value
  }

  return [bool]::Parse([string]$Value)
}

function Test-Sha256 {
  param([string]$Value)

  return -not [string]::IsNullOrWhiteSpace($Value) -and ($Value -match $sha256Pattern)
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Sample run evidence record '$InputPath' was not found. Run eng/Export-SampleRunEvidenceRecordTemplate.ps1 first or pass -InputPath."
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordKind = Get-StringProperty -Object $record -Name "recordKind"
$templateOnly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "templateOnly")
$sampleName = Get-StringProperty -Object $record -Name "sampleName"
$manifestPath = Get-StringProperty -Object $record -Name "manifestPath"
$proofClassification = Get-StringProperty -Object $record -Name "proofClassification"
$modelPath = Get-StringProperty -Object $record -Name "modelPath"
$modelSha256 = Get-StringProperty -Object $record -Name "modelSha256"
$modelLicense = Get-StringProperty -Object $record -Name "modelLicense"
$labelsPath = Get-StringProperty -Object $record -Name "labelsPath"
$labelsSha256 = Get-StringProperty -Object $record -Name "labelsSha256"
$labelsLicense = Get-StringProperty -Object $record -Name "labelsLicense"
$inputAssetPath = Get-StringProperty -Object $record -Name "inputAssetPath"
$inputAssetSha256 = Get-StringProperty -Object $record -Name "inputAssetSha256"
$inputAssetLicense = Get-StringProperty -Object $record -Name "inputAssetLicense"
$preprocessedInputTensorPath = Get-StringProperty -Object $record -Name "preprocessedInputTensorPath"
$preprocessedInputTensorSha256 = Get-StringProperty -Object $record -Name "preprocessedInputTensorSha256"
$preprocessedInputTensorElementCountText = Get-StringProperty -Object $record -Name "preprocessedInputTensorElementCount"
$evidenceSidecarPath = Get-StringProperty -Object $record -Name "evidenceSidecarPath"
$buildReportPath = Get-StringProperty -Object $record -Name "buildReportPath"
$sampleRunCommand = Get-StringProperty -Object $record -Name "sampleRunCommand"
$sampleRunLogPath = Get-StringProperty -Object $record -Name "sampleRunLogPath"
$sampleRunLogSha256 = Get-StringProperty -Object $record -Name "sampleRunLogSha256"
$stdoutSummary = Get-StringProperty -Object $record -Name "stdoutSummary"
$stderrSummary = Get-StringProperty -Object $record -Name "stderrSummary"
$validatorStateDeclared = Get-StringProperty -Object $record -Name "validatorState"
$isSmokePassed = Test-Truthy (Get-PropertyOrNull -Object $record -Name "isSmokePassed")
$canPromoteRealModelRuntimeDeclared = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPromoteRealModelRuntime")

$proofClassificationKnown = $allowedProofClassifications -contains $proofClassification
$packageConsumerRuntimeForbidden = -not [string]::Equals($proofClassification, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)
$modelHashReady = Test-Sha256 -Value $modelSha256
$modelLicenseReady = -not [string]::IsNullOrWhiteSpace($modelLicense) -and -not [string]::Equals($modelLicense, "owner-required", [System.StringComparison]::OrdinalIgnoreCase)
$labelsHashReady = Test-Sha256 -Value $labelsSha256
$labelsLicenseReady = -not [string]::IsNullOrWhiteSpace($labelsLicense) -and -not [string]::Equals($labelsLicense, "owner-required", [System.StringComparison]::OrdinalIgnoreCase)
$inputHashReady = Test-Sha256 -Value $inputAssetSha256
$inputAssetLicenseReady = -not [string]::IsNullOrWhiteSpace($inputAssetLicense) -and -not [string]::Equals($inputAssetLicense, "owner-required", [System.StringComparison]::OrdinalIgnoreCase)
$preprocessedInputTensorDeclared = -not [string]::IsNullOrWhiteSpace($preprocessedInputTensorPath)
$preprocessedInputTensorHashReady = Test-Sha256 -Value $preprocessedInputTensorSha256
$preprocessedInputTensorElementCount = 0
$preprocessedInputTensorElementCountReady = $false
if (-not [string]::IsNullOrWhiteSpace($preprocessedInputTensorElementCountText)) {
  [int64]$parsedElementCount = 0
  if ([int64]::TryParse($preprocessedInputTensorElementCountText, [ref]$parsedElementCount)) {
    $preprocessedInputTensorElementCount = $parsedElementCount
    $preprocessedInputTensorElementCountReady = $parsedElementCount -gt 0
  }
}
$logHashReady = Test-Sha256 -Value $sampleRunLogSha256
$stdoutStderrSummariesReady = -not [string]::IsNullOrWhiteSpace($stdoutSummary) -or -not [string]::IsNullOrWhiteSpace($stderrSummary)
$sampleRunLogReady = -not [string]::IsNullOrWhiteSpace($sampleRunLogPath)

if ($RequireExistingLog.IsPresent -and $sampleRunLogReady) {
  $sampleRunLogReady = Test-Path -LiteralPath (Resolve-InputPath -Path $sampleRunLogPath) -PathType Leaf
}

$expectedEvidenceLines = @()
$expectedEvidenceValue = Get-PropertyOrNull -Object $record -Name "expectedEvidenceLines"
if ($null -ne $expectedEvidenceValue) {
  $expectedEvidenceLines = @($expectedEvidenceValue)
}

$expectedEvidenceReady = $expectedEvidenceLines.Count -gt 0
$expectedEvidenceInputSourceExternalReady = @($expectedEvidenceLines | Where-Object { ([string]$_).Contains("InputSource=external", [System.StringComparison]::Ordinal) }).Count -gt 0
$sampleRunCommandUsesInputData = $sampleRunCommand.Contains("--input-data", [System.StringComparison]::Ordinal)
$yoloVisionExternalInputEvidenceRequired = [string]::Equals($sampleName, "YoloVision", [System.StringComparison]::Ordinal) -and $sampleRunCommandUsesInputData
$realModelEvidenceReady = $modelHashReady -and
  $modelLicenseReady -and
  $labelsHashReady -and
  $labelsLicenseReady -and
  $inputHashReady -and
  $inputAssetLicenseReady -and
  $logHashReady -and
  $stdoutStderrSummariesReady -and
  $sampleRunLogReady -and
  $expectedEvidenceReady -and
  (-not $yoloVisionExternalInputEvidenceRequired -or $expectedEvidenceInputSourceExternalReady) -and
  (-not $preprocessedInputTensorDeclared -or ($preprocessedInputTensorHashReady -and $preprocessedInputTensorElementCountReady))

$realModelProofRequested = [string]::Equals($proofClassification, "real-model-runtime", [System.StringComparison]::Ordinal)
$canPromoteRealModelRuntime = $realModelProofRequested -and
  $realModelEvidenceReady -and
  $isSmokePassed -and
  $canPromoteRealModelRuntimeDeclared -and
  -not $templateOnly

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ($recordKind -in @("sample-run-evidence-record-template", "sample-run-evidence-record")) -Severity "error" -Detail "recordKind must be sample-run-evidence-record-template or sample-run-evidence-record."
  New-ValidationItem -Id "proof-classification-known" -Passed $proofClassificationKnown -Severity "error" -Detail "proofClassification must be template-only, build-only, dependency-probe-only, synthetic-input-runtime, or real-model-runtime."
  New-ValidationItem -Id "package-consumer-runtime-forbidden" -Passed $packageConsumerRuntimeForbidden -Severity "error" -Detail "package-consumer-runtime belongs to release proof records and is forbidden in sample run evidence records."
  New-ValidationItem -Id "sample-name" -Passed (-not [string]::IsNullOrWhiteSpace($sampleName) -or $templateOnly) -Severity "owner-action-required" -Detail "Real sample records must identify the sample name."
  New-ValidationItem -Id "manifest-path" -Passed (-not [string]::IsNullOrWhiteSpace($manifestPath) -or $templateOnly) -Severity "owner-action-required" -Detail "Real sample records should point to the asset manifest."
  New-ValidationItem -Id "model-path" -Passed (-not [string]::IsNullOrWhiteSpace($modelPath)) -Severity "owner-action-required" -Detail "modelPath is required before owner backfill."
  New-ValidationItem -Id "labels-path" -Passed (-not [string]::IsNullOrWhiteSpace($labelsPath)) -Severity "owner-action-required" -Detail "labelsPath is required before owner backfill."
  New-ValidationItem -Id "input-asset-path" -Passed (-not [string]::IsNullOrWhiteSpace($inputAssetPath)) -Severity "owner-action-required" -Detail "inputAssetPath is required before owner backfill."
  New-ValidationItem -Id "evidence-sidecar-path" -Passed (-not [string]::IsNullOrWhiteSpace($evidenceSidecarPath)) -Severity "owner-action-required" -Detail "evidenceSidecarPath links runner proof to TensorRtExec/OnnxToEngine evidence."
  New-ValidationItem -Id "build-report-path" -Passed (-not [string]::IsNullOrWhiteSpace($buildReportPath)) -Severity "owner-action-required" -Detail "buildReportPath links runner proof to parser/builder evidence."
  New-ValidationItem -Id "sample-run-command" -Passed (-not [string]::IsNullOrWhiteSpace($sampleRunCommand)) -Severity "owner-action-required" -Detail "sampleRunCommand must record the exact runner invocation."
  New-ValidationItem -Id "expected-evidence-lines" -Passed $expectedEvidenceReady -Severity "owner-action-required" -Detail "expectedEvidenceLines should include sample Passed=True and key output markers."
  New-ValidationItem -Id "yolovision-external-input-evidence-line" -Passed ($expectedEvidenceInputSourceExternalReady -or -not $yoloVisionExternalInputEvidenceRequired) -Severity "owner-action-required" -Detail "YoloVision records that use --input-data must include InputSource=external in expectedEvidenceLines."
  New-ValidationItem -Id "model-sha256" -Passed ($modelHashReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires a 64-character modelSha256."
  New-ValidationItem -Id "model-license" -Passed ($modelLicenseReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires modelLicense to be owner-reviewed and not owner-required."
  New-ValidationItem -Id "labels-sha256" -Passed ($labelsHashReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires a 64-character labelsSha256."
  New-ValidationItem -Id "labels-license" -Passed ($labelsLicenseReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires labelsLicense to be owner-reviewed and not owner-required."
  New-ValidationItem -Id "input-asset-sha256" -Passed ($inputHashReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires a 64-character inputAssetSha256."
  New-ValidationItem -Id "input-asset-license" -Passed ($inputAssetLicenseReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires inputAssetLicense to be owner-reviewed and not owner-required."
  New-ValidationItem -Id "preprocessed-input-tensor-sha256" -Passed ($preprocessedInputTensorHashReady -or -not ($realModelProofRequested -and $preprocessedInputTensorDeclared)) -Severity "proof-required" -Detail "real-model-runtime with preprocessedInputTensorPath requires a 64-character preprocessedInputTensorSha256."
  New-ValidationItem -Id "preprocessed-input-tensor-element-count" -Passed ($preprocessedInputTensorElementCountReady -or -not ($realModelProofRequested -and $preprocessedInputTensorDeclared)) -Severity "proof-required" -Detail "real-model-runtime with preprocessedInputTensorPath requires preprocessedInputTensorElementCount > 0."
  New-ValidationItem -Id "sample-run-log" -Passed ($sampleRunLogReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires sampleRunLogPath; -RequireExistingLog also requires the file to exist."
  New-ValidationItem -Id "sample-run-log-sha256" -Passed ($logHashReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires a 64-character sampleRunLogSha256."
  New-ValidationItem -Id "stdout-stderr-summary" -Passed ($stdoutStderrSummariesReady -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires stdoutSummary or stderrSummary."
  New-ValidationItem -Id "smoke-passed" -Passed ($isSmokePassed -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires isSmokePassed=true."
  New-ValidationItem -Id "declared-real-model-promotion" -Passed ($canPromoteRealModelRuntimeDeclared -or -not $realModelProofRequested) -Severity "proof-required" -Detail "real-model-runtime requires canPromoteRealModelRuntime=true in the real record."
  New-ValidationItem -Id "not-template-only" -Passed (-not $templateOnly -or -not $realModelProofRequested) -Severity "proof-required" -Detail "Template records cannot promote to real-model-runtime."
)

$errorItems = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "error" })
$ownerActionItems = @($validationItems | Where-Object { -not $_.passed -and $_.severity -in @("owner-action-required", "proof-required") })

if ($errorItems.Count -gt 0) {
  $validationState = "invalid"
}
elseif ($canPromoteRealModelRuntime) {
  $validationState = "real-model-runtime"
}
elseif ($templateOnly -or $proofClassification -eq "template-only") {
  $validationState = "owner-action-required"
}
elseif ($realModelProofRequested) {
  $validationState = "incomplete-real-model-runtime"
}
else {
  $validationState = "non-promotable-sample-evidence"
}

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
$jsonPath = Join-Path $outputRoot "sample-run-evidence-record-validation.json"
$markdownPath = Join-Path $outputRoot "sample-run-evidence-record-validation.md"

$requiredEvidenceSummary = @(
  "recordKind=sample-run-evidence-record",
  "templateOnly=false",
  "proofClassification=real-model-runtime",
  "modelPath/modelSha256",
  "modelLicense",
  "labelsPath/labelsSha256",
  "labelsLicense",
  "inputAssetPath/inputAssetSha256",
  "inputAssetLicense",
  "preprocessedInputTensorPath/preprocessedInputTensorSha256/preprocessedInputTensorElementCount when --input-data is used",
  "evidenceSidecarPath",
  "buildReportPath",
  "sampleRunCommand",
  "sampleRunLogPath/sampleRunLogSha256",
  "stdoutSummary or stderrSummary",
  "isSmokePassed=true",
  "canPromoteRealModelRuntime=true"
)

$promotionRules = @(
  "template-only sample run evidence records are owner-action-required and not proof.",
  "build-only, dependency-probe-only, and synthetic-input-runtime cannot promote real model sample evidence.",
  "package-consumer-runtime is forbidden in sample run evidence records.",
  "Only real-model-runtime with full hashes, log, summaries, and smoke-passed state can promote sample evidence.",
  "Sample run evidence does not replace release proof records for NuGet/runtime package consumer validation."
)

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "sample-run-evidence-record-validation"
  inputPath = $resolvedInputPath
  validationState = $validationState
  recordKind = $recordKind
  templateOnly = $templateOnly
  sampleName = $sampleName
  proofClassification = $proofClassification
  proofClassificationKnown = $proofClassificationKnown
  packageConsumerRuntimeForbidden = $packageConsumerRuntimeForbidden
  isSmokePassed = $isSmokePassed
  canPromoteRealModelRuntime = $canPromoteRealModelRuntime
  canPromoteRealModelRuntimeDeclared = $canPromoteRealModelRuntimeDeclared
  realModelEvidenceReady = $realModelEvidenceReady
  modelLicenseReady = $modelLicenseReady
  labelsLicenseReady = $labelsLicenseReady
  inputAssetLicenseReady = $inputAssetLicenseReady
  preprocessedInputTensorPath = $preprocessedInputTensorPath
  preprocessedInputTensorSha256Ready = $preprocessedInputTensorHashReady
  preprocessedInputTensorElementCount = $preprocessedInputTensorElementCount
  preprocessedInputTensorElementCountReady = $preprocessedInputTensorElementCountReady
  expectedEvidenceInputSourceExternalReady = $expectedEvidenceInputSourceExternalReady
  stdoutStderrSummariesReady = $stdoutStderrSummariesReady
  sampleRunLogReady = $sampleRunLogReady
  requireExistingLog = $RequireExistingLog.IsPresent
  validatorStateDeclared = $validatorStateDeclared
  validatorStateMatches = [string]::IsNullOrWhiteSpace($validatorStateDeclared) -or [string]::Equals($validatorStateDeclared, $validationState, [System.StringComparison]::OrdinalIgnoreCase)
  errorCount = $errorItems.Count
  ownerActionRequiredCount = $ownerActionItems.Count
  failureReasons = @($validationItems | Where-Object { -not $_.passed } | ForEach-Object { "$($_.id): $($_.detail)" })
  allowedProofClassifications = $allowedProofClassifications
  disallowedProofClassifications = @("package-consumer-runtime")
  requiredEvidenceSummary = $requiredEvidenceSummary
  promotionRules = $promotionRules
  validationItems = @($validationItems)
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Sample Run Evidence Record Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- proof classification: ``$proofClassification``")
$lines.Add("- template only: ``$templateOnly``")
$lines.Add("- sample name: ``$sampleName``")
$lines.Add("- real model evidence ready: ``$realModelEvidenceReady``")
$lines.Add("- can promote real model runtime: ``$canPromoteRealModelRuntime``")
$lines.Add("- error count: $($errorItems.Count)")
$lines.Add("- owner action required count: $($ownerActionItems.Count)")
$lines.Add("- input path: ``$resolvedInputPath``")
$lines.Add("")
$lines.Add("## Required Evidence Summary")
$lines.Add("")
foreach ($item in $requiredEvidenceSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $promotionRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Sample run evidence record validation written to $jsonPath"
Write-Host "Sample run evidence record validation written to $markdownPath"
Write-Host "ValidationState=$validationState ProofClassification=$proofClassification CanPromoteRealModelRuntime=$canPromoteRealModelRuntime ErrorCount=$($errorItems.Count) OwnerActionRequiredCount=$($ownerActionItems.Count)"

if ($errorItems.Count -gt 0) {
  Write-Error "Found $($errorItems.Count) sample run evidence record error(s)."
  exit 1
}

if ($FailOnNotProof.IsPresent -and -not $canPromoteRealModelRuntime) {
  Write-Error "Sample run evidence record is not promotable real-model-runtime proof. ValidationState=$validationState"
  exit 1
}
