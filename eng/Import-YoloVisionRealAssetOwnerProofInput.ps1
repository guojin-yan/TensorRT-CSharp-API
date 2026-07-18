[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$OutputRoot = "artifacts/user-acceptance",
  [string]$RepositoryRoot,
  [switch]$RequireExistingLog
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

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return @()
  }

  if ($Value -is [System.Array]) {
    return @($Value)
  }

  return @($Value)
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or
    $text.Equals("owner-required", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("owner-required-or-no-stderr", [StringComparison]::OrdinalIgnoreCase) -or
    $text -like "<*>"
}

function Test-PositiveInt64 {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [int64]0
  return [int64]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -gt 0
}

function New-CandidateRecord {
  param([object]$Record)

  $globalEvidence = Get-PropertyOrDefault -Object $Record -Name "requiredGlobalEvidence" -DefaultValue $null
  $cases = @(ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name "cases" -DefaultValue @()))
  $caseRecords = New-Object System.Collections.Generic.List[object]

  foreach ($case in $cases) {
    $model = Get-PropertyOrDefault -Object $case -Name "model" -DefaultValue $null
    $labels = Get-PropertyOrDefault -Object $case -Name "labels" -DefaultValue $null
    $input = Get-PropertyOrDefault -Object $case -Name "input" -DefaultValue $null
    $tensorRtExec = Get-PropertyOrDefault -Object $case -Name "tensorRtExec" -DefaultValue $null
    $yoloVision = Get-PropertyOrDefault -Object $case -Name "yoloVision" -DefaultValue $null
    $yoloVisionPreflight = Get-PropertyOrDefault -Object $case -Name "yoloVisionPreflight" -DefaultValue $null
    $ownerReview = Get-PropertyOrDefault -Object $case -Name "ownerReview" -DefaultValue $null

    $caseRecords.Add([pscustomobject]@{
        recordKind = "sample-run-evidence-record"
        templateOnly = $false
        sampleName = "YoloVision"
        caseId = [string](Get-PropertyOrDefault -Object $case -Name "caseId" -DefaultValue "")
        task = [string](Get-PropertyOrDefault -Object $case -Name "task" -DefaultValue "")
        manifestPath = "samples/assets/yolovision-real-asset-owner-backfill-pack.json"
        proofClassification = "real-model-runtime"
        modelPath = [string](Get-PropertyOrDefault -Object $model -Name "onnxPath" -DefaultValue "")
        modelSha256 = [string](Get-PropertyOrDefault -Object $model -Name "onnxSha256" -DefaultValue "")
        modelLicense = [string](Get-PropertyOrDefault -Object $model -Name "license" -DefaultValue "")
        labelsPath = [string](Get-PropertyOrDefault -Object $labels -Name "path" -DefaultValue "")
        labelsSha256 = [string](Get-PropertyOrDefault -Object $labels -Name "sha256" -DefaultValue "")
        labelsLicense = [string](Get-PropertyOrDefault -Object $labels -Name "license" -DefaultValue "")
        inputAssetPath = [string](Get-PropertyOrDefault -Object $input -Name "imagePath" -DefaultValue "")
        inputAssetSha256 = [string](Get-PropertyOrDefault -Object $input -Name "imageSha256" -DefaultValue "")
        inputAssetLicense = [string](Get-PropertyOrDefault -Object $input -Name "imageLicense" -DefaultValue "")
        preprocessedInputTensorPath = [string](Get-PropertyOrDefault -Object $input -Name "preprocessedTensorPath" -DefaultValue "")
        preprocessedInputTensorSha256 = [string](Get-PropertyOrDefault -Object $input -Name "preprocessedTensorSha256" -DefaultValue "")
        preprocessedInputTensorElementCount = [string](Get-PropertyOrDefault -Object $input -Name "preprocessedTensorElementCount" -DefaultValue "")
        evidenceSidecarPath = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "reportPath" -DefaultValue "")
        buildReportPath = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "reportPath" -DefaultValue "")
        sampleRunCommand = [string](Get-PropertyOrDefault -Object $yoloVision -Name "runCommand" -DefaultValue "")
        sampleRunLogPath = [string](Get-PropertyOrDefault -Object $yoloVision -Name "runLogPath" -DefaultValue "")
        sampleRunLogSha256 = [string](Get-PropertyOrDefault -Object $yoloVision -Name "runLogSha256" -DefaultValue "")
        stdoutSummary = [string](Get-PropertyOrDefault -Object $yoloVision -Name "stdoutSummary" -DefaultValue "")
        stderrSummary = [string](Get-PropertyOrDefault -Object $yoloVision -Name "stderrSummary" -DefaultValue "")
        expectedEvidenceLines = @(ConvertTo-Array (Get-PropertyOrDefault -Object $yoloVision -Name "expectedEvidenceLines" -DefaultValue @()))
        isSmokePassed = [bool](Get-PropertyOrDefault -Object $case -Name "isSmokePassed" -DefaultValue $false)
        canPromoteRealModelRuntime = [bool](Get-PropertyOrDefault -Object $case -Name "canPromoteRealModelRuntime" -DefaultValue $false)
        canPromotePackageConsumerRuntime = $false
        outputJsonPath = [string](Get-PropertyOrDefault -Object $yoloVision -Name "outputJsonPath" -DefaultValue "")
        outputJsonSha256 = [string](Get-PropertyOrDefault -Object $yoloVision -Name "outputJsonSha256" -DefaultValue "")
        tensorRtExecReportSha256 = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "reportSha256" -DefaultValue "")
        tensorRtEnginePath = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "enginePath" -DefaultValue "")
        tensorRtEngineSha256 = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "engineSha256" -DefaultValue "")
        yoloVisionPreflight = [pscustomobject]@{
          command = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "command" -DefaultValue "")
          reportPath = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "reportPath" -DefaultValue "")
          reportSha256 = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "reportSha256" -DefaultValue "")
          schemaPath = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "schemaPath" -DefaultValue "")
          schemaVersion = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "schemaVersion" -DefaultValue "")
          proofClassification = [string](Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "proofClassification" -DefaultValue "")
          execution = Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "execution" -DefaultValue $null
          boundary = Get-PropertyOrDefault -Object $yoloVisionPreflight -Name "boundary" -DefaultValue $null
        }
        ownerReview = $ownerReview
        hostEvidence = $globalEvidence
        proofBoundary = "sample-run evidence candidate only; never package-consumer-runtime proof"
      }) | Out-Null
  }

  [pscustomobject]@{
    recordKind = "yolovision-real-asset-owner-sample-run-evidence-candidate"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    sourceOwnerInput = $InputPath
    proofBoundary = "candidate projection only; not release proof; package-consumer-runtime belongs to release validators"
    performsPublish = $false
    canPublishPublicly = $false
    canPromotePackageConsumerRuntime = $false
    cases = @($caseRecords.ToArray())
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "YoloVision real asset owner proof input was not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$outputRootResolved = Resolve-RepositoryPath -Path $OutputRoot
New-Item -ItemType Directory -Path $outputRootResolved -Force | Out-Null

$validationScript = Join-Path $RepositoryRoot "eng\Test-YoloVisionRealAssetOwnerProofInput.ps1"
$validationArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $validationScript, "-InputPath", $InputPath)
if ($RequireExistingLog.IsPresent) {
  $validationArgs += "-RequireExistingFiles"
}
$validationProcess = Start-Process -FilePath "pwsh" -ArgumentList $validationArgs -WorkingDirectory $RepositoryRoot -NoNewWindow -PassThru -Wait -RedirectStandardOutput (Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-import-validation.stdout.log") -RedirectStandardError (Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-import-validation.stderr.log")

$validationPath = Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-input-validation.json"
$validation = $null
if (Test-Path -LiteralPath $validationPath -PathType Leaf) {
  $validation = Get-Content -LiteralPath $validationPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$candidate = New-CandidateRecord -Record $record
$candidatePath = Join-Path $outputRootResolved "yolovision-real-asset-owner-sample-run-evidence.candidate.json"
$candidate | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $candidatePath -Encoding utf8

$candidateReadyForRealModelRuntime = $false
if ($null -ne $validation) {
  $candidateReadyForRealModelRuntime = [bool](Get-PropertyOrDefault -Object $validation -Name "candidateReadyForRealModelRuntime" -DefaultValue $false)
}

$report = [pscustomobject]@{
  recordKind = "yolovision-real-asset-owner-proof-import-report"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceOwnerInput = $InputPath
  candidatePath = "artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json"
  validationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json"
  validationExitCode = $validationProcess.ExitCode
  validationState = if ($null -eq $validation) { "validation-missing" } else { [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "unknown") }
  candidateReadyForRealModelRuntime = $candidateReadyForRealModelRuntime
  caseCount = @($candidate.cases).Count
  performsPublish = $false
  canPublishPublicly = $false
  canPromotePackageConsumerRuntime = $false
  proofBoundary = "import report only; candidate may become real-model-runtime only after owner evidence and validators pass; never package-consumer-runtime proof"
}

$reportPath = Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-import-report.json"
$reportMarkdownPath = Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-import-report.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding utf8

$markdown = @"
# YoloVision Real Asset Owner Proof Import Report

| Field | Value |
| --- | --- |
| validationState | ``$($report.validationState)`` |
| validationExitCode | ``$($report.validationExitCode)`` |
| caseCount | ``$($report.caseCount)`` |
| candidateReadyForRealModelRuntime | ``$($report.candidateReadyForRealModelRuntime)`` |
| canPromotePackageConsumerRuntime | ``$($report.canPromotePackageConsumerRuntime)`` |
| candidatePath | ``$($report.candidatePath)`` |

## Boundary

$($report.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $reportMarkdownPath -Encoding utf8

Write-Host "YoloVision real asset owner proof import written:"
Write-Host "  Candidate=$candidatePath"
Write-Host "  Report=$reportPath"
Write-Host "ValidationState=$($report.validationState) CandidateReadyForRealModelRuntime=$($report.candidateReadyForRealModelRuntime) ValidationExitCode=$($report.validationExitCode)"

if ($validationProcess.ExitCode -ne 0) {
  exit $validationProcess.ExitCode
}
