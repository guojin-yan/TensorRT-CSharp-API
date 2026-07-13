[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$ValidationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
  [string]$ImportReportPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or
    $text.Equals("owner-required", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("owner-required-or-no-stderr", [StringComparison]::OrdinalIgnoreCase) -or
    $text -like "<*>"
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-PlaceholderPath {
  param(
    [AllowNull()][object]$Object,
    [string]$Prefix
  )

  $paths = New-Object System.Collections.Generic.List[string]
  if ($null -eq $Object) {
    return @()
  }

  if ($Object -is [System.Array]) {
    for ($i = 0; $i -lt $Object.Count; $i++) {
      $childPrefix = "$Prefix[$i]"
      foreach ($path in @(Get-PlaceholderPath -Object $Object[$i] -Prefix $childPrefix)) {
        $paths.Add($path) | Out-Null
      }
    }

    return @($paths)
  }

  if ($Object -is [System.Management.Automation.PSCustomObject]) {
    foreach ($property in $Object.PSObject.Properties) {
      $childPrefix = if ([string]::IsNullOrWhiteSpace($Prefix)) { $property.Name } else { "$Prefix.$($property.Name)" }
      foreach ($path in @(Get-PlaceholderPath -Object $property.Value -Prefix $childPrefix)) {
        $paths.Add($path) | Out-Null
      }
    }

    return @($paths)
  }

  if (Test-IsPlaceholder -Value $Object) {
    $paths.Add($Prefix) | Out-Null
  }

  return @($paths)
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "YoloVision owner proof input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$validation = $null
$resolvedValidationPath = Resolve-RepositoryPath -Path $ValidationPath
if (Test-Path -LiteralPath $resolvedValidationPath -PathType Leaf) {
  $validation = Get-Content -LiteralPath $resolvedValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$importReport = $null
$resolvedImportReportPath = Resolve-RepositoryPath -Path $ImportReportPath
if (Test-Path -LiteralPath $resolvedImportReportPath -PathType Leaf) {
  $importReport = Get-Content -LiteralPath $resolvedImportReportPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$cases = @($record.cases)
$forbiddenSubstitutes = @($record.forbiddenSubstitutes)
$globalRequiredFields = @(Get-PlaceholderPath -Object $record.requiredGlobalEvidence -Prefix "requiredGlobalEvidence")

$casePacks = @(
  foreach ($case in $cases) {
    $caseId = [string](Get-PropertyOrDefault -Object $case -Name "caseId" -DefaultValue "unknown")
    $task = [string](Get-PropertyOrDefault -Object $case -Name "task" -DefaultValue "unknown")
    $requiredFields = @(Get-PlaceholderPath -Object $case -Prefix "cases[$caseId]")
    $tensorRtExec = Get-PropertyOrDefault -Object $case -Name "tensorRtExec" -DefaultValue $null
    $yoloVision = Get-PropertyOrDefault -Object $case -Name "yoloVision" -DefaultValue $null

    [pscustomobject]@{
      caseId = $caseId
      task = $task
      article = [string](Get-PropertyOrDefault -Object $case -Name "article" -DefaultValue "")
      ownerInputState = [string](Get-PropertyOrDefault -Object $case -Name "ownerInputState" -DefaultValue "owner-action-required")
      requiredFieldCount = $requiredFields.Count
      requiredFields = @($requiredFields)
      recommendedCommandSequence = @(
        [pscustomobject]@{
          id = "prepare-real-assets"
          description = "Prepare ONNX, labels, input image, preprocessed tensor, licenses, expected output metadata, and SHA256 values."
          command = "owner action"
        }
        [pscustomobject]@{
          id = "run-tensorrtexec-build-report"
          description = "Run TensorRtExec build/report as supporting evidence only; TensorRtExec report is not proof."
          command = [string](Get-PropertyOrDefault -Object $tensorRtExec -Name "buildCommand" -DefaultValue "owner-required")
        }
        [pscustomobject]@{
          id = "run-yolovision-sample"
          description = "Run YoloVision sample against the real model input and capture stdout/stderr/log/output JSON."
          command = [string](Get-PropertyOrDefault -Object $yoloVision -Name "runCommand" -DefaultValue "owner-required")
        }
        [pscustomobject]@{
          id = "hash-runtime-artifacts"
          description = "Calculate SHA256 for model, labels, image, tensor, engine, report, run log, and output JSON."
          command = "Get-FileHash -Algorithm SHA256 <owner-artifact>"
        }
        [pscustomobject]@{
          id = "fill-owner-proof-input"
          description = "Replace owner-required placeholders in yolovision-real-asset-owner-proof-input.template.json."
          command = "owner action"
        }
        [pscustomobject]@{
          id = "validate-and-import-owner-input"
          description = "Validate owner input and project a sample-run-evidence candidate."
          command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1"
        }
        [pscustomobject]@{
          id = "validate-sample-run-evidence"
          description = "Only after real files/logs exist, run sample-run evidence validation with existing log enforcement."
          command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
        }
      )
      expectedEvidenceLines = @((Get-PropertyOrDefault -Object $yoloVision -Name "expectedEvidenceLines" -DefaultValue @()))
      proofBoundary = "TensorRtExec report is not proof; sample-run-evidence candidate is not package-consumer-runtime proof; real-model-runtime requires real logs, output JSON, hashes, host metadata, and owner review."
    }
  }
)

$ownerRequiredFieldCount = ($globalRequiredFields.Count + (($casePacks | ForEach-Object { $_.requiredFieldCount }) | Measure-Object -Sum).Sum)
if ($null -eq $ownerRequiredFieldCount) {
  $ownerRequiredFieldCount = $globalRequiredFields.Count
}

$executionPack = [ordered]@{
  recordKind = "yolovision-real-asset-owner-proof-execution-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packState = "blocked-owner-action-required"
  sampleName = "YoloVision"
  sourceOwnerProofInput = $InputPath
  sourceValidation = $ValidationPath
  sourceImportReport = $ImportReportPath
  validationState = [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "missing-validation")
  importValidationState = [string](Get-PropertyOrDefault -Object $importReport -Name "validationState" -DefaultValue "missing-import-report")
  performsPublish = $false
  canPublishPublicly = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  caseCount = $casePacks.Count
  ownerRequiredFieldCount = [int]$ownerRequiredFieldCount
  globalRequiredFields = @($globalRequiredFields)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  validators = @(
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
  )
  cases = @($casePacks)
  proofBoundary = "This execution pack is owner guidance only. TensorRtExec report is not proof, sample-run-evidence candidate is not package-consumer-runtime proof, and no real-model-runtime proof can promote until real logs, output JSON, hashes, host metadata, and owner review pass strict validation."
}

$jsonPath = Join-Path $OutputRoot "yolovision-real-asset-owner-proof-execution-pack.json"
$markdownPath = Join-Path $OutputRoot "yolovision-real-asset-owner-proof-execution-pack.md"
$executionPack | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$caseRows = foreach ($casePack in $casePacks) {
  "| ``$(ConvertTo-MarkdownCell $casePack.caseId)`` | ``$(ConvertTo-MarkdownCell $casePack.task)`` | $($casePack.requiredFieldCount) | ``owner-action-required`` |"
}
$validatorLines = $executionPack.validators | ForEach-Object { "- ``$_``" }
$forbiddenLines = $forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# YoloVision Real Asset Owner Proof Execution Pack

Generated at: ``$($executionPack.generatedAtUtc)``

## Summary

- recordKind: ``$($executionPack.recordKind)``
- packState: ``$($executionPack.packState)``
- caseCount: ``$($executionPack.caseCount)``
- ownerRequiredFieldCount: ``$($executionPack.ownerRequiredFieldCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canPromoteRealModelRuntime: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canCloseReleaseIssue: ``False``

## Case Required Field Delta

| Case | Task | Required Fields | State |
| --- | --- | ---: | --- |
$($caseRows -join "`r`n")

## Validators

$($validatorLines -join "`r`n")

## Forbidden Substitutes

$($forbiddenLines -join "`r`n")

## Required Command Order

1. Prepare ONNX / labels / image / expected output.
2. Run TensorRtExec build/report.
3. Run YoloVision sample.
4. Calculate log/output/model/engine/report SHA256.
5. Fill owner proof input.
6. Run validator/importer.
7. Run ``Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog``.

## Boundary

$($executionPack.proofBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "YoloVision real asset owner proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($executionPack.packState) CaseCount=$($executionPack.caseCount) OwnerRequiredFieldCount=$($executionPack.ownerRequiredFieldCount)"
