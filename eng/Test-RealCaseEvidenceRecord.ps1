[CmdletBinding()]
param(
  [string]$RecordPath = ".\artifacts\final-release\real-case-evidence-record-template.json",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($RecordPath)) {
  $RecordPath = Join-Path $RepositoryRoot $RecordPath
}

if (-not (Test-Path -LiteralPath $RecordPath -PathType Leaf)) {
  throw "Missing real case evidence record: $RecordPath"
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

function Test-Sha256String {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }

  return ([string]$Value) -match "^[A-Fa-f0-9]{64}$"
}

function Test-OwnerValue {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  return -not [string]::IsNullOrWhiteSpace($text) -and $text -notmatch "owner-action-required|template|TODO|TBD"
}

$record = Get-Content -LiteralPath $RecordPath -Raw -Encoding utf8 | ConvertFrom-Json
$requiredOwnerFields = @("caseId", "modelSource", "modelLicense", "onnxSha256", "engineSha256", "inputArtifactSha256", "outputArtifactSha256", "commandLine", "stdoutLogPath", "stderrLogPath", "screenshotPath", "hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion", "runtimePackageId", "runtimePackageVersion", "runtimePackageSource", "ownerReviewedBy", "ownerReviewedAtUtc")
$shaFields = @("onnxSha256", "engineSha256", "inputArtifactSha256", "outputArtifactSha256")
$fileFields = @("stdoutLogPath", "stderrLogPath", "screenshotPath")

$missingOwnerInputs = New-Object System.Collections.Generic.List[string]
$invalidShaFields = New-Object System.Collections.Generic.List[string]
$missingFiles = New-Object System.Collections.Generic.List[string]

foreach ($field in $requiredOwnerFields) {
  if (-not (Test-OwnerValue (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $null))) {
    $missingOwnerInputs.Add($field)
  }
}

foreach ($field in $shaFields) {
  if (-not (Test-Sha256String (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue $null))) {
    $invalidShaFields.Add($field)
  }
}

if ($RequireExistingFiles) {
  foreach ($field in $fileFields) {
    $value = [string](Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match "owner-action-required|template|TODO|TBD") {
      $missingFiles.Add($field)
      continue
    }

    $candidate = if ([System.IO.Path]::IsPathRooted($value)) { $value } else { Join-Path $RepositoryRoot $value }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
      $missingFiles.Add($field)
    }
  }
}

$templateOnly = [bool](Get-PropertyOrDefault -Object $record -Name "templateOnly" -DefaultValue $false)
$hasMissingInputs = $missingOwnerInputs.Count -gt 0
$hasInvalidSha = $invalidShaFields.Count -gt 0
$hasMissingFiles = $missingFiles.Count -gt 0
$canPromote = -not $templateOnly -and -not $hasMissingInputs -and -not $hasInvalidSha -and -not $hasMissingFiles
$validationState = if ($canPromote) { "validated-real-case-evidence-candidate" } else { "blocked-owner-action-required" }
$proofClassification = if ($canPromote) { "real-model-runtime-candidate" } else { "template-or-incomplete-not-proof" }

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "real-case-evidence-record-validation"
  sourceRecordPath = $RecordPath
  validationState = $validationState
  proofClassification = $proofClassification
  templateOnly = $templateOnly
  requiredOwnerFieldCount = $requiredOwnerFields.Count
  missingOwnerInputCount = $missingOwnerInputs.Count
  missingOwnerInputs = @($missingOwnerInputs.ToArray())
  invalidSha256FieldCount = $invalidShaFields.Count
  invalidSha256Fields = @($invalidShaFields.ToArray())
  missingFileCount = $missingFiles.Count
  missingFiles = @($missingFiles.ToArray())
  canPromoteRealModelRuntime = $canPromote
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Validation never publishes and never converts template, build-only, sidecar, article planning, local feed, ProjectReference, dependency probe, blocked-by-cuda-driver, managed-readiness, precheck-only, dry-run-only, or schema-only records into release proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "real-case-evidence-record-validation.json"
$markdownPath = Join-Path $artifactRoot "real-case-evidence-record-validation.md"

$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$missingLines = if ($missingOwnerInputs.Count -gt 0) { $missingOwnerInputs | ForEach-Object { "- ``$_``" } } else { @("- none") }
$markdown = @"
# Real Case Evidence Record Validation

生成时间：$($validation.generatedAtUtc)

## Summary

- validation state: ``$($validation.validationState)``
- proof classification: ``$($validation.proofClassification)``
- template only: ``$($validation.templateOnly)``
- missing owner input count: ``$($validation.missingOwnerInputCount)``
- invalid SHA256 field count: ``$($validation.invalidSha256FieldCount)``
- missing file count: ``$($validation.missingFileCount)``
- can promote real-model-runtime: ``$($validation.canPromoteRealModelRuntime)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Missing Owner Inputs

$($missingLines -join "`r`n")

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Real case evidence record validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState"
Write-Output "CanPromoteRealModelRuntime=$canPromote"
Write-Output "CanPublishPublicly=False"

if ($FailOnNotProof -and -not $canPromote) {
  throw "Real case evidence record cannot promote real-model-runtime: $validationState"
}
