[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json",
  [string]$ImportedOwnerInputPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input.imported.json",
  [string]$RepositoryRoot,
  [switch]$Strict
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

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
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

function Test-Placeholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Text {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-Placeholder -Value $text) {
    return $true
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or $text.StartsWith("\\", [StringComparison]::Ordinal) -or $text.StartsWith("./", [StringComparison]::Ordinal) -or $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Get-ConsumerProjectScan {
  param([AllowNull()][object]$ProjectPath)

  $pathText = [string]$ProjectPath
  $result = [ordered]@{
    projectExists = $false
    usesProjectReference = $true
    usesLocalFeed = $true
    usesDirectNupkg = $true
    noProjectReference = $false
    noLocalFeed = $false
    noDirectNupkg = $false
  }

  if (Test-Placeholder -Value $pathText) {
    return [pscustomobject]$result
  }

  $resolvedPath = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return [pscustomobject]$result
  }

  $result.projectExists = $true
  $content = Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8
  $repoEscaped = [Regex]::Escape((Resolve-Path -LiteralPath $RepositoryRoot).Path)
  $result.usesProjectReference = $content.Contains("<ProjectReference", [StringComparison]::OrdinalIgnoreCase) -and
    ($content -match $repoEscaped -or $content.Contains("..\src\", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("../src/", [StringComparison]::OrdinalIgnoreCase))
  $result.usesLocalFeed = $content.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and
    ($content.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("local", [StringComparison]::OrdinalIgnoreCase))
  $result.usesDirectNupkg = $content.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
  $result.noProjectReference = -not $result.usesProjectReference
  $result.noLocalFeed = -not $result.usesLocalFeed
  $result.noDirectNupkg = -not $result.usesDirectNupkg
  return [pscustomobject]$result
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Package consumer runtime proof owner input import source not found: $resolvedInputPath"
}

$ownerInput = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordKind = [string](Get-PropertyOrDefault -Object $ownerInput -Name "recordKind" -DefaultValue "")
if ($recordKind -ne "package-consumer-runtime-proof-owner-input") {
  throw "recordKind must be package-consumer-runtime-proof-owner-input."
}

$resolvedImportedOwnerInputPath = Resolve-RepositoryPath -Path $ImportedOwnerInputPath
$importedOwnerInputDirectory = Split-Path -Parent $resolvedImportedOwnerInputPath
New-Item -ItemType Directory -Force -Path $importedOwnerInputDirectory | Out-Null
$ownerInput | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $resolvedImportedOwnerInputPath -Encoding utf8

$consumerProjectScan = Get-ConsumerProjectScan -ProjectPath (Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectPath" -DefaultValue "")
$publicPackageSourceIsLocal = Test-PublicPackageSourceIsLocal -Value (Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageSource" -DefaultValue "")
$hashFields = [ordered]@{
  managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgSha256" -DefaultValue "")
  runtimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNupkgSha256" -DefaultValue "")
  smokeLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogSha256" -DefaultValue "")
}
$hashFormatReady = $true
foreach ($entry in $hashFields.GetEnumerator()) {
  $hashFormatReady = $hashFormatReady -and (Test-Sha256Text -Value $entry.Value)
}

$placeholderFields = New-Object System.Collections.Generic.List[string]
foreach ($field in @(
    "cleanExternalConsumerRoot",
    "consumerProjectPath",
    "publicPackageSourceKind",
    "publicPackageSource",
    "publicPackageFeedUrl",
    "managedPackageUrl",
    "managedPackageVersion",
    "managedNupkgPath",
    "managedNupkgSha256",
    "runtimePackageUrl",
    "runtimePackageVersion",
    "runtimeNupkgPath",
    "runtimeNupkgSha256",
    "ownerName",
    "machineName",
    "hostOs",
    "hostArchitecture",
    "gpuName",
    "cudaDriverVersion",
    "cudaDriverSupportedRuntime",
    "cudaRuntimeVersion",
    "cudnnVersion",
    "tensorRtVersion",
    "tensorRtLine",
    "restoreCommand",
    "buildCommand",
    "smokeCommand",
    "exitCode",
    "startedAtUtc",
    "finishedAtUtc",
    "dependencyProbeStatus",
    "smokeStatus",
    "nativeAssetsCopied",
    "smokeLogPath",
    "smokeLogSha256",
    "sourceRunnerQueueStatus",
    "sourceRunnerInfrastructureStatus",
    "sourceRunnerOwnerAction",
    "stdoutSummary",
    "stderrSummary"
  )) {
  if (Test-Placeholder -Value (Get-PropertyOrDefault -Object $ownerInput -Name $field -DefaultValue "")) {
    $placeholderFields.Add($field) | Out-Null
  }
}

$validationScript = Join-Path $RepositoryRoot "eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1"
& $validationScript -InputPath $ImportedOwnerInputPath -OutputRoot $artifactRoot -RepositoryRoot $RepositoryRoot -Strict:$Strict

$schemaScript = Join-Path $RepositoryRoot "eng\Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1"
& $schemaScript -RepositoryRoot $RepositoryRoot

$forbiddenSubstituteScanScript = Join-Path $RepositoryRoot "eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1"
& $forbiddenSubstituteScanScript -InputPath $ImportedOwnerInputPath -RepositoryRoot $RepositoryRoot

$validationPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input-validation.json"
$validation = Get-Content -LiteralPath $validationPath -Raw -Encoding utf8 | ConvertFrom-Json

$forbiddenSubstituteScanPath = Join-Path $artifactRoot "package-consumer-runtime-proof-forbidden-substitute-scan.json"
$forbiddenSubstituteScan = Get-Content -LiteralPath $forbiddenSubstituteScanPath -Raw -Encoding utf8 | ConvertFrom-Json

$projectionScript = Join-Path $RepositoryRoot "eng\Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1"
& $projectionScript -OwnerInputPath $ImportedOwnerInputPath -RepositoryRoot $RepositoryRoot

$recordPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record.json"
$recordValidationScript = Join-Path $RepositoryRoot "eng\Test-PackageConsumerRuntimeProofRecord.ps1"
& $recordValidationScript -InputPath "artifacts/final-release/package-consumer-runtime-proof-record.json" -Strict:$false
$recordValidationPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record-validation.json"
$recordValidation = Get-Content -LiteralPath $recordValidationPath -Raw -Encoding utf8 | ConvertFrom-Json

$failedBlockerCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedBlockerCount" -DefaultValue 0)
$failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedActionRequiredCount" -DefaultValue 0)
$recordFailedProofItemCount = [int](Get-PropertyOrDefault -Object $recordValidation -Name "failedProofItemCount" -DefaultValue 0)
$detectedForbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $forbiddenSubstituteScan -Name "detectedForbiddenSubstituteCount" -DefaultValue 0)
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $validation -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputForbiddenSubstituteFree = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputForbiddenSubstituteFree" -DefaultValue $false) -and $detectedForbiddenSubstituteCount -eq 0
$ownerInputHashFieldsReady = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputHashFieldsReady" -DefaultValue $false)
$ownerInputPackageHashFilesMatch = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputPackageHashFilesMatch" -DefaultValue $false)
$ownerInputSmokeLogReady = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputSmokeLogReady" -DefaultValue $false)
$ownerInputHostMetadataReady = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputHostMetadataReady" -DefaultValue $false)
$ownerInputCommandEvidenceReady = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputCommandEvidenceReady" -DefaultValue $false)
$ownerInputRunnerInfrastructureReady = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerInputRunnerInfrastructureReady" -DefaultValue $false)
$ownerInputBlockedReason = [string](Get-PropertyOrDefault -Object $validation -Name "ownerInputBlockedReason" -DefaultValue "")
$importState = if ($failedBlockerCount -eq 0 -and $failedActionRequiredCount -eq 0 -and $recordFailedProofItemCount -eq 0) {
  "owner-input-import-ready-for-owner-review"
}
elseif ($failedBlockerCount -eq 0) {
  "blocked-owner-input-import-action-required"
}
else {
  "blocked-owner-input-import-invalid"
}

$import = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-owner-input-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $importState
  sourceInputPath = $resolvedInputPath
  importedOwnerInputPath = $ImportedOwnerInputPath
  validationPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
  schemaPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json"
  forbiddenSubstituteScanPath = "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json"
  projectedRecordPath = "artifacts/final-release/package-consumer-runtime-proof-record.json"
  projectedRecordValidationPath = "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  recordKindValidated = ($recordKind -eq "package-consumer-runtime-proof-owner-input")
  placeholderFieldCount = $placeholderFields.Count
  placeholderFields = @($placeholderFields.ToArray())
  hashFormatReady = $hashFormatReady
  publicPackageSourceIsLocal = $publicPackageSourceIsLocal
  consumerProjectScan = $consumerProjectScan
  forbiddenSubstituteScanState = [string](Get-PropertyOrDefault -Object $forbiddenSubstituteScan -Name "scanState" -DefaultValue "")
  detectedForbiddenSubstituteCount = $detectedForbiddenSubstituteCount
  ownerInputValidationState = [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "")
  ownerInputFailedBlockerCount = $failedBlockerCount
  ownerInputFailedActionRequiredCount = $failedActionRequiredCount
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputPackageHashFilesMatch = $ownerInputPackageHashFilesMatch
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputHostMetadataReady = $ownerInputHostMetadataReady
  ownerInputCommandEvidenceReady = $ownerInputCommandEvidenceReady
  ownerInputRunnerInfrastructureReady = $ownerInputRunnerInfrastructureReady
  ownerInputCanPromoteRuntimeProof = $false
  ownerInputBlockedReason = $ownerInputBlockedReason
  sourceRunnerQueueStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerQueueStatus" -DefaultValue "")
  sourceRunnerInfrastructureStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerInfrastructureStatus" -DefaultValue "")
  sourceRunnerOwnerAction = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerOwnerAction" -DefaultValue "")
  projectedRecordValidationState = [string](Get-PropertyOrDefault -Object $recordValidation -Name "validationState" -DefaultValue "")
  projectedRecordProofClassification = [string](Get-PropertyOrDefault -Object $recordValidation -Name "proofClassification" -DefaultValue "")
  projectedRecordFailedProofItemCount = $recordFailedProofItemCount
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPromoteProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  sourceArtifacts = @(
    $InputPath,
    $ImportedOwnerInputPath,
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
    "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
    "artifacts/final-release/package-consumer-runtime-proof-record.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  )
  safetyBoundary = "Owner input import copies and validates owner-provided package consumer metadata only. It does not publish packages, does not close release issues, does not promote runtime proof, and keeps local feed, ProjectReference, direct .nupkg, template, dry-run, and build-only substitutes blocked. queued GitHub Actions run and missing self-hosted runner remain owner-infra-action states, not proof. Additional forbidden substitutes include repository path leakage, GUI screenshots, and TensorRtExec build reports."
}

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input-import.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input-import.md"
$import | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$placeholderRows = $import.placeholderFields | ForEach-Object { "| ``$(ConvertTo-MarkdownCell $_)`` |" }
if ($null -eq $placeholderRows -or $placeholderRows.Count -eq 0) {
  $placeholderRows = @("| none |")
}

$markdown = @"
# Package Consumer Runtime Proof Owner Input Import

生成时间：$($import.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| importState | ``$($import.importState)`` |
| ownerInputValidationState | ``$($import.ownerInputValidationState)`` |
| ownerInputFailedBlockerCount | ``$($import.ownerInputFailedBlockerCount)`` |
| ownerInputFailedActionRequiredCount | ``$($import.ownerInputFailedActionRequiredCount)`` |
| cleanOwnerInputReady | ``$($import.cleanOwnerInputReady)`` |
| ownerInputForbiddenSubstituteFree | ``$($import.ownerInputForbiddenSubstituteFree)`` |
| ownerInputHashFieldsReady | ``$($import.ownerInputHashFieldsReady)`` |
| ownerInputPackageHashFilesMatch | ``$($import.ownerInputPackageHashFilesMatch)`` |
| ownerInputSmokeLogReady | ``$($import.ownerInputSmokeLogReady)`` |
| ownerInputHostMetadataReady | ``$($import.ownerInputHostMetadataReady)`` |
| ownerInputCommandEvidenceReady | ``$($import.ownerInputCommandEvidenceReady)`` |
| ownerInputRunnerInfrastructureReady | ``$($import.ownerInputRunnerInfrastructureReady)`` |
| sourceRunnerQueueStatus | ``$($import.sourceRunnerQueueStatus)`` |
| sourceRunnerInfrastructureStatus | ``$($import.sourceRunnerInfrastructureStatus)`` |
| sourceRunnerOwnerAction | ``$($import.sourceRunnerOwnerAction)`` |
| ownerInputCanPromoteRuntimeProof | ``$($import.ownerInputCanPromoteRuntimeProof)`` |
| projectedRecordValidationState | ``$($import.projectedRecordValidationState)`` |
| projectedRecordProofClassification | ``$($import.projectedRecordProofClassification)`` |
| projectedRecordFailedProofItemCount | ``$($import.projectedRecordFailedProofItemCount)`` |
| placeholderFieldCount | ``$($import.placeholderFieldCount)`` |
| hashFormatReady | ``$($import.hashFormatReady)`` |
| publicPackageSourceIsLocal | ``$($import.publicPackageSourceIsLocal)`` |
| forbiddenSubstituteScanState | ``$($import.forbiddenSubstituteScanState)`` |
| detectedForbiddenSubstituteCount | ``$($import.detectedForbiddenSubstituteCount)`` |
| performsPublish | ``$($import.performsPublish)`` |
| canPromoteRuntimeProof | ``$($import.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($import.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($import.canCloseReleaseIssue)`` |

## Placeholder Fields

| Field |
|---|
$($placeholderRows -join "`r`n")

## Safety Boundary

$($import.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Package consumer runtime proof owner input import failed with $failedBlockerCount blocker(s)."
}

Write-Host "Package consumer runtime proof owner input import written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ImportState=$($import.importState) OwnerInputValidationState=$($import.ownerInputValidationState) PerformsPublish=False CanPromoteRuntimeProof=False"
