[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Test-JsonTextContainsAny {
  param([string]$JsonText, [string[]]$Needles)
  foreach ($needle in $Needles) {
    if (-not [string]::IsNullOrWhiteSpace($needle) -and $JsonText.Contains($needle, [StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }
  return $false
}

function New-AlignmentField {
  param(
    [string]$Name,
    [string]$Kind,
    [string]$Validator,
    [string[]]$Aliases,
    [string]$ProofRole
  )

  [pscustomobject]@{
    name = $Name
    kind = $Kind
    validator = $Validator
    aliases = @($Aliases)
    proofRole = $ProofRole
    ownerRequired = $true
    proofRequired = $true
  }
}

$ownerTemplate = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json"
$ownerSchema = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input.schema.json"
$ownerValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$runbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$runbookValidation = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook-validation.json"
$collectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalOwnerOneScreenPack = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$cleanChecklist = Read-JsonOrNull "artifacts\final-release\clean-consumer-runtime-proof-execution-checklist.json"

$ownerTemplatePropertyNames = @()
if ($ownerTemplate) {
  $ownerTemplatePropertyNames = @($ownerTemplate.PSObject.Properties.Name)
}

$schemaFieldNames = @()
$schemaFieldByName = @{}
if ($ownerSchema) {
  foreach ($field in @(Get-PropertyOrDefault -Object $ownerSchema -Name "fields" -DefaultValue @())) {
    $fieldName = [string](Get-PropertyOrDefault -Object $field -Name "name" -DefaultValue "")
    if (-not [string]::IsNullOrWhiteSpace($fieldName)) {
      $schemaFieldNames += $fieldName
      $schemaFieldByName[$fieldName] = $field
    }
  }
}

$runbookRequiredOwnerInputFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $runbook -Name "requiredOwnerInputFields" -DefaultValue @())
$runbookForbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $runbook -Name "forbiddenRuntimeSmokeSubstitutes" -DefaultValue @())
$runtimeSmokeStatus = "Smoke=not-requested"
$collectionText = if ($collectionBundle) { $collectionBundle | ConvertTo-Json -Depth 16 } else { "" }
$releaseText = if ($releaseEvidenceBundle) { $releaseEvidenceBundle | ConvertTo-Json -Depth 16 } else { "" }
$finalOwnerText = if ($finalOwnerOneScreenPack) { $finalOwnerOneScreenPack | ConvertTo-Json -Depth 16 } else { "" }
$cleanChecklistText = if ($cleanChecklist) { $cleanChecklist | ConvertTo-Json -Depth 16 } else { "" }

$fields = @(
  New-AlignmentField -Name "cleanExternalConsumerRoot" -Kind "path" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("cleanExternalConsumerRoot", "cleanConsumer.projectRoot", "repository-external clean consumer", "clean consumer project path outside") -ProofRole "repository-external clean consumer root"
  New-AlignmentField -Name "consumerProjectPath" -Kind "path" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("consumerProjectPath", "consumer project", "external consumer project", "clean consumer project path") -ProofRole "external .csproj identity"
  New-AlignmentField -Name "publicPackageSource" -Kind "package-source" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("publicPackageSource", "public package source", "packageSource.url", "public-package-source-url") -ProofRole "public or owner-approved package source"
  New-AlignmentField -Name "managedPackageId" -Kind "package-identity" -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Aliases @("managedPackageId", "managedPackage.id", "managed package id") -ProofRole "managed package identity"
  New-AlignmentField -Name "managedPackageVersion" -Kind "package-identity" -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Aliases @("managedPackageVersion", "managedPackage.id/version", "managed package id/version") -ProofRole "managed package version"
  New-AlignmentField -Name "managedNupkgSha256" -Kind "hash" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("managedNupkgSha256", "managedPackage.id/version/sha256", "managed-nupkg-sha256", "managed nupkg SHA256") -ProofRole "managed nupkg integrity"
  New-AlignmentField -Name "runtimePackageId" -Kind "package-identity" -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Aliases @("runtimePackageId", "runtimePackage.id", "runtime package id") -ProofRole "runtime package identity"
  New-AlignmentField -Name "runtimePackageVersion" -Kind "package-identity" -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Aliases @("runtimePackageVersion", "runtimePackage.id/version", "runtime package id/version") -ProofRole "runtime package version"
  New-AlignmentField -Name "runtimePackageKey" -Kind "package-identity" -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Aliases @("runtimePackageKey", "runtimePackage.id/version/key", "runtime package key") -ProofRole "runtime package key"
  New-AlignmentField -Name "runtimeNupkgSha256" -Kind "hash" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("runtimeNupkgSha256", "runtimePackage.id/version/key/sha256", "runtime-nupkg-sha256", "runtime nupkg SHA256") -ProofRole "runtime nupkg integrity"
  New-AlignmentField -Name "gpuName" -Kind "host-metadata" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("gpuName", "hostMetadata.os/arch/rid/gpu", "gpu-name", "NVIDIA GPU") -ProofRole "GPU identity"
  New-AlignmentField -Name "cudaDriverVersion" -Kind "host-metadata" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("cudaDriverVersion", "driver", "nvidia-driver-version") -ProofRole "NVIDIA driver version"
  New-AlignmentField -Name "cudaRuntimeVersion" -Kind "host-metadata" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("cudaRuntimeVersion", "cuda", "cuda-version") -ProofRole "CUDA runtime version"
  New-AlignmentField -Name "tensorRtVersion" -Kind "host-metadata" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("tensorRtVersion", "tensorrt", "tensorrt-version") -ProofRole "TensorRT runtime version"
  New-AlignmentField -Name "cudnnVersion" -Kind "host-metadata" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("cudnnVersion", "cudnn", "cudnn-version") -ProofRole "cuDNN version"
  New-AlignmentField -Name "restoreCommand" -Kind "command" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("restoreCommand", "restore", "dotnet restore") -ProofRole "restore command"
  New-AlignmentField -Name "buildCommand" -Kind "command" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("buildCommand", "build", "dotnet build") -ProofRole "build command"
  New-AlignmentField -Name "smokeCommand" -Kind "command" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("smokeCommand", "runtime smoke", "Test-PackageConsumer.ps1", "-RunSmoke") -ProofRole "runtime smoke command"
  New-AlignmentField -Name "exitCode" -Kind "result" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("exitCode", "runtimeSmoke.exitCode", "exit code 0") -ProofRole "smoke exit code"
  New-AlignmentField -Name "dependencyProbeStatus" -Kind "result" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("dependencyProbeStatus", "dependencyProbe", "dependency probe") -ProofRole "dependency probe result"
  New-AlignmentField -Name "smokeStatus" -Kind "result" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("smokeStatus", "runtimeSmoke.smokeStatus", "smokeStatus=passed") -ProofRole "runtime smoke status"
  New-AlignmentField -Name "nativeAssetsCopied" -Kind "result" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("nativeAssetsCopied", "native asset", "native assets") -ProofRole "native assets copied"
  New-AlignmentField -Name "smokeLogPath" -Kind "log-path" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("smokeLogPath", "runtimeSmoke.stdoutPath", "runtime smoke logs", "smoke log path") -ProofRole "runtime smoke log path"
  New-AlignmentField -Name "smokeLogSha256" -Kind "hash" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("smokeLogSha256", "runtimeSmoke.stdoutPath/stderrPath/mergedTranscriptPath/sha256", "runtime smoke log/hash", "smoke log SHA256") -ProofRole "runtime smoke log integrity"
  New-AlignmentField -Name "stdoutSummary" -Kind "log-summary" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("stdoutSummary", "stdout/stderr", "stdout summary", "results.stdoutSummary") -ProofRole "reviewed stdout summary"
  New-AlignmentField -Name "stderrSummary" -Kind "log-summary" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("stderrSummary", "stdout/stderr", "stderr summary", "results.stderrSummary", "no-stderr-emitted") -ProofRole "reviewed stderr summary"
  New-AlignmentField -Name "ownerName" -Kind "owner-review" -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("ownerName", "ownerReview.name", "owner review") -ProofRole "owner review identity"
  New-AlignmentField -Name "machineName" -Kind "owner-review" -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("machineName", "ownerReview.name/machine", "machine") -ProofRole "owner proof machine"
  New-AlignmentField -Name "startedAtUtc" -Kind "timestamp" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("startedAtUtc", "started", "started-at-utc") -ProofRole "execution start time"
  New-AlignmentField -Name "finishedAtUtc" -Kind "timestamp" -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Aliases @("finishedAtUtc", "finished", "finished-at-utc") -ProofRole "execution finish time"
)

$fieldRows = foreach ($field in $fields) {
  $schemaField = if ($schemaFieldByName.ContainsKey($field.name)) { $schemaFieldByName[$field.name] } else { $null }
  $aliases = @($field.aliases + $field.name)
  $presentInOwnerTemplate = $ownerTemplatePropertyNames -contains $field.name
  $presentInSchema = $schemaFieldNames -contains $field.name
  $presentInRunbookRequiredOwnerInputFields = $runbookRequiredOwnerInputFields -contains $field.name
  $presentInCollectionBundle = Test-JsonTextContainsAny -JsonText $collectionText -Needles $aliases
  $presentInReleaseEvidence = Test-JsonTextContainsAny -JsonText $releaseText -Needles $aliases
  $presentInFinalOwnerOneScreenPack = Test-JsonTextContainsAny -JsonText $finalOwnerText -Needles $aliases
  $presentInCleanConsumerChecklist = Test-JsonTextContainsAny -JsonText $cleanChecklistText -Needles $aliases
  $coveredSurfaceCount = @(
    $presentInOwnerTemplate,
    $presentInSchema,
    $presentInRunbookRequiredOwnerInputFields,
    $presentInCollectionBundle,
    $presentInReleaseEvidence,
    $presentInFinalOwnerOneScreenPack,
    $presentInCleanConsumerChecklist
  ) | Where-Object { $_ } | Measure-Object | Select-Object -ExpandProperty Count

  [pscustomobject]@{
    name = $field.name
    kind = $field.kind
    ownerRequired = $field.ownerRequired
    proofRequired = $field.proofRequired
    validator = $field.validator
    proofRole = if ($schemaField) { [string](Get-PropertyOrDefault -Object $schemaField -Name "proofRole" -DefaultValue $field.proofRole) } else { $field.proofRole }
    validatorItemId = if ($schemaField) { [string](Get-PropertyOrDefault -Object $schemaField -Name "validatorItemId" -DefaultValue "") } else { "" }
    presentInOwnerTemplate = $presentInOwnerTemplate
    presentInOwnerSchema = $presentInSchema
    presentInRunbookRequiredOwnerInputFields = $presentInRunbookRequiredOwnerInputFields
    presentInCollectionBundle = $presentInCollectionBundle
    presentInReleaseEvidenceBundle = $presentInReleaseEvidence
    presentInFinalOwnerOneScreenPack = $presentInFinalOwnerOneScreenPack
    presentInCleanConsumerChecklist = $presentInCleanConsumerChecklist
    coveredSurfaceCount = $coveredSurfaceCount
    alignmentReady = $presentInOwnerTemplate -and $presentInSchema -and $presentInRunbookRequiredOwnerInputFields -and $presentInCollectionBundle -and ($presentInFinalOwnerOneScreenPack -or $presentInCleanConsumerChecklist)
  }
}

$missingRequired = @($fieldRows | Where-Object { -not $_.alignmentReady })
$runbookValidationState = [string](Get-PropertyOrDefault -Object $runbookValidation -Name "validationState" -DefaultValue "missing-compatible-host-runtime-proof-runbook-validation")
$ownerValidationState = [string](Get-PropertyOrDefault -Object $ownerValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-owner-input-validation")
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$ownerRuntimeSmokeRunbookState = [string](Get-PropertyOrDefault -Object $runbook -Name "ownerRuntimeSmokeRunbookState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookOwnerRuntimeSmokeState" -DefaultValue "missing-owner-runtime-smoke-runbook-state")))

$record = [pscustomobject]@{
  recordKind = "package-consumer-owner-runtime-smoke-field-alignment"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  alignmentState = "blocked-owner-compatible-host-runtime-smoke-field-alignment"
  ownerRuntimeSmokeRunbookState = $ownerRuntimeSmokeRunbookState
  ownerInputValidationState = $ownerValidationState
  compatibleHostRunbookValidationState = $runbookValidationState
  releaseEvidenceBundleState = $releaseEvidenceState
  runtimeSmokeStatus = $runtimeSmokeStatus
  fieldCount = $fieldRows.Count
  missingRequiredFieldCount = $missingRequired.Count
  ownerTemplateCoverageCount = @($fieldRows | Where-Object { $_.presentInOwnerTemplate }).Count
  ownerSchemaCoverageCount = @($fieldRows | Where-Object { $_.presentInOwnerSchema }).Count
  runbookCoverageCount = @($fieldRows | Where-Object { $_.presentInRunbookRequiredOwnerInputFields }).Count
  collectionBundleCoverageCount = @($fieldRows | Where-Object { $_.presentInCollectionBundle }).Count
  releaseEvidenceCoverageCount = @($fieldRows | Where-Object { $_.presentInReleaseEvidenceBundle }).Count
  finalOwnerSurfaceCoverageCount = @($fieldRows | Where-Object { $_.presentInFinalOwnerOneScreenPack -or $_.presentInCleanConsumerChecklist }).Count
  forbiddenRuntimeSmokeSubstitutes = @($runbookForbiddenSubstitutes)
  sourceArtifacts = @(
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook-validation.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/final-owner-execution-one-screen-pack.json",
    "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json"
  )
  fields = @($fieldRows)
  missingRequiredFields = @($missingRequired | Select-Object -ExpandProperty name)
  performsPublish = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  boundary = "This alignment matrix checks field coverage only. It is not runtime proof, package publish approval, public channel proof, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-owner-runtime-smoke-field-alignment.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-owner-runtime-smoke-field-alignment.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($field in $fieldRows) {
  "| ``$($field.name)`` | ``$($field.kind)`` | ``$($field.presentInOwnerSchema)`` | ``$($field.presentInRunbookRequiredOwnerInputFields)`` | ``$($field.presentInCollectionBundle)`` | ``$($field.presentInReleaseEvidenceBundle)`` | ``$($field.presentInFinalOwnerOneScreenPack -or $field.presentInCleanConsumerChecklist)`` | ``$($field.validator)`` |"
}

$markdown = @"
# Package Consumer Owner Runtime Smoke Field Alignment

| Field | Value |
|---|---|
| alignmentState | ``$($record.alignmentState)`` |
| ownerRuntimeSmokeRunbookState | ``$($record.ownerRuntimeSmokeRunbookState)`` |
| ownerInputValidationState | ``$($record.ownerInputValidationState)`` |
| compatibleHostRunbookValidationState | ``$($record.compatibleHostRunbookValidationState)`` |
| runtimeSmokeStatus | ``$($record.runtimeSmokeStatus)`` |
| fieldCount | ``$($record.fieldCount)`` |
| missingRequiredFieldCount | ``$($record.missingRequiredFieldCount)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Fields

| Field | Kind | Schema | Runbook | Collection | Release Evidence | Final Owner Surface | Validator |
|---|---|---:|---:|---:|---:|---:|---|
$($rows -join "`r`n")

## Forbidden Runtime Smoke Substitutes

$($runbookForbiddenSubstitutes | ForEach-Object { "- ``$_``" } | Out-String)

## Boundary

$($record.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer owner runtime smoke field alignment written to $jsonPath"
Write-Host "AlignmentState=$($record.alignmentState) Smoke=$($record.runtimeSmokeStatus.Substring(6)) FieldCount=$($record.fieldCount) MissingRequiredFields=$($record.missingRequiredFieldCount) CanPromoteRuntimeProof=False CanPublishPublicly=False CanCloseReleaseIssue=False"
