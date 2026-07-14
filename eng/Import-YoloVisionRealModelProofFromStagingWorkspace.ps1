[CmdletBinding()]
param(
  [string]$OwnerStagingRoot = "",
  [string]$StagingImportPath = "artifacts\final-release\owner-real-proof-staging-workspace-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Add-YoloFinding {
  param([string]$Id, [string]$Category, [string]$Message)
  $script:findings.Add((New-OwnerFinding $Id "action-required" $Category $Message)) | Out-Null
}

function Get-StringProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  return [string](Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue "")
}

function Test-HttpUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("http://", [StringComparison]::OrdinalIgnoreCase)
}

function Get-LaneMapping {
  param([object[]]$Mappings, [string]$SourcePath)
  $normalized = $SourcePath.Replace("\", "/")
  return @($Mappings | Where-Object { ([string]$_.sourcePath).Replace("\", "/") -eq $normalized } | Select-Object -First 1)[0]
}

function Test-AssetManifestEntry {
  param(
    [AllowNull()][object]$Manifest,
    [AllowNull()][object]$Mapping,
    [string]$PathField,
    [string]$HashField
  )

  if ($null -eq $Mapping) {
    Add-YoloFinding "asset-manifest-$PathField-missing-mapping" "missing-file" "Asset manifest cannot validate $PathField because staging mapping is missing."
    return $false
  }

  $expectedPath = ([string](Get-PropertyOrDefault -Object $Mapping -Name "sourcePath" -DefaultValue "")).Replace("\", "/")
  $expectedHash = [string](Get-PropertyOrDefault -Object $Mapping -Name "computedSha256" -DefaultValue "")
  $actualPath = (Get-StringProperty $Manifest $PathField).Replace("\", "/")
  $actualHash = Get-StringProperty $Manifest $HashField
  $ok = $true
  if ($actualPath -ne $expectedPath) {
    Add-YoloFinding "asset-manifest-$PathField" "manifest-mismatch" "$PathField must equal $expectedPath."
    $ok = $false
  }
  if ($actualHash -ne $expectedHash -or -not (Test-Sha256Text $actualHash)) {
    Add-YoloFinding "asset-manifest-$HashField" "hash-mismatch" "$HashField must match $expectedPath SHA256."
    $ok = $false
  }

  return $ok
}

if (-not [System.IO.Path]::IsPathRooted($StagingImportPath)) {
  $StagingImportPath = Join-Path $RepositoryRoot $StagingImportPath
}

if (-not [string]::IsNullOrWhiteSpace($OwnerStagingRoot)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerRealProofStagingWorkspace.ps1") `
    -RepositoryRoot $RepositoryRoot `
    -OutputRoot $OutputRoot `
    -OwnerStagingRoot $OwnerStagingRoot `
    -RequireExistingFiles:$RequireExistingFiles.IsPresent `
    -RequireHashMatch:$RequireHashMatch.IsPresent | Out-Null
}
elseif (-not (Test-Path -LiteralPath $StagingImportPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerRealProofStagingWorkspace.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$stagingImport = Get-Content -LiteralPath $StagingImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$mappings = @(Convert-ToArray (Get-PropertyOrDefault -Object $stagingImport -Name "mappingResults" -DefaultValue @()))
$yoloMappings = @($mappings | Where-Object { [string]$_.lane -eq "yolovision-real-model" })
$findings = New-Object System.Collections.Generic.List[object]
$script:findings = $findings

$expectedFiles = @(
  "yolovision/task-metadata.json",
  "yolovision/model.onnx",
  "yolovision/model-license.json",
  "yolovision/labels.txt",
  "yolovision/input.bin",
  "yolovision/asset-manifest.json",
  "yolovision/output.json",
  "yolovision/stdout.log",
  "yolovision/stderr.log",
  "yolovision/runtime-transcript.log",
  "yolovision/host-metadata.json",
  "yolovision/real-model-execution-confirmation.json"
)

$mapByPath = @{}
foreach ($path in $expectedFiles) {
  $mapping = Get-LaneMapping $yoloMappings $path
  $mapByPath[$path] = $mapping
  if ($null -eq $mapping -or -not [bool](Get-PropertyOrDefault -Object $mapping -Name "fileExists" -DefaultValue $false)) {
    Add-YoloFinding ($path.Replace("/", "-") + "-missing") "missing-file" "YoloVision staging file is missing: $path"
  }
  elseif (-not [bool](Get-PropertyOrDefault -Object $mapping -Name "hashValid" -DefaultValue $false)) {
    Add-YoloFinding ($path.Replace("/", "-") + "-sha256") "missing-sha256" "YoloVision staging file has no computable SHA256: $path"
  }
}

function Read-MappedJson {
  param([string]$RelativePath, [string]$FindingId)
  $mapping = $mapByPath[$RelativePath]
  $resolvedPath = [string](Get-PropertyOrDefault -Object $mapping -Name "resolvedPath" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  try {
    return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
  }
  catch {
    Add-YoloFinding $FindingId "invalid-json" "$RelativePath must be valid JSON."
    return $null
  }
}

$taskMetadata = Read-MappedJson "yolovision/task-metadata.json" "task-metadata-json"
$license = Read-MappedJson "yolovision/model-license.json" "model-license-json"
$assetManifest = Read-MappedJson "yolovision/asset-manifest.json" "asset-manifest-json"
$outputJson = Read-MappedJson "yolovision/output.json" "output-json"
$hostMetadata = Read-MappedJson "yolovision/host-metadata.json" "host-metadata-json"
$confirmation = Read-MappedJson "yolovision/real-model-execution-confirmation.json" "real-model-execution-confirmation-json"

$taskName = Get-StringProperty $taskMetadata "taskName"
if ([string]::IsNullOrWhiteSpace($taskName)) {
  Add-YoloFinding "task-metadata-task-name" "missing-field" "task-metadata.json must include taskName."
}
foreach ($field in @("modelFamily", "taskType")) {
  if ([string]::IsNullOrWhiteSpace((Get-StringProperty $taskMetadata $field))) {
    Add-YoloFinding "task-metadata-$field" "missing-field" "task-metadata.json must include $field."
  }
}

if (-not (Test-HttpUrl (Get-StringProperty $license "modelSourceUrl"))) {
  Add-YoloFinding "model-license-source-url" "missing-field" "model-license.json must include http(s) modelSourceUrl."
}
if ([string]::IsNullOrWhiteSpace((Get-StringProperty $license "licenseName"))) {
  Add-YoloFinding "model-license-name" "missing-field" "model-license.json must include licenseName."
}
if (-not [bool](Get-PropertyOrDefault -Object $license -Name "redistributionAllowed" -DefaultValue $false)) {
  Add-YoloFinding "model-license-redistribution" "license-not-approved" "model-license.json must set redistributionAllowed=true for publication."
}
if (-not [bool](Get-PropertyOrDefault -Object $license -Name "ownerLicenseReviewed" -DefaultValue $false)) {
  Add-YoloFinding "model-license-owner-reviewed" "missing-owner-review" "model-license.json must set ownerLicenseReviewed=true."
}

$manifestLinkageCount = 0
$manifestLinkagePassedCount = 0
if ($null -ne $assetManifest) {
  $manifestEntries = @(
    [pscustomobject]@{ SourcePath = "yolovision/task-metadata.json"; PathField = "taskMetadataPath"; HashField = "taskMetadataSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/model.onnx"; PathField = "modelPath"; HashField = "modelSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/model-license.json"; PathField = "modelLicensePath"; HashField = "modelLicenseSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/labels.txt"; PathField = "labelsPath"; HashField = "labelsSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/input.bin"; PathField = "inputPath"; HashField = "inputImageSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/output.json"; PathField = "outputJsonPath"; HashField = "outputJsonSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/stdout.log"; PathField = "stdoutLogPath"; HashField = "stdoutLogSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/stderr.log"; PathField = "stderrLogPath"; HashField = "stderrLogSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/runtime-transcript.log"; PathField = "runtimeTranscriptPath"; HashField = "runtimeTranscriptSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/host-metadata.json"; PathField = "hostMetadataPath"; HashField = "hostMetadataSha256" },
    [pscustomobject]@{ SourcePath = "yolovision/real-model-execution-confirmation.json"; PathField = "realModelExecutionConfirmationPath"; HashField = "realModelExecutionConfirmationSha256" }
  )
  foreach ($entry in $manifestEntries) {
    $manifestLinkageCount++
    if (Test-AssetManifestEntry -Manifest $assetManifest -Mapping $mapByPath[$entry.SourcePath] -PathField $entry.PathField -HashField $entry.HashField) {
      $manifestLinkagePassedCount++
    }
  }
}
else {
  Add-YoloFinding "asset-manifest-missing" "missing-file" "asset-manifest.json is required for real model evidence linkage."
}

if ($null -eq $outputJson) {
  Add-YoloFinding "output-json-missing" "missing-file" "output.json must be readable."
}

foreach ($field in @("osDescription", "cudaVersion", "tensorRtVersion")) {
  if ([string]::IsNullOrWhiteSpace((Get-StringProperty $hostMetadata $field))) {
    Add-YoloFinding "host-metadata-$field" "missing-field" "host-metadata.json must include $field."
  }
}

$realExecutionConfirmed = [bool](Get-PropertyOrDefault -Object $confirmation -Name "realModelExecutionConfirmed" -DefaultValue $false)
$notReadinessOnly = [bool](Get-PropertyOrDefault -Object $confirmation -Name "notReadinessOnly" -DefaultValue $false)
$notTutorialOnly = [bool](Get-PropertyOrDefault -Object $confirmation -Name "notTutorialOnly" -DefaultValue $false)
$notMatrixOnly = [bool](Get-PropertyOrDefault -Object $confirmation -Name "notMatrixOnly" -DefaultValue $false)
if (-not $realExecutionConfirmed) {
  Add-YoloFinding "confirmation-real-execution" "missing-owner-confirmation" "real-model-execution-confirmation.json must set realModelExecutionConfirmed=true."
}
if (-not $notReadinessOnly -or -not $notTutorialOnly -or -not $notMatrixOnly) {
  Add-YoloFinding "confirmation-non-substitute" "forbidden-substitute" "real-model-execution-confirmation.json must reject readiness/tutorial/matrix-only evidence."
}
if (-not [string]::IsNullOrWhiteSpace($taskName) -and (Get-StringProperty $confirmation "taskName") -ne $taskName) {
  Add-YoloFinding "confirmation-task-name" "manifest-mismatch" "real-model confirmation taskName must match task metadata."
}

$runtimeTranscriptReady = $false
$runtimeTranscriptPath = [string](Get-PropertyOrDefault -Object $mapByPath["yolovision/runtime-transcript.log"] -Name "resolvedPath" -DefaultValue "")
if (-not [string]::IsNullOrWhiteSpace($runtimeTranscriptPath) -and (Test-Path -LiteralPath $runtimeTranscriptPath -PathType Leaf)) {
  $runtimeTranscript = Get-Content -LiteralPath $runtimeTranscriptPath -Raw -Encoding utf8
  $runtimeTranscriptReady = -not [string]::IsNullOrWhiteSpace($runtimeTranscript)
  if (-not $runtimeTranscriptReady) {
    Add-YoloFinding "runtime-transcript-empty" "missing-real-owner-input" "runtime-transcript.log must not be empty."
  }
  if ($runtimeTranscript.Contains("readiness", [StringComparison]::OrdinalIgnoreCase) -or
    $runtimeTranscript.Contains("tutorial-only", [StringComparison]::OrdinalIgnoreCase) -or
    $runtimeTranscript.Contains("matrix-only", [StringComparison]::OrdinalIgnoreCase)) {
    Add-YoloFinding "runtime-transcript-forbidden-substitute" "forbidden-substitute" "runtime-transcript.log must not be readiness/tutorial/matrix-only evidence."
  }
}

$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$assetFileCount = $expectedFiles.Count
$existingAssetFileCount = @($expectedFiles | Where-Object { [bool](Get-PropertyOrDefault -Object $mapByPath[$_] -Name "fileExists" -DefaultValue $false) }).Count
$sha256ValidFileCount = @($expectedFiles | Where-Object { [bool](Get-PropertyOrDefault -Object $mapByPath[$_] -Name "hashValid" -DefaultValue $false) }).Count
$licenseReady = $null -ne $license -and (Test-HttpUrl (Get-StringProperty $license "modelSourceUrl")) -and -not [string]::IsNullOrWhiteSpace((Get-StringProperty $license "licenseName")) -and [bool](Get-PropertyOrDefault -Object $license -Name "redistributionAllowed" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $license -Name "ownerLicenseReviewed" -DefaultValue $false)
$realExecutionConfirmationReady = $realExecutionConfirmed -and $notReadinessOnly -and $notTutorialOnly -and $notMatrixOnly -and (([string]::IsNullOrWhiteSpace($taskName)) -or (Get-StringProperty $confirmation "taskName") -eq $taskName)
$shapeValid = $existingAssetFileCount -eq $assetFileCount -and
  $sha256ValidFileCount -eq $assetFileCount -and
  $manifestLinkageCount -eq 11 -and $manifestLinkagePassedCount -eq 11 -and
  $licenseReady -and $realExecutionConfirmationReady -and $runtimeTranscriptReady -and
  $failedActionRequired.Count -eq 0
$state = if ($shapeValid) { "yolovision-real-model-staging-shape-valid-non-proof" } else { "blocked-yolovision-real-model-staging-owner-proof-required" }

$record = [pscustomobject]@{
  recordKind = "yolovision-real-model-proof-from-staging-workspace"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $state
  stagingImportState = [string](Get-PropertyOrDefault -Object $stagingImport -Name "importState" -DefaultValue "")
  stagingReadyForStrictImport = [bool](Get-PropertyOrDefault -Object $stagingImport -Name "readyForStrictImport" -DefaultValue $false)
  laneId = "yolovision-real-model"
  taskName = $taskName
  assetFileCount = $assetFileCount
  existingAssetFileCount = $existingAssetFileCount
  sha256RequiredFileCount = $assetFileCount
  sha256ValidFileCount = $sha256ValidFileCount
  assetManifestLinkageCount = $manifestLinkageCount
  assetManifestHashMatchCount = $manifestLinkagePassedCount
  assetManifestLinkageReady = $manifestLinkageCount -eq 11 -and $manifestLinkagePassedCount -eq 11
  modelLicenseReady = $licenseReady
  realModelExecutionConfirmationReady = $realExecutionConfirmationReady
  runtimeTranscriptLinkageReady = $runtimeTranscriptReady
  ownerEvidenceShapeValid = $shapeValid
  rejectsReadinessTutorialMatrixArtifacts = $realExecutionConfirmationReady
  findingCount = $findings.Count
  failedActionRequiredCount = $failedActionRequired.Count
  findings = @($findings.ToArray())
  proofCandidateReady = $false
  ownerActionRequired = -not $shapeValid
  passed = $false
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "YoloVision real model staging admission validates Owner-provided model assets, license, hashes, manifest linkage, host metadata, transcript, and real execution confirmation shape only. It does not run inference and is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "yolovision-real-model-proof-from-staging-workspace.json"
$mdPath = Join-Path $OutputRoot "yolovision-real-model-proof-from-staging-workspace.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)
$findingRows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# YoloVision Real Model Proof From Staging Workspace",
  "",
  "- importState: ``$state``",
  "- assets: ``$existingAssetFileCount/$assetFileCount``",
  "- sha256: ``$sha256ValidFileCount/$assetFileCount``",
  "- manifestLinkage: ``$manifestLinkagePassedCount/$manifestLinkageCount``",
  "- licenseReady: ``$licenseReady``",
  "- realExecutionConfirmationReady: ``$realExecutionConfirmationReady``",
  "- failedActionRequiredCount: ``$($failedActionRequired.Count)``",
  "",
  "| ID | Category | Message |",
  "|---|---|---|",
  @($findingRows),
  "",
  "## Boundary",
  "",
  $record.boundary
)
Write-Host "YoloVisionRealModelProofFromStagingWorkspaceState=$state Assets=$existingAssetFileCount/$assetFileCount Manifest=$manifestLinkagePassedCount/$manifestLinkageCount LicenseReady=$licenseReady ConfirmationReady=$realExecutionConfirmationReady FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $shapeValid) { throw "YoloVision real model staging workspace evidence is not shape-valid." }
