[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$OwnerInputPath,
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

function Read-JsonFileOrNull {
  param([AllowNull()][string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $resolvedPath = if ([System.IO.Path]::IsPathRooted($Path)) {
    $Path
  }
  else {
    Join-Path $RepositoryRoot $Path
  }

  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Find-ProofLine {
  param(
    [AllowNull()][object]$Record,
    [string]$Id
  )

  if ($null -eq $Record) {
    return $null
  }

  foreach ($line in @(Get-PropertyOrDefault -Object $Record -Name "backfillLines" -DefaultValue @())) {
    if ([string](Get-PropertyOrDefault -Object $line -Name "id" -DefaultValue "") -eq $Id) {
      return $line
    }
  }

  foreach ($spec in @(Get-PropertyOrDefault -Object $Record -Name "draftSpecs" -DefaultValue @())) {
    if ([string](Get-PropertyOrDefault -Object $spec -Name "id" -DefaultValue "") -eq $Id) {
      return $spec
    }
  }

  return $null
}

function Test-IsOutsideRepository {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path) -or $Path -like "<*>" -or $Path -eq "missing") {
    return $false
  }

  try {
    $repositoryFullPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidateFullPath = [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return -not $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $false
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) {
    return $true
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or $text.StartsWith("\\", [StringComparison]::Ordinal) -or $text.StartsWith("./", [StringComparison]::Ordinal) -or $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-ConsumerProjectReferenceFlags {
  param([AllowNull()][object]$ProjectPath)

  $pathText = [string]$ProjectPath
  $result = [ordered]@{
    projectExists = $false
    usesProjectReference = $true
    usesLocalFeed = $true
    usesDirectNupkg = $true
  }

  if (Test-IsPlaceholder -Value $pathText) {
    return [pscustomobject]$result
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
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
  return [pscustomobject]$result
}

$orchestrator = Read-JsonOrNull "artifacts\final-release\owner-external-proof-backfill-orchestrator.json"
$draftPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-draft-pack.json"
$repairPack = Read-JsonOrNull "artifacts\final-release\owner-proof-input-repair-pack.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$packageConsumerRuntimeProofRecord = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record.json"
$packageConsumerRuntimeProofRecordValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record-validation.json"
$ownerInput = Read-JsonFileOrNull -Path $OwnerInputPath

$orchestratorLine = Find-ProofLine -Record $orchestrator -Id "package-consumer-runtime"
$draftSpec = Find-ProofLine -Record $draftPack -Id "package-consumer-runtime"

$inputDraftPath = [string](Get-PropertyOrDefault -Object $draftSpec -Name "inputDraftPath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.input-draft.json")
$targetProofRecordPath = [string](Get-PropertyOrDefault -Object $orchestratorLine -Name "targetProofRecordPath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.json")
$strictValidationCommand = [string](Get-PropertyOrDefault -Object $orchestratorLine -Name "strictValidationCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -FailOnNotProof")
$requiredRules = Convert-ToStringArray (Get-PropertyOrDefault -Object $orchestratorLine -Name "requiredRealInputRules" -DefaultValue @(
  "cleanExternalConsumerIdentity",
  "noProjectReference",
  "noLocalFeedAsPublicProof",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "runtimePackageKeyMatches",
  "compatibleHostMetadata",
  "smokeCommandIncludesRuntimePackageKey",
  "smokeLogPath",
  "smokeLogSha256"
))

$cleanExternalConsumerRoot = "<owner-fill-clean-external-consumer-root-outside-repository>"
$consumerProjectPath = "<owner-fill-clean-consumer-csproj-path>"
$managedNupkgPath = "<owner-fill-public-managed-nupkg-path>"
$runtimeNupkgPath = "<owner-fill-public-runtime-nupkg-path>"
$smokeLogPath = "<owner-fill-package-consumer-smoke-log-path>"
$smokeCommand = "dotnet run --project <clean-consumer-project> -- --runtime-package-key $RuntimePackageKey"

if ($null -ne $ownerInput) {
  $cleanExternalConsumerRoot = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cleanExternalConsumerRoot" -DefaultValue $cleanExternalConsumerRoot)
  $consumerProjectPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectPath" -DefaultValue $consumerProjectPath)
  $managedNupkgPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgPath" -DefaultValue $managedNupkgPath)
  $runtimeNupkgPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNupkgPath" -DefaultValue $runtimeNupkgPath)
  $smokeLogPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogPath" -DefaultValue $smokeLogPath)
  $smokeCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeCommand" -DefaultValue $smokeCommand)
  $RuntimePackageKey = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageKey" -DefaultValue $RuntimePackageKey)
}

$cleanExternalConsumerRootIsOutsideRepository = Test-IsOutsideRepository -Path $cleanExternalConsumerRoot
$smokeCommandIncludesRuntimePackageKey = $smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal)
$runtimePackageKeyMatches = $smokeCommand.Contains($RuntimePackageKey, [StringComparison]::Ordinal)
$consumerProjectFlags = Get-ConsumerProjectReferenceFlags -ProjectPath $consumerProjectPath
$ownerInputOverlayApplied = $null -ne $ownerInput
$publicPackageSource = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageSource" -DefaultValue "<owner-fill-public-package-source-url-or-id>") } else { "<owner-fill-public-package-source-url-or-id>" }
$managedPackageVersion = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedPackageVersion" -DefaultValue "<owner-fill-managed-package-version>") } else { "<owner-fill-managed-package-version>" }
$managedNupkgSha256 = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgSha256" -DefaultValue "<owner-fill-managed-nupkg-sha256>") } else { "<owner-fill-managed-nupkg-sha256>" }
$runtimePackageVersion = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageVersion" -DefaultValue "<owner-fill-runtime-package-version>") } else { "<owner-fill-runtime-package-version>" }
$runtimeNupkgSha256 = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNupkgSha256" -DefaultValue "<owner-fill-runtime-nupkg-sha256>") } else { "<owner-fill-runtime-nupkg-sha256>" }
$hostOs = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "hostOs" -DefaultValue "<owner-fill-host-os>") } else { "<owner-fill-host-os>" }
$hostArchitecture = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "hostArchitecture" -DefaultValue "<owner-fill-host-architecture>") } else { "<owner-fill-host-architecture>" }
$cudaDriverVersion = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudaDriverVersion" -DefaultValue "<owner-fill-cuda-driver-version>") } else { "<owner-fill-cuda-driver-version>" }
$cudaRuntimeVersion = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudaRuntimeVersion" -DefaultValue "<owner-fill-cuda-runtime-version>") } else { "<owner-fill-cuda-runtime-version>" }
$tensorRtVersion = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "tensorRtVersion" -DefaultValue "<owner-fill-tensorrt-version>") } else { "<owner-fill-tensorrt-version>" }
$smokeLogSha256 = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogSha256" -DefaultValue "<owner-fill-smoke-log-sha256>") } else { "<owner-fill-smoke-log-sha256>" }
$stdoutSummary = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "stdoutSummary" -DefaultValue "<owner-fill-stdout-summary>") } else { "<owner-fill-stdout-summary>" }
$stderrSummary = if ($null -ne $ownerInput) { [string](Get-PropertyOrDefault -Object $ownerInput -Name "stderrSummary" -DefaultValue "<owner-fill-stderr-summary>") } else { "<owner-fill-stderr-summary>" }
$publicPackageSourceIsLocal = Test-PublicPackageSourceIsLocal -Value $publicPackageSource
$packageConsumerRuntimeProofRecordValidationState = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofRecordValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-record-validation")
$packageConsumerRuntimeProofRecordCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofRecordValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$packageConsumerRuntimeProofRecordFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofRecordValidation -Name "failedActionRequiredCount" -DefaultValue 0)
$packageConsumerRuntimeProofRecordProofClassification = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofRecordValidation -Name "proofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofRecord -Name "proofClassification" -DefaultValue "missing-proof-classification")))

$blockedBy = @(
  "missing clean external consumer root outside repository",
  "missing public package source evidence",
  "missing managed nupkg SHA256",
  "missing runtime nupkg SHA256",
  "missing compatible host metadata",
  "missing smoke log path and SHA256",
  "candidate is not validator-passing runtime proof"
)

$record = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = "blocked-real-package-consumer-smoke-required"
  proofLineId = "package-consumer-runtime"
  inputDraftPath = $inputDraftPath
  orchestratorPath = "artifacts/final-release/owner-external-proof-backfill-orchestrator.json"
  ownerInputPath = if ([string]::IsNullOrWhiteSpace($OwnerInputPath)) { "" } else { $OwnerInputPath }
  ownerInputOverlayApplied = $ownerInputOverlayApplied
  targetProofRecordPath = $targetProofRecordPath
  packageConsumerRuntimeProofRecordPath = "artifacts/final-release/package-consumer-runtime-proof-record.json"
  packageConsumerRuntimeProofRecordValidationPath = "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  packageConsumerRuntimeProofRecordValidationState = $packageConsumerRuntimeProofRecordValidationState
  packageConsumerRuntimeProofRecordCanPromoteRuntimeProof = $packageConsumerRuntimeProofRecordCanPromoteRuntimeProof
  packageConsumerRuntimeProofRecordFailedActionRequiredCount = $packageConsumerRuntimeProofRecordFailedActionRequiredCount
  packageConsumerRuntimeProofRecordProofClassification = $packageConsumerRuntimeProofRecordProofClassification
  cleanExternalConsumerRoot = $cleanExternalConsumerRoot
  cleanExternalConsumerRootIsOutsideRepository = $cleanExternalConsumerRootIsOutsideRepository
  consumerProjectPath = $consumerProjectPath
  consumerProjectExists = $consumerProjectFlags.projectExists
  consumerProjectUsesProjectReference = $consumerProjectFlags.usesProjectReference
  consumerProjectUsesLocalFeed = $consumerProjectFlags.usesLocalFeed
  consumerProjectUsesDirectNupkg = $consumerProjectFlags.usesDirectNupkg
  publicPackageSource = $publicPackageSource
  publicPackageSourceIsLocal = $publicPackageSourceIsLocal
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  managedPackageVersion = $managedPackageVersion
  managedNupkgPath = $managedNupkgPath
  managedNupkgSha256 = $managedNupkgSha256
  runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
  runtimePackageVersion = $runtimePackageVersion
  runtimePackageKey = $RuntimePackageKey
  runtimePackageKeyMatches = $runtimePackageKeyMatches
  runtimeNupkgPath = $runtimeNupkgPath
  runtimeNupkgSha256 = $runtimeNupkgSha256
  compatibleHostMetadata = [pscustomobject]@{
    hostOs = $hostOs
    hostArchitecture = $hostArchitecture
    cudaDriverVersion = $cudaDriverVersion
    cudaRuntimeVersion = $cudaRuntimeVersion
    tensorRtVersion = $tensorRtVersion
  }
  hostOs = $hostOs
  hostArchitecture = $hostArchitecture
  cudaDriverVersion = $cudaDriverVersion
  cudaRuntimeVersion = $cudaRuntimeVersion
  tensorRtVersion = $tensorRtVersion
  smokeCommand = $smokeCommand
  smokeCommandIncludesRuntimePackageKey = $smokeCommandIncludesRuntimePackageKey
  smokeLogPath = $smokeLogPath
  smokeLogSha256 = $smokeLogSha256
  stdoutSummary = $stdoutSummary
  stderrSummary = $stderrSummary
  requiredRealInputRules = $requiredRules
  blockedBy = $blockedBy
  requiredOwnerActions = @(
    "Create a clean external consumer root outside this repository.",
    "Restore packages from the real public package source; do not use ProjectReference, local feed, or direct .nupkg as public proof.",
    "Capture managed/runtime package identity and SHA256 values.",
    "Run runtime smoke with --runtime-package-key $RuntimePackageKey on a compatible CUDA/TensorRT host.",
    "Record smoke log path, smoke log SHA256, stdout/stderr summaries, and host metadata."
  )
  strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict"
  downstreamProofValidationCommand = $strictValidationCommand
  canPromoteProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  repairPackState = [string](Get-PropertyOrDefault -Object $repairPack -Name "repairPackState" -DefaultValue "missing-owner-proof-input-repair-pack")
  sourceArtifacts = @(
    "artifacts/final-release/owner-external-proof-backfill-orchestrator.json",
    "artifacts/final-release/owner-proof-input-draft-pack.json",
    "artifacts/final-release/owner-proof-input-repair-pack.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-record.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-record.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  safetyBoundary = "package-consumer-runtime-proof-candidate is an owner input candidate only. It does not publish packages, cannot use local feed/ProjectReference/direct .nupkg as public proof, and cannot promote runtime proof without real external smoke evidence."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-candidate.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-candidate.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$requiredRuleLines = $requiredRules | ForEach-Object { "- ``$_``" }
$blockedLines = $blockedBy | ForEach-Object { "- $_" }
$actionLines = $record.requiredOwnerActions | ForEach-Object { "- $_" }

$markdown = @"
# Package Consumer Runtime Proof Candidate

生成时间：$($record.generatedAtUtc)

## 总结

``package-consumer-runtime-proof-candidate`` 是 ``package-consumer-runtime`` proof line 的 owner-fill candidate input record。它用于汇总 clean external consumer、public package source、managed/runtime nupkg hash、compatible host metadata、runtime-key smoke command 和 smoke log hash 的真实输入要求。

它不是 proof，不执行发布，不关闭 release issue，也不会把 local feed、ProjectReference 或 direct ``.nupkg`` 提升为 public package proof。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($record.recordKind)`` |
| candidateState | ``$($record.candidateState)`` |
| canPromoteProof | ``$($record.canPromoteProof)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| runtimePackageKey | ``$RuntimePackageKey`` |

## Required Real Input Rules

$($requiredRuleLines -join "`r`n")

## 当前 Blockers

$($blockedLines -join "`r`n")

## Owner Actions

$($actionLines -join "`r`n")

## Strict Validator

```powershell
$($record.strictValidationCommand)
```

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof candidate written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "CandidateState=$($record.candidateState)"
Write-Host "CanPromoteProof=$($record.canPromoteProof)"
Write-Host "PerformsPublish=$($record.performsPublish)"
Write-Host "CanPublishPublicly=$($record.canPublishPublicly)"
Write-Host "CanCloseReleaseIssue=$($record.canCloseReleaseIssue)"
