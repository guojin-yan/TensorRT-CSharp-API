[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json",
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
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
  $result.noProjectReference = -not $result.usesProjectReference
  $result.noLocalFeed = -not $result.usesLocalFeed
  $result.noDirectNupkg = -not $result.usesDirectNupkg
  return [pscustomobject]$result
}

$resolvedOwnerInputPath = Resolve-InputPath -Path $OwnerInputPath
if (-not (Test-Path -LiteralPath $resolvedOwnerInputPath -PathType Leaf)) {
  throw "Owner input file not found: $resolvedOwnerInputPath"
}

$ownerInput = Get-Content -LiteralPath $resolvedOwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$RuntimePackageKey = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageKey" -DefaultValue $RuntimePackageKey)

$cleanExternalConsumerRoot = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cleanExternalConsumerRoot" -DefaultValue "")
$consumerProjectPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectPath" -DefaultValue "")
$publicPackageSourceKind = [string](Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageSourceKind" -DefaultValue "")
$publicPackageSource = [string](Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageSource" -DefaultValue "")
$publicPackageFeedUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageFeedUrl" -DefaultValue "")
$managedPackageUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedPackageUrl" -DefaultValue "")
$runtimePackageUrl = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageUrl" -DefaultValue "")
$managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgSha256" -DefaultValue "")
$runtimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNupkgSha256" -DefaultValue "")
$smokeLogSha256 = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogSha256" -DefaultValue "")
$smokeCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeCommand" -DefaultValue "")
$sourceRunnerQueueStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerQueueStatus" -DefaultValue "")
$sourceRunnerInfrastructureStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerInfrastructureStatus" -DefaultValue "")
$sourceRunnerOwnerAction = [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerOwnerAction" -DefaultValue "")
$scan = Get-ConsumerProjectScan -ProjectPath $consumerProjectPath
$ownerInputForbiddenSubstituteFree = [bool](Get-PropertyOrDefault -Object $scan -Name "noProjectReference" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $scan -Name "noLocalFeed" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $scan -Name "noDirectNupkg" -DefaultValue $false) -and
  -not (Test-PublicPackageSourceIsLocal -Value $publicPackageSource)
$ownerInputHashFieldsReady = -not (Test-IsPlaceholder -Value $managedNupkgSha256) -and
  -not (Test-IsPlaceholder -Value $runtimeNupkgSha256) -and
  -not (Test-IsPlaceholder -Value $smokeLogSha256) -and
  $managedNupkgSha256 -match "^[0-9a-fA-F]{64}$" -and
  $runtimeNupkgSha256 -match "^[0-9a-fA-F]{64}$" -and
  $smokeLogSha256 -match "^[0-9a-fA-F]{64}$"
$ownerInputSmokeLogReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogPath" -DefaultValue "")) -and
  $smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and
  $smokeCommand.Contains($RuntimePackageKey, [StringComparison]::OrdinalIgnoreCase) -and
  ([string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeStatus" -DefaultValue "")) -eq "passed"

$templateLike = (Test-IsPlaceholder -Value $cleanExternalConsumerRoot) -or
  (Test-IsPlaceholder -Value $consumerProjectPath) -or
  (Test-IsPlaceholder -Value $publicPackageSourceKind) -or
  (Test-IsPlaceholder -Value $publicPackageSource) -or
  (Test-IsPlaceholder -Value $publicPackageFeedUrl) -or
  (Test-IsPlaceholder -Value $managedPackageUrl) -or
  (Test-IsPlaceholder -Value $runtimePackageUrl) -or
  (Test-PublicPackageSourceIsLocal -Value $publicPackageSource)
$proofClassification = if ($templateLike) { "template-only" } else { "package-consumer-runtime" }
$cleanOwnerInputReady = -not $templateLike -and $ownerInputForbiddenSubstituteFree -and $ownerInputHashFieldsReady -and $ownerInputSmokeLogReady

$record = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-record"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  proofClassification = $proofClassification
  proofState = if ($templateLike) { "owner-input-incomplete" } else { "owner-input-candidate" }
  templateOnly = $templateLike
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputCanPromoteRuntimeProof = $false
  ownerInputBlockedReason = if ($cleanOwnerInputReady) { "none" } else { "owner-input-not-clean-or-smoke-log-not-ready" }
  isRuntimeExecutionEvidence = $false
  isDependencyProbeOnly = $true
  canPromoteRuntimeProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  ownerInputPath = $OwnerInputPath
  cleanExternalConsumerRoot = $cleanExternalConsumerRoot
  consumerProjectPath = $consumerProjectPath
  consumerProjectScan = $scan
  publicPackageSourceKind = $publicPackageSourceKind
  publicPackageSource = $publicPackageSource
  publicPackageFeedUrl = $publicPackageFeedUrl
  managedPackageUrl = $managedPackageUrl
  managedPackageId = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedPackageId" -DefaultValue "JYPPX.TensorRT.CSharp.API")
  managedPackageVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedPackageVersion" -DefaultValue "")
  managedNupkgPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "managedNupkgPath" -DefaultValue "")
  managedNupkgSha256 = $managedNupkgSha256
  runtimePackageUrl = $runtimePackageUrl
  runtimePackageId = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageId" -DefaultValue "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey")
  runtimePackageVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimePackageVersion" -DefaultValue "")
  runtimeNupkgPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "runtimeNupkgPath" -DefaultValue "")
  runtimeNupkgSha256 = $runtimeNupkgSha256
  host = [pscustomobject]@{
    ownerName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "ownerName" -DefaultValue "<owner-fill-owner-name>")
    machineName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "machineName" -DefaultValue "<owner-fill-machine-name>")
    osDescription = [string](Get-PropertyOrDefault -Object $ownerInput -Name "hostOs" -DefaultValue "")
    hostArchitecture = [string](Get-PropertyOrDefault -Object $ownerInput -Name "hostArchitecture" -DefaultValue "")
    gpuName = [string](Get-PropertyOrDefault -Object $ownerInput -Name "gpuName" -DefaultValue "<owner-fill-gpu-name>")
    driverVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudaDriverVersion" -DefaultValue "")
    cudaDriverSupportedRuntime = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudaDriverSupportedRuntime" -DefaultValue "<owner-fill-cuda-driver-supported-runtime>")
    cudaRuntimeVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudaRuntimeVersion" -DefaultValue "")
    tensorRtRuntimeVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "tensorRtVersion" -DefaultValue "")
    cudnnVersion = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cudnnVersion" -DefaultValue "<owner-fill-cudnn-version>")
    tensorRtLine = [string](Get-PropertyOrDefault -Object $ownerInput -Name "tensorRtLine" -DefaultValue "<owner-fill-tensorrt-line>")
  }
  command = [pscustomobject]@{
    restoreCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "restoreCommand" -DefaultValue "<owner-fill-restore-command>")
    buildCommand = [string](Get-PropertyOrDefault -Object $ownerInput -Name "buildCommand" -DefaultValue "<owner-fill-build-command>")
    smokeCommand = $smokeCommand
    exitCode = Get-PropertyOrDefault -Object $ownerInput -Name "exitCode" -DefaultValue $null
    startedAtUtc = [string](Get-PropertyOrDefault -Object $ownerInput -Name "startedAtUtc" -DefaultValue "<owner-fill-started-at-utc>")
    finishedAtUtc = [string](Get-PropertyOrDefault -Object $ownerInput -Name "finishedAtUtc" -DefaultValue "<owner-fill-finished-at-utc>")
    logPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeLogPath" -DefaultValue "")
    logSha256 = $smokeLogSha256
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "dependencyProbeStatus" -DefaultValue "pending")
    smokeStatus = [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeStatus" -DefaultValue "pending-compatible-host-execution")
    nativeAssetsCopied = Get-PropertyOrDefault -Object $ownerInput -Name "nativeAssetsCopied" -DefaultValue $null
    stdoutSummary = [string](Get-PropertyOrDefault -Object $ownerInput -Name "stdoutSummary" -DefaultValue "")
    stderrSummary = [string](Get-PropertyOrDefault -Object $ownerInput -Name "stderrSummary" -DefaultValue "")
    failureDiagnostic = [string](Get-PropertyOrDefault -Object $ownerInput -Name "failureDiagnostic" -DefaultValue "")
  }
  sourceRunner = [pscustomobject]@{
    queueStatus = $sourceRunnerQueueStatus
    infrastructureStatus = $sourceRunnerInfrastructureStatus
    ownerAction = $sourceRunnerOwnerAction
    canPromoteRuntimeProof = $false
  }
  externalRuntimeProofRecordPath = "artifacts/final-release/external-runtime-proof-record.json"
  strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath artifacts/final-release/package-consumer-runtime-proof-record.json -Strict"
  externalRuntimeProofValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordFromPackageConsumerProof.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof"
  safetyBoundary = "Owner input projection only. This record cannot promote runtime proof until strict validation passes with real package-consumer smoke evidence."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-record.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Package Consumer Runtime Proof Record

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($record.recordKind)`` |
| proofClassification | ``$($record.proofClassification)`` |
| proofState | ``$($record.proofState)`` |
| templateOnly | ``$($record.templateOnly)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Safety Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof record written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ProofClassification=$($record.proofClassification) CanPromoteRuntimeProof=$($record.canPromoteRuntimeProof)"
