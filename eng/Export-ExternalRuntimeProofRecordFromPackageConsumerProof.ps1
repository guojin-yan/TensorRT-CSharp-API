[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-record.json",
  [string]$OutputPath = "artifacts\final-release\external-runtime-proof-record.json",
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

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Package consumer runtime proof record not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$hostInfo = Get-PropertyOrDefault -Object $record -Name "host" -DefaultValue $null
$command = Get-PropertyOrDefault -Object $record -Name "command" -DefaultValue $null
$results = Get-PropertyOrDefault -Object $record -Name "results" -DefaultValue $null
$scan = Get-PropertyOrDefault -Object $record -Name "consumerProjectScan" -DefaultValue $null
$RuntimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue $RuntimePackageKey)

$external = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-record"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = [bool](Get-PropertyOrDefault -Object $record -Name "templateOnly" -DefaultValue $true)
  proofState = [string](Get-PropertyOrDefault -Object $record -Name "proofState" -DefaultValue "owner-input-candidate")
  proofClassification = [string](Get-PropertyOrDefault -Object $record -Name "proofClassification" -DefaultValue "template-only")
  isRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
  isDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $record -Name "isDependencyProbeOnly" -DefaultValue $true)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  host = [pscustomobject]@{
    ownerName = [string](Get-PropertyOrDefault -Object $hostInfo -Name "ownerName" -DefaultValue "")
    machineName = [string](Get-PropertyOrDefault -Object $hostInfo -Name "machineName" -DefaultValue "")
    osDescription = [string](Get-PropertyOrDefault -Object $hostInfo -Name "osDescription" -DefaultValue "")
    gpuName = [string](Get-PropertyOrDefault -Object $hostInfo -Name "gpuName" -DefaultValue "")
    driverVersion = [string](Get-PropertyOrDefault -Object $hostInfo -Name "driverVersion" -DefaultValue "")
    cudaDriverSupportedRuntime = [string](Get-PropertyOrDefault -Object $hostInfo -Name "cudaDriverSupportedRuntime" -DefaultValue "")
    cudaRuntimeVersion = [string](Get-PropertyOrDefault -Object $hostInfo -Name "cudaRuntimeVersion" -DefaultValue "")
    tensorRtRuntimeVersion = [string](Get-PropertyOrDefault -Object $hostInfo -Name "tensorRtRuntimeVersion" -DefaultValue "")
    cudnnVersion = [string](Get-PropertyOrDefault -Object $hostInfo -Name "cudnnVersion" -DefaultValue "")
    tensorRtLine = [string](Get-PropertyOrDefault -Object $hostInfo -Name "tensorRtLine" -DefaultValue "")
  }
  packageSource = [pscustomobject]@{
    managedPackageSource = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSource" -DefaultValue "")
    runtimePackageSource = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSource" -DefaultValue "")
    runtimePackageKey = $RuntimePackageKey
    consumerProjectName = [IO.Path]::GetFileNameWithoutExtension([string](Get-PropertyOrDefault -Object $record -Name "consumerProjectPath" -DefaultValue ""))
    consumerProjectPath = [string](Get-PropertyOrDefault -Object $record -Name "consumerProjectPath" -DefaultValue "")
    managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")
    runtimeNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "runtimeNupkgSha256" -DefaultValue "")
    noProjectReference = [bool](Get-PropertyOrDefault -Object $scan -Name "noProjectReference" -DefaultValue $false)
  }
  command = [pscustomobject]@{
    restoreCommand = [string](Get-PropertyOrDefault -Object $command -Name "restoreCommand" -DefaultValue "")
    buildCommand = [string](Get-PropertyOrDefault -Object $command -Name "buildCommand" -DefaultValue "")
    smokeCommand = [string](Get-PropertyOrDefault -Object $command -Name "smokeCommand" -DefaultValue "")
    exitCode = Get-PropertyOrDefault -Object $command -Name "exitCode" -DefaultValue $null
    startedAtUtc = Get-PropertyOrDefault -Object $command -Name "startedAtUtc" -DefaultValue $null
    finishedAtUtc = Get-PropertyOrDefault -Object $command -Name "finishedAtUtc" -DefaultValue $null
    logPath = [string](Get-PropertyOrDefault -Object $command -Name "logPath" -DefaultValue "")
    logSha256 = [string](Get-PropertyOrDefault -Object $command -Name "logSha256" -DefaultValue "")
  }
  modelEvidence = [pscustomobject]@{
    modelName = ""
    modelSha256 = ""
    modelLicense = ""
    inputAssetName = ""
    inputAssetSha256 = ""
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = [string](Get-PropertyOrDefault -Object $results -Name "dependencyProbeStatus" -DefaultValue "")
    smokeStatus = [string](Get-PropertyOrDefault -Object $results -Name "smokeStatus" -DefaultValue "")
    nativeAssetsCopied = Get-PropertyOrDefault -Object $results -Name "nativeAssetsCopied" -DefaultValue $null
    stdoutSummary = [string](Get-PropertyOrDefault -Object $results -Name "stdoutSummary" -DefaultValue "")
    stderrSummary = [string](Get-PropertyOrDefault -Object $results -Name "stderrSummary" -DefaultValue "")
    failureDiagnostic = [string](Get-PropertyOrDefault -Object $results -Name "failureDiagnostic" -DefaultValue "")
  }
  sourceArtifacts = @(
    "artifacts/final-release/package-consumer-runtime-proof-record.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  )
  safetyBoundary = "Generated bridge record. It cannot promote runtime proof unless Test-ExternalRuntimeProofRecord.ps1 validates it as real-runtime-proof."
}

$resolvedOutputPath = Resolve-InputPath -Path $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$external | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

Write-Host "External runtime proof record bridge written to $resolvedOutputPath"
Write-Host "ProofClassification=$($external.proofClassification) CanPromoteRuntimeProof=$($external.canPromoteRuntimeProof)"
