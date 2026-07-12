[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$PackageConsumerSummaryPath = "artifacts\package-consumer\package-consumer-validation-summary.json",
  [string]$RuntimeReadinessSummaryPath = "artifacts\package-readiness\runtime-package-readiness-summary.json",
  [string]$SmokeLogPath,
  [string]$OutputPath = "artifacts\final-release\external-runtime-proof-record.draft.json",
  [string]$OutputRoot,
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

function Resolve-RepoPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolved = Resolve-RepoPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolved) -or -not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue = $null
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $DefaultValue
}

function Get-PackageConsumerRuntimeProofPreflightMatrix {
  $matrixPath = Join-Path $RepositoryRoot "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
  if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $matrixPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Find-PackageConsumerRuntimeProofPreflightEntry {
  param(
    [object]$Matrix,
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackageKey
  )

  if ($null -eq $Matrix -or $Matrix.PSObject.Properties.Name -notcontains "entries") {
    return $null
  }

  return @($Matrix.entries | Where-Object { $_.runtimePackageKey -eq $RuntimePackageKey } | Select-Object -First 1)[0]
}

function Get-Sha256OrEmpty {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return ""
  }

  return ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash).ToLowerInvariant()
}

function Test-ProjectReferenceFreeConsumer {
  param([string]$ConsumerRoot)

  if ([string]::IsNullOrWhiteSpace($ConsumerRoot) -or -not (Test-Path -LiteralPath $ConsumerRoot -PathType Container)) {
    return $null
  }

  $projectReferences = @(Get-ChildItem -LiteralPath $ConsumerRoot -Recurse -Filter *.csproj | Select-String -Pattern "<ProjectReference" -SimpleMatch)
  return $projectReferences.Count -eq 0
}

function Get-SmokeProjectPath {
  param([string]$SmokeCommand)

  if ([string]::IsNullOrWhiteSpace($SmokeCommand)) {
    return ""
  }

  $match = [System.Text.RegularExpressions.Regex]::Match($SmokeCommand, "--project\s+(?<path>.+?PackageConsumerSmoke\.csproj)", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if (-not $match.Success) {
    return ""
  }

  return $match.Groups["path"].Value.Trim('"')
}

function Test-ProjectReferenceFreeProject {
  param([string]$ProjectPath)

  if ([string]::IsNullOrWhiteSpace($ProjectPath) -or -not (Test-Path -LiteralPath $ProjectPath -PathType Leaf)) {
    return $null
  }

  $matches = @(Select-String -LiteralPath $ProjectPath -Pattern "<ProjectReference" -SimpleMatch)
  return $matches.Count -eq 0
}

function Get-ConsumerRootFromSmokeProject {
  param([string]$SmokeProjectPath)

  if ([string]::IsNullOrWhiteSpace($SmokeProjectPath)) {
    return ""
  }

  $projectRoot = Split-Path -Parent $SmokeProjectPath
  if ([string]::IsNullOrWhiteSpace($projectRoot) -or -not (Test-Path -LiteralPath $projectRoot -PathType Container)) {
    return ""
  }

  return $projectRoot
}

function Get-ConsumerRootFromOutput {
  param([string]$ConsumerOutput)

  if ([string]::IsNullOrWhiteSpace($ConsumerOutput)) {
    return ""
  }

  $fullPath = [IO.Path]::GetFullPath($ConsumerOutput)
  $marker = [IO.Path]::Combine("bin", "Release")
  $index = $fullPath.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase)
  if ($index -le 0) {
    return ""
  }

  $candidate = $fullPath.Substring(0, $index).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  if (Test-Path -LiteralPath (Join-Path $candidate "PackageConsumerSmoke.csproj") -PathType Leaf) {
    return $candidate
  }

  return ""
}

function ConvertTo-RelativeOrOriginal {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  $fullPath = [IO.Path]::GetFullPath($Path)
  $root = [IO.Path]::GetFullPath($RepositoryRoot)
  if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($root.Length).TrimStart([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  }

  return $Path
}

$packageConsumer = Read-JsonOrNull -Path $PackageConsumerSummaryPath
$runtimeReadiness = Read-JsonOrNull -Path $RuntimeReadinessSummaryPath
$runtimeProofPreflightMatrix = Get-PackageConsumerRuntimeProofPreflightMatrix
$runtimeProofPreflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $runtimeProofPreflightMatrix -RuntimePackageKey $RuntimePackageKey

$managedPackage = Get-PropertyOrDefault -Object $runtimeReadiness -Name "managedPackage"
$runtimePackagePath = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "RuntimePackagePath" -DefaultValue "")
$managedPackagePath = [string](Get-PropertyOrDefault -Object $managedPackage -Name "path" -DefaultValue "")
if ([string]::IsNullOrWhiteSpace($managedPackagePath)) {
  $managedPackagePath = Join-Path $RepositoryRoot "artifacts\managed\JYPPX.TensorRT.CSharp.API.4.0.0.nupkg"
}

$resolvedManagedPackagePath = Resolve-RepoPath -Path $managedPackagePath
$resolvedRuntimePackagePath = Resolve-RepoPath -Path $runtimePackagePath
$consumerOutput = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "ConsumerOutput" -DefaultValue "")
$noProjectReference = Test-ProjectReferenceFreeConsumer -ConsumerRoot $consumerOutput
$smokeCommandFromSummary = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeCommand" -DefaultValue "")
$smokeProjectPath = Get-SmokeProjectPath -SmokeCommand $smokeCommandFromSummary
$consumerProjectName = if ([string]::IsNullOrWhiteSpace($smokeProjectPath)) { "" } else { [System.IO.Path]::GetFileNameWithoutExtension($smokeProjectPath) }
$consumerProjectPath = if ([string]::IsNullOrWhiteSpace($smokeProjectPath)) { "" } else { ConvertTo-RelativeOrOriginal -Path $smokeProjectPath }
$noProjectReferenceSource = "consumer-output"
if ($null -eq $noProjectReference) {
  $noProjectReference = Test-ProjectReferenceFreeProject -ProjectPath $smokeProjectPath
  $noProjectReferenceSource = "smoke-project"
}

if ($null -eq $noProjectReference) {
  $consumerRootFromSmokeProject = Get-ConsumerRootFromSmokeProject -SmokeProjectPath $smokeProjectPath
  $noProjectReference = Test-ProjectReferenceFreeConsumer -ConsumerRoot $consumerRootFromSmokeProject
  $noProjectReferenceSource = "smoke-project-root"
}

if ($null -eq $noProjectReference) {
  $consumerRootFromOutput = Get-ConsumerRootFromOutput -ConsumerOutput $consumerOutput
  $noProjectReference = Test-ProjectReferenceFreeConsumer -ConsumerRoot $consumerRootFromOutput
  $noProjectReferenceSource = "consumer-output-derived-project-root"
}

if ($null -eq $noProjectReference) {
  $noProjectReferenceSource = "unknown-consumer-output-not-preserved"
}

if ([string]::IsNullOrWhiteSpace($SmokeLogPath)) {
  $SmokeLogPath = "artifacts\final-release\external-runtime-proof\$RuntimePackageKey\package-consumer-smoke.log"
}

$resolvedSmokeLogPath = Resolve-RepoPath -Path $SmokeLogPath
$smokeOutputLines = @(Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeOutputLines" -DefaultValue @())
if (-not (Test-Path -LiteralPath $resolvedSmokeLogPath -PathType Leaf) -and $smokeOutputLines.Count -gt 0) {
  $smokeDirectory = Split-Path -Parent $resolvedSmokeLogPath
  New-Item -ItemType Directory -Path $smokeDirectory -Force | Out-Null
  $smokeOutputLines | Set-Content -LiteralPath $resolvedSmokeLogPath -Encoding utf8
}

$smokeResult = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-package-consumer-summary")
$smokeExitCode = Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeExitCode" -DefaultValue $null
$smokePassed = [string]::Equals($smokeResult, "passed", [System.StringComparison]::OrdinalIgnoreCase)
$runtimePackageKeyMatches = [string]::Equals(
  [string](Get-PropertyOrDefault -Object $packageConsumer -Name "RuntimePackageKey" -DefaultValue ""),
  $RuntimePackageKey,
  [System.StringComparison]::OrdinalIgnoreCase)
$managedSha = Get-Sha256OrEmpty -Path $resolvedManagedPackagePath
$runtimeSha = Get-Sha256OrEmpty -Path $resolvedRuntimePackagePath
$logSha = Get-Sha256OrEmpty -Path $resolvedSmokeLogPath
$nativeAssetsExpected = [int](Get-PropertyOrDefault -Object $packageConsumer -Name "NativeAssetsExpected" -DefaultValue 0)
$nativeAssetsFound = [int](Get-PropertyOrDefault -Object $packageConsumer -Name "NativeAssetsFound" -DefaultValue 0)
$preflightRuntimePackageId = if ($null -ne $runtimeProofPreflightEntry) { [string]$runtimeProofPreflightEntry.runtimePackageId } else { "" }
$preflightRestoreSourceMode = if ($null -ne $runtimeProofPreflightEntry) { [string]$runtimeProofPreflightEntry.restoreSourceMode } else { "" }
$preflightNativeAssetsExpected = if ($null -ne $runtimeProofPreflightEntry) { [int]$runtimeProofPreflightEntry.nativeAssetCopyExpected } else { 0 }
$preflightEntryFound = $null -ne $runtimeProofPreflightEntry
$preflightNativeAssetsExpectedMatches = $preflightEntryFound -and $nativeAssetsExpected -eq $preflightNativeAssetsExpected
$preflightNativeAssetsFoundMatches = $preflightEntryFound -and $nativeAssetsFound -ge $preflightNativeAssetsExpected
$nativeAssetsCopied = $nativeAssetsExpected -gt 0 -and $nativeAssetsFound -eq $nativeAssetsExpected
$dependencyProbeStatus = if ($nativeAssetsCopied) { "native-assets-copied" } else { "pending" }
$isDependencyProbeOnly = -not $smokePassed
$canPromoteRuntimeProof = $smokePassed -and
  $runtimePackageKeyMatches -and
  -not [string]::IsNullOrWhiteSpace($managedSha) -and
  -not [string]::IsNullOrWhiteSpace($runtimeSha) -and
  -not [string]::IsNullOrWhiteSpace($logSha) -and
  $nativeAssetsCopied -and
  ($noProjectReference -eq $true)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-record-draft"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = -not $canPromoteRuntimeProof
  exampleOnly = $false
  proofState = if ($canPromoteRuntimeProof) { "draft-promotable-after-owner-review" } elseif ($smokeResult -eq "blocked-by-cuda-driver") { "draft-blocked-by-cuda-driver" } else { "draft-incomplete" }
  proofClassification = if ($canPromoteRuntimeProof) { "package-consumer-runtime" } elseif ($smokeResult -eq "blocked-by-cuda-driver") { "dependency-probe-only" } else { "template-only" }
  evidenceClassifications = @(
    "build-only",
    "dependency-probe-only",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
  )
  isRuntimeExecutionEvidence = $canPromoteRuntimeProof
  isDependencyProbeOnly = $isDependencyProbeOnly
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  currentRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "runtimeProofStatus" -DefaultValue $smokeResult)
  host = [pscustomobject]@{
    ownerName = [Environment]::UserName
    machineName = [Environment]::MachineName
    osDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    gpuName = ""
    driverVersion = ""
    cudaDriverSupportedRuntime = ""
    cudaRuntimeVersion = ""
    tensorRtRuntimeVersion = ""
    cudnnVersion = ""
    tensorRtLine = "11"
  }
  packageSource = [pscustomobject]@{
    managedPackageSource = ConvertTo-RelativeOrOriginal -Path $resolvedManagedPackagePath
    runtimePackageSource = ConvertTo-RelativeOrOriginal -Path $resolvedRuntimePackagePath
    runtimePackageKey = $RuntimePackageKey
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    consumerProjectName = $consumerProjectName
    consumerProjectPath = $consumerProjectPath
    managedNupkgSha256 = $managedSha
    runtimeNupkgSha256 = $runtimeSha
    noProjectReference = $noProjectReference
    noProjectReferenceSource = $noProjectReferenceSource
    noProjectReferenceDiagnostic = if ($null -eq $noProjectReference) { "Consumer output or smoke project was not preserved; rerun Test-PackageConsumer.ps1 with -KeepConsumerOutput to verify no ProjectReference." } else { "No ProjectReference check evaluated from $noProjectReferenceSource." }
  }
  command = [pscustomobject]@{
    restoreCommand = "dotnet restore PackageConsumerSmoke.csproj --configfile NuGet.config"
    buildCommand = "dotnet build PackageConsumerSmoke.csproj -c Release --no-restore"
    smokeCommand = $smokeCommandFromSummary
    exitCode = $smokeExitCode
    startedAtUtc = $null
    finishedAtUtc = $null
    logPath = ConvertTo-RelativeOrOriginal -Path $resolvedSmokeLogPath
    logSha256 = $logSha
  }
  modelEvidence = [pscustomobject]@{
    modelName = ""
    modelSha256 = ""
    modelLicense = ""
    inputAssetName = ""
    inputAssetSha256 = ""
  }
  results = [pscustomobject]@{
    dependencyProbeStatus = $dependencyProbeStatus
    smokeStatus = $smokeResult
    nativeAssetsCopied = $nativeAssetsCopied
    nativeAssetsExpected = $nativeAssetsExpected
    nativeAssetsFound = $nativeAssetsFound
    stdoutSummary = if ($smokeOutputLines.Count -gt 0) { ($smokeOutputLines | Select-Object -First 12) -join [Environment]::NewLine } else { "" }
    stderrSummary = if ($smokePassed) { "no-stderr-emitted needs owner review against the preserved smoke log before promotion." } else { "" }
    failureDiagnostic = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeDiagnostic" -DefaultValue "")
  }
  runtimeProofPreflight = [pscustomobject]@{
    matrixFound = $null -ne $runtimeProofPreflightMatrix
    entryFound = $preflightEntryFound
    runtimePackageId = $preflightRuntimePackageId
    restoreSourceMode = $preflightRestoreSourceMode
    nativeAssetCopyExpected = $preflightNativeAssetsExpected
    nativeAssetsExpectedMatches = $preflightNativeAssetsExpectedMatches
    nativeAssetsFoundMatches = $preflightNativeAssetsFoundMatches
    canPromotePackageConsumerRuntimeProof = if ($preflightEntryFound) { [bool]$runtimeProofPreflightEntry.canPromotePackageConsumerRuntimeProof } else { $false }
    ownerActionRequired = if ($preflightEntryFound) { [bool]$runtimeProofPreflightEntry.ownerActionRequired } else { $true }
  }
  draftDiagnostics = @(
    "This draft is generated from local package-consumer and runtime-readiness artifacts.",
    "It is not a promotable proof until copied/reviewed as external-runtime-proof-record.json with recordKind=external-runtime-proof-record and templateOnly=false.",
    "packageSource.runtimePackageId, packageSource.restoreSourceMode, results.nativeAssetsExpected, and results.nativeAssetsFound are copied for RuntimeProofPreflight alignment.",
    "blocked-by-cuda-driver remains non-promotable and must stay visible.",
    "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof remains the promotion gate."
  )
  ownerActionSummary = @(
    "This draft is a prefilled starting point, not runtime proof.",
    "Owner must replace draft state with a real compatible-host package consumer run.",
    "Owner must fill real host metadata, package hashes, smoke log path/hash, stdoutSummary, and stderrSummary.",
    "Owner must run Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof.",
    "If the current state remains blocked-by-cuda-driver or dependency-probe-only, keep release close blocked."
  )
}

$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$resolvedOutputRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  Split-Path -Parent $resolvedOutputPath
}
else {
  Resolve-RepoPath -Path $OutputRoot
}

New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$markdownPath = [IO.Path]::ChangeExtension($resolvedOutputPath, ".md")
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Record Draft")
$lines.Add("")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- proof state: ``$($record.proofState)``")
$lines.Add("- proof classification: ``$($record.proofClassification)``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- smoke status: ``$smokeResult``")
$lines.Add("- native assets copied: ``$nativeAssetsCopied``")
$lines.Add("- native assets expected: ``$nativeAssetsExpected``")
$lines.Add("- native assets found: ``$nativeAssetsFound``")
$lines.Add("- preflight entry found: ``$preflightEntryFound``")
$lines.Add("- preflight native assets expected: ``$preflightNativeAssetsExpected``")
$lines.Add("- preflight native assets expected matches: ``$preflightNativeAssetsExpectedMatches``")
$lines.Add("- preflight native assets found matches: ``$preflightNativeAssetsFoundMatches``")
$lines.Add("- no ProjectReference: ``$noProjectReference``")
$lines.Add("- managed nupkg SHA256 ready: ``$(-not [string]::IsNullOrWhiteSpace($managedSha))``")
$lines.Add("- runtime nupkg SHA256 ready: ``$(-not [string]::IsNullOrWhiteSpace($runtimeSha))``")
$lines.Add("- smoke log SHA256 ready: ``$(-not [string]::IsNullOrWhiteSpace($logSha))``")
$lines.Add("- stdout summary ready: ``$($smokeOutputLines.Count -gt 0)``")
$lines.Add("- stderr summary ready: ``$($smokePassed)``")
$lines.Add("")
$lines.Add("This draft is an automation aid. It is not release proof while ``recordKind`` is ``external-runtime-proof-record-draft`` or ``templateOnly`` is ``true``.")
$lines.Add("Real proof still requires owner-reviewed stdoutSummary and stderrSummary; when stderr is empty, stderrSummary must say so explicitly.")
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $record.ownerActionSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Promotion Gate")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add("Copy-Item -LiteralPath $OutputPath -Destination artifacts\final-release\external-runtime-proof-record.json")
$lines.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\final-release\external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")
$lines.Add('```')
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof draft written to $resolvedOutputPath"
Write-Host "External runtime proof draft written to $markdownPath"
Write-Host "DraftState=$($record.proofState) SmokeStatus=$smokeResult CanPromoteRuntimeProof=$canPromoteRuntimeProof"
