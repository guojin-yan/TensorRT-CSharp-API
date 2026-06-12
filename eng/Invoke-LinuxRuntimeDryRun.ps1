[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [Parameter(Mandatory = $true)]
  [string]$TensorRtRoot,
  [Parameter(Mandatory = $true)]
  [string]$CudaRoot,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$expectedBuildArtifacts = @(
  "build-out/$($package.buildPreset)/bin/Release/$($package.bridgeFile)",
  "build-out/$($package.buildPreset)/lib/Release/"
)

$expectedPackArtifacts = @(
  "artifacts/runtime/$RuntimePackageKey/runtimes/$($package.rid)/native/",
  "artifacts/runtime/$RuntimePackageKey/artifact-manifest.json",
  "artifacts/runtime-nupkg/$($package.packageId).*nupkg"
)

$preflightCommands = @(
  "pwsh -File ./eng/Validate-RuntimeManifest.ps1",
  "pwsh -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot",
  "pwsh -File ./eng/Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot"
)

$executionCommands = @(
  "cmake --preset $($package.buildPreset)",
  "cmake --build --preset $($package.buildPreset)",
  "pwsh -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot",
  "dotnet pack ./pack/runtime/$RuntimePackageKey/$($package.packageId).csproj -c Release -o ./artifacts/runtime-nupkg"
)

$failureScenarios = @(
  "TensorRT root does not match the requested runtime key line or package family.",
  "CUDA root does not match the requested runtime key line or does not expose libcudart under an expected path.",
  "Expected .so wildcard patterns resolve to zero files.",
  "self-hosted Linux runner is missing pwsh, dotnet, or cmake.",
  "Native bridge build succeeds, but Collect-RuntimeAssets still fails because roots point at the wrong unpacked layout.",
  "Artifact upload succeeds while native assets are incomplete because the runtime key and roots were mixed across package lines."
)

$summary = [ordered]@{
  runtimeKey = $package.key
  packageId = $package.packageId
  rid = $package.rid
  buildPreset = $package.buildPreset
  distributionTier = $package.distributionTier
  validationState = $package.validationState
  distributionNotes = $package.distributionNotes
  tensorRtRoot = $TensorRtRoot
  cudaRoot = $CudaRoot
  requiredRunnerLabels = @("self-hosted", "linux", "x64")
  requiredTools = @(".NET 10 SDK", "pwsh", "cmake", "matching TensorRT root", "matching CUDA root")
  expectedTensorRtPatterns = $package.tensorRtFiles
  expectedCudaPatterns = $package.cudaFiles
  preflightCommands = $preflightCommands
  executionCommands = $executionCommands
  expectedBuildArtifacts = $expectedBuildArtifacts
  expectedPackArtifacts = $expectedPackArtifacts
  promotionCriteria = @(
    "Validate-LinuxRuntimeInputs passes on the target Linux runner.",
    "CMake configure and build complete for the manifest build preset.",
    "The native bridge artifact exists at build-out/$($package.buildPreset)/bin/Release/$($package.bridgeFile).",
    "Collect-RuntimeAssets resolves every TensorRT and CUDA asset pattern to at least one file.",
    "The runtime package nupkg is produced under artifacts/runtime-nupkg.",
    "The runtime package is inspected and contains the bridge plus all expected TensorRT/CUDA assets.",
    "A Linux package consumer project can restore the managed package plus this runtime package and build successfully.",
    "If a GPU is available, a Linux smoke runner can load the bridge and query TensorRT/CUDA runtime information."
  )
  failureScenarios = $failureScenarios
}

$summaryPath = Join-Path $outputRoot "linux-runtime-dry-run.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8

$tensorRtPatternLines = ($package.tensorRtFiles | ForEach-Object { "- " + $_ }) -join [Environment]::NewLine
$cudaPatternLines = ($package.cudaFiles | ForEach-Object { "- " + $_ }) -join [Environment]::NewLine
$preflightLines = ($preflightCommands | ForEach-Object { "1. " + $_ })
$executionLines = ($executionCommands | ForEach-Object { "1. " + $_ })
$buildArtifactLines = ($expectedBuildArtifacts | ForEach-Object { "- " + $_ })
$packArtifactLines = ($expectedPackArtifacts | ForEach-Object { "- " + $_ })
$promotionLines = ($summary.promotionCriteria | ForEach-Object { "- " + $_ })
$failureLines = ($failureScenarios | ForEach-Object { "- $_" })
$labelLines = (@("self-hosted", "linux", "x64") | ForEach-Object { "- " + $_ })
$toolLines = (@(".NET 10 SDK", "pwsh", "cmake", "matching TensorRT root", "matching CUDA root") | ForEach-Object { "- $_" })

$readmeLines = @(
  "# Linux Runtime Dry Run",
  "",
  "Runtime package key: $($package.key)",
  "",
  "Package ID: $($package.packageId)",
  "",
  "RID: $($package.rid)",
  "",
  "Build preset: $($package.buildPreset)",
  "",
  "Distribution tier: $($package.distributionTier)",
  "",
  "Validation state: $($package.validationState)",
  "",
  "TensorRT root: $TensorRtRoot",
  "",
  "CUDA root: $CudaRoot",
  "",
  "## Required runner labels",
  "",
  $labelLines,
  "",
  "## Required tools",
  "",
  $toolLines,
  "",
  "## Expected TensorRT patterns",
  "",
  $tensorRtPatternLines,
  "",
  "## Expected CUDA patterns",
  "",
  $cudaPatternLines,
  "",
  "## Preflight commands",
  "",
  $preflightLines,
  "",
  "## Execution commands",
  "",
  $executionLines,
  "",
  "## Expected build artifacts",
  "",
  $buildArtifactLines,
  "",
  "## Expected pack artifacts",
  "",
  $packArtifactLines,
  "",
  "## Promotion criteria",
  "",
  $promotionLines,
  "",
  "## Common failure scenarios",
  "",
  $failureLines
)

$readmePath = Join-Path $outputRoot "README.md"
$readmeLines -join [Environment]::NewLine | Set-Content -LiteralPath $readmePath -Encoding utf8

$checklistLines = @(
  "# Linux Runner Checklist",
  "",
  "Use this checklist before running the real Linux build or runtime pack workflow for $RuntimePackageKey.",
  "",
  "## Preflight",
  "",
  "- [ ] Confirm the runner has labels self-hosted, linux, and x64.",
  "- [ ] Confirm pwsh, dotnet, and cmake are available on PATH.",
  "- [ ] Confirm TensorRT root exists: $TensorRtRoot.",
  "- [ ] Confirm CUDA root exists: $CudaRoot.",
  "- [ ] Run pwsh -File ./eng/Validate-RuntimeManifest.ps1.",
  "- [ ] Run pwsh -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot.",
  "- [ ] Run pwsh -File ./eng/Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot.",
  "- [ ] Review linux-runtime-dry-run.json and confirm the package line, roots, and patterns are correct.",
  "",
  "## Build",
  "",
  "- [ ] Run cmake --preset $($package.buildPreset).",
  "- [ ] Run cmake --build --preset $($package.buildPreset).",
  "- [ ] Confirm build artifact exists: build-out/$($package.buildPreset)/bin/Release/$($package.bridgeFile).",
  "",
  "## Pack",
  "",
  "- [ ] Run pwsh -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey $RuntimePackageKey -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot.",
  "- [ ] Confirm runtime asset root exists: artifacts/runtime/$RuntimePackageKey/runtimes/$($package.rid)/native/.",
  "- [ ] Confirm artifact manifest exists: artifacts/runtime/$RuntimePackageKey/artifact-manifest.json.",
  "- [ ] Run dotnet pack ./pack/runtime/$RuntimePackageKey/$($package.packageId).csproj -c Release -o ./artifacts/runtime-nupkg.",
  "- [ ] Confirm output nupkg exists under artifacts/runtime-nupkg/.",
  "",
  "## Promotion from dry-run-only to local-validated",
  "",
  "- [ ] Validate-LinuxRuntimeInputs passes on the target Linux runner.",
  "- [ ] CMake configure and build complete for the manifest build preset.",
  "- [ ] Native bridge artifact exists under build-out/$($package.buildPreset)/bin/Release/.",
  "- [ ] Collect-RuntimeAssets resolves every TensorRT and CUDA asset pattern to at least one file.",
  "- [ ] Runtime package nupkg is produced and inspected.",
  "- [ ] Linux package consumer project restores and builds with the managed package plus runtime package.",
  "- [ ] If a GPU is available, Linux smoke runner loads the bridge and queries TensorRT/CUDA runtime information.",
  "- [ ] Manifest validationState can be changed from dry-run-only to local-validated only after the evidence above is recorded.",
  "",
  "## First failure points to inspect",
  ""
)

foreach ($failure in $failureScenarios) {
  $checklistLines += "- [ ] $failure"
}

$checklistPath = Join-Path $outputRoot "linux-runner-checklist.md"
$checklistLines -join [Environment]::NewLine | Set-Content -LiteralPath $checklistPath -Encoding utf8

Write-Host "Linux dry run summary written to $outputRoot"
