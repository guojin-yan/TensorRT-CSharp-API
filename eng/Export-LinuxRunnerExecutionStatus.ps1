[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
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
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$isLinux = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)
$isX64 = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [System.Runtime.InteropServices.Architecture]::X64
$isGitHubActions = [string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase)
$runnerOs = $env:RUNNER_OS
$runnerName = $env:RUNNER_NAME
$runnerArch = $env:RUNNER_ARCH

$blockers = New-Object System.Collections.Generic.List[string]
if (-not $isLinux) {
  $blockers.Add("Current host is not Linux. Real validation requires a Linux x64 self-hosted runner.")
}

if (-not $isX64) {
  $blockers.Add("Current host architecture is not x64.")
}

if ($isGitHubActions -and -not [string]::Equals($runnerOs, "Linux", [System.StringComparison]::OrdinalIgnoreCase)) {
  $blockers.Add("GitHub Actions runner is not a Linux runner.")
}

if (-not $isGitHubActions) {
  $blockers.Add("Current session is not a GitHub Actions self-hosted runner session.")
}

$dryRunRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $dryRunRoot -Force | Out-Null

$status = [ordered]@{
  runtimeKey = $RuntimePackageKey
  packageId = $package.packageId
  validationState = $package.validationState
  osDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
  osArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
  processArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
  isLinux = $isLinux
  isX64 = $isX64
  githubActions = $isGitHubActions
  runnerOs = $runnerOs
  runnerName = $runnerName
  runnerArch = $runnerArch
  status = if ($blockers.Count -eq 0) { "linux-runner-ready-for-real-validation" } else { "blocked" }
  blockers = @($blockers)
  requiredNextEvidence = @(
    "Validate-LinuxRuntimeInputs passes on the Linux runner.",
    "CMake configure/build succeeds for the selected preset.",
    "Collect-RuntimeAssets resolves all TensorRT/CUDA .so patterns.",
    "Linux runtime nupkg is produced.",
    "Test-PackageConsumer.ps1 passes without smoke on the Linux runner.",
    "Optional smoke passes when GPU access is available."
  )
}

$jsonPath = Join-Path $dryRunRoot "linux-runner-execution-status.json"
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Runner Execution Status")
$lines.Add("")
$lines.Add("Runtime key: $RuntimePackageKey")
$lines.Add("")
$lines.Add("Status: $($status.status)")
$lines.Add("")
$lines.Add("## Current host")
$lines.Add("")
$lines.Add("- OS: $($status.osDescription)")
$lines.Add("- OS architecture: $($status.osArchitecture)")
$lines.Add("- Process architecture: $($status.processArchitecture)")
$lines.Add("- GitHub Actions: $($status.githubActions)")
$lines.Add("- RUNNER_OS: $($status.runnerOs)")
$lines.Add("- RUNNER_NAME: $($status.runnerName)")
$lines.Add("- RUNNER_ARCH: $($status.runnerArch)")
$lines.Add("")
$lines.Add("## Blockers")
$lines.Add("")
if ($blockers.Count -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($blocker in $blockers) {
    $lines.Add("- $blocker")
  }
}
$lines.Add("")
$lines.Add("## Required next evidence")
$lines.Add("")
foreach ($item in $status.requiredNextEvidence) {
  $lines.Add("- $item")
}

$markdownPath = Join-Path $dryRunRoot "linux-runner-execution-status.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux runner execution status written to $jsonPath"
Write-Host "Linux runner execution status written to $markdownPath"
