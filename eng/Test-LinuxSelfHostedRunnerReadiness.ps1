[CmdletBinding()]
param(
  [string[]]$RuntimeKey = @(),
  [string]$RuntimeKeySet = "self-hosted-ubuntu20",
  [string]$ExpectedDistro = "ubuntu",
  [string]$ExpectedDistroVersion = "20.04",
  [string]$RequiredLabelSet = "self-hosted,linux,x64,ubuntu-20.04",
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$RunnerRoot,
  [switch]$CheckGitHubRunner,
  [switch]$RequireRegisteredRunner,
  [switch]$WarnOnly,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Test-IsLinuxHost {
  return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)
}

function Add-ReadinessCheck {
  param(
    [Parameter(Mandatory = $true)]
    [AllowEmptyCollection()]
    [System.Collections.Generic.List[object]]$Checks,
    [Parameter(Mandatory = $true)]
    [string]$Category,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [bool]$Passed,
    [string]$Detail,
    [ValidateSet("error", "warning")]
    [string]$Severity = "error"
  )

  $Checks.Add([pscustomobject]@{
      category = $Category
      name = $Name
      status = if ($Passed) { "passed" } else { "failed" }
      severity = $Severity
      detail = $Detail
    }) | Out-Null
}

function Get-CommandVersionText {
  param(
    [Parameter(Mandatory = $true)]
    [string]$CommandName,
    [string[]]$Arguments = @("--version")
  )

  $command = Get-Command $CommandName -ErrorAction SilentlyContinue
  if (-not $command) {
    return $null
  }

  try {
    return (& $CommandName @Arguments 2>$null | Select-Object -First 1 | Out-String).Trim()
  }
  catch {
    return "installed"
  }
}

function Get-LinuxOsRelease {
  $osReleasePath = "/etc/os-release"
  if (-not (Test-Path -LiteralPath $osReleasePath -PathType Leaf)) {
    return $null
  }

  $values = @{}
  foreach ($line in Get-Content -LiteralPath $osReleasePath -Encoding utf8) {
    if ($line -notmatch '^([^=]+)=(.*)$') {
      continue
    }

    $values[$Matches[1]] = $Matches[2].Trim('"')
  }

  [pscustomobject]@{
    id = [string]$values["ID"]
    versionId = [string]$values["VERSION_ID"]
    prettyName = [string]$values["PRETTY_NAME"]
  }
}

function Resolve-RuntimeKeyList {
  $arguments = @(
    "-NoProfile",
    "-File",
    (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1"),
    "-Platform",
    "linux",
    "-RuntimeKeySet",
    $RuntimeKeySet,
    "-RunnerMode",
    "self-hosted",
    "-OutputFormat",
    "json",
    "-RepositoryRoot",
    $RepositoryRoot
  )

  foreach ($key in @($RuntimeKey)) {
    if (-not [string]::IsNullOrWhiteSpace($key)) {
      $arguments += @("-RuntimeKey", $key)
    }
  }

  $json = (& pwsh @arguments | Out-String).Trim()
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) {
    throw "Unable to resolve Linux self-hosted runtime keys."
  }

  @($json | ConvertFrom-Json)
}

function Invoke-ValidationIfRootsExist {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackageKey,
    [Parameter(Mandatory = $true)]
    [object]$Roots
  )

  if ([string]::IsNullOrWhiteSpace([string]$Roots.tensorRtRoot) -or
    [string]::IsNullOrWhiteSpace([string]$Roots.cudaRoot) -or
    -not (Test-Path -LiteralPath ([string]$Roots.tensorRtRoot) -PathType Container) -or
    -not (Test-Path -LiteralPath ([string]$Roots.cudaRoot) -PathType Container) -or
    (-not [string]::IsNullOrWhiteSpace([string]$Roots.cudnnRoot) -and -not (Test-Path -LiteralPath ([string]$Roots.cudnnRoot) -PathType Container))) {
    return [pscustomobject]@{
      attempted = $false
      passed = $false
      detail = "Skipped because one or more NVIDIA roots are missing."
    }
  }

  $arguments = @(
    "-NoProfile",
    "-File",
    (Join-Path $RepositoryRoot "eng\Validate-LinuxRuntimeInputs.ps1"),
    "-RuntimePackageKey",
    $RuntimePackageKey,
    "-TensorRtRoot",
    [string]$Roots.tensorRtRoot,
    "-CudaRoot",
    [string]$Roots.cudaRoot,
    "-RepositoryRoot",
    $RepositoryRoot
  )

  if (-not [string]::IsNullOrWhiteSpace([string]$Roots.cudnnRoot)) {
    $arguments += @("-CudnnRoot", [string]$Roots.cudnnRoot)
  }

  $output = (& pwsh @arguments 2>&1 | Out-String).Trim()
  $passed = $LASTEXITCODE -eq 0
  [pscustomobject]@{
    attempted = $true
    passed = $passed
    detail = if ([string]::IsNullOrWhiteSpace($output)) { "Validate-LinuxRuntimeInputs.ps1 exited with code $LASTEXITCODE." } else { $output }
  }
}

$checks = [System.Collections.Generic.List[object]]::new()
$runtimeReports = [System.Collections.Generic.List[object]]::new()

$linuxHost = Test-IsLinuxHost
Add-ReadinessCheck -Checks $checks -Category "host" -Name "Linux operating system" -Passed $linuxHost -Detail $(if ($linuxHost) { "PowerShell reports this host is Linux." } else { "This script is not running on Linux." })

$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
Add-ReadinessCheck -Checks $checks -Category "host" -Name "x64 architecture" -Passed ($architecture -eq "X64") -Detail "Detected architecture: $architecture"

$osRelease = if ($linuxHost) { Get-LinuxOsRelease } else { $null }
$distroMatches = $false
$distroDetail = "Unable to read /etc/os-release."
if ($osRelease) {
  $distroMatches = $osRelease.id -eq $ExpectedDistro -and $osRelease.versionId -eq $ExpectedDistroVersion
  $distroDetail = "Detected: $($osRelease.id) $($osRelease.versionId) ($($osRelease.prettyName)); expected: $ExpectedDistro $ExpectedDistroVersion."
}
Add-ReadinessCheck -Checks $checks -Category "host" -Name "Target Linux distribution" -Passed $distroMatches -Detail $distroDetail

foreach ($tool in @("pwsh", "dotnet", "cmake", "ninja", "git", "gh")) {
  $versionText = Get-CommandVersionText -CommandName $tool
  Add-ReadinessCheck -Checks $checks -Category "tools" -Name $tool -Passed (-not [string]::IsNullOrWhiteSpace($versionText)) -Detail $(if ($versionText) { $versionText } else { "Command was not found on PATH." })
}

$dotnetSdks = $null
if (Get-Command dotnet -ErrorAction SilentlyContinue) {
  $dotnetSdks = @(& dotnet --list-sdks 2>$null)
}
$sdkVersions = @(
  foreach ($line in @($dotnetSdks)) {
    if ($line -match '^([0-9]+\.[0-9]+\.[0-9]+)') {
      [System.Version]$Matches[1]
    }
  }
)
$hasDotNet10 = @($sdkVersions | Where-Object { $_ -ge [System.Version]::new(10, 0, 300) }).Count -gt 0
Add-ReadinessCheck -Checks $checks -Category "tools" -Name ".NET SDK 10.0.300+" -Passed $hasDotNet10 -Detail $(if ($sdkVersions.Count -gt 0) { ($sdkVersions | Sort-Object -Descending | ForEach-Object { $_.ToString() }) -join ", " } else { "No SDK versions were reported." })

if ([string]::IsNullOrWhiteSpace($RunnerRoot)) {
  if ($linuxHost) {
    $homeDirectory = if (-not [string]::IsNullOrWhiteSpace($env:HOME)) { $env:HOME } else { (Get-Location).Path }
    $RunnerRoot = "$homeDirectory/actions-runner-tensorrt-csharp"
  }
  else {
    $RunnerRoot = "/opt/actions-runner-tensorrt-csharp"
  }
}

$runnerConfigured = Test-Path -LiteralPath (Join-Path $RunnerRoot ".runner") -PathType Leaf
Add-ReadinessCheck -Checks $checks -Category "runner" -Name "Local runner configuration" -Passed ($runnerConfigured -or -not $RequireRegisteredRunner.IsPresent) -Severity $(if ($RequireRegisteredRunner.IsPresent) { "error" } else { "warning" }) -Detail $(if ($runnerConfigured) { "Found .runner under $RunnerRoot." } else { "No .runner file found under $RunnerRoot. Run Install-GitHubSelfHostedRunner.ps1 before packaging." })
Add-ReadinessCheck -Checks $checks -Category "runner" -Name "Required release labels" -Passed $true -Detail $RequiredLabelSet

if ($CheckGitHubRunner.IsPresent) {
  $runnerCheckOutput = (& pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Test-GitHubRunnerAvailability.ps1") -Repository $Repository -RequiredLabelSet $RequiredLabelSet -WarnOnly -RepositoryRoot $RepositoryRoot 2>&1 | Out-String).Trim()
  Add-ReadinessCheck -Checks $checks -Category "runner" -Name "GitHub runner availability" -Passed ($LASTEXITCODE -eq 0) -Detail $runnerCheckOutput
}

$runtimeKeys = @()
try {
  $runtimeKeys = @(Resolve-RuntimeKeyList)
  Add-ReadinessCheck -Checks $checks -Category "runtime" -Name "Runtime key set" -Passed ($runtimeKeys.Count -gt 0) -Detail "$RuntimeKeySet => $($runtimeKeys -join ', ')"
}
catch {
  Add-ReadinessCheck -Checks $checks -Category "runtime" -Name "Runtime key set" -Passed $false -Detail $_.Exception.Message
}

foreach ($key in @($runtimeKeys)) {
  $rootsJson = (& pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $key -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  $roots = $rootsJson | ConvertFrom-Json

  $tensorRtRootExists = -not [string]::IsNullOrWhiteSpace([string]$roots.tensorRtRoot) -and (Test-Path -LiteralPath ([string]$roots.tensorRtRoot) -PathType Container)
  $cudaRootExists = -not [string]::IsNullOrWhiteSpace([string]$roots.cudaRoot) -and (Test-Path -LiteralPath ([string]$roots.cudaRoot) -PathType Container)
  $cudnnRootExists = -not [string]::IsNullOrWhiteSpace([string]$roots.cudnnRoot) -and (Test-Path -LiteralPath ([string]$roots.cudnnRoot) -PathType Container)
  $validation = Invoke-ValidationIfRootsExist -RuntimePackageKey $key -Roots $roots

  $runtimeReports.Add([pscustomobject]@{
      runtimeKey = $key
      tensorRtRoot = [string]$roots.tensorRtRoot
      tensorRtRootExists = $tensorRtRootExists
      cudaRoot = [string]$roots.cudaRoot
      cudaRootExists = $cudaRootExists
      cudnnRoot = [string]$roots.cudnnRoot
      cudnnRootExists = $cudnnRootExists
      validationAttempted = $validation.attempted
      validationPassed = $validation.passed
      validationDetail = $validation.detail
    }) | Out-Null

  $rootsReady = $tensorRtRootExists -and $cudaRootExists -and $cudnnRootExists
  Add-ReadinessCheck -Checks $checks -Category "runtime" -Name "$key NVIDIA roots" -Passed $rootsReady -Detail "TensorRT=$($roots.tensorRtRoot); CUDA=$($roots.cudaRoot); cuDNN=$($roots.cudnnRoot)"
  if ($validation.attempted) {
    Add-ReadinessCheck -Checks $checks -Category "runtime" -Name "$key input validation" -Passed $validation.passed -Detail $validation.detail
  }
}

$failedChecks = @($checks | Where-Object { $_.status -ne "passed" -and $_.severity -eq "error" })
$warningChecks = @($checks | Where-Object { $_.status -ne "passed" -and $_.severity -eq "warning" })

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-self-hosted-runner-readiness"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "linux-self-hosted-runner-readiness.json"
$markdownPath = Join-Path $outputRoot "linux-self-hosted-runner-readiness.md"

[pscustomobject]@{
  repository = $Repository
  requiredLabelSet = $RequiredLabelSet
  runnerRoot = $RunnerRoot
  runtimeKeySet = $RuntimeKeySet
  expectedDistro = $ExpectedDistro
  expectedDistroVersion = $ExpectedDistroVersion
  failedCount = $failedChecks.Count
  warningCount = $warningChecks.Count
  checks = @($checks)
  runtimeKeys = @($runtimeKeys)
  runtimeReports = @($runtimeReports)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$codeQuote = [string][char]96
$lines.Add("# Linux Self-Hosted Runner Readiness")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("Required labels: " + $codeQuote + $RequiredLabelSet + $codeQuote)
$lines.Add("")
$lines.Add("| Category | Check | Status | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($check in $checks) {
  $detail = ([string]$check.detail).Replace("|", "\|").Replace("`r", " ").Replace("`n", "<br>")
  $lines.Add("| $($check.category) | $($check.name) | $($check.status) | $($check.severity) | $detail |")
}

$lines.Add("")
$lines.Add("## Runtime Roots")
$lines.Add("")
$lines.Add("| Runtime key | TensorRT | CUDA | cuDNN | Validation |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($report in $runtimeReports) {
  $validationText = if ($report.validationAttempted) { $report.validationPassed } else { "skipped" }
  $lines.Add("| $($report.runtimeKey) | $($report.tensorRtRootExists): $($report.tensorRtRoot) | $($report.cudaRootExists): $($report.cudaRoot) | $($report.cudnnRootExists): $($report.cudnnRoot) | $validationText |")
}

$lines.Add("")
$lines.Add("## Next Commands")
$lines.Add("")
$lines.Add("Run the registration helper on the Ubuntu 20.04 x64 machine if the runner is not configured:")
$lines.Add("")
$lines.Add('```bash')
$lines.Add("pwsh -File ./eng/Install-GitHubSelfHostedRunner.ps1 -Repository $Repository -UseGhRegistrationToken -InstallService")
$lines.Add('```')
$lines.Add("")
$lines.Add('After the runner and NVIDIA roots are ready, dispatch the Ubuntu 20.04 lane with `RunLinuxSelfHostedUbuntu20RuntimePackaging`.')

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux self-hosted runner readiness written to $jsonPath"
Write-Host "Linux self-hosted runner readiness written to $markdownPath"

if ($failedChecks.Count -gt 0) {
  $message = "Linux self-hosted runner readiness has $($failedChecks.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
