[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22",
  [string]$Distribution,
  [string]$RuntimeConsumerEvidencePath,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $null
  }
  return $Object.PSObject.Properties[$Name].Value
}

function Test-Truthy {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }
  if ($Value -is [bool]) {
    return [bool]$Value
  }
  return [string]::Equals([string]$Value, "true", [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-WslDistributions {
  $items = [System.Collections.Generic.List[object]]::new()
  $registryRoot = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss"
  if (Test-Path -LiteralPath $registryRoot) {
    foreach ($key in @(Get-ChildItem -LiteralPath $registryRoot)) {
      $properties = Get-ItemProperty -LiteralPath $key.PSPath
      $name = [string](Get-PropertyOrNull -Object $properties -Name "DistributionName")
      if (-not [string]::IsNullOrWhiteSpace($name)) {
        $items.Add([pscustomobject]@{
          name = $name
          version = [int](Get-PropertyOrNull -Object $properties -Name "Version")
          source = "HKCU/Lxss"
        })
      }
    }
  }

  if ($items.Count -gt 0) {
    return @($items.ToArray() | Sort-Object name -Unique)
  }

  $rawNames = @(& wsl.exe --list --quiet 2>$null)
  foreach ($rawName in $rawNames) {
    $name = ([string]$rawName -replace "`0", "").Trim()
    if (-not [string]::IsNullOrWhiteSpace($name)) {
      $items.Add([pscustomobject]@{
        name = $name
        version = 0
        source = "wsl-list-fallback"
      })
    }
  }
  return @($items.ToArray() | Sort-Object name -Unique)
}

function Invoke-WslProbe {
  param(
    [Parameter(Mandatory = $true)][string]$DistributionName,
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Command
  )

  $stderrPath = [System.IO.Path]::GetTempFileName()
  try {
    $stdoutLines = @(& wsl.exe --distribution $DistributionName -- bash -lc $Command 2>$stderrPath)
    $exitCode = $LASTEXITCODE
    $stdout = (($stdoutLines | ForEach-Object { [string]$_ }) -join "`n").Trim()
    $stderr = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
      (Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
    }
    else {
      ""
    }
    return [pscustomobject]@{
      id = $Id
      command = $Command
      exitCode = $exitCode
      passed = $exitCode -eq 0
      stdout = $stdout
      stderr = $stderr
    }
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

function Get-OsReleaseValue {
  param(
    [AllowEmptyString()][string]$Text,
    [Parameter(Mandatory = $true)][string]$Name
  )

  $line = @($Text -split "`r?`n" | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -First 1)
  if ($line.Count -eq 0) {
    return ""
  }
  return ($line[0].Substring($Name.Length + 1).Trim() -replace '^"|"$', '')
}

$manifestPath = Join-Path $RepositoryRoot "pack/runtime/runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackage = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $runtimePackage) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}
if ([string]$runtimePackage.platform -ne "linux" -or [string]$runtimePackage.architecture -ne "x64") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux x64 package."
}

$hostIsWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
  [System.Runtime.InteropServices.OSPlatform]::Windows)
$wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
$blockers = [System.Collections.Generic.List[string]]::new()
$registeredDistributions = @()
$ubuntuDistributions = @()
$selectedDistribution = $null
$probes = @()

if (-not $hostIsWindows) {
  $blockers.Add("Current host is not Windows. Run this exporter from the Windows side of WSL.")
}
elseif (-not $wslCommand) {
  $blockers.Add("wsl.exe is not available on the current Windows host.")
}
else {
  $registeredDistributions = @(Get-WslDistributions)
  $ubuntuDistributions = @($registeredDistributions | Where-Object {
      $_.name -match '(?i)^ubuntu(?:-|$)' -and $_.name -notmatch '(?i)docker'
    })

  if (-not [string]::IsNullOrWhiteSpace($Distribution)) {
    $selectedDistribution = $registeredDistributions |
      Where-Object { [string]::Equals($_.name, $Distribution, [System.StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1
    if (-not $selectedDistribution) {
      $blockers.Add("Requested WSL distribution '$Distribution' is not registered.")
    }
    elseif ($selectedDistribution.name -match '(?i)docker') {
      $blockers.Add("Docker Desktop's internal WSL distribution cannot be used as INS-002 Ubuntu proof.")
    }
  }
  else {
    $selectedDistribution = $ubuntuDistributions | Select-Object -First 1
    if (-not $selectedDistribution) {
      $blockers.Add("No independent Ubuntu WSL distribution is registered.")
    }
  }
}

if ($selectedDistribution) {
  if ([int]$selectedDistribution.version -ne 2) {
    $blockers.Add("Selected distribution '$($selectedDistribution.name)' is not confirmed as WSL2.")
  }

  $probes = @(
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "architecture" -Command "uname -m"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "kernel" -Command "uname -r"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "os-release" -Command "cat /etc/os-release"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "dxg-device" -Command "test -e /dev/dxg"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "nvidia-smi" -Command "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "dotnet" -Command "dotnet --info"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "powershell" -Command "pwsh --version"
    Invoke-WslProbe -DistributionName $selectedDistribution.name -Id "native-libraries" -Command "ldconfig -p | grep -E 'libcuda|libnvidia-ml|libnvinfer|libnvonnxparser|libcudart|libcudnn'"
  )

  $architectureProbe = $probes | Where-Object id -eq "architecture" | Select-Object -First 1
  $osReleaseProbe = $probes | Where-Object id -eq "os-release" | Select-Object -First 1
  if (-not $architectureProbe.passed -or $architectureProbe.stdout -ne "x86_64") {
    $blockers.Add("Selected WSL distribution did not report x86_64 architecture.")
  }
  $actualDistroVersion = Get-OsReleaseValue -Text $osReleaseProbe.stdout -Name "VERSION_ID"
  if (-not $osReleaseProbe.passed -or $actualDistroVersion -ne [string]$runtimePackage.linuxDistroVersion) {
    $blockers.Add("Selected WSL distribution does not match Ubuntu $($runtimePackage.linuxDistroVersion).")
  }
  foreach ($requiredProbeId in @("dxg-device", "nvidia-smi", "dotnet", "powershell", "native-libraries")) {
    $requiredProbe = $probes | Where-Object id -eq $requiredProbeId | Select-Object -First 1
    if (-not $requiredProbe.passed) {
      $blockers.Add("WSL probe '$requiredProbeId' failed with exit code $($requiredProbe.exitCode).")
    }
  }
}

$runtimeConsumerEvidence = $null
$runtimeConsumerProofAccepted = $false
if (-not [string]::IsNullOrWhiteSpace($RuntimeConsumerEvidencePath)) {
  if (-not [System.IO.Path]::IsPathRooted($RuntimeConsumerEvidencePath)) {
    $RuntimeConsumerEvidencePath = Join-Path $RepositoryRoot $RuntimeConsumerEvidencePath
  }
  if (-not (Test-Path -LiteralPath $RuntimeConsumerEvidencePath -PathType Leaf)) {
    $blockers.Add("Runtime consumer evidence file was not found: $RuntimeConsumerEvidencePath")
  }
  else {
    $runtimeConsumerEvidence = Get-Content -LiteralPath $RuntimeConsumerEvidencePath -Raw -Encoding utf8 | ConvertFrom-Json
    $execution = Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "execution"
    $consumerHost = Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "host"
    $runtimeConsumerProofAccepted =
      [string](Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "recordKind") -eq "minimal-linux-bridge-package-runtime-consumer" -and
      [string](Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "sourceRuntimeKey") -eq $RuntimePackageKey -and
      (Test-Truthy (Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "runtimeExecutionProof")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $runtimeConsumerEvidence -Name "canPromoteWslRuntimeProof")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $consumerHost -Name "isWsl")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $execution -Name "readyForEnqueue")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $execution -Name "enqueueCompleted")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $execution -Name "streamSynchronized")) -and
      (Test-Truthy (Get-PropertyOrNull -Object $execution -Name "identityOutputMatch"))
    if (-not $runtimeConsumerProofAccepted) {
      $blockers.Add("Runtime consumer evidence is present but does not satisfy the WSL runtime proof contract.")
    }
  }
}

$environmentReady = $selectedDistribution -and $blockers.Count -eq 0
$status = if ($runtimeConsumerProofAccepted -and $environmentReady) {
  "wsl-gpu-runtime-proof-candidate"
}
elseif ($environmentReady) {
  "wsl-environment-ready-runtime-proof-required"
}
else {
  "blocked"
}

$reportRoot = Join-Path $RepositoryRoot "artifacts/wsl-runtime/$RuntimePackageKey"
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$jsonPath = Join-Path $reportRoot "wsl-runtime-evidence.json"
$markdownPath = Join-Path $reportRoot "wsl-runtime-evidence.md"

$record = [ordered]@{
  schemaVersion = 1
  recordKind = "wsl-runtime-evidence"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  runtimeKey = $RuntimePackageKey
  packageId = [string]$runtimePackage.packageId
  requiredUbuntuVersion = [string]$runtimePackage.linuxDistroVersion
  hostIsWindows = $hostIsWindows
  wslCommandAvailable = $null -ne $wslCommand
  registeredDistributions = @($registeredDistributions)
  registeredUbuntuDistributionCount = $ubuntuDistributions.Count
  dockerDesktopDistributionPresent = @($registeredDistributions | Where-Object { $_.name -match '(?i)^docker-desktop$' }).Count -gt 0
  selectedDistribution = if ($selectedDistribution) { $selectedDistribution } else { $null }
  probes = @($probes)
  environmentReady = [bool]$environmentReady
  runtimeConsumerEvidencePath = if ([string]::IsNullOrWhiteSpace($RuntimeConsumerEvidencePath)) { "" } else { [System.IO.Path]::GetFullPath($RuntimeConsumerEvidencePath) }
  runtimeConsumerProofAccepted = $runtimeConsumerProofAccepted
  status = $status
  blockerCount = $blockers.Count
  blockers = @($blockers.ToArray())
  performsRuntimeExecution = $false
  performsPublish = $false
  proofBoundary = "This exporter audits WSL configuration and validates an optional minimal consumer report. It does not execute TensorRT itself, does not accept Docker Desktop as an Ubuntu WSL distribution, and cannot turn container or GitHub runner evidence into WSL proof."
}
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockerLines = if ($blockers.Count -eq 0) { @("- none") } else { @($blockers | ForEach-Object { "- $_" }) }
$markdown = @"
# WSL Runtime Evidence

- runtime key: ``$RuntimePackageKey``
- registered Ubuntu distributions: ``$($ubuntuDistributions.Count)``
- selected distribution: ``$(if ($selectedDistribution) { $selectedDistribution.name } else { 'none' })``
- environment ready: ``$environmentReady``
- runtime consumer proof accepted: ``$runtimeConsumerProofAccepted``
- status: ``$status``

## Blockers

$($blockerLines -join "`n")

## Boundary

$($record.proofBoundary)
"@
[System.IO.File]::WriteAllText($markdownPath, $markdown, $utf8)

Write-Host "WSL runtime evidence written to $jsonPath"
Write-Host "WSL runtime evidence written to $markdownPath"
Write-Host "Status=$status"
