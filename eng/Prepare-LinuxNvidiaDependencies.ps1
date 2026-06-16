[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$RepositoryRoot,
  [switch]$DescribeDependencyPlan,
  [switch]$SkipAptInstall,
  [switch]$AllowDistroMismatch
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Invoke-CheckedNativeCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$ArgumentList
  )

  Write-Host "> $FilePath $($ArgumentList -join ' ')"
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
  }
}

function Get-LinuxOsRelease {
  $osReleasePath = "/etc/os-release"
  if (-not (Test-Path -LiteralPath $osReleasePath -PathType Leaf)) {
    throw "Unable to determine the Linux distribution because /etc/os-release was not found."
  }

  $values = @{}
  foreach ($line in Get-Content -LiteralPath $osReleasePath -Encoding utf8) {
    if ($line -notmatch '^([^=]+)=(.*)$') {
      continue
    }

    $key = $Matches[1]
    $value = $Matches[2].Trim('"')
    $values[$key] = $value
  }

  return [pscustomobject]@{
    id = [string]$values["ID"]
    versionId = [string]$values["VERSION_ID"]
  }
}

function Assert-LinuxDistributionMatchesPackage {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package
  )

  $actual = Get-LinuxOsRelease
  $expectedId = if ($Package.PSObject.Properties.Name.Contains("linuxDistro")) { [string]$Package.linuxDistro } else { "ubuntu" }
  $expectedVersion = if ($Package.PSObject.Properties.Name.Contains("linuxDistroVersion")) { [string]$Package.linuxDistroVersion } else { "" }

  if ($actual.id -ne $expectedId -or (-not [string]::IsNullOrWhiteSpace($expectedVersion) -and $actual.versionId -ne $expectedVersion)) {
    $message = "Runtime package '$($Package.key)' targets $expectedId $expectedVersion but this runner is '$($actual.id) $($actual.versionId)'."
    if ($AllowDistroMismatch.IsPresent) {
      Write-Warning $message
      return
    }

    throw "$message Use a matching runner or pass -AllowDistroMismatch only for a deliberate manual dependency-root workflow."
  }
}

function New-AptVersionPin {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Version
  )

  return "=$Version*"
}

function Test-CudaCrtPackageIsAvailable {
  param(
    [Parameter(Mandatory = $true)]
    [string]$CudaPackageVersion
  )

  $parsed = $null
  if (-not [System.Version]::TryParse($CudaPackageVersion, [ref]$parsed)) {
    return $false
  }

  return $parsed -ge [System.Version]::new(12, 9)
}

function Test-TensorRtSafeHeadersPackageIsAvailable {
  param(
    [Parameter(Mandatory = $true)]
    [string]$TensorRtLine,
    [Parameter(Mandatory = $true)]
    [string]$TensorRtVersion
  )

  if ($TensorRtLine -eq "11") {
    return $true
  }

  $parsed = $null
  if ([System.Version]::TryParse($TensorRtVersion, [ref]$parsed)) {
    return $TensorRtLine -eq "10" -and $parsed -ge [System.Version]::new(10, 16)
  }

  return $false
}

function Get-LinuxDependencyPlan {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package
  )

  $cudaPackageVersion = if ($Package.PSObject.Properties.Name.Contains("linuxCudaPackageVersion") -and -not [string]::IsNullOrWhiteSpace([string]$Package.linuxCudaPackageVersion)) {
    [string]$Package.linuxCudaPackageVersion
  }
  else {
    [string]$Package.cudaVersion
  }

  $tensorRtAptCudaVersion = if ($Package.PSObject.Properties.Name.Contains("tensorRtAptCudaVersion") -and -not [string]::IsNullOrWhiteSpace([string]$Package.tensorRtAptCudaVersion)) {
    [string]$Package.tensorRtAptCudaVersion
  }
  else {
    [string]$Package.cudaVersion
  }

  $cudaMinor = $cudaPackageVersion.Replace(".", "-")
  $tensorRtLine = [string]$Package.tensorRtLine
  $tensorRtDebVersion = "$($Package.tensorRtVersion)-1+cuda$tensorRtAptCudaVersion"

  $runtimeTensorRtPackages = @(
    "libnvinfer$tensorRtLine",
    "libnvinfer-lean$tensorRtLine",
    "libnvinfer-plugin$tensorRtLine",
    "libnvinfer-dispatch$tensorRtLine",
    "libnvonnxparsers$tensorRtLine"
  )

  $devTensorRtPackages = @(
    "libnvinfer-headers-dev",
    "libnvinfer-headers-plugin-dev",
    "libnvinfer-dev",
    "libnvinfer-lean-dev",
    "libnvinfer-plugin-dev",
    "libnvinfer-dispatch-dev",
    "libnvonnxparsers-dev",
    "libnvinfer-bin"
  )

  if (Test-TensorRtSafeHeadersPackageIsAvailable -TensorRtLine $tensorRtLine -TensorRtVersion ([string]$Package.tensorRtVersion)) {
    $devTensorRtPackages = @("libnvinfer-safe-headers-dev") + $devTensorRtPackages
  }

  if ($tensorRtLine -eq "8") {
    $runtimeTensorRtPackages += "libnvparsers$tensorRtLine"
    $devTensorRtPackages += "libnvparsers-dev"
  }
  else {
    $runtimeTensorRtPackages += "libnvinfer-vc-plugin$tensorRtLine"
    $devTensorRtPackages += "libnvinfer-vc-plugin-dev"
  }

  $aptPackages = New-Object System.Collections.Generic.List[string]
  $aptPackages.Add("cuda-cudart-dev-$cudaMinor")
  if (Test-CudaCrtPackageIsAvailable -CudaPackageVersion $cudaPackageVersion) {
    $aptPackages.Add("cuda-crt-$cudaMinor")
  }

  foreach ($name in @($runtimeTensorRtPackages + $devTensorRtPackages)) {
    $aptPackages.Add($name + (New-AptVersionPin -Version $tensorRtDebVersion))
  }

  if ([string]$Package.cudnnMajor -eq "9") {
    $cudnnCudaLine = if ($Package.PSObject.Properties.Name.Contains("cudnnAptCudaVersion") -and -not [string]::IsNullOrWhiteSpace([string]$Package.cudnnAptCudaVersion)) {
      (([string]$Package.cudnnAptCudaVersion) -split "\.")[0]
    }
    else {
      [string]$Package.cudaLine
    }
    $cudnnVersion = "$($Package.cudnnVersion).52-1"
    if ([string]$Package.cudnnVersion -eq "9.22.0") {
      $cudnnVersion = "9.22.0.52-1"
    }

    $aptPackages.Add("libcudnn9-cuda-$cudnnCudaLine" + (New-AptVersionPin -Version $cudnnVersion))
  }
  elseif ([string]$Package.cudnnMajor -eq "8") {
    $cudnnAptCudaVersion = if ($Package.PSObject.Properties.Name.Contains("cudnnAptCudaVersion") -and -not [string]::IsNullOrWhiteSpace([string]$Package.cudnnAptCudaVersion)) {
      [string]$Package.cudnnAptCudaVersion
    }
    else {
      [string]$Package.cudaVersion
    }

    $cudnnVersion = "$($Package.cudnnVersion)-1+cuda$cudnnAptCudaVersion"
    $aptPackages.Add("libcudnn8" + (New-AptVersionPin -Version $cudnnVersion))
    $aptPackages.Add("libcudnn8-dev" + (New-AptVersionPin -Version $cudnnVersion))
  }
  else {
    throw "Hosted Linux dependency preparation supports cuDNN 8 and 9 runtime packages. '$($Package.key)' requests cuDNN major '$($Package.cudnnMajor)'."
  }

  return [pscustomobject]@{
    cudaMinor = $cudaMinor
    cudaPackageVersion = $cudaPackageVersion
    tensorRtAptCudaVersion = $tensorRtAptCudaVersion
    tensorRtDebVersion = $tensorRtDebVersion
    aptPackages = @($aptPackages)
  }
}

function Add-NvidiaCudaRepository {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DistributionId,
    [Parameter(Mandatory = $true)]
    [string]$Architecture
  )

  $keyringDeb = "/tmp/cuda-keyring.deb"
  $keyringUrl = "https://developer.download.nvidia.com/compute/cuda/repos/$DistributionId/$Architecture/cuda-keyring_1.1-1_all.deb"
  Invoke-CheckedNativeCommand -FilePath "wget" -ArgumentList @("-q", "-O", $keyringDeb, $keyringUrl)
  Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList @("dpkg", "-i", $keyringDeb)
}

function Resolve-ExistingPath {
  param(
    [string[]]$Candidates
  )

  foreach ($candidate in $Candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
      continue
    }

    if (Test-Path -LiteralPath $candidate -PathType Container) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }

  return $null
}

function Invoke-ReadLink {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  $resolved = & readlink -f $Path
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolved)) {
    throw "Unable to resolve Linux real path for '$Path'."
  }

  return [string]$resolved
}

function Reset-Directory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }

  New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Copy-HeaderFiles {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Patterns,
    [Parameter(Mandatory = $true)]
    [string]$DestinationDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
  $copied = New-Object System.Collections.Generic.List[string]
  foreach ($pattern in $Patterns) {
    foreach ($match in @(Resolve-Path -Path $pattern -ErrorAction SilentlyContinue)) {
      $source = $match.Path
      $destination = Join-Path $DestinationDirectory ([System.IO.Path]::GetFileName($source))
      Copy-Item -LiteralPath $source -Destination $destination -Force
      $copied.Add($destination)
    }
  }

  if ($copied.Count -eq 0) {
    throw "No $Label headers matched any expected patterns: $($Patterns -join ', ')"
  }

  return @($copied.ToArray())
}

function Add-StagedLibraryLinks {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Patterns,
    [Parameter(Mandatory = $true)]
    [string]$DestinationDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )

  New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
  $linked = New-Object System.Collections.Generic.List[string]
  foreach ($pattern in $Patterns) {
    foreach ($match in @(Resolve-Path -Path $pattern -ErrorAction SilentlyContinue)) {
      $source = $match.Path
      $destination = Join-Path $DestinationDirectory ([System.IO.Path]::GetFileName($source))
      if (Test-Path -LiteralPath $destination) {
        Remove-Item -LiteralPath $destination -Force
      }

      $realSource = Invoke-ReadLink -Path $source
      New-Item -ItemType SymbolicLink -Path $destination -Target $realSource | Out-Null
      $linked.Add($destination)
    }
  }

  if ($linked.Count -eq 0) {
    throw "No $Label libraries matched any expected patterns: $($Patterns -join ', ')"
  }

  return @($linked.ToArray())
}

$manifestPath = Join-Path $RepositoryRoot "pack/runtime/runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$dependencyPlan = Get-LinuxDependencyPlan -Package $package
if ($DescribeDependencyPlan.IsPresent) {
  $dependencyPlan | ConvertTo-Json -Depth 5
  exit 0
}

if (-not $SkipAptInstall.IsPresent) {
  Assert-LinuxDistributionMatchesPackage -Package $package
  $distributionId = if ($package.PSObject.Properties.Name.Contains("nvidiaRepoDistroId") -and -not [string]::IsNullOrWhiteSpace([string]$package.nvidiaRepoDistroId)) {
    [string]$package.nvidiaRepoDistroId
  }
  else {
    throw "Linux package '$($package.key)' is missing nvidiaRepoDistroId."
  }

  $repoArchitecture = if ($package.PSObject.Properties.Name.Contains("nvidiaRepoArchitecture") -and -not [string]::IsNullOrWhiteSpace([string]$package.nvidiaRepoArchitecture)) {
    [string]$package.nvidiaRepoArchitecture
  }
  else {
    "x86_64"
  }

  Add-NvidiaCudaRepository -DistributionId $distributionId -Architecture $repoArchitecture
  Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList @("apt-get", "update")
  Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList (@("apt-get", "install", "-y", "--no-install-recommends") + $dependencyPlan.aptPackages)
}

$cudaRoot = Resolve-ExistingPath -Candidates @(
  "/usr/local/cuda-$($dependencyPlan.cudaPackageVersion)",
  "/usr/local/cuda-$($package.cudaVersion)",
  "/usr/local/cuda"
)
$systemTensorRtRoot = Resolve-ExistingPath -Candidates @("/usr")
$systemCudnnRoot = Resolve-ExistingPath -Candidates @("/usr")

if (-not $cudaRoot) {
  throw "CUDA root was not found after preparing Linux NVIDIA dependencies."
}
if (-not $systemTensorRtRoot) {
  throw "TensorRT root was not found after preparing Linux NVIDIA dependencies."
}
if (-not $systemCudnnRoot) {
  throw "cuDNN root was not found after preparing Linux NVIDIA dependencies."
}

$stagingRoot = Join-Path $RepositoryRoot "artifacts/linux-nvidia-root/$RuntimePackageKey"
$tensorRtRoot = Join-Path $stagingRoot "tensorrt"
$cudnnRoot = Join-Path $stagingRoot "cudnn"
$tensorRtIncludeRoot = Join-Path $tensorRtRoot "include"
$tensorRtLibraryRoot = Join-Path $tensorRtRoot "lib"
$cudnnLibraryRoot = Join-Path $cudnnRoot "lib"

Reset-Directory -Path $stagingRoot
Reset-Directory -Path $tensorRtIncludeRoot
Reset-Directory -Path $tensorRtLibraryRoot
Reset-Directory -Path $cudnnLibraryRoot

$tensorRtHeaders = Copy-HeaderFiles `
  -Patterns @(
    "/usr/include/Nv*.h",
    "/usr/include/x86_64-linux-gnu/Nv*.h",
    "/usr/include/aarch64-linux-gnu/Nv*.h"
  ) `
  -DestinationDirectory $tensorRtIncludeRoot `
  -Label "TensorRT"

$tensorRtLibraries = Add-StagedLibraryLinks `
  -Patterns @(
    "/usr/lib/x86_64-linux-gnu/libnvinfer*.so*",
    "/usr/lib/x86_64-linux-gnu/libnvonnxparser*.so*",
    "/usr/lib/aarch64-linux-gnu/libnvinfer*.so*",
    "/usr/lib/aarch64-linux-gnu/libnvonnxparser*.so*"
  ) `
  -DestinationDirectory $tensorRtLibraryRoot `
  -Label "TensorRT"

$cudnnLibraries = Add-StagedLibraryLinks `
  -Patterns @(
    "/usr/lib/x86_64-linux-gnu/libcudnn*.so*",
    "/usr/lib/aarch64-linux-gnu/libcudnn*.so*"
  ) `
  -DestinationDirectory $cudnnLibraryRoot `
  -Label "cuDNN"

$localManifestPath = Join-Path $RepositoryRoot "pack/runtime/runtime-packages.local.json"
$localManifest = [ordered]@{
  packages = @(
    [ordered]@{
      key = $RuntimePackageKey
      defaultTensorRtRoot = $tensorRtRoot
      defaultCudaRoot = $cudaRoot
      defaultCudnnRoot = $cudnnRoot
    }
  )
}
$localManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $localManifestPath -Encoding utf8

$summaryRoot = Join-Path $RepositoryRoot "artifacts/linux-dependencies/$RuntimePackageKey"
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null
$summaryPath = Join-Path $summaryRoot "linux-nvidia-dependencies.json"
[ordered]@{
  runtimeKey = $RuntimePackageKey
  packageId = $package.packageId
  linuxDistro = $package.linuxDistro
  linuxDistroVersion = $package.linuxDistroVersion
  nvidiaRepoDistroId = $package.nvidiaRepoDistroId
  nvidiaRepoArchitecture = $package.nvidiaRepoArchitecture
  cudaPackageVersion = $dependencyPlan.cudaPackageVersion
  tensorRtAptCudaVersion = $dependencyPlan.tensorRtAptCudaVersion
  tensorRtDebVersion = $dependencyPlan.tensorRtDebVersion
  aptPackages = $dependencyPlan.aptPackages
  tensorRtRoot = $tensorRtRoot
  cudaRoot = $cudaRoot
  cudnnRoot = $cudnnRoot
  systemTensorRtRoot = $systemTensorRtRoot
  systemCudnnRoot = $systemCudnnRoot
  stagedTensorRtHeaders = $tensorRtHeaders
  stagedTensorRtLibraries = $tensorRtLibraries
  stagedCudnnLibraries = $cudnnLibraries
  localManifestPath = $localManifestPath
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host "Prepared Linux NVIDIA dependencies for $RuntimePackageKey."
Write-Host "TensorRT root: $tensorRtRoot"
Write-Host "CUDA root    : $cudaRoot"
Write-Host "cuDNN root   : $cudnnRoot"
Write-Host "Staging root : $stagingRoot"
Write-Host "Local manifest: $localManifestPath"
