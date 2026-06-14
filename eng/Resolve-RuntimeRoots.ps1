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

$thirdPartyRoot = Join-Path $RepositoryRoot "third_party\nvidia"

function Expand-RuntimeRootPath {
  param(
    [string]$Path
  )

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $expanded = $Path.Replace("<repo-root>", $RepositoryRoot).Replace("<repository-root>", $RepositoryRoot)
  $expanded = [Environment]::ExpandEnvironmentVariables($expanded)

  if ($expanded -eq "~") {
    $home = if ([string]::IsNullOrWhiteSpace($env:HOME)) { $env:USERPROFILE } else { $env:HOME }
    if (-not [string]::IsNullOrWhiteSpace($home)) {
      return $home
    }
  }

  if ($expanded.StartsWith("~/") -or $expanded.StartsWith("~\")) {
    $home = if ([string]::IsNullOrWhiteSpace($env:HOME)) { $env:USERPROFILE } else { $env:HOME }
    if (-not [string]::IsNullOrWhiteSpace($home)) {
      return (Join-Path $home $expanded.Substring(2))
    }
  }

  return $expanded
}

function Resolve-ExistingPath {
  param(
    [string[]]$Candidates
  )

  foreach ($candidate in @($Candidates)) {
    $expandedCandidate = Expand-RuntimeRootPath -Path $candidate
    if ([string]::IsNullOrWhiteSpace($expandedCandidate)) {
      continue
    }

    if (Test-Path -LiteralPath $expandedCandidate -PathType Container) {
      return (Resolve-Path -LiteralPath $expandedCandidate).Path
    }
  }

  return $null
}

$overrides = $null
$localManifestCandidates = [System.Collections.Generic.List[string]]::new()
$localManifestCandidates.Add((Join-Path $RepositoryRoot "pack\runtime\runtime-packages.local.json"))
if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_RUNTIME_PACKAGE_ROOTS_FILE)) {
  $localManifestCandidates.Add($env:JYPPX_RUNTIME_PACKAGE_ROOTS_FILE)
}

$profileRoot = if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { $env:HOME } else { $env:USERPROFILE }
if (-not [string]::IsNullOrWhiteSpace($profileRoot)) {
  $localManifestCandidates.Add((Join-Path $profileRoot ".jyppx\runtime-packages.local.json"))
}

foreach ($localManifestPath in $localManifestCandidates) {
  if ([string]::IsNullOrWhiteSpace($localManifestPath)) {
    continue
  }

  $expandedLocalManifestPath = Expand-RuntimeRootPath -Path $localManifestPath
  if (-not (Test-Path -LiteralPath $expandedLocalManifestPath -PathType Leaf)) {
    continue
  }

  $localManifest = Get-Content -LiteralPath $expandedLocalManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $overrides = $localManifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
  if ($overrides) {
    break
  }
}

$cudnnMinorVersion = $null
if (-not [string]::IsNullOrWhiteSpace([string]$package.cudnnVersion)) {
  $parts = ([string]$package.cudnnVersion).Split('.')
  if ($parts.Count -ge 2) {
    $cudnnMinorVersion = "v{0}.{1}" -f $parts[0], $parts[1]
  }
}

$tensorRtRoot = Resolve-ExistingPath -Candidates @(
  $overrides.defaultTensorRtRoot,
  $package.defaultTensorRtRoot,
  (Join-Path $thirdPartyRoot $package.tensorRtPackageName)
)

$cudaRootCandidates = [System.Collections.Generic.List[string]]::new()
$cudaRootCandidates.Add($overrides.defaultCudaRoot)
$cudaRootCandidates.Add($package.defaultCudaRoot)
if ($package.platform -eq "windows" -and -not [string]::IsNullOrWhiteSpace(${env:ProgramFiles})) {
  $cudaRootCandidates.Add((Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v$($package.cudaVersion)"))
}

$cudaRoot = Resolve-ExistingPath -Candidates $cudaRootCandidates.ToArray()

$cudnnRoot = Resolve-ExistingPath -Candidates @(
  $overrides.defaultCudnnRoot,
  $package.defaultCudnnRoot,
  (Join-Path (Join-Path $thirdPartyRoot $package.cudnnPackageName) $cudnnMinorVersion),
  (Join-Path $thirdPartyRoot $package.cudnnPackageName)
)

$result = [ordered]@{
  runtimeKey = $package.key
  platform = $package.platform
  tensorRtRoot = $tensorRtRoot
  cudaRoot = $cudaRoot
  cudnnRoot = $cudnnRoot
}

[pscustomobject]$result | ConvertTo-Json -Depth 4 -Compress
