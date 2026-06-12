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

function Resolve-ExistingPath {
  param(
    [string[]]$Candidates
  )

  foreach ($candidate in @($Candidates)) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
      continue
    }

    if (Test-Path -LiteralPath $candidate -PathType Container) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }

  return $null
}

$overrides = $null
$localManifestCandidates = @(
  (Join-Path $RepositoryRoot "pack\runtime\runtime-packages.local.json"),
  $env:JYPPX_RUNTIME_PACKAGE_ROOTS_FILE,
  (Join-Path $env:USERPROFILE ".jyppx\runtime-packages.local.json")
)

foreach ($localManifestPath in $localManifestCandidates) {
  if ([string]::IsNullOrWhiteSpace($localManifestPath)) {
    continue
  }

  if (-not (Test-Path -LiteralPath $localManifestPath -PathType Leaf)) {
    continue
  }

  $localManifest = Get-Content -LiteralPath $localManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
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

$cudaRoot = Resolve-ExistingPath -Candidates @(
  $overrides.defaultCudaRoot,
  $package.defaultCudaRoot,
  (Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v$($package.cudaVersion)")
)

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
