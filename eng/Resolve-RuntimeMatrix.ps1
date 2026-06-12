[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("windows", "linux")]
  [string]$Platform,
  [string[]]$RuntimeKey,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Expand-KeyList {
  param(
    [string[]]$Values
  )

  $keys = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $keys.Add($trimmed)
      }
    }
  }

  return @($keys | Select-Object -Unique)
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$requestedKeys = @(Expand-KeyList -Values $RuntimeKey)
$packages = @($manifest.packages | Where-Object { $_.platform -eq $Platform })

if ($requestedKeys.Count -gt 0) {
  $packages = @($packages | Where-Object { $requestedKeys -contains $_.key })
  foreach ($key in $requestedKeys) {
    if (-not ($packages | Where-Object { $_.key -eq $key })) {
      throw "Runtime package key '$key' was not found for platform '$Platform'."
    }
  }
}

$matrix = foreach ($package in $packages) {
  [pscustomobject]@{
    key = $package.key
    packageId = $package.packageId
    rid = $package.rid
    buildPreset = $package.buildPreset
    tensorRtLine = $package.tensorRtLine
    tensorRtVersion = $package.tensorRtVersion
    cudaVersion = $package.cudaVersion
    cudnnVersion = $package.cudnnVersion
    distributionTier = $package.distributionTier
    validationState = $package.validationState
  }
}

$matrix | ConvertTo-Json -Depth 5 -Compress
