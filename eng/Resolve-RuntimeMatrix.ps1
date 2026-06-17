[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("windows", "linux")]
  [string]$Platform,
  [string[]]$RuntimeKey,
  [ValidateSet("any", "hosted", "self-hosted")]
  [string]$RunnerMode = "any",
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

if ($Platform -eq "linux" -and $RunnerMode -ne "any") {
  $packages = @($packages | Where-Object { $_.runnerMode -eq $RunnerMode })
  if ($requestedKeys.Count -gt 0) {
    foreach ($key in $requestedKeys) {
      if (-not ($packages | Where-Object { $_.key -eq $key })) {
        throw "Runtime package key '$key' is not available for runner mode '$RunnerMode'."
      }
    }
  }
}

$matrix = foreach ($package in $packages) {
  $runnerLabels = @()
  if ($package.PSObject.Properties.Name.Contains("runnerLabels")) {
    $runnerLabels = @($package.runnerLabels | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
  }

  if ($Platform -eq "linux" -and $runnerLabels.Count -eq 0) {
    $runnerLabels = if ($package.runnerMode -eq "self-hosted") {
      @("self-hosted", "linux", "x64")
    }
    else {
      @("ubuntu-22.04")
    }
  }

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
    linuxDistro = $package.linuxDistro
    linuxDistroVersion = $package.linuxDistroVersion
    architecture = $package.architecture
    runnerMode = $package.runnerMode
    runsOnJson = if ($Platform -eq "linux") { ConvertTo-Json -InputObject @($runnerLabels) -Compress } else { $null }
    nvidiaDependencyMode = $package.nvidiaDependencyMode
    nvidiaRepoDistroId = $package.nvidiaRepoDistroId
    nvidiaRepoArchitecture = $package.nvidiaRepoArchitecture
  }
}

ConvertTo-Json -InputObject @($matrix) -Depth 5 -Compress
