[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("windows", "linux")]
  [string]$Platform,
  [string[]]$RuntimeKey = @(),
  [string]$RuntimeKeySet = "auto",
  [ValidateSet("any", "hosted", "hosted-container", "self-hosted")]
  [string]$RunnerMode = "any",
  [ValidateSet("csv", "json")]
  [string]$OutputFormat = "csv",
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

  @($keys | Select-Object -Unique)
}

$explicitKeys = @(Expand-KeyList -Values $RuntimeKey)
if ($explicitKeys.Count -gt 0) {
  $resolvedKeys = $explicitKeys
}
else {
  $manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Runtime manifest was not found: $manifestPath"
  }

  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $packages = @($manifest.packages | Where-Object { $_.platform -eq $Platform })
  $normalizedSet = if ([string]::IsNullOrWhiteSpace($RuntimeKeySet)) { "auto" } else { $RuntimeKeySet.Trim().ToLowerInvariant() }

  if ($Platform -eq "windows") {
    switch ($normalizedSet) {
      { $_ -in @("auto", "default", "all", "win-x64-all") } {
        $resolvedKeys = @($packages | Select-Object -ExpandProperty key)
        break
      }
      "custom" {
        throw "RuntimeKeySet 'custom' requires explicit RuntimeKey values."
      }
      default {
        throw "Unsupported Windows runtime key set '$RuntimeKeySet'. Supported values: auto, default, all, win-x64-all, custom."
      }
    }
  }
  else {
    $targetCatalogPath = Join-Path $RepositoryRoot "pack\runtime\linux-runtime-targets.manifest.json"
    if (Test-Path -LiteralPath $targetCatalogPath -PathType Leaf) {
      $targetCatalog = Get-Content -LiteralPath $targetCatalogPath -Raw -Encoding utf8 | ConvertFrom-Json
      $futureTarget = @(
        foreach ($target in @($targetCatalog.futureTargets)) {
          $aliases = @($target.keySetAliases | ForEach-Object { ([string]$_).Trim().ToLowerInvariant() })
          if ($aliases -contains $normalizedSet -or ([string]$target.target).Trim().ToLowerInvariant() -eq $normalizedSet) {
            $target
          }
        }
      ) | Select-Object -First 1

      if ($null -ne $futureTarget) {
        $requiredEvidence = @($futureTarget.requiredEvidenceItems | ForEach-Object { [string]$_ }) -join "; "
        throw "Linux runtime key set '$RuntimeKeySet' maps to future package line '$($futureTarget.target)' from pack/runtime/linux-runtime-targets.manifest.json, but that line is not dispatchable yet. Package identity rule: $($futureTarget.packageIdentityRule) Required evidence before enabling: $requiredEvidence"
      }
    }

    switch ($normalizedSet) {
      "auto" {
        if ($RunnerMode -eq "hosted-container") {
          $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted-container" -and $_.linuxDistro -eq "ubuntu" -and $_.linuxDistroVersion -eq "20.04" } | Select-Object -ExpandProperty key)
        }
        elseif ($RunnerMode -eq "self-hosted") {
          $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "self-hosted" } | Select-Object -ExpandProperty key)
        }
        else {
          $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted" -and $_.linuxDistro -eq "ubuntu" -and $_.linuxDistroVersion -eq "22.04" } | Select-Object -ExpandProperty key)
        }
        break
      }
      { $_ -in @("default", "ubuntu22-hosted", "ubuntu22.04-hosted") } {
        $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted" -and $_.linuxDistro -eq "ubuntu" -and $_.linuxDistroVersion -eq "22.04" } | Select-Object -ExpandProperty key)
        break
      }
      { $_ -in @("hosted-all", "ubuntu-hosted-all") } {
        $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted" } | Select-Object -ExpandProperty key)
        break
      }
      { $_ -in @("ubuntu24-hosted", "ubuntu24.04-hosted") } {
        $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted" -and $_.linuxDistro -eq "ubuntu" -and $_.linuxDistroVersion -eq "24.04" } | Select-Object -ExpandProperty key)
        break
      }
      { $_ -in @("hosted-container-ubuntu20", "ubuntu20-hosted-container", "ubuntu20.04-hosted-container") } {
        $resolvedKeys = @($packages | Where-Object { $_.runnerMode -eq "hosted-container" -and $_.linuxDistro -eq "ubuntu" -and $_.linuxDistroVersion -eq "20.04" } | Select-Object -ExpandProperty key)
        break
      }
      "all" {
        $resolvedKeys = @($packages | Select-Object -ExpandProperty key)
        break
      }
      "custom" {
        throw "RuntimeKeySet 'custom' requires explicit RuntimeKey values."
      }
      default {
        throw "Unsupported Linux runtime key set '$RuntimeKeySet'. Supported values: auto, default, ubuntu22-hosted, hosted-all, ubuntu24-hosted, hosted-container-ubuntu20, all, custom. Ubuntu 20.04 now uses hosted-container-ubuntu20 with runner_mode='hosted-container'. Known future lines such as arm64-sbsa, jetson-l4t, and non-ubuntu are tracked in pack/runtime/linux-runtime-targets.manifest.json and require dedicated manifest entries, runners, NVIDIA dependency plans, and package consumer evidence before dispatch."
      }
    }
  }
}

$resolvedKeys = @($resolvedKeys | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
if ($resolvedKeys.Count -eq 0) {
  throw "Runtime key set '$RuntimeKeySet' resolved to no '$Platform' runtime keys."
}

if ($Platform -eq "linux" -and $RunnerMode -ne "any") {
  $manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $modeMismatches = @(
    foreach ($key in $resolvedKeys) {
      $package = $manifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
      if ($null -eq $package) {
        throw "Runtime package key '$key' was not found."
      }

      if ($package.runnerMode -ne $RunnerMode) {
        $key
      }
    }
  )

  if ($modeMismatches.Count -gt 0) {
    throw "Runtime key set '$RuntimeKeySet' includes keys that require a different runner mode than '$RunnerMode': $($modeMismatches -join ', '). Use runner_mode='any' only for inspection, or dispatch separate hosted/hosted-container/self-hosted runs."
  }
}

if ($OutputFormat -eq "json") {
  ConvertTo-Json -InputObject @($resolvedKeys) -Compress
}
else {
  $resolvedKeys -join ","
}
