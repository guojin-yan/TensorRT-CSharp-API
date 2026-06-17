[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$SourceRuntimeKey,
  [string]$Version,
  [string]$MetaPackageVersion,
  [string]$MetaPackageVersionMap,
  [string]$BridgePackageVersion,
  [string]$BridgePackageVersionMap,
  [string]$CudaCudnnPackageVersion,
  [string]$CudaCudnnPackageVersionMap,
  [string]$CudaCudnnPackageReleaseTag,
  [string]$CudaCudnnPackageReleaseTagMap,
  [string]$TensorRtPackageVersion,
  [string]$TensorRtPackageVersionMap,
  [string]$TensorRtPackageReleaseTag,
  [string]$TensorRtPackageReleaseTagMap,
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

function ConvertFrom-PinMap {
  param(
    [AllowEmptyString()]
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  $map = [ordered]@{}
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $map
  }

  $trimmed = $Value.Trim()
  if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed -eq "{}") {
    return $map
  }

  if ($trimmed.StartsWith("{", [System.StringComparison]::Ordinal)) {
    $json = $trimmed | ConvertFrom-Json
    foreach ($property in $json.PSObject.Properties) {
      $key = ([string]$property.Name).Trim()
      $propertyValue = ([string]$property.Value).Trim()
      if (-not [string]::IsNullOrWhiteSpace($key) -and -not [string]::IsNullOrWhiteSpace($propertyValue)) {
        $map[$key] = $propertyValue
      }
    }

    return $map
  }

  foreach ($entry in ($trimmed -split "[`r`n;]")) {
    $item = $entry.Trim()
    if ([string]::IsNullOrWhiteSpace($item)) {
      continue
    }

    $separatorIndex = $item.IndexOf("=")
    if ($separatorIndex -lt 1) {
      $separatorIndex = $item.IndexOf(":")
    }

    if ($separatorIndex -lt 1) {
      throw "$Name map entry '$item' must use key=value or JSON object syntax."
    }

    $key = $item.Substring(0, $separatorIndex).Trim()
    $propertyValue = $item.Substring($separatorIndex + 1).Trim()
    if (-not [string]::IsNullOrWhiteSpace($key) -and -not [string]::IsNullOrWhiteSpace($propertyValue)) {
      $map[$key] = $propertyValue
    }
  }

  return $map
}

function Resolve-PinValue {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey,
    [AllowEmptyString()]
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [System.Collections.IDictionary]$Map
  )

  if ($Map.Contains($RuntimeKey)) {
    return [pscustomobject]@{
      value = [string]$Map[$RuntimeKey]
      source = "map-exact"
      provided = $true
    }
  }

  $wildcardMatches = New-Object System.Collections.Generic.List[object]
  foreach ($entry in $Map.GetEnumerator()) {
    $pattern = [string]$entry.Key
    if (($pattern.Contains("*") -or $pattern.Contains("?")) -and $RuntimeKey -like $pattern) {
      $wildcardMatches.Add([pscustomobject]@{
          pattern = $pattern
          value = [string]$entry.Value
        }) | Out-Null
    }
  }

  if ($wildcardMatches.Count -gt 1) {
    $patterns = @($wildcardMatches | ForEach-Object { $_.pattern }) -join ", "
    throw "Runtime key '$RuntimeKey' matches multiple version-map patterns: $patterns"
  }

  if ($wildcardMatches.Count -eq 1) {
    return [pscustomobject]@{
      value = [string]$wildcardMatches[0].value
      source = "map-wildcard:$($wildcardMatches[0].pattern)"
      provided = $true
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($Value)) {
    return [pscustomobject]@{
      value = $Value.Trim()
      source = "input"
      provided = $true
    }
  }

  return [pscustomobject]@{
    value = ""
    source = "default"
    provided = $false
  }
}

function Resolve-ReleaseTag {
  param(
    [object]$TagPin,
    [object]$VersionPin
  )

  if ($TagPin.provided -and -not [string]::IsNullOrWhiteSpace([string]$TagPin.value)) {
    return [pscustomobject]@{
      value = [string]$TagPin.value
      source = [string]$TagPin.source
      provided = $true
    }
  }

  if ($VersionPin.provided -and -not [string]::IsNullOrWhiteSpace([string]$VersionPin.value)) {
    return [pscustomobject]@{
      value = "v$($VersionPin.value)"
      source = "version-default"
      provided = $false
    }
  }

  return [pscustomobject]@{
    value = ""
    source = "default"
    provided = $false
  }
}

function Resolve-PackageVersionOrFallback {
  param(
    [object]$Pin,
    [Parameter(Mandatory = $true)]
    [string]$Fallback
  )

  if ($Pin.provided -and -not [string]::IsNullOrWhiteSpace([string]$Pin.value)) {
    return [string]$Pin.value
  }

  return $Fallback
}

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1")
}

$metaMap = ConvertFrom-PinMap -Value $MetaPackageVersionMap -Name "MetaPackageVersionMap"
$bridgeMap = ConvertFrom-PinMap -Value $BridgePackageVersionMap -Name "BridgePackageVersionMap"
$cudaCudnnMap = ConvertFrom-PinMap -Value $CudaCudnnPackageVersionMap -Name "CudaCudnnPackageVersionMap"
$cudaCudnnReleaseTagMap = ConvertFrom-PinMap -Value $CudaCudnnPackageReleaseTagMap -Name "CudaCudnnPackageReleaseTagMap"
$tensorRtMap = ConvertFrom-PinMap -Value $TensorRtPackageVersionMap -Name "TensorRtPackageVersionMap"
$tensorRtReleaseTagMap = ConvertFrom-PinMap -Value $TensorRtPackageReleaseTagMap -Name "TensorRtPackageReleaseTagMap"

$metaPin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $MetaPackageVersion -Map $metaMap
$bridgePin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $BridgePackageVersion -Map $bridgeMap
$cudaCudnnPin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $CudaCudnnPackageVersion -Map $cudaCudnnMap
$cudaCudnnTagPin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $CudaCudnnPackageReleaseTag -Map $cudaCudnnReleaseTagMap
$tensorRtPin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $TensorRtPackageVersion -Map $tensorRtMap
$tensorRtTagPin = Resolve-PinValue -RuntimeKey $SourceRuntimeKey -Value $TensorRtPackageReleaseTag -Map $tensorRtReleaseTagMap
$resolvedCudaCudnnTag = Resolve-ReleaseTag -TagPin $cudaCudnnTagPin -VersionPin $cudaCudnnPin
$resolvedTensorRtTag = Resolve-ReleaseTag -TagPin $tensorRtTagPin -VersionPin $tensorRtPin

[pscustomobject]@{
  runtimeKey = $SourceRuntimeKey
  version = $Version
  metaPackageVersion = Resolve-PackageVersionOrFallback -Pin $metaPin -Fallback $Version
  metaPackageVersionProvided = [bool]$metaPin.provided
  metaPackageVersionSource = [string]$metaPin.source
  bridgePackageVersion = Resolve-PackageVersionOrFallback -Pin $bridgePin -Fallback $Version
  bridgePackageVersionProvided = [bool]$bridgePin.provided
  bridgePackageVersionSource = [string]$bridgePin.source
  cudaCudnnPackageVersion = Resolve-PackageVersionOrFallback -Pin $cudaCudnnPin -Fallback $Version
  cudaCudnnPackageVersionProvided = [bool]$cudaCudnnPin.provided
  cudaCudnnPackageVersionSource = [string]$cudaCudnnPin.source
  cudaCudnnPackageReleaseTag = [string]$resolvedCudaCudnnTag.value
  cudaCudnnPackageReleaseTagProvided = [bool]$resolvedCudaCudnnTag.provided
  cudaCudnnPackageReleaseTagSource = [string]$resolvedCudaCudnnTag.source
  tensorRtPackageVersion = Resolve-PackageVersionOrFallback -Pin $tensorRtPin -Fallback $Version
  tensorRtPackageVersionProvided = [bool]$tensorRtPin.provided
  tensorRtPackageVersionSource = [string]$tensorRtPin.source
  tensorRtPackageReleaseTag = [string]$resolvedTensorRtTag.value
  tensorRtPackageReleaseTagProvided = [bool]$resolvedTensorRtTag.provided
  tensorRtPackageReleaseTagSource = [string]$resolvedTensorRtTag.source
} | ConvertTo-Json -Depth 5 -Compress
