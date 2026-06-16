[CmdletBinding()]
param(
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
if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Manifest was not found: $manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (-not $manifest.packages) {
  throw "The runtime manifest does not contain any packages."
}

$requiredCommon = @("key", "packageId", "rid", "platform", "tensorRtLine", "cudaLine", "tensorRtVersion", "cudaVersion", "cudnnMajor", "cudnnVersion", "distributionTier", "validationState", "distributionNotes", "buildPreset", "bridgeConfiguration", "bridgeFile")
$allowedDistributionTiers = @("public-sample", "private-feed", "split-delivery-candidate")
$allowedValidationStates = @("local-validated", "pending-local-validation", "dry-run-only")
$allowedLinuxRunnerModes = @("hosted", "self-hosted")
$allowedLinuxArchitectures = @("x64", "arm64")
$errors = New-Object System.Collections.Generic.List[string]

function Get-MajorMinorVersionLabel {
  param(
    [string]$Version,
    [string]$PropertyName,
    [string]$PackageKey
  )

  if ($Version -match "^(\d+)\.(\d+)") {
    return "$($Matches[1]).$($Matches[2])"
  }

  $errors.Add("Package '$PackageKey' has invalid $PropertyName '$Version'. Use an exact version that contains at least major.minor.")
  return $null
}

foreach ($package in $manifest.packages) {
  foreach ($property in $requiredCommon) {
    if (-not $package.PSObject.Properties.Name.Contains($property) -or [string]::IsNullOrWhiteSpace([string]$package.$property)) {
      $errors.Add("Package '$($package.key)' is missing required property '$property'.")
    }
  }

  if ($package.PSObject.Properties.Name.Contains("distributionTier") -and $package.distributionTier -notin $allowedDistributionTiers) {
    $errors.Add("Package '$($package.key)' has unsupported distributionTier '$($package.distributionTier)'.")
  }

  if ($package.PSObject.Properties.Name.Contains("validationState") -and $package.validationState -notin $allowedValidationStates) {
    $errors.Add("Package '$($package.key)' has unsupported validationState '$($package.validationState)'.")
  }

  if ($package.key -match "^(win|linux)-x64-trt(8|10|11)-cuda(11|12|13)$") {
    $errors.Add("Package '$($package.key)' uses an ambiguous runtime key. Include the exact CUDA minor version, for example cuda11.8 or cuda13.2.")
  }

  if ($package.packageId -match "\.cuda(11|12|13)$") {
    $errors.Add("Package '$($package.key)' uses an ambiguous packageId '$($package.packageId)'. Include the exact CUDA minor version.")
  }

  $tensorRtLabel = Get-MajorMinorVersionLabel -Version ([string]$package.tensorRtVersion) -PropertyName "tensorRtVersion" -PackageKey $package.key
  $cudaLabel = Get-MajorMinorVersionLabel -Version ([string]$package.cudaVersion) -PropertyName "cudaVersion" -PackageKey $package.key
  $cudnnLabel = Get-MajorMinorVersionLabel -Version ([string]$package.cudnnVersion) -PropertyName "cudnnVersion" -PackageKey $package.key
  if ($tensorRtLabel -and $cudaLabel -and $cudnnLabel) {
    $expectedKeyFragment = "trt$tensorRtLabel-cuda$cudaLabel-cudnn$cudnnLabel"
    if ($package.key -notlike "*$expectedKeyFragment*") {
      $errors.Add("Package '$($package.key)' must include dependency version fragment '$expectedKeyFragment'.")
    }

    $expectedPackageIdFragment = "trt$tensorRtLabel.cuda$cudaLabel.cudnn$cudnnLabel"
    if ($package.packageId -notlike "*$expectedPackageIdFragment*") {
      $errors.Add("Package '$($package.key)' packageId '$($package.packageId)' must include dependency version fragment '$expectedPackageIdFragment'.")
    }
  }

  if ($package.cudaVersion -eq "12.9" -and $package.PSObject.Properties.Name.Contains("localBuildCudaVersion") -and $package.localBuildCudaVersion -ne "12.9") {
    if ($package.validationState -eq "local-validated") {
      $errors.Add("Package '$($package.key)' targets CUDA 12.9 but localBuildCudaVersion is '$($package.localBuildCudaVersion)'; it must not be local-validated until CUDA 12.9 is installed and used.")
    }
  }

  if ($package.platform -eq "windows") {
    foreach ($property in @("defaultTensorRtRoot", "defaultCudaRoot")) {
      if ($package.PSObject.Properties.Name.Contains($property) -and $package.$property -match "^[A-Za-z]:\\") {
        $errors.Add("Windows package '$($package.key)' must not store local absolute path '$property' in runtime-packages.manifest.json. Use runtime-packages.local.json instead.")
      }
    }
  }

  if ($package.platform -eq "linux") {
    foreach ($property in @("defaultTensorRtRoot", "defaultCudaRoot", "linuxDistro", "linuxDistroVersion", "architecture", "nvidiaRepoDistroId", "nvidiaRepoArchitecture", "runnerMode")) {
      if (-not $package.PSObject.Properties.Name.Contains($property) -or [string]::IsNullOrWhiteSpace([string]$package.$property)) {
        $errors.Add("Linux package '$($package.key)' is missing '$property'.")
      }
    }

    if ($package.key -match "^linux-(x64|arm64)-trt") {
      $errors.Add("Linux package '$($package.key)' is ambiguous. Include the distribution version, for example linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22.")
    }

    if ($package.packageId -match "\.Runtime\.linux-(x64|arm64)\.trt") {
      $errors.Add("Linux package '$($package.key)' packageId '$($package.packageId)' is ambiguous. Include the distribution version, for example .linux-x64.ubuntu22.04.trt11.0.")
    }

    if ($package.PSObject.Properties.Name.Contains("linuxDistro") -and $package.linuxDistro -ne "ubuntu") {
      $errors.Add("Linux package '$($package.key)' uses unsupported linuxDistro '$($package.linuxDistro)'. Add explicit preparation logic before enabling it.")
    }

    if ($package.PSObject.Properties.Name.Contains("architecture") -and $package.architecture -notin $allowedLinuxArchitectures) {
      $errors.Add("Linux package '$($package.key)' uses unsupported architecture '$($package.architecture)'.")
    }

    if ($package.PSObject.Properties.Name.Contains("runnerMode") -and $package.runnerMode -notin $allowedLinuxRunnerModes) {
      $errors.Add("Linux package '$($package.key)' uses unsupported runnerMode '$($package.runnerMode)'.")
    }

    if ($package.PSObject.Properties.Name.Contains("runnerLabels")) {
      $runnerLabels = @($package.runnerLabels | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
      if ($runnerLabels.Count -eq 0) {
        $errors.Add("Linux package '$($package.key)' has empty runnerLabels.")
      }
    }
    else {
      $errors.Add("Linux package '$($package.key)' is missing runnerLabels.")
    }

    if ($package.runnerMode -eq "hosted") {
      $runnerLabels = @($package.runnerLabels | ForEach-Object { [string]$_ })
      if ($package.linuxDistroVersion -eq "20.04") {
        $errors.Add("Linux package '$($package.key)' targets Ubuntu 20.04 but is marked hosted. GitHub-hosted Ubuntu 20.04 is not available for this runtime workflow.")
      }

      $expectedHostedLabel = "ubuntu-$($package.linuxDistroVersion)"
      if ($runnerLabels -notcontains $expectedHostedLabel) {
        $errors.Add("Linux hosted package '$($package.key)' must include runner label '$expectedHostedLabel'.")
      }
    }

    if (-not ($package.tensorRtFiles | Where-Object { $_ -like "*.so*" })) {
      $errors.Add("Linux package '$($package.key)' must contain at least one TensorRT .so pattern.")
    }

    if (-not ($package.cudaFiles | Where-Object { $_ -like "*.so*" })) {
      $errors.Add("Linux package '$($package.key)' must contain at least one CUDA .so pattern.")
    }
  }
}

if ($errors.Count -gt 0) {
  $errors | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Host "Runtime manifest validation passed for $($manifest.packages.Count) packages."
