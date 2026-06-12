[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @("win-x64-trt11.0-cuda12.9-cudnn9.22"),
  [string]$Version = "4.0.0",
  [string]$Configuration = "Release",
  [switch]$SkipManagedPack,
  [switch]$SkipNativeBuild,
  [switch]$SkipRuntimePack,
  [switch]$SkipConsumerValidation,
  [switch]$RunSmoke,
  [switch]$SignConsumerOutput,
  [switch]$TrustConsumerSigningCertificate,
  [switch]$TrustConsumerSigningCertificateRoot,
  [string]$CertificateThumbprint,
  [string]$CertificateSubject = "CN=JYPPX TensorRtSharp Local Dev Code Signing",
  [string]$SigntoolPath,
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

function Invoke-CheckedCommand {
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

function Get-PlatformName {
  try {
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
      return "windows"
    }

    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
      return "linux"
    }
  }
  catch {
  }

  if ($env:OS -eq "Windows_NT") {
    return "windows"
  }

  return "unknown"
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$hostPlatform = Get-PlatformName
$runtimeKeys = @(Expand-KeyList -Values $RuntimePackageKey)
if ($runtimeKeys.Count -eq 0) {
  throw "At least one runtime package key is required."
}

$summaryRoot = Join-Path $RepositoryRoot "artifacts\local-runtime-validation"
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null

if ($SkipRuntimePack.IsPresent -and -not $SkipConsumerValidation.IsPresent) {
  throw "SkipRuntimePack requires SkipConsumerValidation because no runtime nupkg will be produced."
}

if (-not $SkipManagedPack.IsPresent) {
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "pack",
    (Join-Path $RepositoryRoot "pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj"),
    "-c",
    $Configuration,
    "-o",
    (Join-Path $RepositoryRoot "artifacts\managed"),
    "-p:JYPPXPackageVersion=$resolvedVersion"
  )

  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-ManagedPackageContent.ps1")
  )
}

$results = New-Object System.Collections.Generic.List[object]

foreach ($key in $runtimeKeys) {
  $package = $manifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $package) {
    throw "Runtime package key '$key' was not found."
  }

  if ($package.platform -ne $hostPlatform) {
    throw "Runtime package '$key' targets platform '$($package.platform)', but the current host is '$hostPlatform'."
  }

  $resolvedRoots = & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $key -RepositoryRoot $RepositoryRoot | ConvertFrom-Json
  $runtimeProjectPath = Join-Path $RepositoryRoot "pack\runtime\$key\$($package.packageId).csproj"
  $runtimeOutputRoot = Join-Path $RepositoryRoot "artifacts\runtime\$key"
  $runtimePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"

  if ($package.platform -eq "windows") {
    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Validate-WindowsRuntimeInputs.ps1"),
      "-RuntimePackageKey",
      $key,
      "-TensorRtRoot",
      $resolvedRoots.tensorRtRoot,
      "-CudaRoot",
      $resolvedRoots.cudaRoot,
      "-CudnnRoot",
      $resolvedRoots.cudnnRoot
    )

    if (-not $SkipNativeBuild.IsPresent) {
      Invoke-CheckedCommand -FilePath "cmake" -ArgumentList @(
        "--preset",
        $package.buildPreset,
        "-DJYPPX_TENSORRT_ROOT=$($resolvedRoots.tensorRtRoot)",
        "-DCUDAToolkit_ROOT=$($resolvedRoots.cudaRoot)",
        "-DJYPPX_CUDA_ROOT=$($resolvedRoots.cudaRoot)",
        "-DJYPPX_CUDNN_ROOT=$($resolvedRoots.cudnnRoot)"
      )

      Invoke-CheckedCommand -FilePath "cmake" -ArgumentList @(
        "--build",
        "--preset",
        $package.buildPreset
      )
    }

    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Collect-RuntimeAssets.ps1"),
      "-RuntimePackageKey",
      $key,
      "-TensorRtRoot",
      $resolvedRoots.tensorRtRoot,
      "-CudaRoot",
      $resolvedRoots.cudaRoot,
      "-CudnnRoot",
      $resolvedRoots.cudnnRoot,
      "-RepositoryRoot",
      $RepositoryRoot
    )
  }
  elseif ($package.platform -eq "linux") {
    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Validate-LinuxRuntimeInputs.ps1"),
      "-RuntimePackageKey",
      $key,
      "-TensorRtRoot",
      $resolvedRoots.tensorRtRoot,
      "-CudaRoot",
      $resolvedRoots.cudaRoot
    )

    if (-not $SkipNativeBuild.IsPresent) {
      Invoke-CheckedCommand -FilePath "cmake" -ArgumentList @(
        "--preset",
        $package.buildPreset,
        "-DJYPPX_TENSORRT_ROOT=$($resolvedRoots.tensorRtRoot)",
        "-DCUDAToolkit_ROOT=$($resolvedRoots.cudaRoot)",
        "-DJYPPX_CUDA_ROOT=$($resolvedRoots.cudaRoot)",
        "-DJYPPX_CUDNN_ROOT=$($resolvedRoots.cudnnRoot)"
      )

      Invoke-CheckedCommand -FilePath "cmake" -ArgumentList @(
        "--build",
        "--preset",
        $package.buildPreset
      )
    }

    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Invoke-LinuxRuntimeDryRun.ps1"),
      "-RuntimePackageKey",
      $key,
      "-TensorRtRoot",
      $resolvedRoots.tensorRtRoot,
      "-CudaRoot",
      $resolvedRoots.cudaRoot
    )

    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Collect-RuntimeAssets.ps1"),
      "-RuntimePackageKey",
      $key,
      "-TensorRtRoot",
      $resolvedRoots.tensorRtRoot,
      "-CudaRoot",
      $resolvedRoots.cudaRoot,
      "-CudnnRoot",
      $resolvedRoots.cudnnRoot,
      "-RepositoryRoot",
      $RepositoryRoot
    )
  }
  else {
    throw "Unsupported runtime platform '$($package.platform)' for package '$key'."
  }

  if (-not $SkipRuntimePack.IsPresent) {
    Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
      "pack",
      $runtimeProjectPath,
      "-c",
      $Configuration,
      "-o",
      $runtimePackageDirectory,
      "-p:JYPPXPackageVersion=$resolvedVersion"
    )
  }

  if (-not $SkipConsumerValidation.IsPresent) {
    $consumerArguments = @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Test-PackageConsumer.ps1"),
      "-RuntimePackageKey",
      $key,
      "-ManagedPackageDirectory",
      (Join-Path $RepositoryRoot "artifacts\managed"),
      "-RuntimePackageDirectory",
      $runtimePackageDirectory
    )

    if ($RunSmoke.IsPresent) {
      $consumerArguments += "-RunSmoke"
    }

    if ($SignConsumerOutput.IsPresent) {
      $consumerArguments += "-SignConsumerOutput"
    }

    if ($TrustConsumerSigningCertificate.IsPresent -or $TrustConsumerSigningCertificateRoot.IsPresent) {
      $consumerArguments += "-TrustSigningCertificate"
    }

    if ($TrustConsumerSigningCertificateRoot.IsPresent) {
      $consumerArguments += "-TrustSigningCertificateRoot"
    }

    if (-not [string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
      $consumerArguments += @("-CertificateThumbprint", $CertificateThumbprint)
    }

    if (-not [string]::IsNullOrWhiteSpace($CertificateSubject)) {
      $consumerArguments += @("-CertificateSubject", $CertificateSubject)
    }

    if (-not [string]::IsNullOrWhiteSpace($SigntoolPath)) {
      $consumerArguments += @("-SigntoolPath", $SigntoolPath)
    }

    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList $consumerArguments
  }

  $runtimePackageFile = $null
  if (-not $SkipRuntimePack.IsPresent) {
    $runtimePackageFile = Get-ChildItem -LiteralPath $runtimePackageDirectory -Filter "$($package.packageId).$resolvedVersion.nupkg" |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if (-not $runtimePackageFile) {
      throw "The packed runtime nupkg was not found for '$key'."
    }
  }

  $result = [pscustomobject]@{
    runtimeKey = $key
    packageId = $package.packageId
    version = $resolvedVersion
    platform = $package.platform
    buildPreset = $package.buildPreset
    runtimeOutputRoot = $runtimeOutputRoot
    runtimePackagePath = if ($runtimePackageFile) { $runtimePackageFile.FullName } else { $null }
    runtimePackageSizeMb = if ($runtimePackageFile) { [Math]::Round($runtimePackageFile.Length / 1MB, 2) } else { $null }
    fitsGithubNugetRegistry = if ($runtimePackageFile) { ($runtimePackageFile.Length -lt 2147000000) } else { $null }
    fitsGithubReleaseAsset = if ($runtimePackageFile) { ($runtimePackageFile.Length -lt 2GB) } else { $null }
    skippedRuntimePack = $SkipRuntimePack.IsPresent
    ranSmoke = $RunSmoke.IsPresent
    tensorRtRoot = $resolvedRoots.tensorRtRoot
    cudaRoot = $resolvedRoots.cudaRoot
    cudnnRoot = $resolvedRoots.cudnnRoot
  }

  $results.Add($result)
}

$jsonPath = Join-Path $summaryRoot "local-runtime-validation-summary.json"
$markdownPath = Join-Path $summaryRoot "local-runtime-validation-summary.md"

$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Local Runtime Validation Summary")
$lines.Add("")
$lines.Add("| Runtime key | Package | Size (MB) | GitHub NuGet | GitHub Release | Smoke |")
$lines.Add("| --- | --- | ---: | --- | --- | --- |")
foreach ($result in $results) {
  $packageLabel = '`' + $result.packageId + ' ' + $result.version + '`'
  $sizeLabel = if ($null -eq $result.runtimePackageSizeMb) { "skipped" } else { [string]$result.runtimePackageSizeMb }
  $nugetLabel = if ($null -eq $result.fitsGithubNugetRegistry) { "skipped" } else { [string]$result.fitsGithubNugetRegistry }
  $releaseLabel = if ($null -eq $result.fitsGithubReleaseAsset) { "skipped" } else { [string]$result.fitsGithubReleaseAsset }
  $lines.Add([string]::Format(
      "| {0} | {1} | {2} | {3} | {4} | {5} |",
      $result.runtimeKey,
      $packageLabel,
      $sizeLabel,
      $nugetLabel,
      $releaseLabel,
      $result.ranSmoke))
}
$lines.Add("")
$lines.Add("Generated by `eng/Invoke-LocalRuntimePackage.ps1`.")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Local runtime validation summary written to $jsonPath"
Write-Host "Local runtime validation summary written to $markdownPath"
