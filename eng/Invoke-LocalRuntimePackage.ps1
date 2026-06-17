[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @(),
  [string]$Version = "4.0.0",
  [string]$Configuration = "Release",
  [switch]$SkipManagedPack,
  [switch]$SkipNativeBuild,
  [switch]$SkipRuntimePack,
  [switch]$SkipConsumerValidation,
  [switch]$ResolveOnly,
  [switch]$RunSmoke,
  [string[]]$SmokeRuntimePackageKey = @(),
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
$powerShellCommand = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }

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

function Get-ArchitectureName {
  try {
    switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
      "X64" { return "x64" }
      "Arm64" { return "arm64" }
      default { return ([string][System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture).ToLowerInvariant() }
    }
  }
  catch {
    if ($env:PROCESSOR_ARCHITECTURE -match "^(AMD64|x86_64)$") {
      return "x64"
    }
  }

  return "unknown"
}

function Get-LinuxOsRelease {
  $values = @{}
  $path = "/etc/os-release"
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $values
  }

  foreach ($line in (Get-Content -LiteralPath $path -ErrorAction SilentlyContinue)) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith("#")) {
      continue
    }

    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) {
      continue
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"')
    if (-not [string]::IsNullOrWhiteSpace($name)) {
      $values[$name] = $value
    }
  }

  return $values
}

function Resolve-DefaultRuntimeKeys {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Manifest,
    [Parameter(Mandatory = $true)]
    [string]$Platform
  )

  $architecture = Get-ArchitectureName
  if ($Platform -eq "windows") {
    $keys = @(
      $Manifest.packages |
        Where-Object { $_.platform -eq "windows" -and $_.rid -eq "win-$architecture" } |
        Select-Object -ExpandProperty key
    )

    if ($keys.Count -gt 0) {
      return @($keys)
    }

    throw "No default Windows runtime keys are modeled for architecture '$architecture'. Pass -RuntimePackageKey explicitly."
  }

  if ($Platform -eq "linux") {
    $osRelease = Get-LinuxOsRelease
    $linuxDistro = if ($osRelease.ContainsKey("ID")) { [string]$osRelease["ID"] } else { "" }
    $linuxDistroVersion = if ($osRelease.ContainsKey("VERSION_ID")) { [string]$osRelease["VERSION_ID"] } else { "" }

    if ([string]::IsNullOrWhiteSpace($linuxDistro) -or [string]::IsNullOrWhiteSpace($linuxDistroVersion)) {
      throw "Unable to determine the Linux distribution from /etc/os-release. Pass -RuntimePackageKey explicitly."
    }

    $keys = @(
      $Manifest.packages |
        Where-Object {
          $_.platform -eq "linux" -and
          $_.architecture -eq $architecture -and
          $_.linuxDistro -eq $linuxDistro -and
          $_.linuxDistroVersion -eq $linuxDistroVersion
        } |
        Select-Object -ExpandProperty key
    )

    if ($keys.Count -gt 0) {
      return @($keys)
    }

    throw "No default Linux runtime keys are modeled for '$linuxDistro $linuxDistroVersion $architecture'. Pass -RuntimePackageKey explicitly for future package lines such as ARM/SBSA, Jetson/L4T, or non-Ubuntu targets."
  }

  throw "Unable to choose default runtime keys for host platform '$Platform'. Pass -RuntimePackageKey explicitly."
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$hostPlatform = Get-PlatformName
$runtimeKeys = @(Expand-KeyList -Values $RuntimePackageKey)
$smokeRuntimeKeys = @(Expand-KeyList -Values $SmokeRuntimePackageKey)
if ($runtimeKeys.Count -eq 0) {
  $runtimeKeys = @(Resolve-DefaultRuntimeKeys -Manifest $manifest -Platform $hostPlatform)
  Write-Host "No -RuntimePackageKey was provided. Resolved local host runtime matrix: $($runtimeKeys -join ', ')"
}

$summaryRoot = Join-Path $RepositoryRoot "artifacts\local-runtime-validation"
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null

if ($ResolveOnly.IsPresent) {
  $resolvedRows = @(
    foreach ($key in $runtimeKeys) {
      $package = $manifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
      if (-not $package) {
        throw "Runtime package key '$key' was not found."
      }

      [pscustomobject]@{
        runtimeKey = [string]$package.key
        packageId = [string]$package.packageId
        platform = [string]$package.platform
        rid = [string]$package.rid
        linuxDistro = [string]$package.linuxDistro
        linuxDistroVersion = [string]$package.linuxDistroVersion
        architecture = [string]$package.architecture
        validationState = [string]$package.validationState
      }
    }
  )

  $jsonPath = Join-Path $summaryRoot "local-runtime-key-resolution.json"
  $markdownPath = Join-Path $summaryRoot "local-runtime-key-resolution.md"
  $resolvedRows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = [System.Collections.Generic.List[string]]::new()
  $lines.Add("# Local Runtime Key Resolution")
  $lines.Add("")
  $lines.Add("Host platform: ``$hostPlatform``")
  $lines.Add("")
  $lines.Add("| Runtime key | Package | Platform | Validation |")
  $lines.Add("| --- | --- | --- | --- |")
  foreach ($row in $resolvedRows) {
    $lines.Add("| ``$($row.runtimeKey)`` | ``$($row.packageId)`` | ``$($row.platform)`` | ``$($row.validationState)`` |")
  }
  $lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

  Write-Host "Local runtime key resolution written to $jsonPath"
  Write-Host "Local runtime key resolution written to $markdownPath"
  return
}

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

  Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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
    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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

    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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
    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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

    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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

    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
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

      if ($smokeRuntimeKeys.Count -gt 0) {
        $consumerArguments += @("-SmokeRuntimePackageKey", ($smokeRuntimeKeys -join ","))
      }
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

    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList $consumerArguments
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
