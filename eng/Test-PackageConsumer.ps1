[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @("win-x64-trt8.6-cuda11.8-cudnn8.9"),
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$RuntimePackageDirectory,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$RunSmoke,
  [string[]]$SmokeRuntimePackageKey = @(),
  [switch]$SignConsumerOutput,
  [string]$CertificateThumbprint,
  [string]$CertificateSubject = "CN=JYPPX TensorRtSharp Local Dev Code Signing",
  [string]$SigntoolPath,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}

if ([string]::IsNullOrWhiteSpace($RuntimePackageDirectory)) {
  $RuntimePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "build-out\package-consumer"
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\package-consumer"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

Add-Type -AssemblyName System.IO.Compression.FileSystem

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

function Join-PathMany {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Parts
  )

  if ($Parts.Count -eq 0) {
    throw "At least one path part is required."
  }

  $path = $Parts[0]
  for ($i = 1; $i -lt $Parts.Count; $i++) {
    $path = Join-Path $path $Parts[$i]
  }

  return $path
}

function Get-NupkgMetadata {
  param(
    [string]$Path
  )

  $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
    if (-not $nuspec) {
      throw "Package does not contain a nuspec: $Path"
    }

    $stream = $nuspec.Open()
    try {
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
      try {
        [xml]$xml = $reader.ReadToEnd()
      }
      finally {
        $reader.Dispose()
      }
    }
    finally {
      $stream.Dispose()
    }

    $namespaceManager = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace("n", $xml.package.NamespaceURI)
    $id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
    $version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
    return [pscustomobject]@{
      Path = $Path
      Id = $id
      Version = $version
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Package {
  param(
    [string]$Directory,
    [string]$PackageId
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }

  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($package in Get-ChildItem -LiteralPath $Directory -Filter *.nupkg) {
    $metadata = Get-NupkgMetadata -Path $package.FullName
    if ($metadata.Id -eq $PackageId) {
      $matches.Add($metadata)
    }
  }

  if ($matches.Count -eq 0) {
    throw "Package '$PackageId' was not found under $Directory."
  }

  return @($matches | Sort-Object Version -Descending)[0]
}

function Invoke-CheckedDotNet {
  param(
    [string[]]$Arguments
  )

  $dotnetOutput = & dotnet @Arguments 2>&1
  foreach ($line in @($dotnetOutput)) {
    Write-Host $line
  }

  if ($LASTEXITCODE -ne 0) {
    throw "dotnet $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
  }
}

function Find-Signtool {
  param(
    [string]$PreferredPath
  )

  if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
    if (-not (Test-Path -LiteralPath $PreferredPath -PathType Leaf)) {
      throw "signtool.exe was not found at the specified path: $PreferredPath"
    }

    return (Resolve-Path -LiteralPath $PreferredPath).Path
  }

  $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $kitsRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
  if (Test-Path -LiteralPath $kitsRoot -PathType Container) {
    $matches = @(Get-ChildItem -LiteralPath $kitsRoot -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
      Sort-Object FullName -Descending)
    if ($matches.Count -gt 0) {
      return $matches[0].FullName
    }
  }

  throw "signtool.exe was not found. Install Windows SDK or pass -SigntoolPath."
}

function Get-OrCreate-CodeSigningCertificate {
  param(
    [string]$Thumbprint,
    [string]$Subject
  )

  if (-not [string]::IsNullOrWhiteSpace($Thumbprint)) {
    $normalizedThumbprint = ($Thumbprint -replace "\s", "").ToUpperInvariant()
    $certificate = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
      Where-Object { $_.Thumbprint -eq $normalizedThumbprint } |
      Select-Object -First 1
    if (-not $certificate) {
      throw "Code signing certificate was not found in Cert:\CurrentUser\My: $normalizedThumbprint"
    }

    return $certificate
  }

  $certificate = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
    Where-Object { $_.Subject -eq $Subject } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1
  if ($certificate) {
    return $certificate
  }

  $newSelfSignedCertificate = Get-Command New-SelfSignedCertificate -ErrorAction SilentlyContinue
  if (-not $newSelfSignedCertificate) {
    throw "No matching code signing certificate was found and New-SelfSignedCertificate is unavailable."
  }

  Write-Host "Creating local development code signing certificate: $Subject"
  return New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject $Subject `
    -CertStoreLocation Cert:\CurrentUser\My `
    -KeyExportPolicy Exportable `
    -KeyUsage DigitalSignature `
    -NotAfter (Get-Date).AddYears(5)
}

function Sign-ConsumerOutput {
  param(
    [string]$OutputDirectory
  )

  $resolvedSigntoolPath = Find-Signtool -PreferredPath $SigntoolPath
  $certificate = Get-OrCreate-CodeSigningCertificate -Thumbprint $CertificateThumbprint -Subject $CertificateSubject
  $candidates = @(Get-ChildItem -LiteralPath $OutputDirectory -File -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -like "JYPPX*.dll" -or
      $_.Name -eq "PackageConsumerSmoke.dll" -or
      $_.Name -eq "PackageConsumerSmoke.exe" -or
      $_.Name -eq "jyppxtrtbridge.dll"
    })

  if ($candidates.Count -eq 0) {
    throw "No consumer output files were found for signing under $OutputDirectory."
  }

  foreach ($file in $candidates) {
    Write-Host "Signing consumer output: $($file.Name)"
    $signOutput = & $resolvedSigntoolPath sign /fd SHA256 /sha1 $certificate.Thumbprint /tr http://timestamp.digicert.com /td SHA256 $file.FullName 2>&1
    foreach ($line in @($signOutput)) {
      Write-Host $line
    }

    if ($LASTEXITCODE -ne 0) {
      throw "signtool failed for '$($file.FullName)' with exit code $LASTEXITCODE."
    }
  }

  return [int]$candidates.Count
}

function Unblock-ConsumerRuntimeAssets {
  param(
    [string]$OutputDirectory,
    [string[]]$FileNames
  )

  $unblockCommand = Get-Command Unblock-File -ErrorAction SilentlyContinue
  if (-not $unblockCommand) {
    return
  }

  foreach ($fileName in @($FileNames)) {
    if ([string]::IsNullOrWhiteSpace($fileName)) {
      continue
    }

    foreach ($match in @(Get-ChildItem -LiteralPath $OutputDirectory -Recurse -Filter $fileName -File -ErrorAction SilentlyContinue)) {
      try {
        Unblock-File -LiteralPath $match.FullName -ErrorAction Stop
      }
      catch {
        Write-Warning "Unable to unblock consumer runtime asset '$($match.FullName)': $($_.Exception.Message)"
      }
    }
  }
}

function Get-ExpectedNativeFileNames {
  param(
    [object]$RuntimePackage
  )

  $expectedNativeFiles = @($RuntimePackage.bridgeFile)
  foreach ($relativePath in @($RuntimePackage.tensorRtFiles + $RuntimePackage.cudaFiles + $RuntimePackage.cudnnFiles)) {
    if ([string]::IsNullOrWhiteSpace([string]$relativePath)) {
      continue
    }

    $expectedNativeFiles += [System.IO.Path]::GetFileName($relativePath)
  }

  return @($expectedNativeFiles | Sort-Object -Unique)
}

function Write-ValidationReports {
  param(
    [object[]]$Results,
    [string]$Directory
  )

  New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  $jsonPath = Join-Path $Directory "package-consumer-validation-summary.json"
  $markdownPath = Join-Path $Directory "package-consumer-validation-summary.md"

  $Results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Package Consumer Validation Summary")
  $lines.Add("")
  $lines.Add("| Runtime key | Package | Native assets | Missing native assets | Smoke | Signed consumer output | Elapsed |")
  $lines.Add("| --- | --- | ---: | --- | --- | --- | ---: |")
  foreach ($result in $Results) {
    $missing = if ($result.MissingNativeAssets.Count -eq 0) { "none" } else { ($result.MissingNativeAssets -join ", ") }
    $package = '`' + $result.RuntimePackageId + ' ' + $result.RuntimePackageVersion + '`'
    $lines.Add("| $($result.RuntimePackageKey) | $package | $($result.NativeAssetsFound)/$($result.NativeAssetsExpected) | $missing | $($result.SmokeResult) | $($result.ConsumerOutputSigned) | $($result.ElapsedSeconds)s |")
  }

  $lines.Add("")
  $lines.Add('Generated by `eng/Test-PackageConsumer.ps1`.')
  Set-Content -LiteralPath $markdownPath -Value $lines -Encoding utf8

  Write-Host "Package consumer summary written to $jsonPath"
  Write-Host "Package consumer summary written to $markdownPath"
}

function Remove-ConsumerDirectory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  $lastError = $null
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  if ($fullPath.StartsWith("\\", [System.StringComparison]::Ordinal)) {
    $extendedPath = "\\?\UNC\" + $fullPath.Substring(2)
  }
  else {
    $extendedPath = "\\?\" + $fullPath
  }

  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      $lastError = $_
    }

    try {
      # Long runtime package ids can push the local NuGet cache path near the
      # legacy MAX_PATH boundary on Windows PowerShell. Fall back to an extended
      # path delete after releasing handles from the previous restore/build.
      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()

      if ($attempt -eq 1) {
        try {
          & dotnet build-server shutdown *> $null
        }
        catch {
          # Build server shutdown is best-effort only; deletion retries below are authoritative.
        }
      }

      [System.IO.Directory]::Delete($extendedPath, $true)
      return
    }
    catch {
      $lastError = $_
      Start-Sleep -Milliseconds (250 * $attempt)
    }

    if (-not (Test-Path -LiteralPath $Path)) {
      return
    }
  }

  if (Test-Path -LiteralPath $Path) {
    $reason = if ($lastError) { $lastError.Exception.Message } else { "unknown error" }
    throw "Unable to remove package consumer directory: $Path. Last error: $reason"
  }
}

function Invoke-PackageConsumerValidation {
  param(
    [string]$Key,
    [object]$RuntimePackage,
    [object]$ManagedPackage,
    [object]$RuntimeNupkg,
    [bool]$ShouldRunSmoke
  )

  $timer = [System.Diagnostics.Stopwatch]::StartNew()
  $consumerRoot = Join-Path $OutputRoot $Key
  $resolvedConsumerRoot = [System.IO.Path]::GetFullPath($consumerRoot)
  $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
  if (-not $resolvedConsumerRoot.StartsWith($resolvedOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean consumer path outside output root: $resolvedConsumerRoot"
  }

  if (Test-Path -LiteralPath $resolvedConsumerRoot) {
    Remove-ConsumerDirectory -Path $resolvedConsumerRoot
  }

  New-Item -ItemType Directory -Path $resolvedConsumerRoot -Force | Out-Null

  $nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed" value="$ManagedPackageDirectory" />
    <add key="jyppx-runtime" value="$RuntimePackageDirectory" />
  </packageSources>
</configuration>
"@

  $project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>$($RuntimePackage.rid)</RuntimeIdentifier>
    <RestorePackagesPath>`$(MSBuildProjectDirectory)\.nuget\packages</RestorePackagesPath>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="$($ManagedPackage.Id)" Version="$($ManagedPackage.Version)" />
    <PackageReference Include="$($RuntimeNupkg.Id)" Version="$($RuntimeNupkg.Version)" />
  </ItemGroup>
</Project>
"@

  $program = @"
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;

Console.WriteLine("TensorRtAssemblyBridge=" + TensorRtSharpInfo.NativeBridgeLibraryName);
Console.WriteLine("CudaAssemblyBridge=" + CudaSharpInfo.NativeBridgeLibraryName);

var tensorRt = TensorRtEnvironmentProbe.GetCurrent();
Console.WriteLine("Bridge=" + tensorRt.BuildInfo.BridgeName + " TRT=" + tensorRt.BuildInfo.TensorRtVersion + " CUDA=" + tensorRt.BuildInfo.CudaToolkitVersion);

var cuda = CudaEnvironmentProbe.GetCurrent();
Console.WriteLine("CudaDevices=" + cuda.CudaRuntimeInfo.DeviceCount + " Vendor=" + cuda.CudaRuntimeInfo.VendorDependencyAvailable);
"@

  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "NuGet.config") -Value $nugetConfig -Encoding utf8
  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj") -Value $project -Encoding utf8
  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "Program.cs") -Value $program -Encoding utf8

  Invoke-CheckedDotNet -Arguments @("restore", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "--configfile", (Join-Path $resolvedConsumerRoot "NuGet.config"))
  Invoke-CheckedDotNet -Arguments @("build", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "-c", "Release", "--no-restore")

  $outputDirectory = Join-PathMany -Parts @($resolvedConsumerRoot, "bin", "Release", $TargetFramework, $RuntimePackage.rid)
  if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    throw "Consumer output directory was not found: $outputDirectory"
  }

  $expectedManagedAssemblies = @("JYPPX.Shared.dll", "JYPPX.TensorRtSharp.dll", "JYPPX.CudaSharp.dll")
  foreach ($assembly in $expectedManagedAssemblies) {
    $matches = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $assembly)
    if ($matches.Count -eq 0) {
      throw "Expected managed assembly was not copied to the consumer output: $assembly"
    }
  }

  $expectedNativeFiles = @(Get-ExpectedNativeFileNames -RuntimePackage $RuntimePackage)
  $missingNativeFiles = New-Object System.Collections.Generic.List[string]
  $foundNativeFileCount = 0
  foreach ($fileName in $expectedNativeFiles) {
    $matches = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $fileName)
    if ($matches.Count -eq 0) {
      $missingNativeFiles.Add($fileName)
    }
    else {
      $foundNativeFileCount++
    }
  }

  if ($missingNativeFiles.Count -gt 0) {
    throw "Expected native runtime assets were not copied to the consumer output for ${Key}: $($missingNativeFiles -join ', ')"
  }

  Unblock-ConsumerRuntimeAssets -OutputDirectory $outputDirectory -FileNames $expectedNativeFiles
  $signedConsumerOutputCount = 0
  if ($SignConsumerOutput.IsPresent) {
    $signedConsumerOutputCount = Sign-ConsumerOutput -OutputDirectory $outputDirectory
  }
  Unblock-ConsumerRuntimeAssets -OutputDirectory $outputDirectory -FileNames @(
    $expectedNativeFiles +
    $expectedManagedAssemblies +
    @("PackageConsumerSmoke.dll", "PackageConsumerSmoke.exe")
  )

  $smokeResult = "not-requested"
  if ($ShouldRunSmoke) {
    Invoke-CheckedDotNet -Arguments @("run", "--project", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "-c", "Release", "--no-build")
    $smokeResult = "passed"
  }

  $timer.Stop()
  $elapsedSeconds = [Math]::Round($timer.Elapsed.TotalSeconds, 2)

  Write-Host "Package consumer validation passed for $Key."
  Write-Host "  Managed package: $($ManagedPackage.Id) $($ManagedPackage.Version)"
  Write-Host "  Runtime package: $($RuntimeNupkg.Id) $($RuntimeNupkg.Version)"
  Write-Host "  Native assets: $foundNativeFileCount/$($expectedNativeFiles.Count)"
  Write-Host "  Smoke: $smokeResult"
  Write-Host "  Signed consumer output: $signedConsumerOutputCount"
  Write-Host "  Elapsed: ${elapsedSeconds}s"
  Write-Host "  Consumer output: $outputDirectory"

  return [pscustomobject]@{
    RuntimePackageKey = $Key
    RuntimePackageId = $RuntimeNupkg.Id
    RuntimePackageVersion = $RuntimeNupkg.Version
    RuntimePackagePath = $RuntimeNupkg.Path
    ManagedPackageId = $ManagedPackage.Id
    ManagedPackageVersion = $ManagedPackage.Version
    TargetFramework = $TargetFramework
    RuntimeIdentifier = $RuntimePackage.rid
    DistributionTier = $RuntimePackage.distributionTier
    ValidationState = $RuntimePackage.validationState
    NativeAssetsExpected = $expectedNativeFiles.Count
    NativeAssetsFound = $foundNativeFileCount
    MissingNativeAssets = @($missingNativeFiles.ToArray())
    SmokeResult = $smokeResult
    ConsumerOutputSigned = $SignConsumerOutput.IsPresent
    ConsumerOutputSignedFileCount = $signedConsumerOutputCount
    ElapsedSeconds = $elapsedSeconds
    ConsumerOutput = $outputDirectory
  }
}

$keys = @(Expand-KeyList -Values $RuntimePackageKey)
if ($keys.Count -eq 0) {
  throw "At least one runtime package key is required."
}

$smokeKeys = @(Expand-KeyList -Values $SmokeRuntimePackageKey)
$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
$results = New-Object System.Collections.Generic.List[object]

foreach ($key in $keys) {
  $runtimePackage = $manifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Runtime package key '$key' was not found."
  }

  $runtimeNupkg = Find-Package -Directory $RuntimePackageDirectory -PackageId $runtimePackage.packageId
  $shouldRunSmoke = $RunSmoke.IsPresent -and ($smokeKeys.Count -eq 0 -or $smokeKeys -contains $key)
  $results.Add((Invoke-PackageConsumerValidation -Key $key -RuntimePackage $runtimePackage -ManagedPackage $managedPackage -RuntimeNupkg $runtimeNupkg -ShouldRunSmoke $shouldRunSmoke))
}

Write-ValidationReports -Results @($results.ToArray()) -Directory $ReportDirectory
