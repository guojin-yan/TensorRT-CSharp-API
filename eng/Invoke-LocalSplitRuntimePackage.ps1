[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "win-x64-trt11.0-cuda12.9-cudnn9.22",
  [string]$Version = "4.0.0",
  [string[]]$SplitPackageRole = @("all"),
  [string]$MetaPackageVersion,
  [string]$BridgePackageVersion,
  [string]$CudaCudnnPackageVersion,
  [string]$TensorRtPackageVersion,
  [string]$Configuration = "Release",
  [switch]$SkipManagedPack,
  [switch]$SkipBaseRuntimeBuild,
  [switch]$IncludeMetaPackage,
  [switch]$SkipConsumerValidation,
  [string[]]$AdditionalPackageSource = @(),
  [string]$AdditionalPackageSourceUsername,
  [string]$AdditionalPackageSourcePassword,
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

function Resolve-RolePackageVersion {
  param(
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [string]$Fallback
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $Fallback
  }

  return & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Value
}

function Get-SplitPackageVersion {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitPackage
  )

  switch -Regex ([string]$SplitPackage.role) {
    '^bridge$' { return $resolvedBridgePackageVersion }
    '^cuda-cudnn$' { return $resolvedCudaCudnnPackageVersion }
    '^tensorrt$' { return $resolvedTensorRtPackageVersion }
    default { return $resolvedVersion }
  }
}

function Test-SplitPackageRequested {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string[]]$RequestedRoles
  )

  $role = ([string]$SplitPackage.role).ToLowerInvariant()
  $key = ([string]$SplitPackage.key).ToLowerInvariant()

  foreach ($requestedRole in $RequestedRoles) {
    switch ($requestedRole) {
      "all" { return $true }
      "stable-dependencies" {
        if ($role -eq "cuda-cudnn" -or $role -eq "tensorrt") {
          return $true
        }
      }
      "nvidia-dependencies" {
        if ($role -eq "cuda-cudnn" -or $role -eq "tensorrt") {
          return $true
        }
      }
      default {
        if ($role -eq $requestedRole -or $key -eq $requestedRole) {
          return $true
        }
      }
    }
  }

  return $false
}

function New-PackageVersionProperties {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageVersion
  )

  return @(
    "-p:JYPPXPackageVersion=$PackageVersion",
    "-p:JYPPXBridgePackageVersion=$resolvedBridgePackageVersion",
    "-p:JYPPXCudaCudnnPackageVersion=$resolvedCudaCudnnPackageVersion",
    "-p:JYPPXTensorRtPackageVersion=$resolvedTensorRtPackageVersion"
  )
}

function ConvertTo-XmlAttributeValue {
  param(
    [string]$Value
  )

  return [System.Security.SecurityElement]::Escape($Value)
}

function Resolve-PackageSourceValue {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source
  )

  if ($Source -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') {
    return $Source
  }

  if ([System.IO.Path]::IsPathRooted($Source)) {
    return [System.IO.Path]::GetFullPath($Source)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Source))
}

function Format-CommandForLog {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$ArgumentList
  )

  $safeArguments = New-Object System.Collections.Generic.List[string]
  $redactNext = $false
  foreach ($argument in $ArgumentList) {
    if ($redactNext) {
      $safeArguments.Add("***")
      $redactNext = $false
      continue
    }

    $safeArguments.Add($argument)
    if ($argument -eq "-AdditionalPackageSourcePassword") {
      $redactNext = $true
    }
  }

  return "$FilePath $($safeArguments -join ' ')"
}

function Invoke-CheckedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$ArgumentList
  )

  $safeCommand = Format-CommandForLog -FilePath $FilePath -ArgumentList $ArgumentList
  Write-Host "> $safeCommand"
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $safeCommand"
  }
}

function Remove-OptionalPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LiteralPath
  )

  if (-not (Test-Path -LiteralPath $LiteralPath)) {
    return
  }

  try {
    Remove-Item -LiteralPath $LiteralPath -Recurse -Force -ErrorAction Stop
    Write-Host "Removed temporary path: $LiteralPath"
  }
  catch {
    if ($env:OS -eq "Windows_NT") {
      try {
        $fullPath = [System.IO.Path]::GetFullPath($LiteralPath)
        $extendedPath = if ($fullPath.StartsWith("\\?\", [System.StringComparison]::Ordinal)) {
          $fullPath
        }
        elseif ($fullPath.StartsWith("\\", [System.StringComparison]::Ordinal)) {
          "\\?\UNC\" + $fullPath.Substring(2)
        }
        else {
          "\\?\" + $fullPath
        }

        if ([System.IO.Directory]::Exists($extendedPath)) {
          [System.IO.Directory]::Delete($extendedPath, $true)
          Write-Host "Removed temporary path: $LiteralPath"
          return
        }

        if ([System.IO.File]::Exists($extendedPath)) {
          [System.IO.File]::Delete($extendedPath)
          Write-Host "Removed temporary path: $LiteralPath"
          return
        }
      }
      catch {
        Write-Warning "Failed to remove temporary path '$LiteralPath' with extended path fallback: $($_.Exception.Message)"
        return
      }
    }

    Write-Warning "Failed to remove temporary path '$LiteralPath': $($_.Exception.Message)"
  }
}

function Remove-BaseRuntimeIntermediatePaths {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey,
    [object]$RuntimePackage
  )

  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "artifacts\runtime\$RuntimeKey")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\$RuntimeKey\assets")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\$RuntimeKey\bin")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\$RuntimeKey\obj")

  if ($RuntimePackage -and -not [string]::IsNullOrWhiteSpace([string]$RuntimePackage.buildPreset)) {
    Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "build-out\$($RuntimePackage.buildPreset)")
  }
}

function Get-SafePathName {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  return ($Value -replace '[^A-Za-z0-9._-]', '-')
}

function Get-LocalPackageCachePath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey
  )

  $safeKey = Get-SafePathName -Value $RuntimeKey
  if ($env:RUNNER_TEMP) {
    return Join-Path $env:RUNNER_TEMP "jyppx-split-packages\$safeKey"
  }

  return Join-Path ([System.IO.Path]::GetTempPath()) "jyppx-split-packages\$safeKey"
}

function Remove-SplitPackageIntermediatePaths {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitPackage
  )

  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "artifacts\runtime-split\$($SplitPackage.key)")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\$($SplitPackage.key)\assets")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\$($SplitPackage.key)\bin")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\$($SplitPackage.key)\obj")
}

function Write-DiskSpaceSummary {
  param(
    [string]$Label
  )

  Write-Host $Label
  Get-PSDrive -PSProvider FileSystem |
    Select-Object Name,Root,@{Name='FreeGB';Expression={[math]::Round($_.Free/1GB,2)}} |
    Format-Table -AutoSize
}

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $SourceRuntimeKey } | Select-Object -First 1
if (-not $sourcePackage) {
  throw "Runtime package key '$SourceRuntimeKey' was not found."
}

$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$resolvedCudaCudnnPackageVersion = Resolve-RolePackageVersion -Value $CudaCudnnPackageVersion -Fallback $resolvedVersion
$resolvedTensorRtPackageVersion = Resolve-RolePackageVersion -Value $TensorRtPackageVersion -Fallback $resolvedVersion
$resolvedMetaPackageVersion = Resolve-RolePackageVersion -Value $MetaPackageVersion -Fallback $resolvedVersion
$resolvedBridgePackageVersion = Resolve-RolePackageVersion -Value $BridgePackageVersion -Fallback $resolvedVersion
$smokeRuntimeKeys = @(Expand-KeyList -Values $SmokeRuntimePackageKey)
$requestedSplitRoles = @(Expand-KeyList -Values $SplitPackageRole | ForEach-Object { $_.ToLowerInvariant() })
if ($requestedSplitRoles.Count -eq 0) {
  $requestedSplitRoles = @("all")
}

$allSplitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $SourceRuntimeKey })
if ($allSplitPackages.Count -eq 0) {
  throw "No split runtime packages were defined for source runtime '$SourceRuntimeKey'."
}

$includeAllSplitRoles = $requestedSplitRoles -contains "all"
$shouldPackMetaPackage = $IncludeMetaPackage.IsPresent -or $includeAllSplitRoles -or ($requestedSplitRoles -contains "meta") -or ($requestedSplitRoles -contains "collection")

if ($includeAllSplitRoles) {
  $splitPackages = @($allSplitPackages)
}
else {
  $splitPackages = @($allSplitPackages | Where-Object { Test-SplitPackageRequested -SplitPackage $_ -RequestedRoles $requestedSplitRoles })
}

if ($splitPackages.Count -eq 0 -and -not $shouldPackMetaPackage) {
  throw "No split runtime packages matched roles '$($requestedSplitRoles -join ', ')' for source runtime '$SourceRuntimeKey'."
}

if ($splitPackages.Count -ne $allSplitPackages.Count -and $RunSmoke.IsPresent) {
  Write-Host "Skipping smoke request because only a subset of split runtime packages is being built."
}
$shouldRunSmokeForSplitSet = $RunSmoke.IsPresent -and ($splitPackages.Count -eq $allSplitPackages.Count)

if ($shouldPackMetaPackage) {
  $metaProjectPath = Join-Path $RepositoryRoot "pack\runtime-split\$SourceRuntimeKey-meta\$($sourcePackage.packageId).csproj"
  if (-not (Test-Path -LiteralPath $metaProjectPath -PathType Leaf)) {
    throw "Split meta package project was not found: $metaProjectPath"
  }

  $missingComponentPackages = @($allSplitPackages | Where-Object {
      $candidateKey = [string]$_.key
      -not ($splitPackages | Where-Object { [string]$_.key -eq $candidateKey } | Select-Object -First 1)
    })

  if ($missingComponentPackages.Count -gt 0) {
    $missingCudaCudnn = @($missingComponentPackages | Where-Object { [string]$_.role -eq "cuda-cudnn" })
    $missingTensorRt = @($missingComponentPackages | Where-Object { [string]$_.role -eq "tensorrt" })
    if ($missingCudaCudnn.Count -gt 0 -and [string]::IsNullOrWhiteSpace($CudaCudnnPackageVersion)) {
      throw "The split meta package would reference a non-built CudaCudnn package at version '$resolvedCudaCudnnPackageVersion'. Pass -CudaCudnnPackageVersion to pin an already-published CUDA/cuDNN package explicitly, or build with -SplitPackageRole all/cuda-cudnn first."
    }

    if ($missingTensorRt.Count -gt 0 -and [string]::IsNullOrWhiteSpace($TensorRtPackageVersion)) {
      throw "The split meta package would reference non-built TensorRT component package(s) at version '$resolvedTensorRtPackageVersion'. Pass -TensorRtPackageVersion to pin already-published TensorRT packages explicitly, or build with -SplitPackageRole all/tensorrt first."
    }

    if ((Expand-KeyList -Values $AdditionalPackageSource).Count -eq 0) {
      Write-Warning "The split meta package depends on stable runtime dependency packages that are not being built in this run. Add -AdditionalPackageSource for the feed that already contains those packages, or build with -SplitPackageRole all/cuda-cudnn,tensorrt first."
    }
  }
}

$shouldRunBaseRuntimeBuild = (-not $SkipBaseRuntimeBuild.IsPresent) -and ($splitPackages.Count -gt 0)
if ($shouldRunBaseRuntimeBuild) {
  $baseArguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Invoke-LocalRuntimePackage.ps1"),
    "-RuntimePackageKey",
    $SourceRuntimeKey,
    "-Version",
    $resolvedVersion,
    "-Configuration",
    $Configuration,
    "-SkipRuntimePack",
    "-SkipConsumerValidation"
  )

  if ($SkipManagedPack.IsPresent) {
    $baseArguments += "-SkipManagedPack"
  }

  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList $baseArguments
}
elseif (-not $SkipBaseRuntimeBuild.IsPresent) {
  Write-Host "Skipping base runtime build because only the split meta package is being produced."
}

$splitOutputDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$SourceRuntimeKey"
if (Test-Path -LiteralPath $splitOutputDirectory) {
  Remove-Item -LiteralPath $splitOutputDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $splitOutputDirectory -Force | Out-Null
$localPackageCachePath = Get-LocalPackageCachePath -RuntimeKey $SourceRuntimeKey
Remove-OptionalPath -LiteralPath $localPackageCachePath
New-Item -ItemType Directory -Path $localPackageCachePath -Force | Out-Null
$lastSplitPackageIndex = $splitPackages.Count - 1

for ($splitPackageIndex = 0; $splitPackageIndex -lt $splitPackages.Count; $splitPackageIndex++) {
  $splitPackage = $splitPackages[$splitPackageIndex]
  Write-DiskSpaceSummary -Label "Disk space before packing split role '$($splitPackage.role)' for $SourceRuntimeKey"
  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Collect-SplitRuntimeAssets.ps1"),
    "-SplitPackageKey",
    $splitPackage.key,
    "-RepositoryRoot",
    $RepositoryRoot
  )

  if ($splitPackageIndex -eq $lastSplitPackageIndex) {
    Remove-BaseRuntimeIntermediatePaths -RuntimeKey $SourceRuntimeKey -RuntimePackage $sourcePackage
    Write-DiskSpaceSummary -Label "Disk space after pruning base runtime intermediates for $SourceRuntimeKey"
  }

  $projectPath = Join-Path $RepositoryRoot "pack\runtime-split\$($splitPackage.key)\$($splitPackage.packageId).csproj"
  $splitPackageVersion = Get-SplitPackageVersion -SplitPackage $splitPackage
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "restore",
    $projectPath
  )

  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "pack",
    $projectPath,
    "-c",
    $Configuration,
    "-o",
    $splitOutputDirectory,
    "-p:JYPPXPackageVersion=$splitPackageVersion",
    "-p:NoBuild=true",
    "--no-restore"
  )

  Remove-SplitPackageIntermediatePaths -SplitPackage $splitPackage
  Write-DiskSpaceSummary -Label "Disk space after packing split role '$($splitPackage.role)' for $SourceRuntimeKey"
}

$nugetConfigPath = Join-Path $splitOutputDirectory "NuGet.config"
$packageSources = New-Object System.Collections.Generic.List[string]
$packageSources.Add('    <clear />')
$packageSources.Add('    <add key="local-split" value="' + (ConvertTo-XmlAttributeValue -Value $splitOutputDirectory) + '" />')
$sourceIndex = 1
foreach ($source in @(Expand-KeyList -Values $AdditionalPackageSource)) {
  $resolvedSource = Resolve-PackageSourceValue -Source $source
  $packageSources.Add('    <add key="additional-' + $sourceIndex + '" value="' + (ConvertTo-XmlAttributeValue -Value $resolvedSource) + '" />')
  $sourceIndex++
}
$packageSources.Add('    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />')

$packageSourceCredentials = New-Object System.Collections.Generic.List[string]
if (-not [string]::IsNullOrWhiteSpace($AdditionalPackageSourcePassword)) {
  $credentialUserName = if ([string]::IsNullOrWhiteSpace($AdditionalPackageSourceUsername)) { "github" } else { $AdditionalPackageSourceUsername }
  $additionalSourceCount = $sourceIndex - 1
  for ($credentialIndex = 1; $credentialIndex -le $additionalSourceCount; $credentialIndex++) {
    $packageSourceCredentials.Add('    <additional-' + $credentialIndex + '>')
    $packageSourceCredentials.Add('      <add key="Username" value="' + (ConvertTo-XmlAttributeValue -Value $credentialUserName) + '" />')
    $packageSourceCredentials.Add('      <add key="ClearTextPassword" value="' + (ConvertTo-XmlAttributeValue -Value $AdditionalPackageSourcePassword) + '" />')
    $packageSourceCredentials.Add('    </additional-' + $credentialIndex + '>')
  }
}

$packageSourceCredentialBlock = if ($packageSourceCredentials.Count -gt 0) {
  "  <packageSourceCredentials>`r`n$($packageSourceCredentials -join "`r`n")`r`n  </packageSourceCredentials>"
}
else {
  ""
}

$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
$($packageSources -join "`r`n")
  </packageSources>
$packageSourceCredentialBlock
</configuration>
"@
Set-Content -LiteralPath $nugetConfigPath -Value $nugetConfig -Encoding utf8

if ($shouldPackMetaPackage) {
  $metaRestoreArguments = @(
    "restore",
    $metaProjectPath,
    "--configfile",
    $nugetConfigPath
  ) + (New-PackageVersionProperties -PackageVersion $resolvedMetaPackageVersion)
  $metaRestoreArguments += @("-p:RestorePackagesPath=$localPackageCachePath")
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList $metaRestoreArguments

  $metaPackArguments = @(
    "pack",
    $metaProjectPath,
    "-c",
    $Configuration,
    "-o",
    $splitOutputDirectory
  ) + (New-PackageVersionProperties -PackageVersion $resolvedMetaPackageVersion) + @(
    "-p:RestorePackagesPath=$localPackageCachePath",
    "--no-restore"
  )
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList $metaPackArguments

  Remove-OptionalPath -LiteralPath $localPackageCachePath
  Write-DiskSpaceSummary -Label "Disk space after packing split meta package for $SourceRuntimeKey"
}

if (-not $SkipConsumerValidation.IsPresent -and $shouldPackMetaPackage) {
  $consumerArguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-PackageConsumer.ps1"),
    "-RuntimePackageKey",
    $SourceRuntimeKey,
    "-ManagedPackageDirectory",
    (Join-Path $RepositoryRoot "artifacts\managed"),
    "-RuntimePackageDirectory",
    $splitOutputDirectory
  )

  if ($shouldRunSmokeForSplitSet -and ($smokeRuntimeKeys.Count -eq 0 -or $smokeRuntimeKeys -contains $SourceRuntimeKey)) {
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

  $consumerAdditionalPackageSources = @(Expand-KeyList -Values $AdditionalPackageSource)
  if ($consumerAdditionalPackageSources.Count -gt 0) {
    $consumerArguments += @("-AdditionalPackageSource", ($consumerAdditionalPackageSources -join ","))
  }

  if (-not [string]::IsNullOrWhiteSpace($AdditionalPackageSourceUsername)) {
    $consumerArguments += @("-AdditionalPackageSourceUsername", $AdditionalPackageSourceUsername)
  }

  if (-not [string]::IsNullOrWhiteSpace($AdditionalPackageSourcePassword)) {
    $consumerArguments += @("-AdditionalPackageSourcePassword", $AdditionalPackageSourcePassword)
  }

  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList $consumerArguments
}
elseif (-not $shouldPackMetaPackage) {
  Write-Host "Skipping package consumer validation because no split meta package was produced."
}
else {
  Write-Host "Skipping package consumer validation by request."
}

$packageFiles = @(Get-ChildItem -LiteralPath $splitOutputDirectory -Filter *.nupkg | Sort-Object Name)
$summaryRoot = Join-Path $RepositoryRoot "artifacts\local-runtime-validation"
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null
$jsonPath = Join-Path $summaryRoot ("local-split-runtime-validation-" + $SourceRuntimeKey + ".json")
$markdownPath = Join-Path $summaryRoot ("local-split-runtime-validation-" + $SourceRuntimeKey + ".md")

$rows = foreach ($packageFile in $packageFiles) {
  [pscustomobject]@{
    packageFile = $packageFile.Name
    sizeMb = [Math]::Round($packageFile.Length / 1MB, 2)
    fitsNugetOrg = ($packageFile.Length -le 250MB)
    fitsGithubNugetRegistry = ($packageFile.Length -lt 2147000000)
    fitsGithubReleaseAsset = ($packageFile.Length -lt 2GB)
  }
}

$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Local Split Runtime Validation")
$lines.Add("")
$lines.Add('Source runtime: `' + $SourceRuntimeKey + '`')
$lines.Add("")
$lines.Add("| Package | Size (MB) | nuget.org | GitHub NuGet | GitHub Release |")
$lines.Add("| --- | ---: | --- | --- | --- |")
foreach ($row in $rows) {
  $packageLabel = '`' + $row.packageFile + '`'
  $lines.Add([string]::Format(
      "| {0} | {1} | {2} | {3} | {4} |",
      $packageLabel,
      $row.sizeMb,
      $row.fitsNugetOrg,
      $row.fitsGithubNugetRegistry,
      $row.fitsGithubReleaseAsset))
}
$lines.Add("")
$lines.Add("Generated by `eng/Invoke-LocalSplitRuntimePackage.ps1`.")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Local split runtime validation summary written to $jsonPath"
Write-Host "Local split runtime validation summary written to $markdownPath"

Remove-OptionalPath -LiteralPath $localPackageCachePath
foreach ($splitPackage in $splitPackages) {
  Remove-SplitPackageIntermediatePaths -SplitPackage $splitPackage
}
Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\$SourceRuntimeKey-meta\bin")
Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\$SourceRuntimeKey-meta\obj")
Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "build-out\package-consumer\$SourceRuntimeKey")

if ($shouldRunBaseRuntimeBuild) {
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "artifacts\runtime\$SourceRuntimeKey")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\$SourceRuntimeKey\assets")
  Remove-OptionalPath -LiteralPath (Join-Path $RepositoryRoot "artifacts\runtime-nupkg\$($sourcePackage.packageId).$resolvedVersion.nupkg")
}
