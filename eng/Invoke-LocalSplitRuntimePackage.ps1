[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "win-x64-trt11.0-cuda12.9-cudnn9.22",
  [string]$Version = "4.0.0",
  [string]$Configuration = "Release",
  [switch]$SkipManagedPack,
  [switch]$SkipBaseRuntimeBuild,
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

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $SourceRuntimeKey } | Select-Object -First 1
if (-not $sourcePackage) {
  throw "Runtime package key '$SourceRuntimeKey' was not found."
}

$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$splitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $SourceRuntimeKey })
if ($splitPackages.Count -eq 0) {
  throw "No split runtime packages were defined for source runtime '$SourceRuntimeKey'."
}

$metaProjectPath = Join-Path $RepositoryRoot "pack\runtime-split\$SourceRuntimeKey-meta\$($sourcePackage.packageId).csproj"
if (-not (Test-Path -LiteralPath $metaProjectPath -PathType Leaf)) {
  throw "Split meta package project was not found: $metaProjectPath"
}

if (-not $SkipBaseRuntimeBuild.IsPresent) {
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

$splitOutputDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$SourceRuntimeKey"
if (Test-Path -LiteralPath $splitOutputDirectory) {
  Remove-Item -LiteralPath $splitOutputDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $splitOutputDirectory -Force | Out-Null

foreach ($splitPackage in $splitPackages) {
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

  $projectPath = Join-Path $RepositoryRoot "pack\runtime-split\$($splitPackage.key)\$($splitPackage.packageId).csproj"
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
    "-p:JYPPXPackageVersion=$resolvedVersion",
    "-p:NoBuild=true",
    "--no-restore"
  )
}

$nugetConfigPath = Join-Path $splitOutputDirectory "NuGet.config"
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local-split" value="$splitOutputDirectory" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
"@
Set-Content -LiteralPath $nugetConfigPath -Value $nugetConfig -Encoding utf8

Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
  "restore",
  $metaProjectPath,
  "--configfile",
  $nugetConfigPath
)

Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
  "pack",
  $metaProjectPath,
  "-c",
  $Configuration,
  "-o",
  $splitOutputDirectory,
  "-p:JYPPXPackageVersion=$resolvedVersion",
  "--no-restore"
)

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

$packageFiles = @(Get-ChildItem -LiteralPath $splitOutputDirectory -Filter *.nupkg | Sort-Object Name)
$summaryRoot = Join-Path $RepositoryRoot "artifacts\local-runtime-validation"
New-Item -ItemType Directory -Path $summaryRoot -Force | Out-Null
$jsonPath = Join-Path $summaryRoot ("local-split-runtime-validation-" + $SourceRuntimeKey + ".json")
$markdownPath = Join-Path $summaryRoot ("local-split-runtime-validation-" + $SourceRuntimeKey + ".md")

$rows = foreach ($packageFile in $packageFiles) {
  [pscustomobject]@{
    packageFile = $packageFile.Name
    sizeMb = [Math]::Round($packageFile.Length / 1MB, 2)
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
$lines.Add("| Package | Size (MB) | GitHub NuGet | GitHub Release |")
$lines.Add("| --- | ---: | --- | --- |")
foreach ($row in $rows) {
  $packageLabel = '`' + $row.packageFile + '`'
  $lines.Add([string]::Format(
      "| {0} | {1} | {2} | {3} |",
      $packageLabel,
      $row.sizeMb,
      $row.fitsGithubNugetRegistry,
      $row.fitsGithubReleaseAsset))
}
$lines.Add("")
$lines.Add("Generated by `eng/Invoke-LocalSplitRuntimePackage.ps1`.")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Local split runtime validation summary written to $jsonPath"
Write-Host "Local split runtime validation summary written to $markdownPath"
