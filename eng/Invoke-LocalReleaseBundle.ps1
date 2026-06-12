[CmdletBinding()]
param(
  [string]$Version = "4.0.0",
  [string]$Configuration = "Release",
  [string[]]$WindowsRuntimeKeys = @("win-x64-trt11.0-cuda12.9-cudnn9.22"),
  [ValidateSet("full", "split")]
  [string]$WindowsRuntimeDeliveryMode = "full",
  [switch]$SkipDocs,
  [switch]$SkipManagedPack,
  [switch]$SkipWindowsRuntime,
  [switch]$RunWindowsSmoke,
  [switch]$SignWindowsConsumerOutput,
  [switch]$TrustWindowsConsumerSigningCertificate,
  [switch]$TrustWindowsConsumerSigningCertificateRoot,
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

$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$windowsKeys = @(Expand-KeyList -Values $WindowsRuntimeKeys)
if ($windowsKeys.Count -eq 0) {
  $windowsKeys = @("win-x64-trt11.0-cuda12.9-cudnn9.22")
}

if (-not $SkipDocs.IsPresent) {
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @("tool", "restore")
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @("docfx", (Join-Path $RepositoryRoot "docs\docfx.json"))
}

if (-not $SkipManagedPack.IsPresent) {
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @("restore", (Join-Path $RepositoryRoot "TensorRtSharp.sln"))
  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Validate-RuntimeManifest.ps1")
  )
  Invoke-CheckedCommand -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-BindingGeneratorOutputs.ps1")
  )
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "build",
    (Join-Path $RepositoryRoot "TensorRtSharp.sln"),
    "-c",
    $Configuration,
    "--no-restore"
  )
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "test",
    (Join-Path $RepositoryRoot "tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj"),
    "-c",
    $Configuration,
    "--no-build"
  )
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

if (-not $SkipWindowsRuntime.IsPresent) {
  if ($WindowsRuntimeDeliveryMode -eq "split") {
    foreach ($key in $windowsKeys) {
      $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepositoryRoot "eng\Invoke-LocalSplitRuntimePackage.ps1"),
        "-SourceRuntimeKey",
        $key,
        "-Version",
        $resolvedVersion,
        "-Configuration",
        $Configuration
      )

      if (-not $SkipManagedPack.IsPresent) {
        $arguments += "-SkipManagedPack"
      }

      if ($RunWindowsSmoke.IsPresent) {
        $arguments += "-RunSmoke"
      }

      if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent) {
        $arguments += "-SignConsumerOutput"
      }

      if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent -and $TrustWindowsConsumerSigningCertificate.IsPresent) {
        $arguments += "-TrustConsumerSigningCertificate"
      }

      if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent -and $TrustWindowsConsumerSigningCertificateRoot.IsPresent) {
        $arguments += "-TrustConsumerSigningCertificateRoot"
      }

      if (-not [string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
        $arguments += @("-CertificateThumbprint", $CertificateThumbprint)
      }

      if (-not [string]::IsNullOrWhiteSpace($CertificateSubject)) {
        $arguments += @("-CertificateSubject", $CertificateSubject)
      }

      if (-not [string]::IsNullOrWhiteSpace($SigntoolPath)) {
        $arguments += @("-SigntoolPath", $SigntoolPath)
      }

      Invoke-CheckedCommand -FilePath "powershell" -ArgumentList $arguments
    }
  }
  else {
    $arguments = @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Invoke-LocalRuntimePackage.ps1"),
      "-RuntimePackageKey",
      ($windowsKeys -join ","),
      "-Version",
      $resolvedVersion,
      "-Configuration",
      $Configuration
    )

    if (-not $SkipManagedPack.IsPresent) {
      $arguments += "-SkipManagedPack"
    }

    if ($RunWindowsSmoke.IsPresent) {
      $arguments += "-RunSmoke"
    }

    if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent) {
      $arguments += "-SignConsumerOutput"
    }

    if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent -and $TrustWindowsConsumerSigningCertificate.IsPresent) {
      $arguments += "-TrustConsumerSigningCertificate"
    }

    if ($RunWindowsSmoke.IsPresent -and $SignWindowsConsumerOutput.IsPresent -and $TrustWindowsConsumerSigningCertificateRoot.IsPresent) {
      $arguments += "-TrustConsumerSigningCertificateRoot"
    }

    if (-not [string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
      $arguments += @("-CertificateThumbprint", $CertificateThumbprint)
    }

    if (-not [string]::IsNullOrWhiteSpace($CertificateSubject)) {
      $arguments += @("-CertificateSubject", $CertificateSubject)
    }

    if (-not [string]::IsNullOrWhiteSpace($SigntoolPath)) {
      $arguments += @("-SigntoolPath", $SigntoolPath)
    }

    Invoke-CheckedCommand -FilePath "powershell" -ArgumentList $arguments
  }
}

Write-Host "Local release bundle finished."
Write-Host "  Version: $resolvedVersion"
Write-Host "  Configuration: $Configuration"
Write-Host "  RunDocs: $(-not $SkipDocs.IsPresent)"
Write-Host "  RunManagedPack: $(-not $SkipManagedPack.IsPresent)"
Write-Host "  RunWindowsRuntime: $(-not $SkipWindowsRuntime.IsPresent)"
Write-Host "  RunWindowsSmoke: $($RunWindowsSmoke.IsPresent)"
Write-Host "  WindowsRuntimeDeliveryMode: $WindowsRuntimeDeliveryMode"
Write-Host "  WindowsRuntimeKeys: $($windowsKeys -join ', ')"
