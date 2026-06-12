[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @(),
  [string]$Configuration,
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

function Find-Signtool {
  param(
    [string]$PreferredPath
  )

  if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
    if (-not (Test-Path -LiteralPath $PreferredPath -PathType Leaf)) {
      throw "signtool was not found at '$PreferredPath'."
    }

    return (Resolve-Path -LiteralPath $PreferredPath).Path
  }

  $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $kitRoot = "C:\Program Files (x86)\Windows Kits\10\bin"
  if (Test-Path -LiteralPath $kitRoot -PathType Container) {
    $candidate = Get-ChildItem -LiteralPath $kitRoot -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
      Sort-Object FullName -Descending |
      Select-Object -First 1
    if ($candidate) {
      return $candidate.FullName
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
    $cert = Get-ChildItem Cert:\CurrentUser\My |
      Where-Object { $_.Thumbprint -eq $Thumbprint -and $_.HasPrivateKey } |
      Select-Object -First 1
    if (-not $cert) {
      throw "Code signing certificate with thumbprint '$Thumbprint' was not found under Cert:\CurrentUser\My."
    }

    return $cert
  }

  $existing = Get-ChildItem Cert:\CurrentUser\My |
    Where-Object { $_.Subject -eq $Subject -and $_.HasPrivateKey } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1
  if ($existing) {
    return $existing
  }

  $newSelfSignedCertificate = Get-Command New-SelfSignedCertificate -ErrorAction SilentlyContinue
  if (-not $newSelfSignedCertificate) {
    throw "New-SelfSignedCertificate is not available. Pass -CertificateThumbprint for an existing code signing certificate."
  }

  return New-SelfSignedCertificate `
    -Subject $Subject `
    -Type CodeSigningCert `
    -CertStoreLocation Cert:\CurrentUser\My `
    -KeyExportPolicy Exportable `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddYears(2)
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$packages = @($manifest.packages | Where-Object { $_.platform -eq "windows" })
if ($RuntimePackageKey.Count -gt 0) {
  $selectedKeys = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($value in @($RuntimePackageKey)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        [void]$selectedKeys.Add($trimmed)
      }
    }
  }

  $packages = @($packages | Where-Object { $selectedKeys.Contains([string]$_.key) })
}

if ($packages.Count -eq 0) {
  throw "No Windows runtime packages were selected."
}

$resolvedSigntool = Find-Signtool -PreferredPath $SigntoolPath
$certificate = Get-OrCreate-CodeSigningCertificate -Thumbprint $CertificateThumbprint -Subject $CertificateSubject

$signed = New-Object System.Collections.Generic.List[object]
foreach ($package in $packages) {
  $buildPreset = if ($package.buildPreset) { $package.buildPreset } else { $package.key }
  $bridgeConfiguration = if ($Configuration) { $Configuration } elseif ($package.bridgeConfiguration) { $package.bridgeConfiguration } else { "Release" }
  $bridgePath = Join-Path $RepositoryRoot "build-out\$buildPreset\bin\$bridgeConfiguration\$($package.bridgeFile)"
  if (-not (Test-Path -LiteralPath $bridgePath -PathType Leaf)) {
    throw "Bridge binary was not found: $bridgePath"
  }

  $signature = Get-AuthenticodeSignature -LiteralPath $bridgePath
  if ($signature.SignerCertificate -and $signature.SignerCertificate.Thumbprint -eq $certificate.Thumbprint) {
    Write-Host "Bridge already signed for $($package.key): $bridgePath"
  }
  else {
    & $resolvedSigntool sign /fd SHA256 /sha1 $certificate.Thumbprint $bridgePath
    if ($LASTEXITCODE -ne 0) {
      throw "signtool failed for '$bridgePath' with exit code $LASTEXITCODE."
    }
  }

  $finalSignature = Get-AuthenticodeSignature -LiteralPath $bridgePath
  $signed.Add([pscustomobject]@{
    key = $package.key
    path = $bridgePath
    signerThumbprint = if ($finalSignature.SignerCertificate) { $finalSignature.SignerCertificate.Thumbprint } else { $null }
    signatureStatus = [string]$finalSignature.Status
  })
}

$reportDirectory = Join-Path $RepositoryRoot "artifacts\signing"
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
$reportPath = Join-Path $reportDirectory "windows-bridge-signing-summary.json"
$signed | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "Signed Windows bridge binaries with certificate $($certificate.Thumbprint)."
Write-Host "Signing summary written to $reportPath"
