[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release")]
  [string]$Configuration = "Debug",
  [string]$CertificateThumbprint,
  [string]$CertificateSubject = "CN=JYPPX TensorRtSharp Local Dev Code Signing",
  [string]$SigntoolPath,
  [string]$RepositoryRoot,
  [switch]$IncludeSrc = $true,
  [switch]$IncludeSamples = $true,
  [switch]$IncludeTests = $true
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Find-Signtool {
  param([string]$PreferredPath)

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

function Get-ManagedSigningCandidates {
  param(
    [string]$Root,
    [string]$Config,
    [bool]$ScanSrc,
    [bool]$ScanSamples,
    [bool]$ScanTests
  )

  $roots = New-Object System.Collections.Generic.List[string]
  if ($ScanSrc) { $roots.Add((Join-Path $Root "src")) }
  if ($ScanSamples) { $roots.Add((Join-Path $Root "samples")) }
  if ($ScanTests) { $roots.Add((Join-Path $Root "tests")) }

  $files = New-Object System.Collections.Generic.List[System.IO.FileInfo]
  foreach ($scanRoot in $roots) {
    if (-not (Test-Path -LiteralPath $scanRoot -PathType Container)) {
      continue
    }

    $candidates = Get-ChildItem -LiteralPath $scanRoot -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object {
        $_.FullName -match "\\bin\\$([regex]::Escape($Config))\\" -and
        ($_.Extension -eq ".dll" -or $_.Extension -eq ".exe") -and
        $_.FullName -notmatch "\\runtimes\\" -and
        $_.Name -notmatch "^(nvinfer|nvonnxparser|nvinfer_plugin|cudart|cublas|cudnn|jyppxtrtbridge)"
      } |
      Where-Object {
        $_.Name -like "JYPPX*.dll" -or
        $_.Name -like "*Runner.dll" -or
        $_.Name -like "*Runner.exe" -or
        $_.Name -like "*Tests.dll"
      }

    foreach ($candidate in $candidates) {
      $files.Add($candidate)
    }
  }

  return @($files | Sort-Object FullName -Unique)
}

$resolvedSigntool = Find-Signtool -PreferredPath $SigntoolPath
$certificate = Get-OrCreate-CodeSigningCertificate -Thumbprint $CertificateThumbprint -Subject $CertificateSubject
$candidates = @(Get-ManagedSigningCandidates -Root $RepositoryRoot -Config $Configuration -ScanSrc:$IncludeSrc.IsPresent -ScanSamples:$IncludeSamples.IsPresent -ScanTests:$IncludeTests.IsPresent)

if ($candidates.Count -eq 0) {
  throw "No managed binaries were found for Configuration=$Configuration. Build the solution first."
}

$signed = New-Object System.Collections.Generic.List[object]
foreach ($file in $candidates) {
  $signature = Get-AuthenticodeSignature -LiteralPath $file.FullName
  if ($signature.SignerCertificate -and $signature.SignerCertificate.Thumbprint -eq $certificate.Thumbprint) {
    Write-Host "Managed binary already signed: $($file.FullName)"
  }
  else {
    & $resolvedSigntool sign /fd SHA256 /sha1 $certificate.Thumbprint $file.FullName
    if ($LASTEXITCODE -ne 0) {
      throw "signtool failed for '$($file.FullName)' with exit code $LASTEXITCODE."
    }
  }

  $finalSignature = Get-AuthenticodeSignature -LiteralPath $file.FullName
  $signed.Add([pscustomobject]@{
    path = $file.FullName
    relativePath = $file.FullName.Substring($RepositoryRoot.Length + 1).Replace("\", "/")
    signerThumbprint = if ($finalSignature.SignerCertificate) { $finalSignature.SignerCertificate.Thumbprint } else { $null }
    signatureStatus = [string]$finalSignature.Status
  })
}

$reportDirectory = Join-Path $RepositoryRoot "artifacts\signing"
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
$reportPath = Join-Path $reportDirectory "windows-managed-signing-summary-$($Configuration.ToLowerInvariant()).json"
$signed | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "Signed $($signed.Count) managed binaries with certificate $($certificate.Thumbprint)."
Write-Host "Signing summary written to $reportPath"
