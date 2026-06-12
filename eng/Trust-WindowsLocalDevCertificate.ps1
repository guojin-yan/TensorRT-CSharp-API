[CmdletBinding()]
param(
  [string]$CertificateThumbprint,
  [string]$CertificateSubject = "CN=JYPPX TensorRtSharp Local Dev Code Signing",
  [switch]$TrustRoot
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-LocalDevCertificate {
  param(
    [string]$Thumbprint,
    [string]$Subject
  )

  if (-not [string]::IsNullOrWhiteSpace($Thumbprint)) {
    $byThumbprint = Get-ChildItem Cert:\CurrentUser\My |
      Where-Object { $_.Thumbprint -eq $Thumbprint -and $_.HasPrivateKey } |
      Select-Object -First 1
    if (-not $byThumbprint) {
      throw "Code signing certificate with thumbprint '$Thumbprint' was not found under Cert:\CurrentUser\My."
    }

    return $byThumbprint
  }

  $bySubject = Get-ChildItem Cert:\CurrentUser\My |
    Where-Object { $_.Subject -eq $Subject -and $_.HasPrivateKey } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1
  if (-not $bySubject) {
    throw "Code signing certificate with subject '$Subject' was not found under Cert:\CurrentUser\My. Run Sign-WindowsManagedBinaries.ps1 or Sign-WindowsBridgeBinaries.ps1 first to create it."
  }

  return $bySubject
}

function Add-CertificateToStore {
  param(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [System.Security.Cryptography.X509Certificates.StoreName]$StoreName
  )

  $store = [System.Security.Cryptography.X509Certificates.X509Store]::new($StoreName, [System.Security.Cryptography.X509Certificates.StoreLocation]::CurrentUser)
  try {
    $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
    $existing = $store.Certificates |
      Where-Object { $_.Thumbprint -eq $Certificate.Thumbprint } |
      Select-Object -First 1
    if ($existing) {
      return "already-present"
    }

    $store.Add($Certificate)
    return "added"
  }
  finally {
    $store.Close()
  }
}

$certificate = Get-LocalDevCertificate -Thumbprint $CertificateThumbprint -Subject $CertificateSubject

$rootStatus = "skipped"
if ($TrustRoot) {
  $rootStatus = Add-CertificateToStore -Certificate $certificate -StoreName ([System.Security.Cryptography.X509Certificates.StoreName]::Root)
}
$publisherStatus = Add-CertificateToStore -Certificate $certificate -StoreName ([System.Security.Cryptography.X509Certificates.StoreName]::TrustedPublisher)

$result = [pscustomobject]@{
  subject = $certificate.Subject
  thumbprint = $certificate.Thumbprint
  notAfter = $certificate.NotAfter
  currentUserRoot = $rootStatus
  currentUserTrustedPublisher = $publisherStatus
  rootTrustNote = if ($TrustRoot) { "Root trust was requested. Some Windows policies may still require an interactive confirmation or administrator policy." } else { "Root trust was not requested. Use -TrustRoot only when you intentionally want this CurrentUser self-signed certificate to become a trusted root for local smoke validation." }
}

$result | Format-List
