[CmdletBinding()]
param(
  [string]$PackageRoot,
  [string]$PackagePattern = "*.nupkg",
  [switch]$Recurse,
  [string]$Source,
  [string]$ApiKey,
  [int]$TimeoutSeconds = 3600,
  [int]$MaxAttempts = 3,
  [int]$RetryDelaySeconds = 20,
  [ValidateSet("name", "size-ascending", "size-descending")]
  [string]$SortMode = "name"
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
  throw "PackageRoot is required."
}

if ([string]::IsNullOrWhiteSpace($Source)) {
  throw "Source is required."
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
  throw "ApiKey is required."
}

if ($TimeoutSeconds -lt 1) {
  throw "TimeoutSeconds must be greater than zero."
}

if ($MaxAttempts -lt 1) {
  throw "MaxAttempts must be greater than zero."
}

if (-not (Test-Path -LiteralPath $PackageRoot -PathType Container)) {
  throw "PackageRoot does not exist: $PackageRoot"
}

$packages = if ($Recurse.IsPresent) {
  @(Get-ChildItem -LiteralPath $PackageRoot -Filter $PackagePattern -File -Recurse)
}
else {
  @(Get-ChildItem -LiteralPath $PackageRoot -Filter $PackagePattern -File)
}

if ($packages.Count -eq 0) {
  throw "No packages matching '$PackagePattern' were found under $PackageRoot."
}

switch ($SortMode) {
  "size-ascending" {
    $packages = @($packages | Sort-Object Length, Name)
    break
  }
  "size-descending" {
    $packages = @($packages | Sort-Object Length, Name -Descending)
    break
  }
  default {
    $packages = @($packages | Sort-Object Name)
    break
  }
}

foreach ($package in $packages) {
  $attempt = 1
  $pushed = $false

  while (-not $pushed -and $attempt -le $MaxAttempts) {
    Write-Host ("Pushing package attempt {0}/{1}: {2} ({3} MB)" -f $attempt, $MaxAttempts, $package.FullName, [Math]::Round($package.Length / 1MB, 2))

    $arguments = @(
      "nuget",
      "push",
      $package.FullName,
      "--source",
      $Source,
      "--api-key",
      $ApiKey,
      "--timeout",
      $TimeoutSeconds,
      "--skip-duplicate"
    )

    $output = & dotnet @arguments 2>&1
    foreach ($line in @($output)) {
      Write-Host $line
    }

    if ($LASTEXITCODE -eq 0) {
      $pushed = $true
      break
    }

    if ($attempt -ge $MaxAttempts) {
      throw "Failed to push package after $MaxAttempts attempts: $($package.FullName)"
    }

    Write-Warning ("Push failed for {0} on attempt {1}. Retrying in {2}s." -f $package.Name, $attempt, $RetryDelaySeconds)
    Start-Sleep -Seconds $RetryDelaySeconds
    $attempt++
  }
}
