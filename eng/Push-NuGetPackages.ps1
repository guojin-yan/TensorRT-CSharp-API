[CmdletBinding()]
param(
  [string]$PackageRoot,
  [string]$PackagePattern = "*.nupkg",
  [switch]$Recurse,
  [string]$Source,
  [string]$ApiKey,
  [string]$ApiKeyEnvironmentVariable,
  [string]$SourceName,
  [string]$SourceUserName,
  [string]$SourcePasswordEnvironmentVariable,
  [string]$PushApiKey = "GitHub",
  [int]$TimeoutSeconds = 3600,
  [int]$MaxAttempts = 3,
  [int]$RetryDelaySeconds = 20,
  [int]$DisableBufferingAboveMB = 100,
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

if ($TimeoutSeconds -lt 1) {
  throw "TimeoutSeconds must be greater than zero."
}

if ($MaxAttempts -lt 1) {
  throw "MaxAttempts must be greater than zero."
}

if ($DisableBufferingAboveMB -lt 0) {
  throw "DisableBufferingAboveMB must be zero or greater."
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

function Test-IsAsciiText {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  foreach ($character in $Value.ToCharArray()) {
    if ([int][char]$character -gt 127) {
      return $false
    }
  }

  return $true
}

function Get-SecretFromEnvironment {
  param(
    [string]$Name
  )

  if ([string]::IsNullOrWhiteSpace($Name)) {
    return $null
  }

  [Environment]::GetEnvironmentVariable($Name)
}

function Format-SafeOutputLine {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Line,
    [string[]]$Secrets
  )

  $result = $Line
  foreach ($secret in @($Secrets)) {
    if (-not [string]::IsNullOrEmpty($secret)) {
      $result = $result.Replace($secret, "***")
    }
  }

  $result
}

$ApiKey = if ([string]::IsNullOrWhiteSpace($ApiKey)) { Get-SecretFromEnvironment -Name $ApiKeyEnvironmentVariable } else { $ApiKey }
$sourcePassword = Get-SecretFromEnvironment -Name $SourcePasswordEnvironmentVariable
$hasApiKey = -not [string]::IsNullOrWhiteSpace($ApiKey)
$hasSourceCredentials = -not [string]::IsNullOrWhiteSpace($SourceName) -and
  -not [string]::IsNullOrWhiteSpace($SourceUserName) -and
  -not [string]::IsNullOrWhiteSpace($sourcePassword)

if ($hasApiKey -and -not (Test-IsAsciiText -Value $ApiKey)) {
  throw "ApiKey contains non-ASCII characters. Provide the plain-text package API key instead of an encrypted credential blob or other formatted secret."
}

if ($hasSourceCredentials -and -not (Test-IsAsciiText -Value $sourcePassword)) {
  throw "Source password contains non-ASCII characters. Provide the plain-text package token instead of an encrypted credential blob or other formatted secret."
}

$nugetConfigPath = $null
try {
  if ($hasSourceCredentials) {
    $nugetConfigPath = Join-Path ([IO.Path]::GetTempPath()) ("jyppx-nuget-{0}.config" -f [Guid]::NewGuid().ToString("N"))
    $settings = [System.Xml.XmlWriterSettings]::new()
    $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
    $settings.Indent = $true
    $writer = [System.Xml.XmlWriter]::Create($nugetConfigPath, $settings)
    try {
      $writer.WriteStartDocument()
      $writer.WriteStartElement("configuration")
      $writer.WriteStartElement("packageSources")
      $writer.WriteStartElement("clear")
      $writer.WriteEndElement()
      $writer.WriteStartElement("add")
      $writer.WriteAttributeString("key", $SourceName)
      $writer.WriteAttributeString("value", $Source)
      $writer.WriteAttributeString("protocolVersion", "3")
      $writer.WriteEndElement()
      $writer.WriteEndElement()
      $writer.WriteStartElement("packageSourceCredentials")
      $writer.WriteStartElement($SourceName)
      $writer.WriteStartElement("add")
      $writer.WriteAttributeString("key", "Username")
      $writer.WriteAttributeString("value", $SourceUserName)
      $writer.WriteEndElement()
      $writer.WriteStartElement("add")
      $writer.WriteAttributeString("key", "ClearTextPassword")
      $writer.WriteAttributeString("value", $sourcePassword)
      $writer.WriteEndElement()
      $writer.WriteEndElement()
      $writer.WriteEndElement()
      $writer.WriteEndElement()
      $writer.WriteEndDocument()
    }
    finally {
      $writer.Dispose()
    }

    Write-Host "Using temporary NuGet.config source credentials for '$SourceName'."
  }

  foreach ($package in $packages) {
    $attempt = 1
    $pushed = $false

    while (-not $pushed -and $attempt -le $MaxAttempts) {
      $packageSizeMb = [Math]::Round($package.Length / 1MB, 2)
      $disableBuffering = $DisableBufferingAboveMB -gt 0 -and $package.Length -ge ($DisableBufferingAboveMB * 1MB)
      Write-Host ("Pushing package attempt {0}/{1}: {2} ({3} MB) ApiKey={4} SourceCredentials={5} DisableBuffering={6}" -f $attempt, $MaxAttempts, $package.FullName, $packageSizeMb, ($hasApiKey -or $hasSourceCredentials), $hasSourceCredentials, $disableBuffering)

      $pushSource = if ($hasSourceCredentials) { $SourceName } else { $Source }
      $effectiveApiKey = if ($hasSourceCredentials) { $PushApiKey } else { $ApiKey }
      $arguments = @(
        "nuget",
        "push",
        $package.FullName,
        "--source",
        $pushSource,
        "--timeout",
        $TimeoutSeconds,
        "--skip-duplicate"
      )
      if ($disableBuffering) {
        $arguments += "--disable-buffering"
      }
      if (-not [string]::IsNullOrWhiteSpace($effectiveApiKey)) {
        $arguments += @("--api-key", $effectiveApiKey)
      }
      if ($hasSourceCredentials) {
        $arguments += @("--configfile", $nugetConfigPath)
      }

      $output = & dotnet @arguments 2>&1
      $secrets = @($ApiKey, $sourcePassword)
      foreach ($line in @($output)) {
        Write-Host (Format-SafeOutputLine -Line ([string]$line) -Secrets $secrets)
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
}
finally {
  if (-not [string]::IsNullOrWhiteSpace($nugetConfigPath) -and (Test-Path -LiteralPath $nugetConfigPath)) {
    Remove-Item -LiteralPath $nugetConfigPath -Force -ErrorAction SilentlyContinue
  }
}
