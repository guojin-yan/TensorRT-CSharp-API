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

function Get-UniqueStringList {
  param(
    [string[]]$Values
  )

  $set = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $result = [System.Collections.Generic.List[string]]::new()
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    $trimmed = $value.Trim()
    if ($set.Add($trimmed)) {
      $result.Add($trimmed)
    }

    $withoutTrailingSlash = $trimmed.TrimEnd("/")
    if ($set.Add($withoutTrailingSlash)) {
      $result.Add($withoutTrailingSlash)
    }
  }

  @($result)
}

function Get-NuGetApiKeyAliases {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source
  )

  $aliases = @($Source)
  if ($Source -match "nuget\.org") {
    $aliases += @(
      "nuget.org",
      "https://www.nuget.org",
      "https://www.nuget.org/",
      "https://api.nuget.org/v3/index.json",
      "https://api.nuget.org/v3/index.json/",
      "https://www.nuget.org/api/v2/package",
      "https://www.nuget.org/api/v2/package/",
      "https://nuget.org/api/v2/package",
      "https://nuget.org/api/v2/package/"
    )
  }

  Get-UniqueStringList -Values $aliases
}

function Get-LocalNuGetConfigPath {
  if ([string]::IsNullOrWhiteSpace($env:APPDATA)) {
    return $null
  }

  Join-Path $env:APPDATA "NuGet\NuGet.Config"
}

function Get-LocalNuGetApiKeyEntries {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source
  )

  $configPath = Get-LocalNuGetConfigPath
  if ([string]::IsNullOrWhiteSpace($configPath) -or -not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    return @()
  }

  [xml]$xml = Get-Content -LiteralPath $configPath -Raw
  $nodes = @($xml.SelectNodes("//*[translate(local-name(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'apikeys']/*[translate(local-name(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'add']"))
  if ($nodes.Count -eq 0) {
    return @()
  }

  $aliases = @(Get-NuGetApiKeyAliases -Source $Source)
  $matchingValues = [System.Collections.Generic.List[string]]::new()
  foreach ($node in $nodes) {
    $key = [string]$node.GetAttribute("key")
    $value = [string]$node.GetAttribute("value")
    if ([string]::IsNullOrWhiteSpace($key) -or [string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    $isMatch = $false
    foreach ($alias in $aliases) {
      if ([StringComparer]::OrdinalIgnoreCase.Equals($key.TrimEnd("/"), $alias.TrimEnd("/"))) {
        $isMatch = $true
        break
      }
    }

    if (-not $isMatch -and $Source -match "nuget\.org" -and $key -match "nuget\.org") {
      $isMatch = $true
    }

    if ($isMatch -and -not $matchingValues.Contains($value)) {
      $matchingValues.Add($value)
    }
  }

  if ($matchingValues.Count -eq 0) {
    return @()
  }

  $entries = [System.Collections.Generic.List[object]]::new()
  foreach ($value in @($matchingValues)) {
    if (-not (Test-IsAsciiText -Value $value)) {
      throw "A local NuGet API key entry for '$Source' contains non-ASCII characters. Recreate the local key with NuGet tooling or provide NUGET_API_KEY as a plain-text secret."
    }

    foreach ($alias in $aliases) {
      $entries.Add([pscustomobject]@{
          Key   = $alias
          Value = $value
        })
    }
  }

  @($entries)
}

function New-TemporaryNuGetConfig {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$PackageSourceName,
    [Parameter(Mandatory = $true)]
    [string]$PackageSource,
    [object[]]$ApiKeyEntries,
    [string]$SourceUserName,
    [string]$SourcePassword
  )

  $settings = [System.Xml.XmlWriterSettings]::new()
  $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
  $settings.Indent = $true
  $writer = [System.Xml.XmlWriter]::Create($Path, $settings)
  try {
    $writer.WriteStartDocument()
    $writer.WriteStartElement("configuration")

    $writer.WriteStartElement("packageSources")
    $writer.WriteStartElement("clear")
    $writer.WriteEndElement()
    $writer.WriteStartElement("add")
    $writer.WriteAttributeString("key", $PackageSourceName)
    $writer.WriteAttributeString("value", $PackageSource)
    $writer.WriteAttributeString("protocolVersion", "3")
    $writer.WriteEndElement()
    $writer.WriteEndElement()

    if (-not [string]::IsNullOrWhiteSpace($SourceUserName) -and -not [string]::IsNullOrWhiteSpace($SourcePassword)) {
      $writer.WriteStartElement("packageSourceCredentials")
      $writer.WriteStartElement($PackageSourceName)
      $writer.WriteStartElement("add")
      $writer.WriteAttributeString("key", "Username")
      $writer.WriteAttributeString("value", $SourceUserName)
      $writer.WriteEndElement()
      $writer.WriteStartElement("add")
      $writer.WriteAttributeString("key", "ClearTextPassword")
      $writer.WriteAttributeString("value", $SourcePassword)
      $writer.WriteEndElement()
      $writer.WriteEndElement()
      $writer.WriteEndElement()
    }

    if ($ApiKeyEntries.Count -gt 0) {
      $writer.WriteStartElement("apikeys")
      foreach ($entry in @($ApiKeyEntries)) {
        $writer.WriteStartElement("add")
        $writer.WriteAttributeString("key", [string]$entry.Key)
        $writer.WriteAttributeString("value", [string]$entry.Value)
        $writer.WriteEndElement()
      }

      $writer.WriteEndElement()
    }

    $writer.WriteEndElement()
    $writer.WriteEndDocument()
  }
  finally {
    $writer.Dispose()
  }
}

function Resolve-NuGetExePath {
  $command = Get-Command nuget.exe -ErrorAction SilentlyContinue
  if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
    return $command.Source
  }

  $candidatePaths = @()
  if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    $candidatePaths += (Join-Path $env:USERPROFILE "bin\nuget.exe")
  }

  if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    $candidatePaths += (Join-Path $env:RUNNER_TEMP "nuget.exe")
  }

  $candidatePaths += (Join-Path ([IO.Path]::GetTempPath()) "nuget.exe")

  foreach ($candidatePath in $candidatePaths) {
    if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
      return $candidatePath
    }
  }

  $downloadRoot = if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    $env:RUNNER_TEMP
  }
  else {
    [IO.Path]::GetTempPath()
  }

  if (-not (Test-Path -LiteralPath $downloadRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
  }

  $downloadPath = Join-Path $downloadRoot "nuget.exe"
  Write-Host "Downloading nuget.exe for local NuGet.config API key fallback."
  Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $downloadPath
  $downloadPath
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
  $pushSource = $Source
  $usingLocalApiKeyFallback = $false

  if ($hasSourceCredentials) {
    $nugetConfigPath = Join-Path ([IO.Path]::GetTempPath()) ("jyppx-nuget-{0}.config" -f [Guid]::NewGuid().ToString("N"))
    New-TemporaryNuGetConfig -Path $nugetConfigPath -PackageSourceName $SourceName -PackageSource $Source -SourceUserName $SourceUserName -SourcePassword $sourcePassword
    $pushSource = $SourceName
    Write-Host "Using temporary NuGet.config source credentials for '$SourceName'."
  }
  elseif (-not $hasApiKey) {
    $localApiKeyEntries = @(Get-LocalNuGetApiKeyEntries -Source $Source)
    if ($localApiKeyEntries.Count -gt 0) {
      if ($Source -match "nuget\.org") {
        $pushSource = "nuget.org"
      }

      $usingLocalApiKeyFallback = $true
      Write-Host "Using the current user's NuGet.config local API key entries for '$pushSource'."
    }
  }

  foreach ($package in $packages) {
    $attempt = 1
    $pushed = $false

    while (-not $pushed -and $attempt -le $MaxAttempts) {
      $packageSizeMb = [Math]::Round($package.Length / 1MB, 2)
      $disableBuffering = $DisableBufferingAboveMB -gt 0 -and $package.Length -ge ($DisableBufferingAboveMB * 1MB)
      $useNuGetExe = $usingLocalApiKeyFallback -and
        -not $hasApiKey -and
        -not $hasSourceCredentials -and
        $Source -match "nuget\.org"
      $clientName = if ($useNuGetExe) { "nuget.exe" } else { "dotnet" }
      Write-Host ("Pushing package attempt {0}/{1}: {2} ({3} MB) Client={4} ApiKey={5} SourceCredentials={6} LocalConfigApiKey={7} DisableBuffering={8}" -f $attempt, $MaxAttempts, $package.FullName, $packageSizeMb, $clientName, $hasApiKey, $hasSourceCredentials, $usingLocalApiKeyFallback, $disableBuffering)

      $effectiveApiKey = if ($hasSourceCredentials) { $PushApiKey } else { $ApiKey }
      if ($useNuGetExe) {
        $nugetExePath = Resolve-NuGetExePath
        $arguments = @(
          "push",
          $package.FullName,
          "-Source",
          $pushSource,
          "-Timeout",
          $TimeoutSeconds,
          "-SkipDuplicate",
          "-NonInteractive"
        )
        if ($disableBuffering) {
          $arguments += "-DisableBuffering"
        }
        if (-not [string]::IsNullOrWhiteSpace($effectiveApiKey)) {
          $arguments += @("-ApiKey", $effectiveApiKey)
        }

        $output = & $nugetExePath @arguments 2>&1
      }
      else {
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
      }

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
