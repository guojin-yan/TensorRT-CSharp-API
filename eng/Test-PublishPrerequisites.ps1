[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("nuget.org", "github-packages")]
  [string]$Target,
  [string]$ApiKey,
  [switch]$AllowMissingApiKey
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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

function Test-ContainsWhitespace {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  return [string]::Join("", $Value.ToCharArray()) -match "\s"
}

function Get-NormalizedSecretValue {
  param(
    [string]$Value,
    [string]$Description
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $null
  }

  $trimmed = $Value.Trim()
  if ($trimmed.Length -ne $Value.Length) {
    Write-Warning "$Description had leading or trailing whitespace; using the trimmed value."
  }

  $trimmed
}

switch ($Target) {
  "github-packages" {
    Write-Host "Publish prerequisites passed for github-packages."
    exit 0
  }
  "nuget.org" {
    $normalizedApiKey = Get-NormalizedSecretValue -Value $ApiKey -Description "The nuget.org API key"

    if ([string]::IsNullOrWhiteSpace($normalizedApiKey)) {
      if ($AllowMissingApiKey.IsPresent) {
        Write-Warning "No API key was provided for nuget.org. The caller must rely on the current machine's local NuGet configuration."
        exit 0
      }

      throw "No nuget.org API key was provided."
    }

    if (-not (Test-IsAsciiText -Value $normalizedApiKey)) {
      throw "The nuget.org API key contains non-ASCII characters. Store the plain-text NuGet API key in NUGET_API_KEY instead of an encrypted credential blob or other machine-generated token."
    }

    if (Test-ContainsWhitespace -Value $normalizedApiKey) {
      throw "The nuget.org API key contains internal whitespace. Store the exact plain-text NuGet API key value without embedded spaces or line breaks."
    }

    Write-Host "Publish prerequisites passed for nuget.org."
    exit 0
  }
}
