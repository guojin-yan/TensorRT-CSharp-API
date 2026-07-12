$script:OwnerRealProofCommonLoaded = $true

function Initialize-OwnerRealProofScript {
  param(
    [AllowNull()][string]$RepositoryRoot,
    [string]$OutputRoot
  )

  if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
    $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
  }

  if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
  }

  New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

  $utf8 = [System.Text.UTF8Encoding]::new($false)
  [Console]::OutputEncoding = $utf8
  $OutputEncoding = $utf8
  $script:utf8 = $utf8

  [pscustomobject]@{
    RepositoryRoot = $RepositoryRoot
    OutputRoot = $OutputRoot
  }
}

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Read-JsonOrNull {
  param([string]$RepositoryRoot, [string]$RelativePath)
  $path = if ([System.IO.Path]::IsPathRooted($RelativePath)) { $RelativePath } else { Join-Path $RepositoryRoot $RelativePath }
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-OwnerPlaceholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or
    $text.StartsWith("<owner-", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("<external-", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("example-not-real-proof", [StringComparison]::OrdinalIgnoreCase)
}

function Test-Sha256Text {
  param([AllowNull()][object]$Value)
  return [System.Text.RegularExpressions.Regex]::IsMatch([string]$Value, "^[0-9a-fA-F]{64}$")
}

function New-OwnerFinding {
  param([string]$Id, [string]$Severity, [string]$Category, [string]$Message)
  [pscustomobject]@{
    id = $Id
    severity = $Severity
    category = $Category
    message = $Message
    ownerActionRequired = $true
  }
}

function New-OwnerValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Resolve-OwnerPath {
  param([string]$BaseRoot, [string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $BaseRoot $Path
}
