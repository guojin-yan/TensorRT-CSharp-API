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

function Get-OwnerUtf8Encoding {
  if ($null -eq $script:utf8) {
    $script:utf8 = [System.Text.UTF8Encoding]::new($false)
  }

  return $script:utf8
}

function Write-Utf8FileAtomic {
  param([string]$LiteralPath, [string]$Content)

  $encoding = Get-OwnerUtf8Encoding
  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $Content, $encoding)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  Write-Utf8FileAtomic -LiteralPath $LiteralPath -Content (($lines -join [Environment]::NewLine) + [Environment]::NewLine)
}

function Read-JsonOrNull {
  param([string]$RepositoryRoot, [string]$RelativePath)
  $path = if ([System.IO.Path]::IsPathRooted($RelativePath)) { $RelativePath } else { Join-Path $RepositoryRoot $RelativePath }
  $lastError = $null
  for ($attempt = 1; $attempt -le 8; $attempt++) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try {
      $json = [System.IO.File]::ReadAllText($path, (Get-OwnerUtf8Encoding))
      if ([string]::IsNullOrWhiteSpace($json)) {
        throw "JSON file is empty: $path"
      }

      return $json | ConvertFrom-Json
    }
    catch {
      $lastError = $_
      if ($attempt -eq 8) { throw }
      Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
    }
  }

  throw $lastError
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
    $text.IndexOf("example-not-real-proof", [StringComparison]::OrdinalIgnoreCase) -ge 0
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
