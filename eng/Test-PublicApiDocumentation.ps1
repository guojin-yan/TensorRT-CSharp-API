[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Invoke-DocAuditBuild {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath
  )

  $rawOutput = & dotnet build $ProjectPath -c Release -t:Rebuild -p:TargetFramework=net8.0 -p:JYPPXSuppressMissingXmlDocs=false 2>&1
  $exitCode = $LASTEXITCODE
  $normalizedOutput = New-Object System.Collections.Generic.List[string]
  foreach ($line in @($rawOutput)) {
    $lineText = [string]$line
    $lineText = [System.Text.RegularExpressions.Regex]::Replace($lineText, '\x1B\[[0-9;]*[A-Za-z]', '')
    $normalizedOutput.Add($lineText)
    Write-Host $lineText
  }

  if ($exitCode -ne 0) {
    throw "Public API documentation build failed for $ProjectPath with exit code $exitCode."
  }

  $warnings = @($normalizedOutput | Where-Object { $_ -match 'warning CS1591:' })
  return $warnings
}

function Parse-WarningLine {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Line,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRootPath
  )

  if ($Line -notmatch '^\s*(?<file>[A-Za-z]:\\[^()]+)\((?<line>\d+),(?<column>\d+)\): warning CS1591: (?<message>.+)$') {
    return $null
  }

  $fullPath = $Matches.file
  $relativePath = $fullPath
  if ($relativePath.StartsWith($RepositoryRootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    $relativePath = $relativePath.Substring($RepositoryRootPath.Length).TrimStart('\')
  }

  return [pscustomobject]@{
    file = $relativePath
    line = [int]$Matches.line
    column = [int]$Matches.column
    message = $Matches.message
  }
}

function Resolve-ProjectNameFromFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RelativeFile
  )

  if ($RelativeFile.StartsWith("src\JYPPX.Shared\", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "JYPPX.Shared"
  }

  if ($RelativeFile.StartsWith("src\JYPPX.CudaSharp\", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "JYPPX.CudaSharp"
  }

  if ($RelativeFile.StartsWith("src\JYPPX.TensorRtSharp\", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "JYPPX.TensorRtSharp"
  }

  return "Unknown"
}

$projects = @(
  [pscustomobject]@{
    name = "JYPPX.Shared"
    path = Join-Path $RepositoryRoot "src\JYPPX.Shared\JYPPX.Shared.csproj"
  }
  [pscustomobject]@{
    name = "JYPPX.CudaSharp"
    path = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\JYPPX.CudaSharp.csproj"
  }
  [pscustomobject]@{
    name = "JYPPX.TensorRtSharp"
    path = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\JYPPX.TensorRtSharp.csproj"
  }
)

$rows = New-Object System.Collections.Generic.List[object]
$uniqueWarnings = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($project in $projects) {
  $warnings = @(Invoke-DocAuditBuild -ProjectPath $project.path)
  $parsed = @($warnings | ForEach-Object { Parse-WarningLine -Line $_ -RepositoryRootPath $RepositoryRoot } | Where-Object { $_ -ne $null })

  foreach ($item in $parsed) {
    $key = "$($item.file)|$($item.line)|$($item.column)|$($item.message)"
    if ($uniqueWarnings.ContainsKey($key)) {
      continue
    }

    $row = [pscustomobject]@{
      project = Resolve-ProjectNameFromFile -RelativeFile $item.file
      file = $item.file
      line = $item.line
      column = $item.column
      message = $item.message
    }

    $uniqueWarnings[$key] = $row
    $rows.Add($row)
  }
}

$grouped = @($rows | Group-Object file | Sort-Object Count -Descending)
$projectSummaries = @($rows | Group-Object project | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
      project = $_.Name
      warningCount = $_.Count
    }
  })

$outputRoot = Join-Path $RepositoryRoot "artifacts\api-doc-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "public-api-documentation-audit.json"
$markdownPath = Join-Path $outputRoot "public-api-documentation-audit.md"

[pscustomobject]@{
  generatedOn = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  projectSummaries = $projectSummaries
  warningCount = $rows.Count
  fileSummaries = @($grouped | ForEach-Object {
      [pscustomobject]@{
        file = $_.Name
        count = $_.Count
      }
    })
  warnings = $rows
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Public API Documentation Audit")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("## Project Summary")
$lines.Add("")
$lines.Add("| Project | Missing XML doc warnings |")
$lines.Add("| --- | ---: |")
foreach ($summary in $projectSummaries) {
  $lines.Add("| $($summary.project) | $($summary.warningCount) |")
}

$lines.Add("")
$lines.Add("## Top Files")
$lines.Add("")
$lines.Add("| File | Missing XML doc warnings |")
$lines.Add("| --- | ---: |")
foreach ($group in $grouped | Select-Object -First 50) {
  $lines.Add("| " + $codeQuote + $group.Name + $codeQuote + " | " + $group.Count + " |")
}

$lines.Add("")
$lines.Add("## First 200 Warnings")
$lines.Add("")
$lines.Add("| Project | File | Line | Message |")
$lines.Add("| --- | --- | ---: | --- |")
foreach ($row in ($rows | Select-Object -First 200)) {
  $message = $row.message.Replace("|", "\|")
  $lines.Add("| " + $row.project + " | " + $codeQuote + $row.file + $codeQuote + " | " + $row.line + " | " + $message + " |")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public API documentation audit written to $jsonPath"
Write-Host "Public API documentation audit written to $markdownPath"

if ($rows.Count -gt 0) {
  Write-Warning "Found $($rows.Count) compiler-reported CS1591 warning(s) across the public API surface."
  exit 1
}
