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

  $output = & dotnet build $ProjectPath -c Release -p:TargetFramework=net8.0 -p:NoWarn= 2>&1
  $warnings = @($output | Where-Object { $_ -match 'warning CS1591:' })

  foreach ($line in @($output)) {
    Write-Host $line
  }

  return $warnings
}

function Parse-WarningLine {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Line,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRootPath
  )

  if ($Line -notmatch '^(?<file>[A-Za-z]:\\[^()]+)\((?<line>\d+),(?<column>\d+)\): warning CS1591: (?<message>.+)$') {
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
$projectSummaries = New-Object System.Collections.Generic.List[object]
foreach ($project in $projects) {
  $warnings = @(Invoke-DocAuditBuild -ProjectPath $project.path)
  $parsed = @($warnings | ForEach-Object { Parse-WarningLine -Line $_ -RepositoryRootPath $RepositoryRoot } | Where-Object { $_ -ne $null })

  $projectSummaries.Add([pscustomobject]@{
    project = $project.name
    warningCount = $parsed.Count
  })

  foreach ($item in $parsed) {
    $rows.Add([pscustomobject]@{
      project = $project.name
      file = $item.file
      line = $item.line
      column = $item.column
      message = $item.message
    })
  }
}

$grouped = @($rows | Group-Object file | Sort-Object Count -Descending)

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
