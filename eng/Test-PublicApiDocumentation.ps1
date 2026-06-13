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

function Test-IsCandidateDeclaration {
  param(
    [string]$Line
  )

  if ($Line -match '^\s*//') {
    return $false
  }

  if ($Line -notmatch '\bpublic\b') {
    return $false
  }

  if ($Line -match '\b(public\s+const|public\s+static\s+readonly)\b') {
    return $false
  }

  return $Line -match '\b(public\s+(sealed\s+|static\s+|abstract\s+|partial\s+|readonly\s+|ref\s+)*' +
    '(class|struct|record|interface|enum|delegate)\b|' +
    'public\s+.*\b(get|set|init)\s*;' +
    '|public\s+.*\(' +
    '|public\s+.*\{)'
}

function Test-HasXmlSummary {
  param(
    [string[]]$Lines,
    [int]$Index
  )

  for ($i = $Index - 1; $i -ge 0; $i--) {
    $trimmed = $Lines[$i].Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
      continue
    }

    if ($trimmed -like '/// <summary>*') {
      return $true
    }

    if ($trimmed.StartsWith('/// ')) {
      continue
    }

    return $false
  }

  return $false
}

$sourceRoot = Join-Path $RepositoryRoot "src"
$files = @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter *.cs -File | Where-Object {
    $_.FullName -notlike '*\bin\*' -and
    $_.FullName -notlike '*\obj\*' -and
    $_.FullName -notlike '*\Generated\*'
  })

$missing = New-Object System.Collections.Generic.List[object]
foreach ($file in $files) {
  $lines = Get-Content -LiteralPath $file.FullName -Encoding utf8
  for ($index = 0; $index -lt $lines.Count; $index++) {
    if (-not (Test-IsCandidateDeclaration -Line $lines[$index])) {
      continue
    }

    if (Test-HasXmlSummary -Lines $lines -Index $index) {
      continue
    }

    $relativePath = $file.FullName.Substring($RepositoryRoot.Length).TrimStart('\')
    $missing.Add([pscustomobject]@{
      file = $relativePath
      line = $index + 1
      declaration = $lines[$index].Trim()
    })
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\api-doc-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "public-api-documentation-audit.json"
$markdownPath = Join-Path $outputRoot "public-api-documentation-audit.md"

$missing | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Public API Documentation Audit")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("- Missing XML summary count: $($missing.Count)")
$lines.Add("")
$lines.Add("| File | Line | Declaration |")
$lines.Add("| --- | ---: | --- |")
foreach ($item in $missing) {
  $declaration = $item.declaration.Replace("|", "\|")
  $lines.Add("| `$($item.file)` | $($item.line) | `$declaration` |")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public API documentation audit written to $jsonPath"
Write-Host "Public API documentation audit written to $markdownPath"

if ($missing.Count -gt 0) {
  Write-Warning "Found $($missing.Count) public declaration(s) without an XML summary comment."
  exit 1
}
