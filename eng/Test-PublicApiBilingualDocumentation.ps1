[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [switch]$SkipBuild
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-DocumentationText {
  param(
    [object]$Value
  )

  if ($null -eq $Value) {
    return ""
  }

  if ($Value -is [System.Xml.XmlNode]) {
    return ([string]$Value.InnerText -replace '\s+', ' ').Trim()
  }

  return ([string]$Value -replace '\s+', ' ').Trim()
}

function Test-BilingualText {
  param(
    [string]$Text
  )

  [pscustomobject]@{
    hasEnglish = $Text -match '[A-Za-z]'
    hasChinese = $Text -match '\p{IsCJKUnifiedIdeographs}'
  }
}

if (-not $SkipBuild) {
  & (Join-Path $RepositoryRoot "eng\Test-PublicApiDocumentation.ps1") -RepositoryRoot $RepositoryRoot
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

$projects = @(
  [pscustomobject]@{
    name = "JYPPX.Shared"
    xml = Join-Path $RepositoryRoot "src\JYPPX.Shared\bin\Release\net8.0\JYPPX.Shared.xml"
  }
  [pscustomobject]@{
    name = "JYPPX.CudaSharp"
    xml = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\bin\Release\net8.0\JYPPX.CudaSharp.xml"
  }
  [pscustomobject]@{
    name = "JYPPX.TensorRtSharp"
    xml = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\bin\Release\net8.0\JYPPX.TensorRtSharp.xml"
  }
)

$rows = New-Object System.Collections.Generic.List[object]
foreach ($project in $projects) {
  if (-not (Test-Path -LiteralPath $project.xml -PathType Leaf)) {
    throw "Documentation XML was not found for $($project.name): $($project.xml)"
  }

  [xml]$doc = Get-Content -LiteralPath $project.xml -Raw -Encoding utf8
  foreach ($member in @($doc.doc.members.member)) {
    foreach ($elementName in @("summary", "returns", "remarks")) {
      $text = Get-DocumentationText -Value $member.$elementName
      if ([string]::IsNullOrWhiteSpace($text)) {
        continue
      }

      $result = Test-BilingualText -Text $text
      if (-not $result.hasEnglish -or -not $result.hasChinese) {
        $rows.Add([pscustomobject]@{
          project = $project.name
          member = [string]$member.name
          element = $elementName
          name = ""
          hasEnglish = $result.hasEnglish
          hasChinese = $result.hasChinese
          text = $text
        })
      }
    }

    foreach ($param in @($member.param)) {
      if ($null -eq $param) {
        continue
      }

      $text = Get-DocumentationText -Value $param
      if ([string]::IsNullOrWhiteSpace($text)) {
        continue
      }

      $result = Test-BilingualText -Text $text
      if (-not $result.hasEnglish -or -not $result.hasChinese) {
        $rows.Add([pscustomobject]@{
          project = $project.name
          member = [string]$member.name
          element = "param"
          name = [string]$param.name
          hasEnglish = $result.hasEnglish
          hasChinese = $result.hasChinese
          text = $text
        })
      }
    }
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\api-doc-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "public-api-bilingual-documentation-audit.json"
$markdownPath = Join-Path $outputRoot "public-api-bilingual-documentation-audit.md"

$projectSummaries = @($rows | Group-Object project | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
      project = $_.Name
      findingCount = $_.Count
    }
  })

[pscustomobject]@{
  generatedOn = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  findingCount = $rows.Count
  projectSummaries = $projectSummaries
  findings = $rows
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Public API Bilingual Documentation Audit")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("Checks public documentation XML for English and Chinese text in summary, param, returns, and remarks elements.")
$lines.Add("")
$lines.Add("## Project Summary")
$lines.Add("")
$lines.Add("| Project | Finding count |")
$lines.Add("| --- | ---: |")
foreach ($summary in $projectSummaries) {
  $lines.Add("| $($summary.project) | $($summary.findingCount) |")
}

$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
$lines.Add("| Project | Member | Element | Name | Missing | Text |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($row in ($rows | Select-Object -First 300)) {
  $missing = @()
  if (-not $row.hasEnglish) { $missing += "English" }
  if (-not $row.hasChinese) { $missing += "Chinese" }
  $text = $row.text.Replace("|", "\|")
  $lines.Add("| $($row.project) | " + $codeQuote + $row.member + $codeQuote + " | $($row.element) | $($row.name) | " + ($missing -join ", ") + " | $text |")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public API bilingual documentation audit written to $jsonPath"
Write-Host "Public API bilingual documentation audit written to $markdownPath"

if ($rows.Count -gt 0) {
  Write-Warning "Found $($rows.Count) public documentation element(s) that are not bilingual."
  exit 1
}
