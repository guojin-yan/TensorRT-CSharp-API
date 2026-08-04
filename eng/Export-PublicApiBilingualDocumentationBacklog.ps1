[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-MemberBody {
  param([string]$Member)

  if ([string]::IsNullOrWhiteSpace($Member)) {
    return ""
  }

  if ($Member.Length -gt 2 -and $Member[1] -eq ':') {
    return $Member.Substring(2)
  }

  return $Member
}

function Get-TypeNameFromMember {
  param(
    [string]$Project,
    [string]$Member
  )

  $body = Get-MemberBody -Member $Member
  if ([string]::IsNullOrWhiteSpace($body)) {
    return ""
  }

  $namespacePrefix = "JYPPX.TensorRtSharp."
  if ([string]::Equals($Project, "JYPPX.CudaSharp", [System.StringComparison]::OrdinalIgnoreCase)) {
    $namespacePrefix = "JYPPX.CudaSharp."
  }
  elseif ([string]::Equals($Project, "JYPPX.Shared", [System.StringComparison]::OrdinalIgnoreCase)) {
    $namespacePrefix = "JYPPX.TensorRtSharp.Shared."
  }

  if ($body.StartsWith($namespacePrefix, [System.StringComparison]::Ordinal)) {
    $body = $body.Substring($namespacePrefix.Length)
  }

  $paren = $body.IndexOf('(')
  if ($paren -ge 0) {
    $body = $body.Substring(0, $paren)
  }

  $parts = @($body.Split('.') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($parts.Count -eq 0) {
    return $body
  }

  return $parts[0]
}

function Get-RecommendedBatch {
  param(
    [string]$Project,
    [string]$TypeName
  )

  if ([string]::Equals($Project, "JYPPX.CudaSharp", [System.StringComparison]::OrdinalIgnoreCase)) {
    return "P0-CUDA-remaining-cleanup"
  }

  $highValueWrappers = @(
    "TensorRtLogger",
    "TensorRtRuntime",
    "TensorRtBuilder",
    "TensorRtBuilderConfig",
    "TensorRtNetworkDefinition",
    "TensorRtEngine",
    "TensorRtExecutionContext",
    "TensorRtPluginRegistryInventory",
    "TensorRtOnnxParser",
    "TensorRtOnnxParserRefitter",
    "TensorRtRefitter",
    "TensorRtHostMemory",
    "TensorRtOptimizationProfile",
    "TensorRtInferenceBindings",
    "TensorRtInferenceBuffer"
  )

  if ($highValueWrappers -contains $TypeName) {
    return "P1-high-value-wrapper-docs"
  }

  if ($TypeName -match 'DebugListener|OutputAllocator|Allocator') {
    return "P2-callback-allocator-boundary-docs"
  }

  if ($TypeName -match 'Logger|Profiler|ProgressMonitor|Callback') {
    return "P3-managed-callback-surface-docs"
  }

  if ($TypeName -match 'Gate|Proof|Precheck|Snapshot|Readiness|Boundary|Ledger|DryRun') {
    return "P4-diagnostic-gate-docs"
  }

  return "P5-remaining-tensorrt-surface-docs"
}

function Resolve-SourceHint {
  param(
    [string]$Project,
    [string]$TypeName,
    [hashtable]$Cache
  )

  $key = "$Project|$TypeName"
  if ($Cache.ContainsKey($key)) {
    return $Cache[$key]
  }

  if ([string]::IsNullOrWhiteSpace($TypeName)) {
    $Cache[$key] = ""
    return ""
  }

  $projectDirectory = Join-Path $RepositoryRoot "src\$Project"
  if (-not (Test-Path -LiteralPath $projectDirectory -PathType Container)) {
    $Cache[$key] = ""
    return ""
  }

  $exact = Join-Path $projectDirectory "$TypeName.cs"
  if (Test-Path -LiteralPath $exact -PathType Leaf) {
    $Cache[$key] = ("src\$Project\$TypeName.cs")
    return $Cache[$key]
  }

  $candidate = Get-ChildItem -LiteralPath $projectDirectory -Filter "$TypeName*.cs" -File -ErrorAction SilentlyContinue |
    Sort-Object Name |
    Select-Object -First 1
  if ($candidate) {
    $relative = $candidate.FullName.Substring($RepositoryRoot.Length).TrimStart('\')
    $Cache[$key] = $relative
    return $relative
  }

  $escapedTypeName = [System.Text.RegularExpressions.Regex]::Escape($TypeName)
  $typeDeclarationPattern = "\b(class|struct|enum|delegate|interface)\s+$escapedTypeName\b"
  $declaringFile = Get-ChildItem -LiteralPath $projectDirectory -Filter "*.cs" -File -ErrorAction SilentlyContinue |
    Select-String -Pattern $typeDeclarationPattern -List -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($declaringFile) {
    $relative = $declaringFile.Path.Substring($RepositoryRoot.Length).TrimStart('\')
    $Cache[$key] = $relative
    return $relative
  }

  $Cache[$key] = ""
  return ""
}

$auditPath = Join-Path $RepositoryRoot "artifacts\api-doc-audit\public-api-bilingual-documentation-audit.json"
if (-not (Test-Path -LiteralPath $auditPath -PathType Leaf)) {
  throw "Public API bilingual documentation audit was not found: $auditPath"
}

$audit = Get-Content -LiteralPath $auditPath -Raw -Encoding utf8 | ConvertFrom-Json
$sourceHints = @{}
$rows = New-Object System.Collections.Generic.List[object]

foreach ($finding in @($audit.findings)) {
  $project = [string]$finding.project
  $member = [string]$finding.member
  $typeName = Get-TypeNameFromMember -Project $project -Member $member
  $missing = @()
  if (-not [bool]$finding.hasEnglish) { $missing += "English" }
  if (-not [bool]$finding.hasChinese) { $missing += "Chinese" }
  $sourceHint = Resolve-SourceHint -Project $project -TypeName $typeName -Cache $sourceHints

  $rows.Add([pscustomobject]@{
      recommendedBatch = Get-RecommendedBatch -Project $project -TypeName $typeName
      project = $project
      typeName = $typeName
      sourceHint = $sourceHint
      member = $member
      element = [string]$finding.element
      name = [string]$finding.name
      missing = ($missing -join ", ")
      text = [string]$finding.text
    })
}

$projectSummaries = @($rows | Group-Object project | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
      project = $_.Name
      findingCount = $_.Count
    }
  })

$elementSummaries = @($rows | Group-Object element | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
      element = $_.Name
      findingCount = $_.Count
    }
  })

$batchSummaries = @($rows | Group-Object recommendedBatch | Sort-Object Name | ForEach-Object {
    $topTypes = @($_.Group | Group-Object typeName | Sort-Object @{ Expression = "Count"; Descending = $true }, Name | Select-Object -First 12 | ForEach-Object {
        [pscustomobject]@{
          typeName = $_.Name
          findingCount = $_.Count
        }
      })

    [pscustomobject]@{
      recommendedBatch = $_.Name
      findingCount = $_.Count
      topTypes = $topTypes
    }
  })

$topTypeSummaries = @($rows | Group-Object project, typeName | Sort-Object Count -Descending | Select-Object -First 100 | ForEach-Object {
    $sample = $_.Group | Select-Object -First 1
    [pscustomobject]@{
      project = $sample.project
      typeName = $sample.typeName
      recommendedBatch = $sample.recommendedBatch
      sourceHint = $sample.sourceHint
      findingCount = $_.Count
    }
  })

$outputRoot = Join-Path $RepositoryRoot "artifacts\api-doc-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "public-api-bilingual-documentation-backlog.json"
$markdownPath = Join-Path $outputRoot "public-api-bilingual-documentation-backlog.md"

[pscustomobject]@{
  generatedOn = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  sourceAudit = "artifacts/api-doc-audit/public-api-bilingual-documentation-audit.json"
  inputFindingCount = [int]$audit.findingCount
  backlogFindingCount = $rows.Count
  projectSummaries = $projectSummaries
  elementSummaries = $elementSummaries
  batchSummaries = $batchSummaries
  topTypeSummaries = $topTypeSummaries
  backlogItems = @($rows.ToArray())
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$codeQuote = [string][char]96
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Public API Bilingual Documentation Backlog")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("Source audit: " + $codeQuote + "artifacts/api-doc-audit/public-api-bilingual-documentation-audit.json" + $codeQuote)
$lines.Add("")
$lines.Add("This backlog groups public documentation elements that are not bilingual. It is a work queue, not a release pass. A finding remains open until " + $codeQuote + "Test-PublicApiBilingualDocumentation.ps1" + $codeQuote + " reports zero findings.")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- input findings: $([int]$audit.findingCount)")
$lines.Add("- backlog findings: $($rows.Count)")
$lines.Add("")
$lines.Add("## Project Summary")
$lines.Add("")
$lines.Add("| Project | Findings |")
$lines.Add("| --- | ---: |")
foreach ($summary in $projectSummaries) {
  $lines.Add("| $($summary.project) | $($summary.findingCount) |")
}

$lines.Add("")
$lines.Add("## Recommended Batches")
$lines.Add("")
$lines.Add("| Batch | Findings | Top types |")
$lines.Add("| --- | ---: | --- |")
foreach ($summary in $batchSummaries) {
  $topTypes = @($summary.topTypes | ForEach-Object { "$($_.typeName) ($($_.findingCount))" }) -join ", "
  $lines.Add("| $($summary.recommendedBatch) | $($summary.findingCount) | $(ConvertTo-MarkdownCell $topTypes) |")
}

$lines.Add("")
$lines.Add("## Top Type Backlog")
$lines.Add("")
$lines.Add("| Project | Type | Batch | Findings | Source hint |")
$lines.Add("| --- | --- | --- | ---: | --- |")
foreach ($summary in ($topTypeSummaries | Select-Object -First 60)) {
  $lines.Add("| $($summary.project) | " + $codeQuote + "$($summary.typeName)" + $codeQuote + " | $($summary.recommendedBatch) | $($summary.findingCount) | " + $codeQuote + "$($summary.sourceHint)" + $codeQuote + " |")
}

$lines.Add("")
$lines.Add("## First 400 Backlog Items")
$lines.Add("")
$lines.Add("| Batch | Project | Type | Element | Name | Missing | Member | Text |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- |")
foreach ($row in ($rows | Select-Object -First 400)) {
  $lines.Add("| $($row.recommendedBatch) | $($row.project) | " + $codeQuote + "$($row.typeName)" + $codeQuote + " | $($row.element) | $(ConvertTo-MarkdownCell $row.name) | $($row.missing) | " + $codeQuote + "$($row.member)" + $codeQuote + " | $(ConvertTo-MarkdownCell $row.text) |")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public API bilingual documentation backlog written to $jsonPath"
Write-Host "Public API bilingual documentation backlog written to $markdownPath"
