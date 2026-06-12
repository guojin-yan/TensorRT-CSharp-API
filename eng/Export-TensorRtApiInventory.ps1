[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\api-inventory"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$manifestRoot = Join-Path $RepositoryRoot "native\manifests\tensorrt"
if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
  throw "TensorRT manifest root was not found: $manifestRoot"
}

function Get-ApiCategory {
  param([string]$EntryPoint)

  if ($EntryPoint -match "_onnx_parser_") { return "onnx-parser" }
  if ($EntryPoint -match "_optimization_profile_") { return "optimization-profile" }
  if ($EntryPoint -match "_builder_config_") { return "builder-config" }
  if ($EntryPoint -match "_execution_context_") { return "execution-context" }
  if ($EntryPoint -match "_engine_inspector_") { return "engine-inspector" }
  if ($EntryPoint -match "_engine_") { return "engine" }
  if ($EntryPoint -match "_network_") { return "network" }
  if ($EntryPoint -match "_runtime_") { return "runtime" }
  if ($EntryPoint -match "_builder_") { return "builder" }
  if ($EntryPoint -match "_host_memory_") { return "host-memory" }
  if ($EntryPoint -match "_timing_cache_") { return "timing-cache" }
  if ($EntryPoint -match "_logger_") { return "logger" }
  if ($EntryPoint -match "_query_adapter_info$") { return "adapter-info" }
  if ($EntryPoint -eq "jyppx_trt_object_destroy") { return "object-lifetime" }
  return "other"
}

function Get-DeploymentRole {
  param([string]$Category)

  switch ($Category) {
    "adapter-info" { "environment-query" }
    "logger" { "minimal-lifecycle" }
    "runtime" { "minimal-lifecycle" }
    "builder" { "minimal-lifecycle" }
    "builder-config" { "deployment-critical" }
    "network" { "deployment-critical" }
    "optimization-profile" { "dynamic-shape-critical" }
    "onnx-parser" { "model-import-critical" }
    "host-memory" { "serialization-critical" }
    "timing-cache" { "build-performance-critical" }
    "engine" { "deployment-critical" }
    "engine-inspector" { "debugging-critical" }
    "execution-context" { "inference-critical" }
    "object-lifetime" { "minimal-lifecycle" }
    default { "supporting" }
  }
}

function Get-ApiLine {
  param(
    [object]$Document,
    [string]$EntryPoint
  )

  if ($EntryPoint -match "^jyppx_trt(?<line>\d+)_") {
    return $Matches["line"]
  }

  if (-not [string]::IsNullOrWhiteSpace([string]$Document.versionLine)) {
    return [string]$Document.versionLine
  }

  return "common"
}

function Get-NormalizedEntryPoint {
  param([string]$EntryPoint)

  return ($EntryPoint -replace "^jyppx_trt(8|10|11)_", "jyppx_trtX_")
}

$apis = New-Object System.Collections.Generic.List[object]
$manifestFiles = @(Get-ChildItem -Path $manifestRoot -Recurse -Filter "*.manifest.json" | Sort-Object FullName)
foreach ($file in $manifestFiles) {
  $document = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($api in @($document.apis)) {
    $entryPoint = [string]$api.entryPoint
    $category = Get-ApiCategory -EntryPoint $entryPoint
    $line = Get-ApiLine -Document $document -EntryPoint $entryPoint

    $apis.Add([pscustomobject]@{
      id = [string]$api.id
      line = $line
      module = [string]$document.module
      category = $category
      deploymentRole = Get-DeploymentRole -Category $category
      entryPoint = $entryPoint
      normalizedEntryPoint = Get-NormalizedEntryPoint -EntryPoint $entryPoint
      ownership = [string]$api.ownership
      manualOverride = [bool]$api.manualOverride
      versionGuard = [string]$api.versionGuard
      manifest = $file.FullName.Substring($RepositoryRoot.Length + 1).Replace("\", "/")
      parameterCount = @($api.parameters).Count
    })
  }
}

if ($apis.Count -eq 0) {
  throw "No TensorRT API records were found under $manifestRoot."
}

$lineSummaries = @(
  $apis |
    Group-Object line |
    Sort-Object Name |
    ForEach-Object {
      [pscustomobject]@{
        line = $_.Name
        count = $_.Count
        deploymentCriticalCount = @($_.Group | Where-Object { $_.deploymentRole -match "critical" -or $_.deploymentRole -eq "minimal-lifecycle" }).Count
        manualOverrideCount = @($_.Group | Where-Object { $_.manualOverride }).Count
      }
    }
)

$categorySummaries = @(
  $apis |
    Group-Object line, category |
    Sort-Object Name |
    ForEach-Object {
      $parts = $_.Name -split ", "
      [pscustomobject]@{
        line = $parts[0]
        category = $parts[1]
        count = $_.Count
      }
    }
)

$lines = @($apis | Where-Object { $_.line -ne "common" } | Select-Object -ExpandProperty line -Unique | Sort-Object)
$normalizedNames = @($apis | Where-Object { $_.line -ne "common" } | Select-Object -ExpandProperty normalizedEntryPoint -Unique | Sort-Object)
$normalizedCoverage = New-Object System.Collections.Generic.List[object]
foreach ($name in $normalizedNames) {
  $present = @($apis | Where-Object { $_.normalizedEntryPoint -eq $name } | Select-Object -ExpandProperty line -Unique | Sort-Object)
  $missing = @($lines | Where-Object { $present -notcontains $_ })
  $category = @($apis | Where-Object { $_.normalizedEntryPoint -eq $name } | Select-Object -First 1 -ExpandProperty category)
  $role = @($apis | Where-Object { $_.normalizedEntryPoint -eq $name } | Select-Object -First 1 -ExpandProperty deploymentRole)
  $normalizedCoverage.Add([pscustomobject]@{
    normalizedEntryPoint = $name
    category = if ($category.Count -gt 0) { $category[0] } else { "unknown" }
    deploymentRole = if ($role.Count -gt 0) { $role[0] } else { "unknown" }
    presentLines = $present
    missingLines = $missing
  })
}

$nextMigrationCandidates = @(
  $normalizedCoverage |
    Where-Object {
      $_.missingLines.Count -gt 0 -and
      ($_.deploymentRole -match "critical" -or $_.deploymentRole -eq "minimal-lifecycle")
    } |
    Sort-Object category, normalizedEntryPoint |
    Select-Object -First 40
)

$report = [pscustomobject]@{
  generatedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  totalTensorRtApiCount = $apis.Count
  lineSummaries = $lineSummaries
  categorySummaries = $categorySummaries
  normalizedCoverage = $normalizedCoverage
  nextMigrationCandidates = $nextMigrationCandidates
}

$jsonPath = Join-Path $OutputDirectory "tensorrt-api-inventory.json"
$markdownPath = Join-Path $OutputDirectory "tensorrt-api-inventory.md"

$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# TensorRT API Inventory")
$md.Add("")
$md.Add("Generated: $($report.generatedAt)")
$md.Add("")
$md.Add("## Summary By Line")
$md.Add("")
$md.Add("| Line | API Count | Deployment-Critical / Lifecycle | Manual Override |")
$md.Add("| --- | ---: | ---: | ---: |")
foreach ($row in $lineSummaries) {
  $md.Add("| ``$($row.line)`` | $($row.count) | $($row.deploymentCriticalCount) | $($row.manualOverrideCount) |")
}

$md.Add("")
$md.Add("## Category Coverage")
$md.Add("")
$md.Add("| Line | Category | API Count |")
$md.Add("| --- | --- | ---: |")
foreach ($row in $categorySummaries) {
  $md.Add("| ``$($row.line)`` | $($row.category) | $($row.count) |")
}

$md.Add("")
$md.Add("## Cross-Line Gap Candidates")
$md.Add("")
if ($nextMigrationCandidates.Count -eq 0) {
  $md.Add("No deployment-critical cross-line gaps were detected from manifest metadata.")
} else {
  $md.Add("| Normalized Entrypoint | Category | Present Lines | Missing Lines |")
  $md.Add("| --- | --- | --- | --- |")
  foreach ($row in $nextMigrationCandidates) {
    $md.Add("| ``$($row.normalizedEntryPoint)`` | $($row.category) | " + (@($row.presentLines) -join ", ") + " | " + (@($row.missingLines) -join ", ") + " |")
  }
}

$md.Add("")
$md.Add("## Decision")
$md.Add("")
$md.Add("Use this report as the tactical map for Windows-first API completion. New APIs should enter manifests first, then generated NativeMethods, managed helper/wrapper routing, and smoke validation.")

[System.IO.File]::WriteAllLines($markdownPath, $md, $utf8)

Write-Host "TensorRT API inventory report written to $jsonPath"
Write-Host "TensorRT API inventory report written to $markdownPath"
