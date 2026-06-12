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

$outputPath = Join-Path $RepositoryRoot "artifacts\interop-comparison"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

function Get-DllImportMethods {
  param(
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return @()
  }

  $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
  $regex = [regex]::new("\[DllImport\([^\]]+\)\]\s*internal\s+static\s+extern\s+[^\(]+\s+(?<name>jyppx_[A-Za-z0-9_]+)\s*\(", [System.Text.RegularExpressions.RegexOptions]::Singleline)
  $matches = $regex.Matches($content)

  return @(
    $matches |
      ForEach-Object { $_.Groups["name"].Value } |
      Sort-Object -Unique
  )
}

function Get-ModuleLabel {
  param(
    [string]$Module
  )

  switch ($Module.ToLowerInvariant()) {
    "tensorrt" { return "TensorRT" }
    "cuda" { return "CUDA" }
    "common" { return "Common" }
    default { return $Module }
  }
}

$manifestRoot = Join-Path $RepositoryRoot "native\manifests"
$manifestEntries = New-Object System.Collections.Generic.List[object]

Get-ChildItem -Path $manifestRoot -Filter *.manifest.json -Recurse | Sort-Object FullName | ForEach-Object {
  $document = Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  $moduleLabel = Get-ModuleLabel -Module $document.module

  foreach ($api in $document.apis) {
    $manifestEntries.Add([pscustomobject]@{
      module = $moduleLabel
      versionLine = [string]$document.versionLine
      id = [string]$api.id
      entryPoint = [string]$api.entryPoint
    })
  }
}

$pairs = @(
  [pscustomobject]@{
    generated = "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/GeneratedTensorRtManifestNativeMethods.g.cs"
    partial = "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsTensorRt.Generated.g.cs"
    existing = "src/JYPPX.TensorRtSharp/Internal/Interop/NativeMethodsTensorRt.cs"
    module = "TensorRT"
    manifestModule = "TensorRT"
  },
  [pscustomobject]@{
    generated = "src/JYPPX.CudaSharp/Internal/Interop/Generated/GeneratedCudaManifestNativeMethods.g.cs"
    partial = "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCuda.Generated.g.cs"
    existing = "src/JYPPX.CudaSharp/Internal/Interop/NativeMethodsCuda.cs"
    module = "CUDA"
    manifestModule = "CUDA"
  },
  [pscustomobject]@{
    generated = "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs"
    partial = "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs"
    existing = "src/JYPPX.TensorRtSharp/Internal/Interop/NativeMethodsCommon.cs"
    module = "Common (TensorRtSharp)"
    manifestModule = "Common"
  },
  [pscustomobject]@{
    generated = "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs"
    partial = "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs"
    existing = "src/JYPPX.CudaSharp/Internal/Interop/NativeMethodsCommon.cs"
    module = "Common (CudaSharp)"
    manifestModule = "Common"
  }
)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# NativeMethods Comparison")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("Status meanings:")
$lines.Add("")
$lines.Add("- generated-partial: same-name partial NativeMethods already owns this entrypoint.")
$lines.Add("- handwritten-only: manifest covers the entrypoint, but only the handwritten file currently declares it.")
$lines.Add("- generated-helper-only: manifest-generated helper file has the declaration, but the same-name partial file has not taken it over yet.")
$lines.Add("- missing: manifest describes the entrypoint, but neither handwritten nor generated files currently expose it.")
$lines.Add("")

foreach ($pair in $pairs) {
  $generatedPath = Join-Path $RepositoryRoot $pair.generated
  $partialPath = Join-Path $RepositoryRoot $pair.partial
  $existingPath = Join-Path $RepositoryRoot $pair.existing

  $generatedMethods = @(Get-DllImportMethods -Path $generatedPath)
  $partialMethods = @(Get-DllImportMethods -Path $partialPath)
  $existingMethods = @(Get-DllImportMethods -Path $existingPath)

  $moduleEntries = @(
    $manifestEntries |
      Where-Object { $_.module -eq $pair.manifestModule } |
      Sort-Object versionLine, id
  )

  $statusRows = foreach ($entry in $moduleEntries) {
    $inExisting = $existingMethods -contains $entry.entryPoint
    $inPartial = $partialMethods -contains $entry.entryPoint
    $inGenerated = $generatedMethods -contains $entry.entryPoint

    $status = if ($inPartial) {
      "generated-partial"
    }
    elseif ($inExisting) {
      "handwritten-only"
    }
    elseif ($inGenerated) {
      "generated-helper-only"
    }
    else {
      "missing"
    }

    [pscustomobject]@{
      id = $entry.id
      entryPoint = $entry.entryPoint
      versionLine = $entry.versionLine
      handwritten = $inExisting
      partialGenerated = $inPartial
      generatedHelper = $inGenerated
      status = $status
    }
  }

  $handwrittenOnlyNotInManifest = @(
    $existingMethods |
      Where-Object { $_ -notin $moduleEntries.entryPoint } |
      Sort-Object -Unique
  )

  $coveragePercent = if ($moduleEntries.Count -eq 0) {
    0
  }
  else {
    [math]::Round((($statusRows | Where-Object { $_.status -eq 'generated-partial' }).Count / $moduleEntries.Count) * 100, 2)
  }

  $nextCandidates = @(
    $statusRows |
      Where-Object { $_.status -ne 'generated-partial' } |
      Select-Object -ExpandProperty entryPoint
  )

  if ($nextCandidates.Count -eq 0 -and $handwrittenOnlyNotInManifest.Count -gt 0) {
    $nextCandidates = $handwrittenOnlyNotInManifest
  }

  $lines.Add("## $($pair.module)")
  $lines.Add("")
  $lines.Add("- Existing: $($pair.existing)")
  $lines.Add("- Generated helper: $($pair.generated)")
  $lines.Add("- Partial generated: $($pair.partial)")
  $lines.Add("- Manifest entry count: $($moduleEntries.Count)")
  $lines.Add("- Generated partial count: $(($statusRows | Where-Object { $_.status -eq 'generated-partial' }).Count)")
  $lines.Add("- Handwritten-only count: $(($statusRows | Where-Object { $_.status -eq 'handwritten-only' }).Count)")
  $lines.Add("- Generated-helper-only count: $(($statusRows | Where-Object { $_.status -eq 'generated-helper-only' }).Count)")
  $lines.Add("- Missing count: $(($statusRows | Where-Object { $_.status -eq 'missing' }).Count)")
  $lines.Add("- Generated partial coverage: $coveragePercent%")
  $lines.Add("")
  $lines.Add("| ID | EntryPoint | Line | Handwritten | Partial | Helper | Status |")
  $lines.Add("| --- | --- | --- | --- | --- | --- | --- |")

  foreach ($row in $statusRows) {
    $lines.Add("| $($row.id) | $($row.entryPoint) | $($row.versionLine) | $($row.handwritten) | $($row.partialGenerated) | $($row.generatedHelper) | $($row.status) |")
  }

  $lines.Add("")

  if ($handwrittenOnlyNotInManifest.Count -gt 0) {
    $lines.Add("### Handwritten methods not yet described by manifest")
    $lines.Add("")
    foreach ($method in $handwrittenOnlyNotInManifest) {
      $lines.Add("- $method")
    }
    $lines.Add("")
  }

  $lines.Add("### Next migration candidates")
  $lines.Add("")
  if ($nextCandidates.Count -eq 0) {
    $lines.Add("- none")
  }
  else {
    foreach ($candidate in $nextCandidates | Sort-Object -Unique) {
      $lines.Add("- $candidate")
    }
  }
  $lines.Add("")
}

$reportPath = Join-Path $outputPath "native-methods-comparison.md"
$lines | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "NativeMethods comparison report written to $reportPath"
