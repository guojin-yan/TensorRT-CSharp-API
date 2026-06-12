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

function Get-PublicStaticMethodCount {
  param(
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return 0
  }

  $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
  return ([regex]::Matches($content, "public\s+static\s+[^\(]+\s+[A-Za-z0-9_]+\s*\(")).Count
}

function Get-PrivateStaticMethodCount {
  param(
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return 0
  }

  $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
  return ([regex]::Matches($content, "private\s+static\s+[^\(]+\s+[A-Za-z0-9_]+\s*\(")).Count
}

$manifestRoot = Join-Path $RepositoryRoot "native\manifests"
$manifestFiles = @(Get-ChildItem -Path $manifestRoot -Filter *.manifest.json -Recurse | Sort-Object FullName)
$apiRows = New-Object System.Collections.Generic.List[object]

foreach ($file in $manifestFiles) {
  $document = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($api in @($document.apis)) {
    $apiRows.Add([pscustomobject]@{
      module = [string]$document.module
      versionLine = [string]$document.versionLine
      id = [string]$api.id
      entryPoint = [string]$api.entryPoint
      wrapperKind = [string]$api.wrapperKind
      bindingRole = [string]$api.bindingRole
    })
  }
}

$generatedFiles = @(
  "native/generated/bridge_api_catalog.g.h",
  "native/generated/bridge_entrypoints.g.h",
  "src/JYPPX.Shared/Generated/GeneratedApiCatalog.g.cs",
  "src/JYPPX.Shared/Generated/GeneratedEntryPointNames.g.cs",
  "src/JYPPX.Shared/Generated/GeneratedNativeMethods.g.cs",
  "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsTensorRt.Generated.g.cs",
  "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs",
  "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeBridgeApi.Common.Generated.g.cs",
  "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeBridgeApi.TensorRtBindings.Generated.g.cs",
  "src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeBridgeApi.TensorRtHelpers.Generated.g.cs",
  "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCuda.Generated.g.cs",
  "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCommon.Generated.g.cs",
  "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeBridgeApi.Common.Generated.g.cs",
  "src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeCudaApi.Generated.g.cs"
)

$fileRows = foreach ($relativePath in $generatedFiles) {
  $fullPath = Join-Path $RepositoryRoot $relativePath
  [pscustomobject]@{
    path = $relativePath
    exists = Test-Path -LiteralPath $fullPath -PathType Leaf
    bytes = if (Test-Path -LiteralPath $fullPath -PathType Leaf) { (Get-Item -LiteralPath $fullPath).Length } else { 0 }
  }
}

$summary = [ordered]@{
  manifestFileCount = $manifestFiles.Count
  manifestApiCount = $apiRows.Count
  byModule = @($apiRows | Group-Object module | Sort-Object Name | ForEach-Object { [pscustomobject]@{ module = $_.Name; apiCount = $_.Count } })
  byVersionLine = @($apiRows | Group-Object versionLine | Sort-Object Name | ForEach-Object { [pscustomobject]@{ versionLine = $_.Name; apiCount = $_.Count } })
  wrapperKindCount = @($apiRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.wrapperKind) }).Count
  bindingRoleCount = @($apiRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.bindingRole) }).Count
  nativeCudaGeneratedWrapperCount = Get-PublicStaticMethodCount -Path (Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeCudaApi.Generated.g.cs")
  tensorRtGeneratedHelperCoreCount = Get-PrivateStaticMethodCount -Path (Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.TensorRtHelpers.Generated.g.cs")
  nextCoverageCandidates = @(
    "TensorRT helper: metadata-driven TryCreateRuntime/TryCreateBuilder public wrapper skeletons.",
    "TensorRT helper: builder config/network create helper flow descriptors.",
    "CUDA helper: richer stream/event/memory operation metadata beyond thin wrappers.",
    "Split runtime: generated consumer validation metadata once split packages become validated candidates."
  )
  generationBoundaries = @(
    "TensorRT multi-step build chain remains manual until lifecycle metadata captures ownership and disposal order.",
    "CUDA pinned host-buffer copy remains descriptor-backed but not fully generated beyond the current wrapper boundary.",
    "Runtime split package consumer validation remains outside API generation until package-set metadata is finalized."
  )
  generatedFiles = @($fileRows)
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\interop-comparison"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "generated-api-coverage.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Generated API Coverage")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("- Manifest files: $($summary.manifestFileCount)")
$lines.Add("- Manifest API records: $($summary.manifestApiCount)")
$lines.Add("- APIs with wrapperKind metadata: $($summary.wrapperKindCount)")
$lines.Add("- APIs with bindingRole metadata: $($summary.bindingRoleCount)")
$lines.Add("- NativeCudaApi generated public wrapper count: $($summary.nativeCudaGeneratedWrapperCount)")
$lines.Add("- TensorRt helper generated private core count: $($summary.tensorRtGeneratedHelperCoreCount)")
$lines.Add("")
$lines.Add("## Module counts")
$lines.Add("")
$lines.Add("| Module | API count |")
$lines.Add("| --- | ---: |")
foreach ($row in $summary.byModule) {
  $lines.Add("| $($row.module) | $($row.apiCount) |")
}
$lines.Add("")
$lines.Add("## Version line counts")
$lines.Add("")
$lines.Add("| Version line | API count |")
$lines.Add("| --- | ---: |")
foreach ($row in $summary.byVersionLine) {
  $lines.Add("| $($row.versionLine) | $($row.apiCount) |")
}
$lines.Add("")
$lines.Add("## Generated files")
$lines.Add("")
$lines.Add("| File | Exists | Bytes |")
$lines.Add("| --- | --- | ---: |")
foreach ($file in $summary.generatedFiles) {
  $lines.Add("| $($file.path) | $($file.exists) | $($file.bytes) |")
}
$lines.Add("")
$lines.Add("## Engineering interpretation")
$lines.Add("")
$lines.Add("- NativeMethods partial generation covers the manifest entrypoint layer.")
$lines.Add("- NativeCudaApi thin wrappers are generated from manifest wrapperKind metadata.")
$lines.Add("- TensorRtSharp line bindings and helper core methods are generated from bindingRole metadata.")
$lines.Add("- Multi-step TensorRT build-chain orchestration remains manual until lifecycle metadata becomes richer.")
$lines.Add("")
$lines.Add("## Next coverage candidates")
$lines.Add("")
foreach ($candidate in $summary.nextCoverageCandidates) {
  $lines.Add("- $candidate")
}
$lines.Add("")
$lines.Add("## Current generation boundaries")
$lines.Add("")
foreach ($boundary in $summary.generationBoundaries) {
  $lines.Add("- $boundary")
}

$markdownPath = Join-Path $outputRoot "generated-api-coverage.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Generated API coverage report written to $jsonPath"
Write-Host "Generated API coverage report written to $markdownPath"
