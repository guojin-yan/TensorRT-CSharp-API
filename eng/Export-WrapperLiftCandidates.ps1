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

function Get-PublicMethodNames {
  param(
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return @()
  }

  $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
  $regex = [regex]::new("public\s+static\s+[^\(]+\s+(?<name>[A-Za-z0-9_]+)\s*\(", [System.Text.RegularExpressions.RegexOptions]::Singleline)
  return @(
    $regex.Matches($content) |
      ForEach-Object { $_.Groups["name"].Value } |
      Sort-Object -Unique
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\interop-comparison"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$nativeCudaGeneratedPath = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeCudaApi.Generated.g.cs"
$nativeCudaManualPath = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\Internal\Interop\NativeCudaApi.cs"
$nativeBridgeCommonTensorRtPath = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.Common.Generated.g.cs"
$nativeBridgeCommonCudaPath = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeBridgeApi.Common.Generated.g.cs"
$tensorRtBindingsGeneratedPath = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.TensorRtBindings.Generated.g.cs"
$tensorRtHelpersGeneratedPath = Join-Path $RepositoryRoot "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.TensorRtHelpers.Generated.g.cs"

$nativeCudaGeneratedMethods = @(Get-PublicMethodNames -Path $nativeCudaGeneratedPath)
$nativeCudaManualMethods = @(Get-PublicMethodNames -Path $nativeCudaManualPath)
$nativeBridgeCommonTensorRtMethods = @(Get-PublicMethodNames -Path $nativeBridgeCommonTensorRtPath)
$nativeBridgeCommonCudaMethods = @(Get-PublicMethodNames -Path $nativeBridgeCommonCudaPath)
$tensorRtBindingLines = if (Test-Path -LiteralPath $tensorRtBindingsGeneratedPath) { @((Get-Content -LiteralPath $tensorRtBindingsGeneratedPath -Encoding utf8) | Where-Object { $_ -match "TensorRtLineBindings\s+Trt(8|10)Bindings" }) } else { @() }
$tensorRtHelperMethods = @(Get-PublicMethodNames -Path $tensorRtHelpersGeneratedPath)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Wrapper Lift Candidates")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("This report identifies the next thin wrapper layers that are good candidates for generated or semi-generated lift.")
$lines.Add("")
$lines.Add("## Confirmed completed lift foundations")
$lines.Add("")
$lines.Add("- NativeMethodsTensorRt manifest coverage is complete.")
$lines.Add("- NativeMethodsCuda manifest coverage is complete.")
$lines.Add("- NativeMethodsCommon manifest coverage is complete in both TensorRtSharp and CudaSharp.")
$lines.Add("- NativeBridgeApi.Common.Generated.g.cs now owns the stable common bridge helper methods in both namespaces.")
$lines.Add("")
$lines.Add("## Current generated wrapper lift status")
$lines.Add("")
$lines.Add("### NativeBridgeApi.Common")
$lines.Add("")
$lines.Add("- TensorRtSharp generated helper count: $($nativeBridgeCommonTensorRtMethods.Count)")
$lines.Add("- CudaSharp generated helper count: $($nativeBridgeCommonCudaMethods.Count)")
$lines.Add("")
$lines.Add("### NativeCudaApi")
$lines.Add("")
$lines.Add("- Generated thin wrapper count: $($nativeCudaGeneratedMethods.Count)")
$lines.Add("- Manual thin wrapper count: $($nativeCudaManualMethods.Count)")
$lines.Add("")
if ($nativeCudaGeneratedMethods.Count -gt 0) {
  $lines.Add("Generated wrappers:")
  $lines.Add("")
  foreach ($method in $nativeCudaGeneratedMethods) {
    $lines.Add("- $method")
  }
  $lines.Add("")
}

if ($nativeCudaManualMethods.Count -gt 0) {
  $lines.Add("Manual wrappers still kept:")
  $lines.Add("")
  foreach ($method in $nativeCudaManualMethods) {
    $lines.Add("- $method")
  }
  $lines.Add("")
}

$lines.Add("### NativeBridgeApi.TensorRtLineBindings")
$lines.Add("")
$lines.Add("- Generated line binding count: $($tensorRtBindingLines.Count)")
$lines.Add("")

$lines.Add("### NativeBridgeApi.TensorRtHelpers")
$lines.Add("")
$lines.Add("- Generated helper method count: $($tensorRtHelperMethods.Count)")
$lines.Add("")

$lines.Add("## Next likely lift targets")
$lines.Add("")
$lines.Add("### CudaSharp")
$lines.Add("")
$lines.Add("- PinnedByteBufferScope metadata model")
$lines.Add("- Generic pinned host buffer descriptor")
$lines.Add("")
$lines.Add("The remaining pinning boundary has already been isolated; the next step is to make that boundary metadata-driven instead of open-coded.")
$lines.Add("")
$lines.Add("### TensorRtSharp")
$lines.Add("")
$lines.Add("- adapter info query helper flow")
$lines.Add("- logger/runtime/builder create helper flow")
$lines.Add("- minimal build chain orchestration should remain manual until there is a stronger manifest model for object lifecycle and multi-step flows")
$lines.Add("")
$lines.Add("## Recommended next lift order")
$lines.Add("")
$lines.Add("1. Metadata-driven pinned host buffer model for NativeCudaApi copy helpers")
$lines.Add("2. TensorRtSharp adapter-line query helpers")
$lines.Add("3. TensorRtSharp logger/runtime/builder create helpers")
$lines.Add("4. keep multi-step TensorRT build chains manual until lifecycle metadata is richer")

$reportPath = Join-Path $outputRoot "wrapper-lift-candidates.md"
$lines | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "Wrapper lift candidate report written to $reportPath"
