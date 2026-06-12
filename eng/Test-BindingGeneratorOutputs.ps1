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

$requiredManifestProperties = @("module", "versionLine", "apis")
$requiredApiProperties = @("id", "entryPoint", "returnType", "ownership", "manualOverride", "parameters")
$requiredParameterProperties = @("name", "type", "direction")

function Assert-Property {
  param(
    [object]$Object,
    [string]$PropertyName,
    [string]$Context
  )

  $property = $Object.PSObject.Properties[$PropertyName]
  if ($null -eq $property) {
    throw "$Context is missing required property '$PropertyName'."
  }

  if ($null -eq $property.Value) {
    throw "$Context has an empty required property '$PropertyName'."
  }

  if ($property.Value -is [string] -and [string]::IsNullOrWhiteSpace($property.Value)) {
    throw "$Context has an empty required property '$PropertyName'."
  }
}

function Get-GeneratedFileHashes {
  param(
    [string[]]$Paths
  )

  $hashes = @{}
  foreach ($path in $Paths) {
    $fullPath = Join-Path $RepositoryRoot $path
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
      throw "Expected generated file was not found: $fullPath"
    }

    $file = Get-Item -LiteralPath $fullPath
    if ($file.Length -le 0) {
      throw "Expected generated file is empty: $fullPath"
    }

    $hashes[$path] = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
  }

  return $hashes
}

$manifestRoot = Join-Path $RepositoryRoot "native\manifests"
$manifestFiles = @(Get-ChildItem -Path $manifestRoot -Filter *.manifest.json -Recurse | Sort-Object FullName)
if ($manifestFiles.Count -eq 0) {
  throw "No binding manifest files were found under $manifestRoot."
}

$apiIds = New-Object System.Collections.Generic.HashSet[string]
$entryPoints = New-Object System.Collections.Generic.HashSet[string]
$apiCount = 0

foreach ($file in $manifestFiles) {
  $document = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($propertyName in $requiredManifestProperties) {
    Assert-Property -Object $document -PropertyName $propertyName -Context $file.FullName
  }

  foreach ($api in @($document.apis)) {
    $apiCount++
    $context = "$($file.FullName) api"
    foreach ($propertyName in $requiredApiProperties) {
      Assert-Property -Object $api -PropertyName $propertyName -Context $context
    }

    if (-not $apiIds.Add([string]$api.id)) {
      throw "Duplicate manifest API id: $($api.id)"
    }

    if (-not $entryPoints.Add([string]$api.entryPoint)) {
      throw "Duplicate manifest entry point: $($api.entryPoint)"
    }

    foreach ($parameter in @($api.parameters)) {
      foreach ($propertyName in $requiredParameterProperties) {
        Assert-Property -Object $parameter -PropertyName $propertyName -Context "$context $($api.id) parameter"
      }
    }
  }
}

if ($apiCount -lt 1) {
  throw "Binding manifests did not contain any API records."
}

$generatedFiles = @(
  "native\generated\bridge_api_catalog.g.h",
  "native\generated\bridge_entrypoints.g.h",
  "src\JYPPX.Shared\Generated\GeneratedApiCatalog.g.cs",
  "src\JYPPX.Shared\Generated\GeneratedEntryPointNames.g.cs",
  "src\JYPPX.Shared\Generated\GeneratedNativeMethods.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\GeneratedTensorRtManifestNativeMethods.g.cs",
  "src\JYPPX.CudaSharp\Internal\Interop\Generated\GeneratedCudaManifestNativeMethods.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeMethodsCommon.Generated.g.cs",
  "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeMethodsCommon.Generated.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeMethodsTensorRt.Generated.g.cs",
  "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeMethodsCuda.Generated.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.Common.Generated.g.cs",
  "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeBridgeApi.Common.Generated.g.cs",
  "src\JYPPX.CudaSharp\Internal\Interop\Generated\NativeCudaApi.Generated.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.TensorRtBindings.Generated.g.cs",
  "src\JYPPX.TensorRtSharp\Internal\Interop\Generated\NativeBridgeApi.TensorRtHelpers.Generated.g.cs"
)

& (Join-Path $RepositoryRoot "eng\Generate-Bindings.ps1") -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$firstHashes = Get-GeneratedFileHashes -Paths $generatedFiles

& (Join-Path $RepositoryRoot "eng\Generate-Bindings.ps1") -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$secondHashes = Get-GeneratedFileHashes -Paths $generatedFiles

foreach ($path in $generatedFiles) {
  if ($firstHashes[$path] -ne $secondHashes[$path]) {
    throw "Binding generator output is not deterministic for $path."
  }
}

& (Join-Path $RepositoryRoot "eng\Export-NativeMethodsComparison.ps1") -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& (Join-Path $RepositoryRoot "eng\Export-WrapperLiftCandidates.ps1") -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& (Join-Path $RepositoryRoot "eng\Export-GeneratedApiCoverage.ps1") -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$reportFiles = @(
  "artifacts\interop-comparison\native-methods-comparison.md",
  "artifacts\interop-comparison\wrapper-lift-candidates.md",
  "artifacts\interop-comparison\generated-api-coverage.md"
)

foreach ($path in $reportFiles) {
  $fullPath = Join-Path $RepositoryRoot $path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    throw "Expected generator report was not found: $fullPath"
  }
}

Write-Host "Binding generator output validation passed for $apiCount API records across $($manifestFiles.Count) manifests."
