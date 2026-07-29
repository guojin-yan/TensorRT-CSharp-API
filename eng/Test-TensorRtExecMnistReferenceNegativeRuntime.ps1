[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$SourcePlanPath,
  [string]$SourceInputPath,
  [string]$BaseReferencePath,
  [string]$ExpectedPlanSha256 = "14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89",
  [string]$ExpectedInputSha256 = "81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564",
  [string]$ExpectedOutputSha256 = "0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5",
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimeArtifactDirectory,
  [string]$OutputRoot,
  [string]$CompactOutputDirectory,
  [switch]$KeepWorkspace,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else { $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot) }

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RelativePath {
  param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Path
  )
  return [IO.Path]::GetRelativePath([IO.Path]::GetFullPath($Root), [IO.Path]::GetFullPath($Path)).Replace('\', '/')
}

function Test-PathWithin {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Parent
  )
  $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $fullParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $fullPath.StartsWith($fullParent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Remove-SafeDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$AllowedRoot
  )
  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove a directory outside the allowed root: $Path"
  }
  if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
}

function ConvertTo-XmlAttributeValue {
  param([Parameter(Mandatory = $true)][string]$Value)
  return [Security.SecurityElement]::Escape($Value)
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)
  $zip = [IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries |
      Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1
    if (-not $nuspec) { throw "Package does not contain a nuspec: $Path" }
    $stream = $nuspec.Open()
    try {
      $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
      try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
    }
    finally { $stream.Dispose() }

    $namespaceManager = [Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace("n", $xml.package.NamespaceURI)
    return [pscustomobject]@{
      Path = [IO.Path]::GetFullPath($Path)
      Id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
      Version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
      Length = (Get-Item -LiteralPath $Path).Length
      LastWriteTime = (Get-Item -LiteralPath $Path).LastWriteTimeUtc
      Sha256 = Get-Sha256 -Path $Path
    }
  }
  finally { $zip.Dispose() }
}

function Find-Package {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId
  )
  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }
  $matches = foreach ($file in Get-ChildItem -LiteralPath $Directory -File -Filter *.nupkg) {
    $metadata = Get-NupkgMetadata -Path $file.FullName
    if ([string]::Equals($metadata.Id, $PackageId, [StringComparison]::OrdinalIgnoreCase)) { $metadata }
  }
  if (@($matches).Count -eq 0) { throw "Package '$PackageId' was not found under $Directory." }
  return @($matches | Sort-Object LastWriteTime, Version -Descending)[0]
}

function Get-MarkerValue {
  param(
    [string[]]$Lines,
    [Parameter(Mandatory = $true)][string]$Prefix
  )
  foreach ($line in @($Lines)) {
    if ([string]$line -like "$Prefix*") { return ([string]$line).Substring($Prefix.Length) }
  }
  return ""
}

function ConvertTo-Int32 {
  param([string]$Value)
  $result = 0
  [void][int]::TryParse($Value, [ref]$result)
  return $result
}

function Assert-Condition {
  param(
    [Parameter(Mandatory = $true)][bool]$Condition,
    [Parameter(Mandatory = $true)][string]$Message
  )
  if (-not $Condition) { throw $Message }
}

function New-ReferenceVariant {
  param(
    [Parameter(Mandatory = $true)][object]$BaseReference,
    [Parameter(Mandatory = $true)][string]$CaseId,
    [Parameter(Mandatory = $true)][string]$OutputPath
  )
  $tensorName = [string]$BaseReference.tensorName
  $shape = @($BaseReference.shape | ForEach-Object { [int]$_ })
  [object[]]$values = @($BaseReference.values | ForEach-Object { [single]$_ })
  switch ($CaseId) {
    "name-mismatch" { $tensorName = "Wrong_Output_0" }
    "shape-mismatch" { $shape = @(1, 2, 5) }
    "value-count-mismatch" { $values = @($values[0..8]) }
    "nan-reject" { $values[0] = "NaN" }
    "infinity-reject" { $values[0] = "Infinity" }
    default { throw "Unknown negative reference case: $CaseId" }
  }

  $document = [ordered]@{
    schemaVersion = 1
    tensorName = $tensorName
    shape = @($shape)
    values = @($values)
    sourceClassification = "controlled-negative-from-onnxruntime-cpu-1.23.2-unreviewed"
  }
  [IO.File]::WriteAllText(
    $OutputPath,
    ($document | ConvertTo-Json -Depth 10) + [Environment]::NewLine,
    $utf8)
}

function Get-ExpectedCaseShape {
  param([Parameter(Mandatory = $true)][string]$CaseId)
  $metadataCase = $CaseId -in @("name-mismatch", "shape-mismatch", "value-count-mismatch")
  return [pscustomobject]@{
    ValidationCompleted = -not $metadataCase
    ComparedElementCount = if ($metadataCase) { 0 } else { 10 }
    MismatchCount = if ($metadataCase) { 0 } else { 1 }
    FirstMismatchIndex = if ($metadataCase) { -1 } else { 0 }
    Diagnostic = switch ($CaseId) {
      "name-mismatch" { "reference tensorName does not match engine output name" }
      "shape-mismatch" { "reference shape does not match runtime output shape" }
      "value-count-mismatch" { "reference value count does not match runtime output element count" }
      "nan-reject" { "exceeded tolerance or special-value policy" }
      "infinity-reject" { "exceeded tolerance or special-value policy" }
    }
  }
}

function Invoke-SourceTreeCase {
  param(
    [Parameter(Mandatory = $true)][object]$Case,
    [Parameter(Mandatory = $true)][string]$ReferencePath,
    [Parameter(Mandatory = $true)][string]$ApplicationPath,
    [Parameter(Mandatory = $true)][string]$ArtifactRoot
  )
  $caseRoot = Join-Path $ArtifactRoot $Case.Id
  New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null
  $rawPath = Join-Path $caseRoot "output.raw"
  $outputPath = Join-Path $caseRoot "output.json"
  $reportPath = Join-Path $caseRoot "report.json"
  $stdoutPath = Join-Path $caseRoot "stdout.log"
  $stderrPath = Join-Path $caseRoot "stderr.log"

  $arguments = @(
    $ApplicationPath,
    "--tensor-rt-line", "10",
    "--loadEngine", $SourcePlanPath,
    "--iterations", "1",
    "--warmUp", "0",
    "--duration", "0",
    "--streams", "1",
    "--loadInputs", "Input3:$SourceInputPath",
    "--dumpOutput",
    "--dumpRawBindingsToFile", $rawPath,
    "--exportOutput", $outputPath,
    "--exportReport", $reportPath,
    "--referenceOutputs", "Plus214_Output_0:$ReferencePath",
    "--referenceAbsTolerance", "0.0001",
    "--referenceRelTolerance", "0.0001",
    "--referenceNaNPolicy", "reject",
    "--referenceInfinityPolicy", "reject"
  )
  & dotnet @arguments 1> $stdoutPath 2> $stderrPath
  $exitCode = $LASTEXITCODE

  foreach ($requiredPath in @($rawPath, $outputPath, $reportPath, $stdoutPath, $stderrPath)) {
    Assert-Condition (Test-Path -LiteralPath $requiredPath -PathType Leaf) "Source-tree case '$($Case.Id)' did not create $requiredPath."
  }
  $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  $output = Get-Content -LiteralPath $outputPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  $comparison = @($output.ReferenceValidation.TensorComparisons)[0]
  $stdout = Get-Content -LiteralPath $stdoutPath -Raw -Encoding utf8
  Assert-Condition ($exitCode -eq 2) "Source-tree case '$($Case.Id)' must fail closed with exit code 2; actual=$exitCode."
  Assert-Condition (-not [bool]$report.Success -and [bool]$report.InferenceRan -and -not [bool]$report.OutputValidated) "Source-tree report contract failed for '$($Case.Id)'."
  Assert-Condition ([bool]$output.OutputCaptureAvailable -and [int]$output.OutputTensorCount -eq 1 -and -not [bool]$output.OutputValidated) "Source-tree output capture contract failed for '$($Case.Id)'."
  Assert-Condition ([bool]$output.ReferenceValidation.Requested -and -not [bool]$output.ReferenceValidation.Passed) "Source-tree validation must be requested and fail for '$($Case.Id)'."
  Assert-Condition ([bool]$output.ReferenceValidation.Completed -eq [bool]$Case.Expected.ValidationCompleted) "Source-tree completed state mismatch for '$($Case.Id)'."
  Assert-Condition ([int]$comparison.ComparedElementCount -eq [int]$Case.Expected.ComparedElementCount) "Source-tree compared count mismatch for '$($Case.Id)'."
  Assert-Condition ([int]$comparison.MismatchCount -eq [int]$Case.Expected.MismatchCount) "Source-tree mismatch count mismatch for '$($Case.Id)'."
  Assert-Condition ([int]$comparison.FirstMismatchIndex -eq [int]$Case.Expected.FirstMismatchIndex) "Source-tree first mismatch mismatch for '$($Case.Id)'."
  Assert-Condition ([string]$comparison.Diagnostic -like "*$($Case.Expected.Diagnostic)*") "Source-tree diagnostic mismatch for '$($Case.Id)': $($comparison.Diagnostic)"
  Assert-Condition ((Get-Sha256 -Path $rawPath) -eq $ExpectedOutputSha256) "Source-tree output hash mismatch for '$($Case.Id)'."
  Assert-Condition ($stdout.Contains("BoundedRuntime Attempted=True Succeeded=True", [StringComparison]::Ordinal) -and $stdout.Contains("OutputValidated=False", [StringComparison]::Ordinal)) "Source-tree enqueue/fail-closed markers are missing for '$($Case.Id)'."

  return [pscustomobject][ordered]@{
    exitCode = $exitCode
    state = [string]$report.State
    inferenceRan = [bool]$report.InferenceRan
    outputCaptureAvailable = [bool]$output.OutputCaptureAvailable
    outputValidated = [bool]$output.OutputValidated
    validationCompleted = [bool]$output.ReferenceValidation.Completed
    validationPassed = [bool]$output.ReferenceValidation.Passed
    comparedElementCount = [int]$comparison.ComparedElementCount
    mismatchCount = [int]$comparison.MismatchCount
    firstMismatchIndex = [int]$comparison.FirstMismatchIndex
    diagnostic = [string]$comparison.Diagnostic
    outputSha256 = Get-Sha256 -Path $rawPath
    referenceSha256 = Get-Sha256 -Path $ReferencePath
    reportSha256 = Get-Sha256 -Path $reportPath
    outputArtifactSha256 = Get-Sha256 -Path $outputPath
    stdoutSha256 = Get-Sha256 -Path $stdoutPath
    stderrSha256 = Get-Sha256 -Path $stderrPath
    artifactPrefix = "source-tree/$($Case.Id)"
  }
}

function Invoke-PackageConsumerCase {
  param(
    [Parameter(Mandatory = $true)][object]$Case,
    [Parameter(Mandatory = $true)][string]$ReferencePath,
    [Parameter(Mandatory = $true)][string]$ProjectPath,
    [Parameter(Mandatory = $true)][string]$PlanPath,
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$ArtifactRoot
  )
  $caseRoot = Join-Path $ArtifactRoot $Case.Id
  New-Item -ItemType Directory -Path $caseRoot -Force | Out-Null
  $rawPath = Join-Path $caseRoot "output.raw"
  $stdoutPath = Join-Path $caseRoot "stdout.log"
  $stderrPath = Join-Path $caseRoot "stderr.log"

  & dotnet run --project $ProjectPath -c Release --no-build -- `
    $PlanPath $InputPath $rawPath $ExpectedOutputSha256 $ReferencePath 0.0001 0.0001 reject reject `
    1> $stdoutPath 2> $stderrPath
  $exitCode = $LASTEXITCODE
  foreach ($requiredPath in @($rawPath, $stdoutPath, $stderrPath)) {
    Assert-Condition (Test-Path -LiteralPath $requiredPath -PathType Leaf) "Package consumer case '$($Case.Id)' did not create $requiredPath."
  }

  $stdoutLines = @(Get-Content -LiteralPath $stdoutPath -Encoding utf8)
  $stderr = Get-Content -LiteralPath $stderrPath -Raw -Encoding utf8
  $completed = (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationCompleted=") -eq "True"
  $passed = (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationPassed=") -eq "True"
  $outputValidated = (Get-MarkerValue -Lines $stdoutLines -Prefix "OutputValidated=") -eq "True"
  $diagnostic = Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceDiagnostic="
  $comparedCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceComparedElementCount=")
  $mismatchCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceMismatchCount=")
  $firstMismatch = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceFirstMismatchIndex=")
  $enqueueCompleted = (Get-MarkerValue -Lines $stdoutLines -Prefix "EnqueueCompleted=") -eq "True"
  $ownerScopeExited = (Get-MarkerValue -Lines $stdoutLines -Prefix "OwnerScopeExited=") -eq "True"

  Assert-Condition ($exitCode -eq 1) "Package consumer case '$($Case.Id)' must fail closed with exit code 1; actual=$exitCode."
  Assert-Condition $enqueueCompleted "Package consumer enqueue marker is missing for '$($Case.Id)'."
  Assert-Condition $ownerScopeExited "Package consumer owner scope marker is missing for '$($Case.Id)'."
  Assert-Condition (-not $outputValidated -and -not $passed) "Package consumer validation must fail closed for '$($Case.Id)'."
  Assert-Condition ($completed -eq [bool]$Case.Expected.ValidationCompleted) "Package consumer completed state mismatch for '$($Case.Id)'."
  Assert-Condition ($comparedCount -eq [int]$Case.Expected.ComparedElementCount) "Package consumer compared count mismatch for '$($Case.Id)'."
  Assert-Condition ($mismatchCount -eq [int]$Case.Expected.MismatchCount) "Package consumer mismatch count mismatch for '$($Case.Id)'."
  Assert-Condition ($firstMismatch -eq [int]$Case.Expected.FirstMismatchIndex) "Package consumer first mismatch mismatch for '$($Case.Id)'."
  Assert-Condition ($diagnostic -like "*$($Case.Expected.Diagnostic)*") "Package consumer diagnostic mismatch for '$($Case.Id)': $diagnostic"
  Assert-Condition ((Get-Sha256 -Path $rawPath) -eq $ExpectedOutputSha256) "Package consumer output hash mismatch for '$($Case.Id)'."
  Assert-Condition ($stderr.Contains("PackageConsumerRuntime=Failed", [StringComparison]::Ordinal)) "Package consumer failure marker is missing for '$($Case.Id)'."
  Assert-Condition ((Get-MarkerValue -Lines $stdoutLines -Prefix "PackageReferenceOnly=") -eq "True") "PackageReference-only marker is missing for '$($Case.Id)'."
  Assert-Condition ((Get-MarkerValue -Lines $stdoutLines -Prefix "ProjectReference=") -eq "False") "ProjectReference boundary failed for '$($Case.Id)'."

  return [pscustomobject][ordered]@{
    exitCode = $exitCode
    enqueueCompleted = $enqueueCompleted
    ownerScopeExited = $ownerScopeExited
    outputValidated = $outputValidated
    validationCompleted = $completed
    validationPassed = $passed
    comparedElementCount = $comparedCount
    mismatchCount = $mismatchCount
    firstMismatchIndex = $firstMismatch
    diagnostic = $diagnostic
    outputSha256 = Get-Sha256 -Path $rawPath
    referenceSha256 = Get-Sha256 -Path $ReferencePath
    stdoutSha256 = Get-Sha256 -Path $stdoutPath
    stderrSha256 = Get-Sha256 -Path $stderrPath
    artifactPrefix = "package-consumer/$($Case.Id)"
  }
}

if ([string]::IsNullOrWhiteSpace($SourcePlanPath)) {
  $SourcePlanPath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\reference-validation\mnist-build-reference.plan"
}
else { $SourcePlanPath = [IO.Path]::GetFullPath($SourcePlanPath) }
if ([string]::IsNullOrWhiteSpace($SourceInputPath)) {
  $SourceInputPath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7-input-f32.bin"
}
else { $SourceInputPath = [IO.Path]::GetFullPath($SourceInputPath) }
if ([string]::IsNullOrWhiteSpace($BaseReferencePath)) {
  $BaseReferencePath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7.onnxruntime-cpu.reference.json"
}
else { $BaseReferencePath = [IO.Path]::GetFullPath($BaseReferencePath) }
if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
else { $ManagedPackageDirectory = [IO.Path]::GetFullPath($ManagedPackageDirectory) }
if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey"
}
else { $BridgePackageDirectory = [IO.Path]::GetFullPath($BridgePackageDirectory) }
if ([string]::IsNullOrWhiteSpace($RuntimeArtifactDirectory)) {
  $RuntimeArtifactDirectory = Join-Path $RepositoryRoot "artifacts\real-case\tensorrtexec-mnist-reference-negative-runtime"
}
else { $RuntimeArtifactDirectory = [IO.Path]::GetFullPath($RuntimeArtifactDirectory) }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path (Split-Path -Parent $RepositoryRoot) ".reference-negative-runtime-work"
}
else { $OutputRoot = [IO.Path]::GetFullPath($OutputRoot) }
if ([string]::IsNullOrWhiteSpace($CompactOutputDirectory)) {
  $CompactOutputDirectory = Join-Path $RepositoryRoot "artifacts\interface-coverage"
}
else { $CompactOutputDirectory = [IO.Path]::GetFullPath($CompactOutputDirectory) }

foreach ($path in @($SourcePlanPath, $SourceInputPath, $BaseReferencePath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required runtime artifact is missing: $path" }
}
Assert-Condition ((Get-Sha256 -Path $SourcePlanPath) -eq $ExpectedPlanSha256) "Source plan SHA256 mismatch."
Assert-Condition ((Get-Sha256 -Path $SourceInputPath) -eq $ExpectedInputSha256) "Source input SHA256 mismatch."
Assert-Condition ([IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\') -ne $env:SystemDrive.TrimEnd('\')) "Negative runtime workspace must not use the system drive: $OutputRoot"
Assert-Condition (Test-PathWithin -Path $RuntimeArtifactDirectory -Parent (Join-Path $RepositoryRoot "artifacts\real-case")) "Runtime artifacts must stay below artifacts/real-case."

$baseReference = Get-Content -LiteralPath $BaseReferencePath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 20
Assert-Condition ([int]$baseReference.schemaVersion -eq 1 -and [string]$baseReference.tensorName -eq "Plus214_Output_0") "Base reference schema/name mismatch."
Assert-Condition ((@($baseReference.shape) -join ",") -eq "1,10" -and @($baseReference.values).Count -eq 10) "Base reference shape/value count mismatch."
Assert-Condition ([string]$baseReference.sourceClassification -eq "onnxruntime-cpu-1.23.2-derived-unreviewed") "Base reference classification mismatch."

$runtimeRoots = & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") `
  -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | ConvertFrom-Json
$runtimeSearchDirectories = @(
  (Join-Path $runtimeRoots.tensorRtRoot "bin"),
  (Join-Path $runtimeRoots.tensorRtRoot "lib"),
  (Join-Path $runtimeRoots.cudaRoot "bin"),
  (Join-Path $runtimeRoots.cudaRoot "bin\x64"),
  (Join-Path $runtimeRoots.cudnnRoot "bin")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -Unique
Assert-Condition ($runtimeSearchDirectories.Count -ge 2) "TensorRT/CUDA runtime search directories were not resolved."

$bridgeManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$bridgeManifest = Get-Content -LiteralPath $bridgeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 30
$bridgeDefinition = $bridgeManifest.packages |
  Where-Object { $_.sourceRuntimeKey -eq $RuntimePackageKey -and $_.role -eq "bridge" } |
  Select-Object -First 1
if (-not $bridgeDefinition) { throw "Bridge package definition was not found for $RuntimePackageKey." }
$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
$bridgePackage = Find-Package -Directory $BridgePackageDirectory -PackageId ([string]$bridgeDefinition.packageId)

$artifactAllowedRoot = Join-Path $RepositoryRoot "artifacts\real-case"
Remove-SafeDirectory -Path $RuntimeArtifactDirectory -AllowedRoot $artifactAllowedRoot
New-Item -ItemType Directory -Path $RuntimeArtifactDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $CompactOutputDirectory -Force | Out-Null
$referenceDirectory = Join-Path $RuntimeArtifactDirectory "references"
$sourceTreeArtifactRoot = Join-Path $RuntimeArtifactDirectory "source-tree"
$packageArtifactRoot = Join-Path $RuntimeArtifactDirectory "package-consumer"
New-Item -ItemType Directory -Path $referenceDirectory, $sourceTreeArtifactRoot, $packageArtifactRoot -Force | Out-Null

$caseDefinitions = foreach ($caseId in @("name-mismatch", "shape-mismatch", "value-count-mismatch", "nan-reject", "infinity-reject")) {
  $referencePath = Join-Path $referenceDirectory "$caseId.reference.json"
  New-ReferenceVariant -BaseReference $baseReference -CaseId $caseId -OutputPath $referencePath
  [pscustomobject]@{
    Id = $caseId
    Mutation = switch ($caseId) {
      "name-mismatch" { "tensorName differs from the engine output" }
      "shape-mismatch" { "shape is [1,2,5] instead of [1,10]" }
      "value-count-mismatch" { "values contain 9 elements instead of 10" }
      "nan-reject" { "values[0] is NaN under reject policy" }
      "infinity-reject" { "values[0] is positive infinity under reject policy" }
    }
    ReferencePath = $referencePath
    Expected = Get-ExpectedCaseShape -CaseId $caseId
  }
}

$applicationProject = Join-Path $RepositoryRoot "applications\TensorRtExec\TensorRtExec.csproj"
& dotnet build $applicationProject -c Debug --no-restore --nologo
$applicationBuildExitCode = $LASTEXITCODE
Assert-Condition ($applicationBuildExitCode -eq 0) "TensorRtExec build failed with exit code $applicationBuildExitCode."
$applicationPath = Join-Path $RepositoryRoot "applications\TensorRtExec\bin\Debug\net8.0-windows\TensorRtExec.dll"
$sourceBridgePath = Join-Path $RepositoryRoot "build-out\win-x64-trt10-cuda12-release\bin\Release\jyppxtrtbridge.dll"
foreach ($path in @($applicationPath, $sourceBridgePath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required source-tree runtime binary is missing: $path" }
}

$previousPath = $env:PATH
$previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
$previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
$sourceTreeResults = [Collections.Generic.List[object]]::new()
try {
  $env:PATH = (@($runtimeSearchDirectories) + @($previousPath)) -join [IO.Path]::PathSeparator
  $env:JYPPX_NATIVE_BRIDGE_PATH = $sourceBridgePath
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "0"
  foreach ($case in $caseDefinitions) {
    Write-Output "SourceTreeNegativeCase=$($case.Id)"
    $sourceTreeResults.Add((Invoke-SourceTreeCase -Case $case -ReferencePath $case.ReferencePath -ApplicationPath $applicationPath -ArtifactRoot $sourceTreeArtifactRoot))
  }
}
finally {
  $env:PATH = $previousPath
  $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
}

$workspaceRoot = Join-Path $OutputRoot "tensorrtexec-mnist-reference-negative"
Remove-SafeDirectory -Path $workspaceRoot -AllowedRoot $OutputRoot
New-Item -ItemType Directory -Path $workspaceRoot -Force | Out-Null
$projectPath = Join-Path $workspaceRoot "RefittedPlan.PackageConsumer.csproj"
$programPath = Join-Path $workspaceRoot "Program.cs"
$nugetConfigPath = Join-Path $workspaceRoot "NuGet.config"
$restorePath = Join-Path $workspaceRoot ".nuget\packages"
$inputRoot = Join-Path $workspaceRoot "input"
New-Item -ItemType Directory -Path $inputRoot, $restorePath -Force | Out-Null
$consumerPlanPath = Join-Path $inputRoot "mnist.plan"
$consumerInputPath = Join-Path $inputRoot "input-f32.bin"
Copy-Item -LiteralPath $SourcePlanPath -Destination $consumerPlanPath -Force
Copy-Item -LiteralPath $SourceInputPath -Destination $consumerInputPath -Force
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "samples\RefittedPlan.PackageConsumer\Program.cs") -Destination $programPath -Force

$template = Get-Content -LiteralPath (Join-Path $RepositoryRoot "samples\RefittedPlan.PackageConsumer\RefittedPlan.PackageConsumer.csproj.template") -Raw -Encoding utf8
$project = $template.Replace("__TARGET_FRAMEWORK__", "net8.0")
$project = $project.Replace("__RUNTIME_IDENTIFIER__", [string]$bridgeDefinition.rid)
$project = $project.Replace("__RESTORE_PACKAGES_PATH__", (ConvertTo-XmlAttributeValue $restorePath))
$project = $project.Replace("__MANAGED_PACKAGE_ID__", (ConvertTo-XmlAttributeValue $managedPackage.Id))
$project = $project.Replace("__MANAGED_PACKAGE_VERSION__", (ConvertTo-XmlAttributeValue $managedPackage.Version))
$project = $project.Replace("__BRIDGE_PACKAGE_ID__", (ConvertTo-XmlAttributeValue $bridgePackage.Id))
$project = $project.Replace("__BRIDGE_PACKAGE_VERSION__", (ConvertTo-XmlAttributeValue $bridgePackage.Version))
[IO.File]::WriteAllText($projectPath, $project, $utf8)
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="managed-local" value="$(ConvertTo-XmlAttributeValue $ManagedPackageDirectory)" />
    <add key="bridge-local" value="$(ConvertTo-XmlAttributeValue $BridgePackageDirectory)" />
  </packageSources>
</configuration>
"@
[IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)
Assert-Condition (-not $project.Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase)) "Consumer project must not contain ProjectReference."

$previousNugetPackages = $env:NUGET_PACKAGES
$previousDotnetHome = $env:DOTNET_CLI_HOME
$previousSkipFirstTime = $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE
$previousDotnetNoLogo = $env:DOTNET_NOLOGO
$previousTelemetry = $env:DOTNET_CLI_TELEMETRY_OPTOUT
$consumerResults = [Collections.Generic.List[object]]::new()
$workspaceRemoved = $false
try {
  $env:NUGET_PACKAGES = $restorePath
  $env:DOTNET_CLI_HOME = Join-Path $workspaceRoot ".dotnet-home"
  $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
  $env:DOTNET_NOLOGO = "1"
  $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
  & dotnet restore $projectPath --configfile $nugetConfigPath --packages $restorePath --force --no-cache
  $restoreExitCode = $LASTEXITCODE
  Assert-Condition ($restoreExitCode -eq 0) "Package consumer restore failed with exit code $restoreExitCode."
  & dotnet build $projectPath -c Release --no-restore --nologo
  $consumerBuildExitCode = $LASTEXITCODE
  Assert-Condition ($consumerBuildExitCode -eq 0) "Package consumer build failed with exit code $consumerBuildExitCode."

  $consumerOutputDirectory = Join-Path $workspaceRoot "bin\Release\net8.0\$($bridgeDefinition.rid)"
  $consumerBridgePath = Join-Path $consumerOutputDirectory ([string]@($bridgeDefinition.assets)[0])
  if (-not (Test-Path -LiteralPath $consumerBridgePath -PathType Leaf)) {
    $consumerBridgePath = Get-ChildItem -LiteralPath $consumerOutputDirectory -Recurse -File -Filter ([string]@($bridgeDefinition.assets)[0]) |
      Select-Object -First 1 -ExpandProperty FullName
  }
  Assert-Condition (-not [string]::IsNullOrWhiteSpace($consumerBridgePath) -and (Test-Path -LiteralPath $consumerBridgePath -PathType Leaf)) "Package consumer bridge asset is missing below $consumerOutputDirectory."
  $managedAssemblyPath = Join-Path $consumerOutputDirectory "JYPPX.TensorRtSharp.dll"
  Assert-Condition (Test-Path -LiteralPath $managedAssemblyPath -PathType Leaf) "Package consumer managed assembly is missing."

  $previousPath = $env:PATH
  $previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
  $previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
  try {
    $env:PATH = (@($runtimeSearchDirectories) + @($previousPath)) -join [IO.Path]::PathSeparator
    $env:JYPPX_NATIVE_BRIDGE_PATH = $null
    $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $null
    foreach ($case in $caseDefinitions) {
      $consumerReferencePath = Join-Path $inputRoot "$($case.Id).reference.json"
      Copy-Item -LiteralPath $case.ReferencePath -Destination $consumerReferencePath -Force
      Write-Output "PackageConsumerNegativeCase=$($case.Id)"
      $consumerResults.Add((Invoke-PackageConsumerCase -Case $case -ReferencePath $consumerReferencePath -ProjectPath $projectPath -PlanPath $consumerPlanPath -InputPath $consumerInputPath -ArtifactRoot $packageArtifactRoot))
    }
  }
  finally {
    $env:PATH = $previousPath
    $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
    $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
  }
}
finally {
  $env:NUGET_PACKAGES = $previousNugetPackages
  $env:DOTNET_CLI_HOME = $previousDotnetHome
  $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = $previousSkipFirstTime
  $env:DOTNET_NOLOGO = $previousDotnetNoLogo
  $env:DOTNET_CLI_TELEMETRY_OPTOUT = $previousTelemetry
  if (-not $KeepWorkspace -and (Test-Path -LiteralPath $workspaceRoot)) {
    Remove-SafeDirectory -Path $workspaceRoot -AllowedRoot $OutputRoot
    $workspaceRemoved = -not (Test-Path -LiteralPath $workspaceRoot)
    if ($workspaceRemoved -and (Test-Path -LiteralPath $OutputRoot -PathType Container) -and @(Get-ChildItem -LiteralPath $OutputRoot -Force).Count -eq 0) {
      Remove-Item -LiteralPath $OutputRoot -Force
    }
  }
}
if ($Strict -and -not $KeepWorkspace -and -not $workspaceRemoved) {
  throw "Package consumer workspace cleanup was required but did not complete."
}

$records = for ($index = 0; $index -lt $caseDefinitions.Count; $index++) {
  $case = $caseDefinitions[$index]
  [pscustomobject][ordered]@{
    id = $case.Id
    mutation = $case.Mutation
    expectedDiagnostic = $case.Expected.Diagnostic
    sourceTree = $sourceTreeResults[$index]
    localPackageConsumer = $consumerResults[$index]
  }
}
$nvidiaLine = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
$nvidiaParts = if ([string]::IsNullOrWhiteSpace([string]$nvidiaLine)) { @("", "") } else { ([string]$nvidiaLine).Split(',', 2) }
$evidence = [ordered]@{
  schemaVersion = "tensorrtexec-mnist-reference-negative-runtime-evidence.v1"
  capturedDate = [DateTime]::Now.ToString("yyyy-MM-dd")
  state = "controlled-reference-negative-runtime-passed"
  evidenceClassification = "controlled-negative-runtime"
  runtime = [ordered]@{
    runtimeKey = $RuntimePackageKey
    tensorRtLine = 10
    gpuName = $nvidiaParts[0].Trim()
    driverVersion = if ($nvidiaParts.Count -gt 1) { $nvidiaParts[1].Trim() } else { "" }
    applicationBuildExitCode = $applicationBuildExitCode
    consumerRestoreExitCode = $restoreExitCode
    consumerBuildExitCode = $consumerBuildExitCode
  }
  baseReference = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $BaseReferencePath
    sha256 = Get-Sha256 -Path $BaseReferencePath
    sourceClassification = [string]$baseReference.sourceClassification
    ownerReviewedGolden = $false
  }
  engine = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $SourcePlanPath
    sha256 = Get-Sha256 -Path $SourcePlanPath
  }
  input = [ordered]@{
    path = Get-RelativePath -Root $RepositoryRoot -Path $SourceInputPath
    sha256 = Get-Sha256 -Path $SourceInputPath
  }
  localPackageConsumer = [ordered]@{
    usesPackageReferenceOnly = $true
    usesProjectReference = $false
    remoteSourcesCleared = $true
    isolatedRestoreCache = $true
    workspaceOnSystemDrive = $false
    workspaceRemovedAfterValidation = $workspaceRemoved
    managedPackage = [ordered]@{ id = $managedPackage.Id; version = $managedPackage.Version; length = $managedPackage.Length; sha256 = $managedPackage.Sha256 }
    bridgePackage = [ordered]@{ id = $bridgePackage.Id; version = $bridgePackage.Version; length = $bridgePackage.Length; sha256 = $bridgePackage.Sha256 }
    programSha256 = Get-Sha256 -Path (Join-Path $RepositoryRoot "samples\RefittedPlan.PackageConsumer\Program.cs")
  }
  caseCount = $records.Count
  sourceTreeFailClosedCount = @($records | Where-Object { $_.sourceTree.inferenceRan -and -not $_.sourceTree.outputValidated -and $_.sourceTree.exitCode -eq 2 }).Count
  localPackageConsumerFailClosedCount = @($records | Where-Object { $_.localPackageConsumer.enqueueCompleted -and -not $_.localPackageConsumer.outputValidated -and $_.localPackageConsumer.exitCode -eq 1 }).Count
  cases = @($records)
  proofBoundary = [ordered]@{
    provesRealEnqueueBeforeReferenceRejection = $true
    provesSourceTreeFailClosedValidation = $true
    provesIsolatedLocalPackageConsumerFailClosedValidation = $true
    ownerReviewedGolden = $false
    publicPackageProof = $false
    postPublishProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    statement = "Controlled malformed references prove source-tree and isolated local PackageReference consumers fail closed only after real TensorRT enqueue and output readback. They do not promote the unreviewed ONNX Runtime reference to an Owner golden, public-package, post-publish, or release proof."
  }
}
$compactJson = $evidence | ConvertTo-Json -Depth 30
if ($compactJson -match '(?i)[A-Z]:[\\/]') { throw "Compact negative runtime evidence contains an absolute Windows path." }
$compactPath = Join-Path $CompactOutputDirectory "tensorrtexec-mnist-reference-negative-runtime-evidence.json"
[IO.File]::WriteAllText($compactPath, $compactJson + [Environment]::NewLine, $utf8)

Write-Output "TensorRtExecMnistReferenceNegativeRuntime=Passed"
Write-Output "Cases=$($records.Count)"
Write-Output "SourceTreeFailClosed=$($evidence.sourceTreeFailClosedCount)"
Write-Output "PackageConsumerFailClosed=$($evidence.localPackageConsumerFailClosedCount)"
Write-Output "WorkspaceRemoved=$workspaceRemoved"
Write-Output "Evidence=$compactPath"
