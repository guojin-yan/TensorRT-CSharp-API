[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$ManifestPath,
  [Parameter(Mandatory = $true)][string]$ExpectedClassIndexPath,
  [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$utf8 = [Text.UTF8Encoding]::new($false)
$findings = [Collections.Generic.List[string]]::new()

function Add-Finding {
  param([Parameter(Mandatory = $true)][string]$Value)
  $findings.Add($Value) | Out-Null
}

$resolvedManifestPath = [IO.Path]::GetFullPath($ManifestPath)
$resolvedExpectedPath = [IO.Path]::GetFullPath($ExpectedClassIndexPath)
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = [IO.Path]::ChangeExtension($resolvedManifestPath, ".validation.json")
}
$resolvedOutputPath = [IO.Path]::GetFullPath($OutputPath)
$manifest = $null
$artifactPath = ""
$artifactSha256 = ""
$expectedSha256 = ""
$histogramMatches = $false
$classIndexMatches = $false

try {
  if (-not (Test-Path -LiteralPath $resolvedManifestPath -PathType Leaf)) {
    throw "Semantic artifact manifest does not exist: $resolvedManifestPath"
  }
  if (-not (Test-Path -LiteralPath $resolvedExpectedPath -PathType Leaf)) {
    throw "Expected semantic class-index artifact does not exist: $resolvedExpectedPath"
  }

  $manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  if ([string]$manifest.schemaVersion -ne "yolovision-semantic-map-artifacts.v1") { Add-Finding "schema-version" }
  if ([string]$manifest.task -ne "sem") { Add-Finding "task" }
  if ([int]$manifest.classCount -le 0 -or [int]$manifest.width -le 0 -or [int]$manifest.height -le 0) { Add-Finding "dimensions" }

  $pixelCount = [long]$manifest.width * [long]$manifest.height
  if ([long]$manifest.pixelCount -ne $pixelCount) { Add-Finding "pixel-count" }
  $artifact = $manifest.classIndexArtifact
  if ($null -eq $artifact) {
    Add-Finding "class-index-artifact"
  }
  else {
    $artifactPath = [string]$artifact.path
    if (-not [IO.Path]::IsPathRooted($artifactPath)) {
      $artifactPath = Join-Path (Split-Path -Parent $resolvedManifestPath) $artifactPath
    }
    $artifactPath = [IO.Path]::GetFullPath($artifactPath)
    if ([string]$artifact.role -ne "semantic-class-index-map") { Add-Finding "artifact-role" }
    if ([string]$artifact.dataType -ne "int32-little-endian") { Add-Finding "artifact-data-type" }
    if ([string]$artifact.layout -ne "row-major-hw") { Add-Finding "artifact-layout" }
    if ((@($artifact.shape) -join 'x') -ne "$($manifest.height)x$($manifest.width)") { Add-Finding "artifact-shape" }
    if ([long]$artifact.elementCount -ne $pixelCount -or [long]$artifact.byteLength -ne ($pixelCount * 4)) { Add-Finding "artifact-length-contract" }
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
      Add-Finding "artifact-missing"
    }
    else {
      $artifactLength = (Get-Item -LiteralPath $artifactPath).Length
      $artifactSha256 = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
      $expectedSha256 = (Get-FileHash -LiteralPath $resolvedExpectedPath -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($artifactLength -ne ($pixelCount * 4)) { Add-Finding "artifact-file-length" }
      if (-not [string]::Equals($artifactSha256, [string]$artifact.sha256, [StringComparison]::Ordinal)) { Add-Finding "artifact-sha256" }
      $classIndexMatches = [string]::Equals($artifactSha256, $expectedSha256, [StringComparison]::Ordinal)
      if (-not $classIndexMatches) { Add-Finding "reference-class-index-sha256" }

      if ($artifactLength -eq ($pixelCount * 4)) {
        $counts = [long[]]::new([int]$manifest.classCount)
        $stream = [IO.File]::OpenRead($artifactPath)
        $reader = [IO.BinaryReader]::new($stream)
        try {
          for ($index = 0L; $index -lt $pixelCount; $index++) {
            $classId = $reader.ReadInt32()
            if ($classId -lt 0 -or $classId -ge $counts.Length) {
              Add-Finding "class-index-range"
              break
            }
            $counts[$classId]++
          }
        }
        finally {
          $reader.Dispose()
          $stream.Dispose()
        }

        $manifestHistogram = @($manifest.classHistogram)
        $histogramMatches = $manifestHistogram.Count -eq $counts.Length
        for ($classId = 0; $histogramMatches -and $classId -lt $counts.Length; $classId++) {
          $row = @($manifestHistogram | Where-Object { [int]$_.classId -eq $classId })
          $histogramMatches = $row.Count -eq 1 -and [long]$row[0].pixelCount -eq $counts[$classId]
        }
        if (-not $histogramMatches) { Add-Finding "class-histogram" }
      }
    }
  }
}
catch {
  Add-Finding ("exception: " + $_.Exception.Message)
}

$passed = $findings.Count -eq 0
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-semantic-map-artifact-validation"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = if ($passed) { "passed" } else { "failed" }
  manifestPath = $resolvedManifestPath
  manifestSha256 = if (Test-Path -LiteralPath $resolvedManifestPath -PathType Leaf) { (Get-FileHash -LiteralPath $resolvedManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
  artifactPath = $artifactPath
  artifactSha256 = $artifactSha256
  expectedClassIndexPath = $resolvedExpectedPath
  expectedClassIndexSha256 = $expectedSha256
  classIndexMatches = $classIndexMatches
  histogramMatches = $histogramMatches
  findings = @($findings)
  failedFindingCount = $findings.Count
  passed = $passed
  boundary = [pscustomobject][ordered]@{
    isArtifactIntegrityValidation = $true
    isRuntimeProof = $false
    isPackageConsumerProof = $false
    isPublicPackageProof = $false
    performsPublish = $false
    uploadsAssets = $false
  }
}

$outputDirectory = Split-Path -Parent $resolvedOutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[IO.File]::WriteAllText($resolvedOutputPath, ($report | ConvertTo-Json -Depth 8) + "`n", $utf8)
Write-Host "ValidationState=$($report.validationState) ClassIndexMatches=$classIndexMatches HistogramMatches=$histogramMatches FailedFindingCount=$($findings.Count)"
Write-Host "Report=$resolvedOutputPath"
if (-not $passed) {
  exit 1
}
