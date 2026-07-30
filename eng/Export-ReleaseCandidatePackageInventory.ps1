[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string[]]$CompatibleBridgeRuntimePackageKey = @("win-x64-trt10.11-cuda12.9-cudnn9.22"),
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

Add-Type -AssemblyName System.IO.Compression.FileSystem

function ConvertTo-RelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $root = [System.IO.Path]::GetFullPath($RepositoryRoot)
  if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($root.Length).TrimStart('\', '/')
  }

  return $Path
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Read-JsonOrNull {
  param([Parameter(Mandatory = $true)][string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Test-Sha256Value {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match '^[A-Fa-f0-9]{64}$'
}

function Read-NupkgMetadata {
  param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

  $metadata = @{
    id = ""
    version = ""
    targetFrameworks = @()
  }

  $archive = [System.IO.Compression.ZipFile]::OpenRead($File.FullName)
  try {
    $nuspec = $archive.Entries |
      Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1

    if ($nuspec) {
      $stream = $nuspec.Open()
      try {
        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
        try {
          [xml]$xml = $reader.ReadToEnd()
          $metadata.id = [string]$xml.package.metadata.id
          $metadata.version = [string]$xml.package.metadata.version
        }
        finally {
          $reader.Dispose()
        }
      }
      finally {
        $stream.Dispose()
      }
    }

    $frameworks = @(
      $archive.Entries |
        Where-Object { $_.FullName.StartsWith("lib/", [System.StringComparison]::OrdinalIgnoreCase) -or $_.FullName.StartsWith("ref/", [System.StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object {
          $parts = $_.FullName.Split('/')
          if ($parts.Length -ge 2 -and -not [string]::IsNullOrWhiteSpace($parts[1])) {
            $parts[1]
          }
        } |
        Sort-Object -Unique
    )
    $metadata.targetFrameworks = @($frameworks)
  }
  finally {
    $archive.Dispose()
  }

  return [pscustomobject]$metadata
}

function Get-PackageRole {
  param(
    [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
    [Parameter(Mandatory = $true)][string]$RootRole
  )

  if ($RootRole -ne "split-runtime") {
    return $RootRole
  }

  if ($File.Name.IndexOf(".Bridge.", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
    return "split-bridge"
  }

  if ($File.Name.IndexOf(".CudaCudnn.", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
    return "split-cuda-cudnn"
  }

  if ($File.Name.IndexOf(".cudnn9.22.TensorRt.", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $File.Name.IndexOf(".cudnn8.9.TensorRt.", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
    return "split-tensorrt"
  }

  return "split-meta"
}

function New-PackageInventoryItem {
  param(
    [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
    [Parameter(Mandatory = $true)][string]$RootRole,
    [string]$RuntimePackageKey = ""
  )

  $metadata = Read-NupkgMetadata -File $File
  $sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  $role = Get-PackageRole -File $File -RootRole $RootRole

  [pscustomobject]@{
    role = $role
    runtimePackageKey = $RuntimePackageKey
    packageId = [string]$metadata.id
    version = [string]$metadata.version
    fileName = $File.Name
    relativePath = ConvertTo-RelativePath -Path $File.FullName
    sizeBytes = $File.Length
    sizeMb = [Math]::Round($File.Length / 1MB, 2)
    sha256 = $sha256
    sha256Ready = ($sha256 -match '^[a-f0-9]{64}$')
    lastWriteTimeUtc = $File.LastWriteTimeUtc.ToString("O")
    targetFrameworks = @($metadata.targetFrameworks)
  }
}

function Get-PackagesFromDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$RootRole,
    [string]$RuntimePackageKey = "",
    [switch]$Recurse
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    return @()
  }

  $searchOption = if ($Recurse.IsPresent) { [System.IO.SearchOption]::AllDirectories } else { [System.IO.SearchOption]::TopDirectoryOnly }
  return @(
    [System.IO.Directory]::EnumerateFiles($Directory, "*.nupkg", $searchOption) |
      ForEach-Object { New-PackageInventoryItem -File (Get-Item -LiteralPath $_) -RootRole $RootRole -RuntimePackageKey $RuntimePackageKey }
  )
}

$managedDirectory = Join-Path $RepositoryRoot "artifacts\managed"
$runtimeDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"
$splitRuntimeRoot = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg"
$selectedSplitDirectory = Join-Path $splitRuntimeRoot $RuntimePackageKey
$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$managedPackages = @(Get-PackagesFromDirectory -Directory $managedDirectory -RootRole "managed")
$runtimePackages = @(Get-PackagesFromDirectory -Directory $runtimeDirectory -RootRole "full-runtime" -RuntimePackageKey $RuntimePackageKey)
$splitRuntimePackages = @(Get-PackagesFromDirectory -Directory $selectedSplitDirectory -RootRole "split-runtime" -RuntimePackageKey $RuntimePackageKey)
$compatibleBridgeRuntimeKeys = @($CompatibleBridgeRuntimePackageKey |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not [string]::Equals($_, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) } |
    Select-Object -Unique)
$compatibleBridgePackages = @(
  foreach ($compatibleRuntimeKey in $compatibleBridgeRuntimeKeys) {
    $compatibleDirectory = Join-Path $splitRuntimeRoot $compatibleRuntimeKey
    Get-PackagesFromDirectory -Directory $compatibleDirectory -RootRole "split-runtime" -RuntimePackageKey $compatibleRuntimeKey |
      Where-Object { $_.role -eq "split-bridge" }
  }
)
$compatibleBridgeRuntimeProofs = @(
  foreach ($compatibleRuntimeKey in $compatibleBridgeRuntimeKeys) {
    $relativePath = "artifacts\package-consumer\bridge-runtime\$compatibleRuntimeKey\bridge-package-runtime-consumer-proof.json"
    $proof = Read-JsonOrNull -RelativePath $relativePath
    $hashesReady = $proof -and
      (Test-Sha256Value $proof.packages.managed.sha256) -and
      (Test-Sha256Value $proof.packages.bridge.sha256) -and
      (Test-Sha256Value $proof.logs.stdoutSha256) -and
      (Test-Sha256Value $proof.logs.stderrSha256) -and
      (Test-Sha256Value $proof.logs.combinedSha256)
    $ready = $proof -and
      [string]$proof.proofClassification -eq "compatible-host-bridge-package-runtime" -and
      [string]$proof.smokeStatus -eq "passed" -and
      [int]$proof.exitCode -eq 0 -and
      [bool]$proof.isRuntimeExecutionProof -and
      -not [bool]$proof.isPackageConsumerRuntimeProof -and
      [bool]$proof.canPromoteCompatibleHostRuntimeProof -and
      -not [bool]$proof.canPublishPublicly -and
      -not [bool]$proof.canCloseReleaseIssue -and
      [bool]$proof.consumerRootOutsideRepository -and
      -not [bool]$proof.consumer.usesProjectReference -and
      $hashesReady
    [pscustomobject]@{
      runtimePackageKey = $compatibleRuntimeKey
      evidencePath = $relativePath
      ready = [bool]$ready
      proofClassification = if ($proof) { [string]$proof.proofClassification } else { "missing" }
      smokeStatus = if ($proof) { [string]$proof.smokeStatus } else { "missing" }
      managedPackageSha256 = if ($proof) { [string]$proof.packages.managed.sha256 } else { "" }
      bridgePackageSha256 = if ($proof) { [string]$proof.packages.bridge.sha256 } else { "" }
      combinedLogSha256 = if ($proof) { [string]$proof.logs.combinedSha256 } else { "" }
      engineSerializedBytes = if ($proof) { [long]$proof.engineSerializedBytes } else { 0L }
      nativeAssetCount = if ($proof) { @($proof.nativeAssets).Count } else { 0 }
    }
  }
)
$compatibleBridgeRuntimeProofReady = $compatibleBridgeRuntimeProofs.Count -gt 0 -and @($compatibleBridgeRuntimeProofs | Where-Object { -not $_.ready }).Count -eq 0
$allPackages = @($managedPackages + $runtimePackages + $splitRuntimePackages + $compatibleBridgePackages) |
  Sort-Object role, packageId, version, fileName

$managedPackageReady = @($allPackages | Where-Object { $_.role -eq "managed" -and $_.packageId -eq "JYPPX.TensorRT.CSharp.API" -and $_.version -eq "4.0.0" }).Count -gt 0
$fullRuntimeReady = $false
$requiredSplitRoles = @("split-bridge")
$presentRoles = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($package in $allPackages) {
  [void]$presentRoles.Add([string]$package.role)
}
$missingSplitRoles = @($requiredSplitRoles | Where-Object { -not $presentRoles.Contains($_) })
$splitBridgePackageReady = $presentRoles.Contains("split-bridge")
$splitRuntimePackagesReady = $missingSplitRoles.Count -eq 0
$allowedPackages = @($allPackages | Where-Object { $_.role -in @("managed", "split-bridge") })
$retiredPackageCandidates = @($allPackages | Where-Object { $_.role -notin @("managed", "split-bridge") })
$sha256Ready = @($allowedPackages | Where-Object { -not $_.sha256Ready }).Count -eq 0 -and $allowedPackages.Count -gt 0
$packageSetReady = $managedPackageReady -and $splitBridgePackageReady -and $splitRuntimePackagesReady -and $sha256Ready -and $retiredPackageCandidates.Count -eq 0

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-candidate-package-inventory"
  runtimePackageKey = $RuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canUseAsPublicPackageProof = $false
  canCloseReleaseIssue = $false
  packageCount = $allPackages.Count
  managedPackageCount = $managedPackages.Count
  fullRuntimePackageCount = $runtimePackages.Count
  allowedPackageCount = $allowedPackages.Count
  retiredPackageCandidateCount = $retiredPackageCandidates.Count
  splitRuntimePackageCount = $splitRuntimePackages.Count
  compatibleBridgePackageCount = $compatibleBridgePackages.Count
  compatibleBridgeRuntimeKeys = $compatibleBridgeRuntimeKeys
  compatibleBridgeRuntimeProofReady = $compatibleBridgeRuntimeProofReady
  compatibleBridgeRuntimeProofs = $compatibleBridgeRuntimeProofs
  managedPackageReady = $managedPackageReady
  fullRuntimePackageReady = $fullRuntimeReady
  fullRuntimePackageRequired = $false
  vendorRuntimePackagesForbidden = $true
  publicationPolicy = "bridge-only"
  requiredSplitRoles = $requiredSplitRoles
  missingSplitRoles = $missingSplitRoles
  splitBridgePackageReady = $splitBridgePackageReady
  splitRuntimePackagesReady = $splitRuntimePackagesReady
  sha256Ready = $sha256Ready
  packageSetReady = $packageSetReady
  packages = @($allPackages)
  proofBoundary = "Local package inventory accepts only managed and bridge candidates. Any full/vendor, collection, or meta candidate blocks packageSetReady. The inventory is not public channel proof, runtime execution proof, post-publish proof, or owner authorization."
  guardrails = @(
    "This inventory does not publish packages.",
    "Local feed and dependency-probe-only evidence are not post-publish proof.",
    "B-tier safe alternative proof is not runtime proof.",
    "NuGet push, GitHub Packages publish, GitHub Release, and release issue close remain forbidden without owner authorization and post-publish proof."
  )
}

$jsonPath = Join-Path $outputRoot "release-candidate-package-inventory.json"
$markdownPath = Join-Path $outputRoot "release-candidate-package-inventory.md"
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Release Candidate Package Inventory")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($record.generatedAtUtc)``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- performs publish: ``$($record.performsPublish)``")
$lines.Add("- can publish publicly: ``$($record.canPublishPublicly)``")
$lines.Add("- can use as public package proof: ``$($record.canUseAsPublicPackageProof)``")
$lines.Add("- can close release issue: ``$($record.canCloseReleaseIssue)``")
$lines.Add("- managed package ready: ``$managedPackageReady``")
$lines.Add("- full/vendor runtime packages required: ``False``")
$lines.Add("- retired package candidates found: ``$($retiredPackageCandidates.Count)``")
$lines.Add("- split bridge package ready: ``$splitBridgePackageReady``")
  $lines.Add("- split runtime packages ready: ``$splitRuntimePackagesReady``")
  $lines.Add("- compatible bridge packages: ``$($compatibleBridgePackages.Count)``")
  $lines.Add("- compatible bridge runtime proof ready: ``$compatibleBridgeRuntimeProofReady``")
$lines.Add("- SHA256 ready: ``$sha256Ready``")
$lines.Add("- complete package set ready: ``$packageSetReady``")
$lines.Add("")
$lines.Add("## Packages")
$lines.Add("")
$lines.Add("| Role | Package ID | Version | Size MB | SHA256 | Path |")
$lines.Add("| --- | --- | --- | ---: | --- | --- |")
foreach ($package in $allPackages) {
  $lines.Add("| $(ConvertTo-MarkdownCell $package.role) | $(ConvertTo-MarkdownCell $package.packageId) | $(ConvertTo-MarkdownCell $package.version) | $(ConvertTo-MarkdownCell $package.sizeMb) | $(ConvertTo-MarkdownCell $package.sha256) | $(ConvertTo-MarkdownCell $package.relativePath) |")
}

$lines.Add("")
$lines.Add("## Compatible Bridge Runtime Proof")
$lines.Add("")
$lines.Add("| Runtime key | Ready | Classification | Smoke | Engine bytes | Native assets | Managed SHA256 | Bridge SHA256 | Combined log SHA256 |")
$lines.Add("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |")
foreach ($proof in $compatibleBridgeRuntimeProofs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $proof.runtimePackageKey) | $($proof.ready) | $(ConvertTo-MarkdownCell $proof.proofClassification) | $(ConvertTo-MarkdownCell $proof.smokeStatus) | $($proof.engineSerializedBytes) | $($proof.nativeAssetCount) | $($proof.managedPackageSha256) | $($proof.bridgePackageSha256) | $($proof.combinedLogSha256) |")
}

$lines.Add("")
$lines.Add("## Proof Boundary")
$lines.Add("")
$lines.Add($record.proofBoundary)
$lines.Add("")
$lines.Add("## Guardrails")
$lines.Add("")
foreach ($guardrail in $record.guardrails) {
  $lines.Add("- $guardrail")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Release candidate package inventory written:"
Write-Output "  $jsonPath"
Write-Output "  $markdownPath"
