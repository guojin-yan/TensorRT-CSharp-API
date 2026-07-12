[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string[]]$CompatibleBridgeRuntimePackageKey = @("win-x64-trt10.11-cuda12.9-cudnn9.22"),
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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $DefaultValue
}

function ConvertTo-RelativePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

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

function Get-NuspecMetadata {
  param([System.IO.FileInfo]$File)

  $metadata = @{
    id = ""
    version = ""
    targetFrameworks = @()
    nativeAssetCount = 0
  }

  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($File.FullName)
    try {
      $nuspec = $archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
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
      $metadata.nativeAssetCount = @(
        $archive.Entries |
          Where-Object {
            $_.FullName -match '^runtimes/[^/]+/native/.+\.(dll|so|dylib)$'
          }
      ).Count
    }
    finally {
      $archive.Dispose()
    }
  }
  catch {
    $metadata.id = ""
    $metadata.version = ""
    $metadata.targetFrameworks = @()
    $metadata.nativeAssetCount = 0
  }

  return [pscustomobject]$metadata
}

function New-PackageReviewItem {
  param(
    [System.IO.FileInfo]$File,
    [string]$Kind,
    [string]$RuntimePackageKey
  )

  $metadata = Get-NuspecMetadata -File $File
  $sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  [pscustomobject]@{
    kind = $Kind
    fileName = $File.Name
    relativePath = ConvertTo-RelativePath -Path $File.FullName
    packageId = [string]$metadata.id
    version = [string]$metadata.version
    runtimePackageKey = $RuntimePackageKey
    sizeBytes = $File.Length
    sizeMb = [Math]::Round($File.Length / 1MB, 2)
    sha256 = $sha256
    sha256Ready = ($sha256 -match '^[a-f0-9]{64}$')
    targetFrameworks = @($metadata.targetFrameworks)
    nativeAssetCount = [int]$metadata.nativeAssetCount
  }
}

function Test-RuntimePackageFileName {
  param(
    [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
    [Parameter(Mandatory = $true)][string]$RuntimePackageKey,
    [string]$RuntimePackageId
  )

  $normalizedFileName = ($File.BaseName -replace '[^A-Za-z0-9]', '').ToLowerInvariant()
  $normalizedRuntimeKey = ($RuntimePackageKey -replace '[^A-Za-z0-9]', '').ToLowerInvariant()
  $normalizedRuntimePackageId = ($RuntimePackageId -replace '[^A-Za-z0-9]', '').ToLowerInvariant()

  return (
    (-not [string]::IsNullOrWhiteSpace($normalizedRuntimeKey) -and $normalizedFileName.Contains($normalizedRuntimeKey)) -or
    (-not [string]::IsNullOrWhiteSpace($normalizedRuntimePackageId) -and $normalizedFileName.Contains($normalizedRuntimePackageId))
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$releaseCandidatePackageInventory = Read-JsonOrNull "artifacts\final-release\release-candidate-package-inventory.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$postPublish = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$freezeSummary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
$artifactManifest = Read-JsonOrNull "artifacts\runtime\$RuntimePackageKey\artifact-manifest.json"
$runtimePackageId = [string](Get-PropertyOrDefault -Object $artifactManifest -Name "packageId" -DefaultValue "")

$managedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
$runtimePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"
$splitPackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey"
$compatibleBridgeRuntimeKeys = @($CompatibleBridgeRuntimePackageKey |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and -not [string]::Equals($_, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) } |
    Select-Object -Unique)

$managedPackages = @()
if (Test-Path -LiteralPath $managedPackageDirectory -PathType Container) {
  $managedPackages = @(Get-ChildItem -LiteralPath $managedPackageDirectory -File -Filter "*.nupkg" | Sort-Object Name | ForEach-Object {
      New-PackageReviewItem -File $_ -Kind "managed" -RuntimePackageKey $RuntimePackageKey
    })
}

$runtimePackages = @()
if (Test-Path -LiteralPath $runtimePackageDirectory -PathType Container) {
  $runtimePackages = @(Get-ChildItem -LiteralPath $runtimePackageDirectory -File -Filter "*.nupkg" | Where-Object {
      Test-RuntimePackageFileName -File $_ -RuntimePackageKey $RuntimePackageKey -RuntimePackageId $runtimePackageId
    } | Sort-Object Name | ForEach-Object {
      New-PackageReviewItem -File $_ -Kind "runtime" -RuntimePackageKey $RuntimePackageKey
    })
}

$splitRuntimePackages = @()
if (Test-Path -LiteralPath $splitPackageDirectory -PathType Container) {
  $splitRuntimePackages = @(Get-ChildItem -LiteralPath $splitPackageDirectory -File -Filter "*.nupkg" | Sort-Object Name | ForEach-Object {
      New-PackageReviewItem -File $_ -Kind "split-runtime" -RuntimePackageKey $RuntimePackageKey
    })
}

$compatibleBridgePackages = @(
  foreach ($compatibleRuntimeKey in $compatibleBridgeRuntimeKeys) {
    $compatibleDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$compatibleRuntimeKey"
    if (Test-Path -LiteralPath $compatibleDirectory -PathType Container) {
      Get-ChildItem -LiteralPath $compatibleDirectory -File -Filter "*.Bridge.*.nupkg" |
        Sort-Object Name |
        ForEach-Object { New-PackageReviewItem -File $_ -Kind "compatible-split-bridge" -RuntimePackageKey $compatibleRuntimeKey }
    }
  }
)
$compatibleBridgeRuntimeProofs = @(
  foreach ($compatibleRuntimeKey in $compatibleBridgeRuntimeKeys) {
    $relativePath = "artifacts\package-consumer\bridge-runtime\$compatibleRuntimeKey\bridge-package-runtime-consumer-proof.json"
    $proof = Read-JsonOrNull $relativePath
    [pscustomobject]@{
      runtimePackageKey = $compatibleRuntimeKey
      evidencePath = $relativePath
      proofClassification = [string](Get-PropertyOrDefault -Object $proof -Name "proofClassification" -DefaultValue "missing")
      smokeStatus = [string](Get-PropertyOrDefault -Object $proof -Name "smokeStatus" -DefaultValue "missing")
      isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $proof -Name "isRuntimeExecutionProof" -DefaultValue $false)
      isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $proof -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
      canPromoteCompatibleHostRuntimeProof = [bool](Get-PropertyOrDefault -Object $proof -Name "canPromoteCompatibleHostRuntimeProof" -DefaultValue $false)
      canPublishPublicly = [bool](Get-PropertyOrDefault -Object $proof -Name "canPublishPublicly" -DefaultValue $false)
      canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $proof -Name "canCloseReleaseIssue" -DefaultValue $false)
      engineSerializedBytes = [long](Get-PropertyOrDefault -Object $proof -Name "engineSerializedBytes" -DefaultValue 0L)
      nativeAssetCount = if ($proof) { @($proof.nativeAssets).Count } else { 0 }
    }
  }
)
$compatibleBridgeRuntimeProofReady = $compatibleBridgeRuntimeProofs.Count -gt 0 -and @($compatibleBridgeRuntimeProofs | Where-Object {
      $_.proofClassification -ne "compatible-host-bridge-package-runtime" -or
      $_.smokeStatus -ne "passed" -or
      -not $_.isRuntimeExecutionProof -or
      $_.isPackageConsumerRuntimeProof -or
      -not $_.canPromoteCompatibleHostRuntimeProof -or
      $_.canPublishPublicly -or
      $_.canCloseReleaseIssue
    }).Count -eq 0
$allPackages = @($managedPackages + $runtimePackages + $splitRuntimePackages + $compatibleBridgePackages)
$nativeAssetCount = if ($runtimePackages.Count -gt 0) {
  [int](($runtimePackages | ForEach-Object nativeAssetCount | Measure-Object -Sum).Sum)
}
else {
  [int](($splitRuntimePackages | ForEach-Object nativeAssetCount | Measure-Object -Sum).Sum)
}

$postPublishStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutSummaryReady" -DefaultValue $false)
$postPublishStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stderrSummaryReady" -DefaultValue $false)
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutStderrSummaryReady" -DefaultValue $false)
$postPublishRestoreLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "restoreLogSha256Matches" -DefaultValue $false)
$postPublishNativeAssetListingSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "nativeAssetListingSha256Matches" -DefaultValue $false)
$postPublishDependencyProbeLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "dependencyProbeLogSha256Matches" -DefaultValue $false)
$postPublishSmokeLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeLogSha256Matches" -DefaultValue $false)
$postPublishAllLogSha256Matches = $postPublishRestoreLogSha256Matches -and $postPublishNativeAssetListingSha256Matches -and $postPublishDependencyProbeLogSha256Matches -and $postPublishSmokeLogSha256Matches
$packageInventoryState = [string](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "recordKind" -DefaultValue "missing-release-candidate-package-inventory")
$packageInventoryPackageCount = [int](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "packageCount" -DefaultValue 0)
$packageInventoryManagedPackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "managedPackageReady" -DefaultValue $false)
$packageInventoryFullRuntimePackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "fullRuntimePackageReady" -DefaultValue $false)
$packageInventorySplitBridgePackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "splitBridgePackageReady" -DefaultValue $false)
$packageInventorySplitRuntimePackagesReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "splitRuntimePackagesReady" -DefaultValue $false)
$packageInventorySha256Ready = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "sha256Ready" -DefaultValue $false)

$guardrails = @(
  "Local package review is not public channel proof.",
  "This script does not publish packages and does not execute dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw.",
  "Templates, drafts, examples, runbooks, collection bundles, dependency-probe-only output, and blocked-by-cuda-driver output are not runtime proof.",
  "Owner authorization is required before any public or private package channel publication.",
  "Post-publish proof requires a real channel, clean consumer, compatible host metadata, runtime-key smoke command, stdout/stderr summaries, and matching log SHA256 values.",
  "canCloseReleaseIssue remains false until real external runtime proof, real post-publish proof, and owner authorization are all present."
)

$sourceEvidence = @(
  "artifacts/final-release/release-candidate-package-inventory.json",
  "artifacts/final-release/release-package-proof-bundle.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/release/release-candidate-freeze-summary.json"
)
foreach ($compatibleProof in $compatibleBridgeRuntimeProofs) {
  $sourceEvidence += $compatibleProof.evidencePath.Replace('\', '/')
}
if ($artifactManifest) {
  $sourceEvidence += "artifacts/runtime/$RuntimePackageKey/artifact-manifest.json"
}

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "final-package-review-bundle"
  runtimePackageKey = $RuntimePackageKey
  bundleState = "owner-review-required"
  performsPublish = $false
  canPublishPublicly = $false
  canUseAsPublicPackageProof = $false
  canCloseReleaseIssue = $false
  managedPackageCount = $managedPackages.Count
  runtimePackageCount = $runtimePackages.Count
  splitRuntimePackageCount = $splitRuntimePackages.Count
  compatibleBridgePackageCount = $compatibleBridgePackages.Count
  compatibleBridgeRuntimeKeys = $compatibleBridgeRuntimeKeys
  compatibleBridgeRuntimeProofReady = $compatibleBridgeRuntimeProofReady
  compatibleBridgeRuntimeProofs = $compatibleBridgeRuntimeProofs
  packageCount = $allPackages.Count
  nativeAssetCount = $nativeAssetCount
  packages = @($allPackages)
  sourceEvidence = $sourceEvidence
  packageInventoryState = $packageInventoryState
  packageInventoryPackageCount = $packageInventoryPackageCount
  packageInventoryManagedPackageReady = $packageInventoryManagedPackageReady
  packageInventoryFullRuntimePackageReady = $packageInventoryFullRuntimePackageReady
  packageInventorySplitBridgePackageReady = $packageInventorySplitBridgePackageReady
  packageInventorySplitRuntimePackagesReady = $packageInventorySplitRuntimePackagesReady
  packageInventorySha256Ready = $packageInventorySha256Ready
  releasePackageProofState = [string](Get-PropertyOrDefault -Object $releasePackageProof -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
  postPublishVerificationState = [string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-validation")
  freezeState = [string](Get-PropertyOrDefault -Object $freezeSummary -Name "freezeState" -DefaultValue "missing-freeze-summary")
  postPublishStdoutSummaryReady = $postPublishStdoutSummaryReady
  postPublishStderrSummaryReady = $postPublishStderrSummaryReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  postPublishRestoreLogSha256Matches = $postPublishRestoreLogSha256Matches
  postPublishNativeAssetListingSha256Matches = $postPublishNativeAssetListingSha256Matches
  postPublishDependencyProbeLogSha256Matches = $postPublishDependencyProbeLogSha256Matches
  postPublishSmokeLogSha256Matches = $postPublishSmokeLogSha256Matches
  postPublishAllLogSha256Matches = $postPublishAllLogSha256Matches
  guardrails = $guardrails
}

$jsonPath = Join-Path $outputRoot "final-package-review-bundle.json"
$markdownPath = Join-Path $outputRoot "final-package-review-bundle.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Final Package Review Bundle")
$lines.Add("")
$lines.Add("Generated at UTC: ``$($record.generatedAtUtc)``")
$lines.Add("")
$lines.Add("## State")
$lines.Add("")
$lines.Add("- record kind: ``$($record.recordKind)``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- bundle state: ``$($record.bundleState)``")
$lines.Add("- performs publish: ``$($record.performsPublish)``")
$lines.Add("- can publish publicly: ``$($record.canPublishPublicly)``")
$lines.Add("- can use as public package proof: ``$($record.canUseAsPublicPackageProof)``")
$lines.Add("- can close release issue: ``$($record.canCloseReleaseIssue)``")
$lines.Add("- native asset count: ``$nativeAssetCount``")
$lines.Add("- package inventory state: ``$packageInventoryState``")
$lines.Add("- package inventory package count: ``$packageInventoryPackageCount``")
$lines.Add("- package inventory managed package ready: ``$packageInventoryManagedPackageReady``")
$lines.Add("- package inventory full runtime package ready: ``$packageInventoryFullRuntimePackageReady``")
$lines.Add("- package inventory split bridge package ready: ``$packageInventorySplitBridgePackageReady``")
$lines.Add("- package inventory split runtime packages ready: ``$packageInventorySplitRuntimePackagesReady``")
  $lines.Add("- package inventory SHA256 ready: ``$packageInventorySha256Ready``")
  $lines.Add("- compatible bridge packages: ``$($compatibleBridgePackages.Count)``")
  $lines.Add("- compatible bridge runtime proof ready: ``$compatibleBridgeRuntimeProofReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish all log SHA256 matches: ``$postPublishAllLogSha256Matches``")
$lines.Add("")
$lines.Add("## Packages")
$lines.Add("")
$lines.Add("| Kind | Package ID | Version | Size MB | SHA256 | Path |")
$lines.Add("| --- | --- | --- | ---: | --- | --- |")
foreach ($package in $allPackages) {
  $lines.Add("| $(ConvertTo-MarkdownCell $package.kind) | $(ConvertTo-MarkdownCell $package.packageId) | $(ConvertTo-MarkdownCell $package.version) | $(ConvertTo-MarkdownCell $package.sizeMb) | $(ConvertTo-MarkdownCell $package.sha256) | $(ConvertTo-MarkdownCell $package.relativePath) |")
}

$lines.Add("")
$lines.Add("## Compatible Bridge Runtime Proof")
$lines.Add("")
$lines.Add("| Runtime key | Classification | Smoke | Runtime execution | Package-consumer proof | Engine bytes | Native assets | Evidence |")
$lines.Add("| --- | --- | --- | --- | --- | ---: | ---: | --- |")
foreach ($proof in $compatibleBridgeRuntimeProofs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $proof.runtimePackageKey) | $(ConvertTo-MarkdownCell $proof.proofClassification) | $(ConvertTo-MarkdownCell $proof.smokeStatus) | $($proof.isRuntimeExecutionProof) | $($proof.isPackageConsumerRuntimeProof) | $($proof.engineSerializedBytes) | $($proof.nativeAssetCount) | ``$($proof.evidencePath)`` |")
}

$lines.Add("")
$lines.Add("## Source Evidence")
$lines.Add("")
foreach ($source in $sourceEvidence) {
  $lines.Add("- ``$source``")
}

$lines.Add("")
$lines.Add("## Guardrails")
$lines.Add("")
foreach ($guardrail in $guardrails) {
  $lines.Add("- $guardrail")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Final package review bundle written:"
Write-Output "  $jsonPath"
Write-Output "  $markdownPath"
