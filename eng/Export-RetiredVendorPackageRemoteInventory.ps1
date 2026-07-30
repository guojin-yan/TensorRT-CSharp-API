[CmdletBinding()]
param(
  [string]$GitHubCliPath = "gh",
  [string]$Owner = "guojin-yan",
  [string]$Repository = "TensorRT-CSharp-API",
  [string]$OutputDirectory,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\release-cleanup"
}

$resolvedGh = Get-Command $GitHubCliPath -ErrorAction Stop
$script:GitHubCliPath = $resolvedGh.Source

function Invoke-GhJson {
  param([string[]]$Arguments)

  $output = @(& $script:GitHubCliPath @Arguments 2>&1)
  $exitCode = $LASTEXITCODE
  $text = ($output -join "`n").Trim()
  if ($exitCode -ne 0) {
    throw "GitHub CLI failed with exit code $exitCode. Arguments: $($Arguments -join ' '). Output: $text"
  }
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $null
  }

  return $text | ConvertFrom-Json
}

function Invoke-GhPagedJson {
  param([string]$Endpoint)

  $pages = Invoke-GhJson -Arguments @("api", "--paginate", "--slurp", $Endpoint)
  $items = New-Object System.Collections.Generic.List[object]
  foreach ($page in @($pages)) {
    foreach ($item in @($page)) {
      $items.Add($item) | Out-Null
    }
  }

  return @($items.ToArray())
}

function Get-RetiredRole {
  param([string]$PackageId)

  if ($PackageId -match "\.CudaCudnn$") {
    return "cuda-cudnn"
  }
  if ($PackageId -match "\.TensorRtBuilder(?:\..+)?$") {
    return "builder-resource"
  }
  if ($PackageId -match "\.TensorRtRuntime$") {
    return "tensorrt"
  }
  if ($PackageId -match "\.TensorRt$") {
    return "tensorrt"
  }

  return "full-runtime"
}

function Format-Bytes {
  param([long]$Value)

  if ($Value -ge 1GB) {
    return "{0:N2} GiB" -f ($Value / 1GB)
  }
  if ($Value -ge 1MB) {
    return "{0:N2} MiB" -f ($Value / 1MB)
  }
  if ($Value -ge 1KB) {
    return "{0:N2} KiB" -f ($Value / 1KB)
  }

  return "$Value bytes"
}

function Get-Sha256Hex {
  param([string]$Value)

  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    return ([BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
  }
  finally {
    $sha256.Dispose()
  }
}

$formalRepository = "$Owner/$Repository"
$viewer = Invoke-GhJson -Arguments @("api", "user")
if ([string]$viewer.login -ne $Owner) {
  throw "The active GitHub account must be '$Owner'. Current account: '$([string]$viewer.login)'."
}

$policyPath = Join-Path $RepositoryRoot "pack\external-vendor-runtime-policy.json"
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$managedPackageIds = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($packageId in @($policy.managedPackageIds)) {
  $managedPackageIds.Add([string]$packageId) | Out-Null
}
$bridgePattern = New-Object System.Text.RegularExpressions.Regex(
  [string]$policy.bridgePackageIdPattern,
  [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

$localCandidates = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($package in @($runtimeManifest.packages)) {
  $localCandidates[[string]$package.packageId] = [pscustomobject]@{
    source = "pack/runtime/runtime-packages.manifest.json"
    role = "full-runtime"
    runtimeKey = [string]$package.key
  }
}
foreach ($package in @($splitManifest.packages | Where-Object { [string]$_.role -ne "bridge" })) {
  $localCandidates[[string]$package.packageId] = [pscustomobject]@{
    source = "pack/runtime-split/split-runtime-packages.manifest.json"
    role = [string]$package.role
    runtimeKey = [string]$package.sourceRuntimeKey
  }
}

$allPackages = @(Invoke-GhPagedJson -Endpoint "users/$Owner/packages?package_type=nuget&per_page=100")
$remotePackages = @($allPackages | Where-Object {
  [string]$_.repository.full_name -eq $formalRepository -and
  [string]$_.name -like "JYPPX.TensorRT.CSharp.API*"
} | Sort-Object name)

$packageRecords = New-Object System.Collections.Generic.List[object]
$packageByName = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
$packageIndex = 0
foreach ($package in $remotePackages) {
  $packageIndex++
  $packageId = [string]$package.name
  Write-Progress -Activity "Inventory GitHub Packages" -Status "$packageIndex/$($remotePackages.Count) $packageId" -PercentComplete (($packageIndex * 100) / [Math]::Max(1, $remotePackages.Count))
  $encodedPackageId = [uri]::EscapeDataString($packageId)
  $versions = @(Invoke-GhPagedJson -Endpoint "users/$Owner/packages/nuget/$encodedPackageId/versions?per_page=100" | ForEach-Object {
    [pscustomobject]@{
      id = [long]$_.id
      version = [string]$_.name
      apiUrl = [string]$_.url
      htmlUrl = [string]$_.html_url
      createdAt = [string]$_.created_at
      updatedAt = [string]$_.updated_at
    }
  })

  $isManaged = $managedPackageIds.Contains($packageId)
  $isBridge = $bridgePattern.IsMatch($packageId)
  $isRetired = -not $isManaged -and -not $isBridge
  $localCandidate = if ($localCandidates.ContainsKey($packageId)) { $localCandidates[$packageId] } else { $null }
  $record = [pscustomobject]@{
    packageId = $packageId
    classification = if ($isManaged) { "managed" } elseif ($isBridge) { "bridge" } else { Get-RetiredRole -PackageId $packageId }
    disposition = if ($isRetired) { "delete-after-owner-review" } else { "preserve" }
    candidateSource = if ($null -ne $localCandidate) { "local-manifest" } elseif ($isRetired) { "remote-discovery" } else { "preserve-policy" }
    localManifestSource = if ($null -ne $localCandidate) { [string]$localCandidate.source } else { $null }
    runtimeKey = if ($null -ne $localCandidate) { [string]$localCandidate.runtimeKey } else { $null }
    visibility = [string]$package.visibility
    apiUrl = [string]$package.url
    htmlUrl = [string]$package.html_url
    createdAt = [string]$package.created_at
    updatedAt = [string]$package.updated_at
    versions = $versions
    releaseAssets = @()
  }
  $packageRecords.Add($record) | Out-Null
  $packageByName[$packageId] = $record
}
Write-Progress -Activity "Inventory GitHub Packages" -Completed

$releases = @(Invoke-GhPagedJson -Endpoint "repos/$formalRepository/releases?per_page=100")
$packageNamesByLength = @($packageByName.Keys | Sort-Object { $_.Length } -Descending)
$assetRecords = New-Object System.Collections.Generic.List[object]
foreach ($release in $releases) {
  foreach ($asset in @($release.assets)) {
    $assetName = [string]$asset.name
    $matchedPackage = $null
    foreach ($packageName in $packageNamesByLength) {
      if ($assetName.StartsWith("$packageName.", [StringComparison]::OrdinalIgnoreCase) -and
          $assetName.EndsWith(".nupkg", [StringComparison]::OrdinalIgnoreCase)) {
        $matchedPackage = $packageByName[$packageName]
        break
      }
    }

    $assetVersion = $null
    $versionKnown = $false
    if ($null -ne $matchedPackage) {
      $assetVersion = $assetName.Substring(
        $matchedPackage.packageId.Length + 1,
        $assetName.Length - $matchedPackage.packageId.Length - 1 - ".nupkg".Length)
      $versionKnown = @($matchedPackage.versions | Where-Object { [string]$_.version -eq $assetVersion }).Count -gt 0
    }

    $assetRecord = [pscustomobject]@{
      id = [long]$asset.id
      releaseId = [long]$release.id
      releaseTag = [string]$release.tag_name
      name = $assetName
      size = [long]$asset.size
      digest = [string]$asset.digest
      apiUrl = [string]$asset.url
      downloadUrl = [string]$asset.browser_download_url
      createdAt = [string]$asset.created_at
      updatedAt = [string]$asset.updated_at
      packageId = if ($null -ne $matchedPackage) { [string]$matchedPackage.packageId } else { $null }
      packageVersion = $assetVersion
      packageVersionFoundInGitHubPackages = $versionKnown
      disposition = if ($null -ne $matchedPackage) { [string]$matchedPackage.disposition } else { "manual-review" }
    }
    $assetRecords.Add($assetRecord) | Out-Null
    if ($null -ne $matchedPackage) {
      $matchedPackage.releaseAssets = @($matchedPackage.releaseAssets) + @($assetRecord)
    }
  }
}

$retiredPackages = @($packageRecords.ToArray() | Where-Object disposition -eq "delete-after-owner-review")
$preservedPackages = @($packageRecords.ToArray() | Where-Object disposition -eq "preserve")
$retiredAssets = @($assetRecords.ToArray() | Where-Object disposition -eq "delete-after-owner-review")
$preservedAssets = @($assetRecords.ToArray() | Where-Object disposition -eq "preserve")
$unmatchedAssets = @($assetRecords.ToArray() | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.packageId) })
$remoteDiscoveredCandidates = @($retiredPackages | Where-Object candidateSource -eq "remote-discovery")
$retiredPackageVersions = @($retiredPackages | ForEach-Object {
  $packageId = [string]$_.packageId
  @($_.versions) | ForEach-Object {
    [pscustomobject]@{
      packageId = $packageId
      id = [long]$_.id
      version = [string]$_.version
    }
  }
})
$localCandidatePackageIds = @($localCandidates.Keys | Sort-Object)
$remotePackageIds = @($packageByName.Keys)
$localCandidatesMissingRemotely = @($localCandidatePackageIds | Where-Object { $remotePackageIds -notcontains $_ })
$retiredAssetBytes = [long](($retiredAssets | Measure-Object -Property size -Sum).Sum)
$reviewLines = New-Object System.Collections.Generic.List[string]
foreach ($version in @($retiredPackageVersions | Sort-Object packageId, id)) {
  $reviewLines.Add("package-version`t$($version.id)`t$($version.packageId)`t$($version.version)") | Out-Null
}
foreach ($asset in @($retiredAssets | Sort-Object id)) {
  $reviewLines.Add("release-asset`t$($asset.id)`t$($asset.releaseTag)`t$($asset.name)`t$($asset.digest)") | Out-Null
}
$reviewFingerprint = "sha256:$(Get-Sha256Hex -Value ($reviewLines -join "`n"))"

$plan = [pscustomobject]@{
  recordKind = "retired-vendor-package-remote-inventory"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  formalRepository = $formalRepository
  authenticatedOwner = [string]$viewer.login
  policy = "pack/external-vendor-runtime-policy.json"
  localPlanCandidateCount = $localCandidates.Count
  remoteTensorRtPackageCount = $packageRecords.Count
  remotePackageVersionCount = @($packageRecords.ToArray() | ForEach-Object { @($_.versions) }).Count
  preservedPackageCount = $preservedPackages.Count
  retiredPackageCount = $retiredPackages.Count
  retiredPackageVersionCount = $retiredPackageVersions.Count
  remoteDiscoveredRetiredPackageCount = $remoteDiscoveredCandidates.Count
  localCandidatesMissingRemotely = $localCandidatesMissingRemotely
  releaseCount = $releases.Count
  releaseAssetCount = $assetRecords.Count
  preservedReleaseAssetCount = $preservedAssets.Count
  retiredReleaseAssetCount = $retiredAssets.Count
  unmatchedReleaseAssetCount = $unmatchedAssets.Count
  retiredReleaseAssetBytes = $retiredAssetBytes
  retiredReleaseAssetGiB = [Math]::Round($retiredAssetBytes / 1GB, 3)
  reviewItemCount = $reviewLines.Count
  reviewFingerprint = $reviewFingerprint
  packages = @($packageRecords.ToArray())
  releaseAssets = @($assetRecords.ToArray())
  remoteInventoryComplete = $true
  ownerReviewRequired = $true
  deleteExecuted = $false
  performsRemoteQuery = $true
  performsDelete = $false
  preservePackageKinds = @("managed", "bridge")
  preserveGitHubGeneratedSourceArchives = $true
  cleanupScript = "eng/Invoke-RetiredVendorPackageCleanup.ps1"
  nextAction = "Run eng/Invoke-RetiredVendorPackageCleanup.ps1 without -ExecuteDeletion for live preflight. After explicit Owner confirmation of this fingerprint, rerun with -ExecuteDeletion and -ExpectedReviewFingerprint."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$jsonPath = Join-Path $OutputDirectory "retired-vendor-package-remote-inventory.json"
$markdownPath = Join-Path $OutputDirectory "retired-vendor-package-remote-inventory.md"
$plan | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$retiredPackageRows = @($retiredPackages | ForEach-Object {
  $versions = @($_.versions | ForEach-Object { $_.version }) -join ", "
  $assetBytes = [long]((@($_.releaseAssets) | Measure-Object -Property size -Sum).Sum)
  "| ``$($_.packageId)`` | ``$($_.classification)`` | ``$($_.candidateSource)`` | ``$versions`` | $(@($_.releaseAssets).Count) | $(Format-Bytes $assetBytes) |"
})
$retiredAssetRows = @($retiredAssets | ForEach-Object {
  "| ``$($_.releaseTag)`` | ``$($_.name)`` | $(Format-Bytes $_.size) | ``$($_.digest)`` |"
})
$preservedRows = @($preservedPackages | ForEach-Object {
  "| ``$($_.packageId)`` | ``$($_.classification)`` | $(@($_.versions).Count) | $(@($_.releaseAssets).Count) |"
})
$markdown = @"
# Retired Vendor Package Remote Inventory

This read-only inventory queried GitHub as ``$([string]$viewer.login)``. It did not delete packages, package versions, releases, or assets.

## Summary

- Formal repository: ``$formalRepository``
- Remote TensorRT packages: ``$($packageRecords.Count)``
- Preserved managed/Bridge packages: ``$($preservedPackages.Count)``
- Retired packages pending Owner review: ``$($retiredPackages.Count)``
- Retired package versions pending Owner review: ``$($retiredPackageVersions.Count)``
- Retired packages discovered only from GitHub: ``$($remoteDiscoveredCandidates.Count)``
- Releases: ``$($releases.Count)``
- Retired Release assets: ``$($retiredAssets.Count)``
- Retired Release asset size: ``$(Format-Bytes $retiredAssetBytes)``
- Unmatched Release assets: ``$($unmatchedAssets.Count)``
- Review item count: ``$($reviewLines.Count)``
- Review fingerprint: ``$reviewFingerprint``

## Retired Packages

| Package ID | Classification | Candidate source | Versions | Release assets | Asset size |
|---|---|---|---|---:|---:|
$($retiredPackageRows -join "`r`n")

## Retired Release Assets

| Release | Asset | Size | Digest |
|---|---|---:|---|
$($retiredAssetRows -join "`r`n")

## Preserved Packages

| Package ID | Classification | Versions | Release assets |
|---|---|---:|---:|
$($preservedRows -join "`r`n")

Owner review is required before deletion. Preserve ``JYPPX.TensorRT.CSharp.API``, every ``.Bridge`` package, and GitHub-generated source archives. Run ``eng/Invoke-RetiredVendorPackageCleanup.ps1`` without ``-ExecuteDeletion`` for a live, side-effect-free preflight.
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Remote inventory JSON written: $jsonPath"
Write-Host "Remote inventory Markdown written: $markdownPath"
$plan | ConvertTo-Json -Depth 12
