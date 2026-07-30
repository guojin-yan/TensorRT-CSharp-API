[CmdletBinding()]
param(
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

$runtimeManifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$candidates = New-Object System.Collections.Generic.List[object]

foreach ($package in @($runtimeManifest.packages)) {
  $candidates.Add([pscustomobject]@{
    packageId = [string]$package.packageId
    role = "full-runtime"
    runtimeKey = [string]$package.key
    source = "pack/runtime/runtime-packages.manifest.json"
    reason = "Retired package identity bundled project bridge and NVIDIA vendor runtime libraries."
    releaseAssetPattern = "$([string]$package.packageId).*.nupkg"
    remoteState = "not-inventoried"
    proposedAction = "delete-after-owner-review"
  })
}

foreach ($package in @($splitManifest.packages | Where-Object { [string]$_.role -ne "bridge" })) {
  $candidates.Add([pscustomobject]@{
    packageId = [string]$package.packageId
    role = [string]$package.role
    runtimeKey = [string]$package.sourceRuntimeKey
    source = "pack/runtime-split/split-runtime-packages.manifest.json"
    reason = "Retired split package identity contains or depends on NVIDIA vendor runtime libraries."
    releaseAssetPattern = "$([string]$package.packageId).*.nupkg"
    remoteState = "not-inventoried"
    proposedAction = "delete-after-owner-review"
  })
}

$uniqueCandidates = @($candidates.ToArray() | Sort-Object packageId -Unique)
$plan = [pscustomobject]@{
  recordKind = "retired-vendor-package-cleanup-plan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  formalRepository = "guojin-yan/TensorRT-CSharp-API"
  validationOnlyAccount = "grape-yan"
  policy = "pack/external-vendor-runtime-policy.json"
  candidateCount = $uniqueCandidates.Count
  candidates = $uniqueCandidates
  preservePackageKinds = @("managed", "bridge")
  preserveGitHubGeneratedSourceArchives = $true
  remoteInventoryComplete = $false
  ownerReviewRequired = $true
  deleteExecuted = $false
  performsRemoteQuery = $false
  performsDelete = $false
  remoteInventoryScript = "eng/Export-RetiredVendorPackageRemoteInventory.ps1"
  nextAction = "Run eng/Export-RetiredVendorPackageRemoteInventory.ps1 as the formal owner, review its exact version/asset IDs and fingerprint, then request Owner confirmation before deletion."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$jsonPath = Join-Path $OutputDirectory "retired-vendor-package-cleanup-plan.json"
$markdownPath = Join-Path $OutputDirectory "retired-vendor-package-cleanup-plan.md"
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = @($uniqueCandidates | ForEach-Object {
  "| ``$($_.packageId)`` | ``$($_.role)`` | ``$($_.runtimeKey)`` | ``$($_.remoteState)`` | ``$($_.proposedAction)`` |"
})
$markdown = @"
# Retired Vendor Package Cleanup Plan

This is a local candidate inventory. It did not query or modify GitHub.

| Package ID | Role | Runtime key | Remote state | Proposed action |
|---|---|---|---|---|
$($rows -join "`r`n")

Preserve the managed package, every `.Bridge` package, and GitHub-generated source archives. Run `eng/Export-RetiredVendorPackageRemoteInventory.ps1` as the formal owner, verify the exact version/asset IDs and review fingerprint, and obtain Owner confirmation before deletion.
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8
$plan | ConvertTo-Json -Depth 8
