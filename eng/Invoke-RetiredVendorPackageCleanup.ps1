[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
  [Parameter(Mandatory = $true)]
  [string]$InventoryPath,
  [string]$GitHubCliPath = "gh",
  [string]$ExpectedReviewFingerprint,
  [string]$OutputDirectory,
  [switch]$ExecuteDeletion
)

$ErrorActionPreference = "Stop"

$resolvedInventoryPath = (Resolve-Path -LiteralPath $InventoryPath).Path
$inventory = Get-Content -LiteralPath $resolvedInventoryPath -Raw -Encoding utf8 | ConvertFrom-Json
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $repositoryRoot "artifacts\release-cleanup"
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

function Add-Finding {
  param([string]$Message)
  $script:Findings.Add($Message) | Out-Null
}

function Write-ExecutionReport {
  $script:Report.updatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  $script:Report.findings = @($script:Findings.ToArray())
  $script:Report.operations = @($script:Operations.ToArray())
  $script:Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $script:JsonPath -Encoding utf8

  $findingRows = @($script:Findings | ForEach-Object { "- $_" })
  if ($findingRows.Count -eq 0) {
    $findingRows = @("- None")
  }
  $operationRows = @($script:Operations | ForEach-Object {
    "| ``$($_.kind)`` | ``$($_.target)`` | ``$($_.state)`` | ``$($_.detail)`` |"
  })
  if ($operationRows.Count -eq 0) {
    $operationRows = @("| - | - | not-started | - |")
  }
  $markdown = @"
# Retired Vendor Package Cleanup Execution

- Mode: ``$($script:Report.mode)``
- State: ``$($script:Report.state)``
- Repository: ``$($script:Report.formalRepository)``
- Review fingerprint: ``$($script:Report.reviewFingerprint)``
- Expected package versions: ``$($script:Report.expectedPackageVersionCount)``
- Expected Release assets: ``$($script:Report.expectedReleaseAssetCount)``
- Live preflight passed: ``$($script:Report.livePreflightPassed)``
- Delete executed: ``$($script:Report.deleteExecuted)``
- Post-delete validation passed: ``$($script:Report.postDeleteValidationPassed)``

## Findings

$($findingRows -join "`r`n")

## Operations

| Kind | Target | State | Detail |
|---|---|---|---|
$($operationRows -join "`r`n")
"@
  $markdown | Set-Content -LiteralPath $script:MarkdownPath -Encoding utf8
}

if ([string]$inventory.recordKind -ne "retired-vendor-package-remote-inventory") {
  throw "Inventory recordKind must be 'retired-vendor-package-remote-inventory'."
}
if (-not [bool]$inventory.remoteInventoryComplete -or
    -not [bool]$inventory.ownerReviewRequired -or
    [bool]$inventory.deleteExecuted -or
    [bool]$inventory.performsDelete) {
  throw "Inventory must be a complete, read-only, not-yet-executed Owner review record."
}
if ([int]$inventory.unmatchedReleaseAssetCount -ne 0) {
  throw "Inventory contains unmatched Release assets and cannot be executed."
}

$formalRepository = [string]$inventory.formalRepository
$repositoryParts = @($formalRepository.Split('/'))
if ($repositoryParts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($repositoryParts[0]) -or [string]::IsNullOrWhiteSpace($repositoryParts[1])) {
  throw "Inventory formalRepository is invalid: '$formalRepository'."
}
$owner = $repositoryParts[0]
$repository = $repositoryParts[1]

$policyPath = Join-Path $repositoryRoot "pack\external-vendor-runtime-policy.json"
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$managedPackageIds = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($packageId in @($policy.managedPackageIds)) {
  $managedPackageIds.Add([string]$packageId) | Out-Null
}
$bridgePattern = New-Object System.Text.RegularExpressions.Regex(
  [string]$policy.bridgePackageIdPattern,
  [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

$retiredPackages = @($inventory.packages | Where-Object disposition -eq "delete-after-owner-review" | Sort-Object packageId)
$preservedPackages = @($inventory.packages | Where-Object disposition -eq "preserve" | Sort-Object packageId)
$retiredAssets = @($inventory.releaseAssets | Where-Object disposition -eq "delete-after-owner-review" | Sort-Object id)
$preservedAssets = @($inventory.releaseAssets | Where-Object disposition -eq "preserve" | Sort-Object id)
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

foreach ($package in $retiredPackages) {
  $packageId = [string]$package.packageId
  if ($managedPackageIds.Contains($packageId) -or $bridgePattern.IsMatch($packageId)) {
    throw "Delete set contains a preserved managed/Bridge package: '$packageId'."
  }
  if (-not $packageId.StartsWith("JYPPX.TensorRT.CSharp.API.Runtime.", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Delete set contains an unexpected package identity: '$packageId'."
  }
}
foreach ($asset in $retiredAssets) {
  if ([long]$asset.id -le 0 -or
      [string]::IsNullOrWhiteSpace([string]$asset.packageId) -or
      [string]::IsNullOrWhiteSpace([string]$asset.digest)) {
    throw "Delete set contains an incomplete Release asset record: '$([string]$asset.name)'."
  }
  if (@($retiredPackages | Where-Object { [string]$_.packageId -eq [string]$asset.packageId }).Count -ne 1) {
    throw "Release asset '$([string]$asset.name)' does not map to exactly one retired package."
  }
}

$duplicateVersionIds = @($retiredPackageVersions | Group-Object id | Where-Object Count -ne 1)
$duplicateAssetIds = @($retiredAssets | Group-Object id | Where-Object Count -ne 1)
if ($duplicateVersionIds.Count -ne 0 -or $duplicateAssetIds.Count -ne 0) {
  throw "Delete set contains duplicate package version or Release asset IDs."
}

$reviewLines = New-Object System.Collections.Generic.List[string]
foreach ($version in @($retiredPackageVersions | Sort-Object packageId, id)) {
  $reviewLines.Add("package-version`t$($version.id)`t$($version.packageId)`t$($version.version)") | Out-Null
}
foreach ($asset in @($retiredAssets | Sort-Object id)) {
  $reviewLines.Add("release-asset`t$($asset.id)`t$($asset.releaseTag)`t$($asset.name)`t$($asset.digest)") | Out-Null
}
$computedFingerprint = "sha256:$(Get-Sha256Hex -Value ($reviewLines -join "`n"))"
if ($computedFingerprint -ne [string]$inventory.reviewFingerprint) {
  throw "Inventory review fingerprint is invalid. Recorded='$([string]$inventory.reviewFingerprint)' Computed='$computedFingerprint'."
}
if ($reviewLines.Count -ne [int]$inventory.reviewItemCount) {
  throw "Inventory review item count does not match the reviewed delete set."
}

if ($ExecuteDeletion.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($ExpectedReviewFingerprint)) {
    throw "-ExpectedReviewFingerprint is required with -ExecuteDeletion."
  }
  if ($ExpectedReviewFingerprint -ne $computedFingerprint) {
    throw "Expected review fingerprint does not match the live inventory delete set."
  }
}

$viewer = Invoke-GhJson -Arguments @("api", "user")
if ([string]$viewer.login -ne $owner) {
  throw "The active GitHub account must be '$owner'. Current account: '$([string]$viewer.login)'."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$script:JsonPath = Join-Path $OutputDirectory "retired-vendor-package-cleanup-execution.json"
$script:MarkdownPath = Join-Path $OutputDirectory "retired-vendor-package-cleanup-execution.md"
$script:Findings = New-Object System.Collections.Generic.List[string]
$script:Operations = New-Object System.Collections.Generic.List[object]
$script:Report = [pscustomobject]@{
  recordKind = "retired-vendor-package-cleanup-execution"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  updatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  mode = if ($ExecuteDeletion.IsPresent) { "execute" } else { "validate-only" }
  state = "preflight-running"
  formalRepository = $formalRepository
  authenticatedOwner = [string]$viewer.login
  inventoryPath = $resolvedInventoryPath
  reviewFingerprint = $computedFingerprint
  expectedPackageVersionCount = $retiredPackageVersions.Count
  expectedReleaseAssetCount = $retiredAssets.Count
  preservedPackageCount = $preservedPackages.Count
  preservedReleaseAssetCount = $preservedAssets.Count
  livePreflightPassed = $false
  deleteExecuted = $false
  deletedPackageVersionCount = 0
  deletedReleaseAssetCount = 0
  postDeleteValidationPassed = $false
  findings = @()
  operations = @()
}
Write-ExecutionReport

$allLivePackages = @(Invoke-GhPagedJson -Endpoint "users/$owner/packages?package_type=nuget&per_page=100")
$liveTensorRtPackages = @($allLivePackages | Where-Object {
  [string]$_.repository.full_name -eq $formalRepository -and
  [string]$_.name -like "JYPPX.TensorRT.CSharp.API*"
})
$liveRetiredPackageIds = @($liveTensorRtPackages | Where-Object {
  $packageId = [string]$_.name
  -not $managedPackageIds.Contains($packageId) -and -not $bridgePattern.IsMatch($packageId)
} | ForEach-Object { [string]$_.name } | Sort-Object)
$livePackageIdsByLength = @($liveTensorRtPackages | ForEach-Object { [string]$_.name } | Sort-Object { $_.Length } -Descending)
$expectedRetiredPackageIds = @($retiredPackages | ForEach-Object { [string]$_.packageId } | Sort-Object)
foreach ($packageId in @($expectedRetiredPackageIds | Where-Object { $liveRetiredPackageIds -notcontains $_ })) {
  Add-Finding "Expected retired package is missing remotely: $packageId"
}
foreach ($packageId in @($liveRetiredPackageIds | Where-Object { $expectedRetiredPackageIds -notcontains $_ })) {
  Add-Finding "Unreviewed retired package exists remotely: $packageId"
}

$versionIndex = 0
foreach ($package in $retiredPackages) {
  $versionIndex++
  $packageId = [string]$package.packageId
  Write-Progress -Activity "Preflight retired package versions" -Status "$versionIndex/$($retiredPackages.Count) $packageId" -PercentComplete (($versionIndex * 100) / [Math]::Max(1, $retiredPackages.Count))
  $expectedVersions = @($package.versions | Sort-Object id)
  try {
    $encodedPackageId = [uri]::EscapeDataString($packageId)
    $liveVersions = @(Invoke-GhPagedJson -Endpoint "users/$owner/packages/nuget/$encodedPackageId/versions?per_page=100" | Sort-Object id)
    foreach ($expectedVersion in $expectedVersions) {
      $match = @($liveVersions | Where-Object { [long]$_.id -eq [long]$expectedVersion.id })
      if ($match.Count -ne 1) {
        Add-Finding "Package version ID is missing or duplicated: $packageId / $([long]$expectedVersion.id)"
      }
      elseif ([string]$match[0].name -ne [string]$expectedVersion.version) {
        Add-Finding "Package version name drifted: $packageId / $([long]$expectedVersion.id)"
      }
    }
    foreach ($liveVersion in $liveVersions) {
      if (@($expectedVersions | Where-Object { [long]$_.id -eq [long]$liveVersion.id }).Count -ne 1) {
        Add-Finding "Unreviewed package version exists: $packageId / $([long]$liveVersion.id) / $([string]$liveVersion.name)"
      }
    }
  }
  catch {
    Add-Finding "Failed to inventory package versions for '$packageId': $($_.Exception.Message)"
  }
}
Write-Progress -Activity "Preflight retired package versions" -Completed

$liveReleases = @(Invoke-GhPagedJson -Endpoint "repos/$formalRepository/releases?per_page=100")
$liveAssets = @($liveReleases | ForEach-Object {
  $releaseTag = [string]$_.tag_name
  @($_.assets) | ForEach-Object {
    [pscustomobject]@{
      id = [long]$_.id
      releaseTag = $releaseTag
      name = [string]$_.name
      size = [long]$_.size
      digest = [string]$_.digest
    }
  }
})
foreach ($expectedAsset in $retiredAssets) {
  $match = @($liveAssets | Where-Object { [long]$_.id -eq [long]$expectedAsset.id })
  if ($match.Count -ne 1) {
    Add-Finding "Release asset ID is missing or duplicated: $([long]$expectedAsset.id) / $([string]$expectedAsset.name)"
    continue
  }
  $liveAsset = $match[0]
  if ([string]$liveAsset.name -ne [string]$expectedAsset.name -or
      [string]$liveAsset.releaseTag -ne [string]$expectedAsset.releaseTag -or
      [long]$liveAsset.size -ne [long]$expectedAsset.size -or
      [string]$liveAsset.digest -ne [string]$expectedAsset.digest) {
    Add-Finding "Release asset metadata drifted: $([long]$expectedAsset.id) / $([string]$expectedAsset.name)"
  }
}
foreach ($liveAsset in $liveAssets) {
  $matchedPackageId = $null
  foreach ($packageId in $livePackageIdsByLength) {
    if ($liveAsset.name.StartsWith("$packageId.", [StringComparison]::OrdinalIgnoreCase) -and
        $liveAsset.name.EndsWith(".nupkg", [StringComparison]::OrdinalIgnoreCase)) {
      $matchedPackageId = $packageId
      break
    }
  }
  if ($null -ne $matchedPackageId -and
      $expectedRetiredPackageIds -contains $matchedPackageId -and
      @($retiredAssets | Where-Object { [long]$_.id -eq [long]$liveAsset.id }).Count -ne 1) {
    Add-Finding "Unreviewed retired Release asset exists: $([long]$liveAsset.id) / $([string]$liveAsset.name)"
  }
}

$script:Report.livePreflightPassed = $script:Findings.Count -eq 0
$script:Report.state = if ($script:Report.livePreflightPassed) { "live-preflight-passed" } else { "live-preflight-failed" }
Write-ExecutionReport
if (-not $script:Report.livePreflightPassed) {
  throw "Live cleanup preflight failed with $($script:Findings.Count) finding(s). See '$script:JsonPath'."
}

if (-not $ExecuteDeletion.IsPresent) {
  Write-Host "Live cleanup preflight passed. No DELETE request was executed."
  Write-Host "Execution report JSON: $script:JsonPath"
  Write-Host "Execution report Markdown: $script:MarkdownPath"
  $script:Report | ConvertTo-Json -Depth 12
  return
}

$targetDescription = "$formalRepository cleanup set $computedFingerprint ($($retiredPackageVersions.Count) package versions and $($retiredAssets.Count) Release assets)"
if (-not $PSCmdlet.ShouldProcess($targetDescription, "Permanently delete reviewed retired vendor package artifacts")) {
  $script:Report.state = "execution-cancelled"
  Write-ExecutionReport
  return
}

$script:Report.state = "deletion-running"
Write-ExecutionReport
foreach ($version in @($retiredPackageVersions | Sort-Object packageId, id)) {
  $encodedPackageId = [uri]::EscapeDataString([string]$version.packageId)
  $endpoint = "users/$owner/packages/nuget/$encodedPackageId/versions/$([long]$version.id)"
  $target = "$([string]$version.packageId)@$([string]$version.version) [$([long]$version.id)]"
  try {
    Invoke-GhJson -Arguments @("api", "--method", "DELETE", $endpoint) | Out-Null
    $script:Operations.Add([pscustomobject]@{ kind = "package-version"; target = $target; state = "deleted"; detail = $endpoint }) | Out-Null
    $script:Report.deletedPackageVersionCount++
    $script:Report.deleteExecuted = $true
    Write-ExecutionReport
  }
  catch {
    $script:Operations.Add([pscustomobject]@{ kind = "package-version"; target = $target; state = "failed"; detail = $_.Exception.Message }) | Out-Null
    $script:Report.state = "partial-delete-failed"
    Write-ExecutionReport
    throw
  }
}

foreach ($asset in @($retiredAssets | Sort-Object id)) {
  $endpoint = "repos/$formalRepository/releases/assets/$([long]$asset.id)"
  $target = "$([string]$asset.releaseTag)/$([string]$asset.name) [$([long]$asset.id)]"
  try {
    Invoke-GhJson -Arguments @("api", "--method", "DELETE", $endpoint) | Out-Null
    $script:Operations.Add([pscustomobject]@{ kind = "release-asset"; target = $target; state = "deleted"; detail = $endpoint }) | Out-Null
    $script:Report.deletedReleaseAssetCount++
    $script:Report.deleteExecuted = $true
    Write-ExecutionReport
  }
  catch {
    $script:Operations.Add([pscustomobject]@{ kind = "release-asset"; target = $target; state = "failed"; detail = $_.Exception.Message }) | Out-Null
    $script:Report.state = "partial-delete-failed"
    Write-ExecutionReport
    throw
  }
}

$remainingPackages = @(Invoke-GhPagedJson -Endpoint "users/$owner/packages?package_type=nuget&per_page=100")
$remainingPackageIds = @($remainingPackages | Where-Object { [string]$_.repository.full_name -eq $formalRepository } | ForEach-Object { [string]$_.name })
foreach ($packageId in $expectedRetiredPackageIds) {
  if ($remainingPackageIds -contains $packageId) {
    Add-Finding "Retired package still exists after deletion: $packageId"
  }
}
foreach ($package in $preservedPackages) {
  if ($remainingPackageIds -notcontains [string]$package.packageId) {
    Add-Finding "Preserved package is missing after deletion: $([string]$package.packageId)"
  }
}

$remainingReleases = @(Invoke-GhPagedJson -Endpoint "repos/$formalRepository/releases?per_page=100")
$remainingAssetIds = @($remainingReleases | ForEach-Object { @($_.assets) | ForEach-Object { [long]$_.id } })
foreach ($asset in $retiredAssets) {
  if ($remainingAssetIds -contains [long]$asset.id) {
    Add-Finding "Retired Release asset still exists after deletion: $([long]$asset.id) / $([string]$asset.name)"
  }
}
foreach ($asset in $preservedAssets) {
  if ($remainingAssetIds -notcontains [long]$asset.id) {
    Add-Finding "Preserved Release asset is missing after deletion: $([long]$asset.id) / $([string]$asset.name)"
  }
}

$script:Report.postDeleteValidationPassed = $script:Findings.Count -eq 0
$script:Report.state = if ($script:Report.postDeleteValidationPassed) { "deletion-completed-and-validated" } else { "deletion-completed-post-validation-failed" }
Write-ExecutionReport
if (-not $script:Report.postDeleteValidationPassed) {
  throw "Deletion completed, but post-delete validation found $($script:Findings.Count) issue(s). See '$script:JsonPath'."
}

Write-Host "Reviewed retired package cleanup completed and validated."
Write-Host "Execution report JSON: $script:JsonPath"
Write-Host "Execution report Markdown: $script:MarkdownPath"
$script:Report | ConvertTo-Json -Depth 12
