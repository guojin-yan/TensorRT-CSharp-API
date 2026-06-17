[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$InventoryJsonPath,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  $Repository = $env:GITHUB_REPOSITORY
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  throw "Repository is required. Pass -Repository owner/name or set GITHUB_REPOSITORY."
}

if ([string]::IsNullOrWhiteSpace($InventoryJsonPath)) {
  $InventoryJsonPath = Join-Path $RepositoryRoot "artifacts\publication-inventory\github-publication-inventory.json"
}

if (-not (Test-Path -LiteralPath $InventoryJsonPath -PathType Leaf)) {
  throw "Publication inventory JSON was not found: $InventoryJsonPath"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$inventory = Get-Content -LiteralPath $InventoryJsonPath -Raw -Encoding utf8 | ConvertFrom-Json

function ConvertFrom-RuntimePackageId {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  $prefix = "JYPPX.TensorRT.CSharp.API.Runtime."
  if (-not $PackageId.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $null
  }

  $runtimeIdentity = $PackageId.Substring($prefix.Length)
  $pattern = '^(?<target>win-x64|linux-x64\.ubuntu\d{2}\.\d{2})\.trt(?<tensorRt>[0-9]+\.[0-9]+)\.cuda(?<cuda>[0-9]+\.[0-9]+)\.cudnn(?<cudnn>[0-9]+\.[0-9]+)(?:\.(?<component>.+))?$'
  if ($runtimeIdentity -notmatch $pattern) {
    return $null
  }

  [pscustomobject]@{
    target = [string]$Matches["target"]
    dependencyCombination = "trt$($Matches["tensorRt"])-cuda$($Matches["cuda"])-cudnn$($Matches["cudnn"])"
    component = if ($Matches.ContainsKey("component") -and -not [string]::IsNullOrWhiteSpace([string]$Matches["component"])) { [string]$Matches["component"] } else { "Base" }
  }
}

$releaseAssetRows = @($inventory.releaseAssets)
$packageVersionRows = @($inventory.packageVersions)
$runtimeMatrixRows = @($inventory.runtimeMatrix)
$runtimeTargetSummaryRows = @($inventory.runtimeTargetSummary)

$releaseTagsByPackageVersion = @{}
foreach ($asset in $releaseAssetRows) {
  $key = "$($asset.packageId)@$($asset.version)"
  if (-not $releaseTagsByPackageVersion.ContainsKey($key)) {
    $releaseTagsByPackageVersion[$key] = [System.Collections.Generic.List[string]]::new()
  }

  if (-not $releaseTagsByPackageVersion[$key].Contains([string]$asset.releaseTag)) {
    $releaseTagsByPackageVersion[$key].Add([string]$asset.releaseTag) | Out-Null
  }
}

$packageRepositoryByKey = @{}
foreach ($packageVersion in $packageVersionRows) {
  $packageRepositoryByKey["$($packageVersion.packageId)@$($packageVersion.version)"] = [string]$packageVersion.repository
}

$runtimeRows = foreach ($matrixRow in $runtimeMatrixRows) {
  $assetMatches = @(
    $releaseAssetRows | Where-Object {
      $identity = ConvertFrom-RuntimePackageId -PackageId ([string]$_.packageId)
      $identity -and
        [string]$identity.target -eq [string]$matrixRow.target -and
        [string]$identity.dependencyCombination -eq [string]$matrixRow.dependencyCombination -and
        [string]$_.version -eq [string]$matrixRow.version
    }
  )

  $releaseTags = @($assetMatches | Select-Object -ExpandProperty releaseTag -Unique | Sort-Object)
  $packageIds = @($assetMatches | Select-Object -ExpandProperty packageId -Unique | Sort-Object)
  $repositories = @(
    foreach ($packageId in $packageIds) {
      $key = "$packageId@$($matrixRow.version)"
      if ($packageRepositoryByKey.ContainsKey($key)) {
        $packageRepositoryByKey[$key]
      }
    }
  ) | Sort-Object -Unique

  [pscustomobject]@{
    target = [string]$matrixRow.target
    dependencyCombination = [string]$matrixRow.dependencyCombination
    version = [string]$matrixRow.version
    releaseTags = @($releaseTags)
    releaseAssetComponentCount = [int]$matrixRow.releaseAssetCount
    githubPackageComponentCount = [int]$matrixRow.githubPackageVersionCount
    releaseComponents = @($matrixRow.releaseComponents)
    githubPackageComponents = @($matrixRow.githubPackageComponents)
    missingGitHubPackageComponents = @($matrixRow.missingGitHubPackageComponents)
    unexpectedGitHubPackageComponents = @($matrixRow.unexpectedGitHubPackageComponents)
    repositories = @($repositories)
    complete = (
      [int]$matrixRow.releaseAssetCount -gt 0 -and
      [int]$matrixRow.githubPackageVersionCount -gt 0 -and
      @($matrixRow.missingGitHubPackageComponents).Count -eq 0 -and
      @($matrixRow.unexpectedGitHubPackageComponents).Count -eq 0
    )
  }
}

$managedRows = @(
  $packageVersionRows |
    Where-Object { [string]$_.packageId -eq "JYPPX.TensorRT.CSharp.API" } |
    Sort-Object version
)

$outputRoot = Join-Path $RepositoryRoot "artifacts\publication-index"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "runtime-publication-index.json"
$markdownPath = Join-Path $outputRoot "runtime-publication-index.md"

[pscustomobject]@{
  repository = $Repository
  inventoryJsonPath = $InventoryJsonPath
  expectedReleaseTags = @($inventory.expectedReleaseTags)
  remoteReleaseTags = @($inventory.remoteReleaseTags)
  failedCount = [int]$inventory.failedCount
  managedPackages = @(
    foreach ($managedRow in $managedRows) {
      $key = "$($managedRow.packageId)@$($managedRow.version)"
      [pscustomobject]@{
        packageId = [string]$managedRow.packageId
        version = [string]$managedRow.version
        repository = [string]$managedRow.repository
        releaseTags = if ($releaseTagsByPackageVersion.ContainsKey($key)) { @($releaseTagsByPackageVersion[$key].ToArray()) } else { @() }
      }
    }
  )
  targets = @($runtimeTargetSummaryRows)
  runtimeRows = @($runtimeRows)
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$codeQuote = [string][char]96
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Runtime Publication Index")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("Expected releases: " + ((@($inventory.expectedReleaseTags) | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
$lines.Add("Remote releases: " + ((@($inventory.remoteReleaseTags) | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
$lines.Add("")
$lines.Add("## Managed Package")
$lines.Add("")
$lines.Add("| Package | Version | Release tag | Repository |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($managedRow in $managedRows) {
  $key = "$($managedRow.packageId)@$($managedRow.version)"
  $tags = if ($releaseTagsByPackageVersion.ContainsKey($key)) { @($releaseTagsByPackageVersion[$key].ToArray()) } else { @() }
  $tagText = if ($tags.Count -gt 0) { (($tags | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", ") } else { "" }
  $lines.Add("| " + $codeQuote + $managedRow.packageId + $codeQuote + " | " + $codeQuote + $managedRow.version + $codeQuote + " | $tagText | " + $codeQuote + $managedRow.repository + $codeQuote + " |")
}

$lines.Add("")
$lines.Add("## Runtime Targets")
$lines.Add("")
$lines.Add("| Target | Versions | Combinations | Release components | GitHub Package components |")
$lines.Add("| --- | --- | ---: | ---: | ---: |")
foreach ($target in $runtimeTargetSummaryRows) {
  $versions = (@($target.versions) | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "
  $lines.Add("| " + $codeQuote + $target.target + $codeQuote + " | $versions | $($target.dependencyCombinationCount) | $($target.releaseAssetComponentCount) | $($target.githubPackageComponentCount) |")
}

$lines.Add("")
$lines.Add("## Runtime Matrix")
$lines.Add("")
$lines.Add("| Target | Combination | Version | Release tag | Release components | GitHub Package components | Complete |")
$lines.Add("| --- | --- | --- | --- | ---: | ---: | --- |")
foreach ($row in ($runtimeRows | Sort-Object target,dependencyCombination,version)) {
  $releaseTagText = if ($row.releaseTags.Count -gt 0) { (($row.releaseTags | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", ") } else { "" }
  $lines.Add("| " + $codeQuote + $row.target + $codeQuote + " | " + $codeQuote + $row.dependencyCombination + $codeQuote + " | " + $codeQuote + $row.version + $codeQuote + " | $releaseTagText | $($row.releaseAssetComponentCount) | $($row.githubPackageComponentCount) | $($row.complete) |")
}

$lines.Add("")
$lines.Add("## Notes")
$lines.Add("")
$lines.Add("- Windows and Linux runtime packages are intentionally spread across runtime release tags. The latest managed release does not necessarily contain every runtime asset.")
$lines.Add("- Ubuntu 20.04, ARM/SBSA, Jetson/L4T, and non-Ubuntu Linux remain separate or infrastructure-blocked package lines until their dedicated runner and NVIDIA dependency strategy are validated.")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime publication index written to $jsonPath"
Write-Host "Runtime publication index written to $markdownPath"
