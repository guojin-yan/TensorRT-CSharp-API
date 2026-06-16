[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ReleaseTag,
  [string]$Repository,
  [string]$PackageOwner,
  [ValidateSet("auto", "user", "org")]
  [string]$PackageOwnerKind = "auto",
  [string[]]$AssetPattern = @(),
  [string[]]$AssetName = @(),
  [switch]$WarnOnly,
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

$repositoryParts = $Repository.Split("/", 2)
if ($repositoryParts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($repositoryParts[0]) -or [string]::IsNullOrWhiteSpace($repositoryParts[1])) {
  throw "Repository must use the 'owner/name' format. Value: $Repository"
}

if ([string]::IsNullOrWhiteSpace($PackageOwner)) {
  $PackageOwner = $repositoryParts[0]
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Expand-TokenList {
  param(
    [AllowNull()]
    [string[]]$Values
  )

  $tokens = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[`r`n,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $tokens.Add($trimmed)
      }
    }
  }

  @($tokens | Select-Object -Unique)
}

function ConvertFrom-NuGetAssetFileName {
  param(
    [Parameter(Mandatory = $true)]
    [object]$AssetFileName
  )

  $assetFileNameText = if ($AssetFileName -is [array]) {
    @($AssetFileName | ForEach-Object { [string]$_ }) -join ""
  }
  else {
    [string]$AssetFileName
  }

  if ([string]::IsNullOrWhiteSpace($assetFileNameText)) {
    throw "Release asset name is empty."
  }

  if (-not $assetFileNameText.EndsWith(".nupkg", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Release asset '$assetFileNameText' is not a .nupkg file."
  }

  $withoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($assetFileNameText)
  if ($withoutExtension -notmatch "^(?<id>.+)\.(?<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?)$") {
    throw "Unable to parse NuGet package id/version from release asset '$assetFileNameText'. Expected '<PackageId>.<version>.nupkg'."
  }

  [pscustomobject]@{
    packageId = $Matches["id"]
    version = $Matches["version"]
  }
}

function Get-GitHubPackageVersions {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$OwnerKind,
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  $encodedPackageId = [Uri]::EscapeDataString($PackageId)
  $pathPrefix = if ($OwnerKind -eq "org") { "orgs" } else { "users" }
  $endpoint = "/$pathPrefix/$Owner/packages/nuget/$encodedPackageId/versions?per_page=100"

  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $lines = @(& gh api $endpoint --paginate --jq '.[] | [.id,.name,.created_at,.updated_at] | @tsv' 2>$stderrPath)
    $exitCode = $LASTEXITCODE
    $stderr = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }

    if ($exitCode -ne 0) {
      if ($stderr -match "404|Package not found|Not Found") {
        return [pscustomobject]@{
          exists = $false
          ownerKind = $OwnerKind
          versions = @()
        }
      }

      throw "Failed to query GitHub Packages for '$PackageId' under $OwnerKind '$Owner'. $stderr"
    }

    $versions = @(
      foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
          continue
        }

        $parts = $line -split "`t"
        [pscustomobject]@{
          id = if ($parts.Count -gt 0) { $parts[0] } else { "" }
          name = if ($parts.Count -gt 1) { $parts[1] } else { "" }
          createdAt = if ($parts.Count -gt 2) { $parts[2] } else { "" }
          updatedAt = if ($parts.Count -gt 3) { $parts[3] } else { "" }
        }
      }
    )

    return [pscustomobject]@{
      exists = $true
      ownerKind = $OwnerKind
      versions = $versions
    }
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

function Resolve-GitHubPackageVersions {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$OwnerKind,
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  if ($OwnerKind -eq "auto") {
    $userResult = Get-GitHubPackageVersions -Owner $Owner -OwnerKind "user" -PackageId $PackageId
    if ($userResult.exists) {
      return $userResult
    }

    $orgResult = Get-GitHubPackageVersions -Owner $Owner -OwnerKind "org" -PackageId $PackageId
    if ($orgResult.exists) {
      return $orgResult
    }

    return $userResult
  }

  Get-GitHubPackageVersions -Owner $Owner -OwnerKind $OwnerKind -PackageId $PackageId
}

$patterns = @(Expand-TokenList -Values $AssetPattern)
$names = @(Expand-TokenList -Values $AssetName)
if ($patterns.Count -eq 0 -and $names.Count -eq 0) {
  $patterns = @("*.nupkg")
}

$releaseJson = gh api "repos/$Repository/releases/tags/$ReleaseTag"
if ($LASTEXITCODE -ne 0) {
  throw "Failed to read release '$ReleaseTag' from repository '$Repository'."
}

$release = $releaseJson | ConvertFrom-Json
$assets = @($release.assets)
$selectedByName = [ordered]@{}
foreach ($asset in $assets) {
  $assetName = [string]$asset.name
  if ([string]::IsNullOrWhiteSpace($assetName)) {
    continue
  }

  $matchedByName = $names.Count -gt 0 -and ($names -contains $assetName)
  $matchedByPattern = $false
  foreach ($pattern in $patterns) {
    if ($assetName -like $pattern) {
      $matchedByPattern = $true
      break
    }
  }

  if (($matchedByName -or $matchedByPattern) -and $assetName.EndsWith(".nupkg", [System.StringComparison]::OrdinalIgnoreCase)) {
    $selectedByName[$assetName] = [pscustomobject]@{
      name = $assetName
      size = [long]$asset.size
    }
  }
}

$selectedAssets = @(
  $selectedByName.GetEnumerator() |
    ForEach-Object { $_.Value } |
    Sort-Object -Property @{ Expression = { [string]$_.name } }
)

if ($selectedAssets.Count -eq 0) {
  throw "No .nupkg release assets matched the requested names/patterns on $ReleaseTag."
}

Write-Host "Auditing GitHub Packages coverage for $($selectedAssets.Count) release asset(s) from $ReleaseTag."

$packageCache = @{}
$rows = New-Object System.Collections.Generic.List[object]
foreach ($asset in $selectedAssets) {
  $assetName = [string]$asset.name
  $identity = ConvertFrom-NuGetAssetFileName -AssetFileName $assetName
  $packageId = [string]$identity.packageId
  $version = [string]$identity.version

  if (-not $packageCache.ContainsKey($packageId)) {
    $packageCache[$packageId] = Resolve-GitHubPackageVersions -Owner $PackageOwner -OwnerKind $PackageOwnerKind -PackageId $packageId
  }

  $packageResult = $packageCache[$packageId]
  $matchingVersion = @($packageResult.versions | Where-Object { [string]$_.name -eq $version } | Select-Object -First 1)
  $rows.Add([pscustomobject]@{
      assetName = $assetName
      assetSize = [long]$asset.size
      packageId = $packageId
      version = $version
      packageOwner = $PackageOwner
      packageOwnerKind = [string]$packageResult.ownerKind
      packageExists = [bool]$packageResult.exists
      versionExists = [bool]($matchingVersion.Count -gt 0)
      packageVersionId = if ($matchingVersion.Count -gt 0) { [string]$matchingVersion[0].id } else { "" }
      packageVersionUpdatedAt = if ($matchingVersion.Count -gt 0) { [string]$matchingVersion[0].updatedAt } else { "" }
    }) | Out-Null
}

$missing = @($rows | Where-Object { -not $_.versionExists })
$outputRoot = Join-Path $RepositoryRoot "artifacts\github-packages-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$safeTag = $ReleaseTag -replace '[^A-Za-z0-9._-]', '-'
$jsonPath = Join-Path $outputRoot "github-packages-coverage-$safeTag.json"
$markdownPath = Join-Path $outputRoot "github-packages-coverage-$safeTag.md"

[pscustomobject]@{
  releaseTag = $ReleaseTag
  repository = $Repository
  packageOwner = $PackageOwner
  packageOwnerKind = $PackageOwnerKind
  expectedCount = $rows.Count
  missingCount = $missing.Count
  rows = @($rows.ToArray())
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# GitHub Packages Coverage Audit")
$lines.Add("")
$lines.Add("Release: " + $codeQuote + $ReleaseTag + $codeQuote)
$lines.Add("")
$lines.Add("| Package | Version | In Packages | Asset |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($row in ($rows | Sort-Object packageId, version)) {
  $lines.Add("| " + $codeQuote + $row.packageId + $codeQuote + " | " + $codeQuote + $row.version + $codeQuote + " | $($row.versionExists) | " + $codeQuote + $row.assetName + $codeQuote + " |")
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Expected package versions: $($rows.Count)")
$lines.Add("- Missing package versions: $($missing.Count)")

if ($missing.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Missing")
  $lines.Add("")
  foreach ($row in ($missing | Sort-Object packageId, version)) {
    $lines.Add("- " + $codeQuote + $row.packageId + $codeQuote + " " + $codeQuote + $row.version + $codeQuote + " from asset " + $codeQuote + $row.assetName + $codeQuote)
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub Packages coverage audit written to $jsonPath"
Write-Host "GitHub Packages coverage audit written to $markdownPath"

if ($missing.Count -gt 0) {
  $message = "GitHub Packages is missing $($missing.Count) selected package version(s) from release '$ReleaseTag'."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
