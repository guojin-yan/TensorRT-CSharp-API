[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$PackageOwner,
  [ValidateSet("auto", "user", "org")]
  [string]$PackageOwnerKind = "auto",
  [string[]]$ExpectedReleaseTag = @(),
  [string]$PackageNamePrefix = "JYPPX.TensorRT.CSharp.API",
  [switch]$RequireOnlyExpectedReleases,
  [switch]$RequireOnlyExpectedPackageVersions,
  [switch]$RequirePackageRepositoryAssociation,
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

function Invoke-GhJsonLines {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments,
    [switch]$AllowFailure
  )

  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $output = @(& gh @Arguments 2>$stderrPath)
    $exitCode = $LASTEXITCODE
    $stderr = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    if ($exitCode -ne 0) {
      if ($AllowFailure.IsPresent) {
        return [pscustomobject]@{
          success = $false
          items = @()
          stderr = $stderr
        }
      }

      throw "gh $($Arguments -join ' ') failed. $stderr"
    }

    $items = @(
      foreach ($line in $output) {
        if ([string]::IsNullOrWhiteSpace($line)) {
          continue
        }

        $line | ConvertFrom-Json
      }
    )

    return [pscustomobject]@{
      success = $true
      items = $items
      stderr = $stderr
    }
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

function ConvertFrom-NuGetAssetFileName {
  param(
    [Parameter(Mandatory = $true)]
    [string]$AssetFileName
  )

  if (-not $AssetFileName.EndsWith(".nupkg", [System.StringComparison]::OrdinalIgnoreCase)) {
    return $null
  }

  $withoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($AssetFileName)
  if ($withoutExtension -notmatch "^(?<id>.+)\.(?<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?)$") {
    throw "Unable to parse NuGet package id/version from release asset '$AssetFileName'. Expected '<PackageId>.<version>.nupkg'."
  }

  [pscustomobject]@{
    packageId = $Matches["id"]
    version = $Matches["version"]
    key = "$($Matches["id"])@$($Matches["version"])"
  }
}

function Get-PackageEndpointPrefix {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$OwnerKind
  )

  if ($OwnerKind -eq "org") {
    return "/orgs/$Owner"
  }

  "/users/$Owner"
}

function Get-GitHubPackages {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$OwnerKind
  )

  $prefix = Get-PackageEndpointPrefix -Owner $Owner -OwnerKind $OwnerKind
  Invoke-GhJsonLines -Arguments @("api", "$prefix/packages?package_type=nuget&per_page=100", "--paginate", "--jq", ".[] | @json") -AllowFailure
}

function Resolve-GitHubPackages {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,
    [Parameter(Mandatory = $true)]
    [string]$OwnerKind
  )

  if ($OwnerKind -eq "auto") {
    $userResult = Get-GitHubPackages -Owner $Owner -OwnerKind "user"
    if ($userResult.success -and @($userResult.items).Count -gt 0) {
      return [pscustomobject]@{
        ownerKind = "user"
        packages = @($userResult.items)
        querySucceeded = $true
        stderr = $userResult.stderr
      }
    }

    $orgResult = Get-GitHubPackages -Owner $Owner -OwnerKind "org"
    if ($orgResult.success) {
      return [pscustomobject]@{
        ownerKind = "org"
        packages = @($orgResult.items)
        querySucceeded = $true
        stderr = $orgResult.stderr
      }
    }

    return [pscustomobject]@{
      ownerKind = "user"
      packages = @()
      querySucceeded = $false
      stderr = ($userResult.stderr + "`n" + $orgResult.stderr).Trim()
    }
  }

  $result = Get-GitHubPackages -Owner $Owner -OwnerKind $OwnerKind
  [pscustomobject]@{
    ownerKind = $OwnerKind
    packages = @($result.items)
    querySucceeded = $result.success
    stderr = $result.stderr
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
  $prefix = Get-PackageEndpointPrefix -Owner $Owner -OwnerKind $OwnerKind
  $endpoint = "$prefix/packages/nuget/$encodedPackageId/versions?per_page=100"
  Invoke-GhJsonLines -Arguments @("api", $endpoint, "--paginate", "--jq", ".[] | @json") -AllowFailure
}

$expectedReleaseTags = @(Expand-TokenList -Values $ExpectedReleaseTag)
if ($expectedReleaseTags.Count -eq 0) {
  throw "At least one ExpectedReleaseTag is required for publication inventory auditing."
}

$releaseResult = Invoke-GhJsonLines -Arguments @("api", "repos/$Repository/releases?per_page=100", "--paginate", "--jq", ".[] | @json")
$remoteReleases = @($releaseResult.items)
$remoteReleaseTags = @($remoteReleases | ForEach-Object { [string]$_.tag_name })
$unexpectedReleaseTags = @($remoteReleaseTags | Where-Object { $expectedReleaseTags -notcontains $_ } | Sort-Object)
$missingReleaseTags = @($expectedReleaseTags | Where-Object { $remoteReleaseTags -notcontains $_ } | Sort-Object)

$expectedPackageVersions = [ordered]@{}
$releaseAssetRows = New-Object System.Collections.Generic.List[object]
foreach ($tag in $expectedReleaseTags) {
  $release = @($remoteReleases | Where-Object { [string]$_.tag_name -eq $tag } | Select-Object -First 1)
  if ($release.Count -eq 0) {
    continue
  }

  foreach ($asset in @($release[0].assets)) {
    $assetName = [string]$asset.name
    if ([string]::IsNullOrWhiteSpace($assetName) -or -not $assetName.EndsWith(".nupkg", [System.StringComparison]::OrdinalIgnoreCase)) {
      continue
    }

    $identity = ConvertFrom-NuGetAssetFileName -AssetFileName $assetName
    if ($null -eq $identity) {
      continue
    }

    $expectedPackageVersions[$identity.key] = $identity
    $releaseAssetRows.Add([pscustomobject]@{
        releaseTag = $tag
        assetName = $assetName
        assetSize = [long]$asset.size
        packageId = $identity.packageId
        version = $identity.version
        key = $identity.key
      }) | Out-Null
  }
}

$packageInventory = Resolve-GitHubPackages -Owner $PackageOwner -OwnerKind $PackageOwnerKind
if (-not $packageInventory.querySucceeded) {
  throw "Failed to list GitHub Packages for '$PackageOwner'. $($packageInventory.stderr)"
}

$remotePackages = @(
  $packageInventory.packages |
    Where-Object {
      ([string]$_.name).StartsWith($PackageNamePrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
      ($null -ne $_.repository -and [string]$_.repository.full_name -eq $Repository)
    } |
    Sort-Object name
)

$packageVersionRows = New-Object System.Collections.Generic.List[object]
foreach ($package in $remotePackages) {
  $packageName = [string]$package.name
  $versionResult = Get-GitHubPackageVersions -Owner $PackageOwner -OwnerKind $packageInventory.ownerKind -PackageId $packageName
  if (-not $versionResult.success) {
    throw "Failed to query versions for GitHub package '$packageName'. $($versionResult.stderr)"
  }

  foreach ($version in @($versionResult.items)) {
    $versionName = [string]$version.name
    $key = "$packageName@$versionName"
    $packageVersionRows.Add([pscustomobject]@{
        packageId = $packageName
        version = $versionName
        key = $key
        versionId = [string]$version.id
        createdAt = [string]$version.created_at
        updatedAt = [string]$version.updated_at
        repository = if ($null -ne $package.repository) { [string]$package.repository.full_name } else { "" }
        expected = $expectedPackageVersions.Contains($key)
      }) | Out-Null
  }
}

$actualPackageVersionKeys = @($packageVersionRows | ForEach-Object { [string]$_.key })
$expectedPackageVersionKeys = @($expectedPackageVersions.Keys)
$missingPackageVersionKeys = @($expectedPackageVersionKeys | Where-Object { $actualPackageVersionKeys -notcontains $_ } | Sort-Object)
$unexpectedPackageVersionRows = @($packageVersionRows | Where-Object { -not $_.expected } | Sort-Object packageId, version)
$unassociatedPackageVersionRows = @(
  $packageVersionRows |
    Where-Object {
      $_.expected -and -not [string]::IsNullOrWhiteSpace([string]$_.repository) -and [string]$_.repository -ne $Repository
    } |
    Sort-Object packageId, version
)
$missingRepositoryAssociationRows = @(
  $packageVersionRows |
    Where-Object {
      $_.expected -and [string]::IsNullOrWhiteSpace([string]$_.repository)
    } |
    Sort-Object packageId, version
)

$failures = New-Object System.Collections.Generic.List[string]
if ($missingReleaseTags.Count -gt 0) {
  $failures.Add("Missing expected GitHub Releases: $($missingReleaseTags -join ', ')") | Out-Null
}

if ($RequireOnlyExpectedReleases.IsPresent -and $unexpectedReleaseTags.Count -gt 0) {
  $failures.Add("Unexpected GitHub Releases: $($unexpectedReleaseTags -join ', ')") | Out-Null
}

if ($missingPackageVersionKeys.Count -gt 0) {
  $failures.Add("Missing expected GitHub Package versions: $($missingPackageVersionKeys -join ', ')") | Out-Null
}

if ($RequireOnlyExpectedPackageVersions.IsPresent -and $unexpectedPackageVersionRows.Count -gt 0) {
  $failures.Add("Unexpected GitHub Package versions: $((@($unexpectedPackageVersionRows | ForEach-Object { $_.key })) -join ', ')") | Out-Null
}

if ($RequirePackageRepositoryAssociation.IsPresent -and ($unassociatedPackageVersionRows.Count -gt 0 -or $missingRepositoryAssociationRows.Count -gt 0)) {
  if ($unassociatedPackageVersionRows.Count -gt 0) {
    $failures.Add("GitHub Package versions associated with another repository: $((@($unassociatedPackageVersionRows | ForEach-Object { "$($_.key) -> $($_.repository)" })) -join ', ')") | Out-Null
  }

  if ($missingRepositoryAssociationRows.Count -gt 0) {
    $failures.Add("GitHub Package versions missing repository association: $((@($missingRepositoryAssociationRows | ForEach-Object { $_.key })) -join ', ')") | Out-Null
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\publication-inventory"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "github-publication-inventory.json"
$markdownPath = Join-Path $outputRoot "github-publication-inventory.md"

[pscustomobject]@{
  repository = $Repository
  packageOwner = $PackageOwner
  packageOwnerKind = $packageInventory.ownerKind
  packageNamePrefix = $PackageNamePrefix
  expectedReleaseTags = @($expectedReleaseTags)
  remoteReleaseTags = @($remoteReleaseTags)
  missingReleaseTags = @($missingReleaseTags)
  unexpectedReleaseTags = @($unexpectedReleaseTags)
  expectedPackageVersionCount = $expectedPackageVersionKeys.Count
  actualPackageVersionCount = $actualPackageVersionKeys.Count
  missingPackageVersionKeys = @($missingPackageVersionKeys)
  unexpectedPackageVersions = @($unexpectedPackageVersionRows)
  packageVersionsAssociatedWithAnotherRepository = @($unassociatedPackageVersionRows)
  packageVersionsMissingRepositoryAssociation = @($missingRepositoryAssociationRows)
  releaseAssets = @($releaseAssetRows.ToArray())
  packageVersions = @($packageVersionRows.ToArray())
  failedCount = $failures.Count
  failures = @($failures.ToArray())
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# GitHub Publication Inventory")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("## Release Summary")
$lines.Add("")
$lines.Add("- Expected releases: " + (($expectedReleaseTags | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
$lines.Add("- Remote releases: " + (($remoteReleaseTags | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
$lines.Add("- Missing releases: $($missingReleaseTags.Count)")
$lines.Add("- Unexpected releases: $($unexpectedReleaseTags.Count)")
$lines.Add("")
$lines.Add("## Package Summary")
$lines.Add("")
$lines.Add("- Expected package versions from release assets: $($expectedPackageVersionKeys.Count)")
$lines.Add("- Actual TensorRT package versions in GitHub Packages: $($actualPackageVersionKeys.Count)")
$lines.Add("- Missing package versions: $($missingPackageVersionKeys.Count)")
$lines.Add("- Unexpected package versions: $($unexpectedPackageVersionRows.Count)")
$lines.Add("- Package versions associated with another repository: $($unassociatedPackageVersionRows.Count)")
$lines.Add("- Package versions missing repository association: $($missingRepositoryAssociationRows.Count)")
$lines.Add("")
$lines.Add("| Package | Version | Expected | Repository |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($row in ($packageVersionRows | Sort-Object packageId, version)) {
  $lines.Add("| " + $codeQuote + $row.packageId + $codeQuote + " | " + $codeQuote + $row.version + $codeQuote + " | $($row.expected) | " + $codeQuote + $row.repository + $codeQuote + " |")
}

if ($failures.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Failures")
  foreach ($failure in $failures) {
    $lines.Add("- $failure")
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub publication inventory written to $jsonPath"
Write-Host "GitHub publication inventory written to $markdownPath"

if ($failures.Count -gt 0) {
  $message = "GitHub publication inventory has $($failures.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
