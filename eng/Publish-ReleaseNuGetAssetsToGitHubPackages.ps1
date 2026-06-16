[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ReleaseTag,
  [string]$Repository,
  [string[]]$AssetPattern = @(),
  [string[]]$AssetName = @(),
  [string]$PackageSource,
  [string]$PackageTokenEnvironmentVariable = "GITHUB_PACKAGES_TOKEN",
  [long]$MaxPackageBytes = 2147000000,
  [int]$DownloadAttempts = 3,
  [int]$RetryDelaySeconds = 20,
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

if ([string]::IsNullOrWhiteSpace($PackageSource)) {
  $PackageSource = "https://nuget.pkg.github.com/$($repositoryParts[0])/index.json"
}

if ($DownloadAttempts -lt 1) {
  throw "DownloadAttempts must be greater than zero."
}

if ($MaxPackageBytes -lt 1) {
  throw "MaxPackageBytes must be greater than zero."
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Expand-TokenList {
  param(
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
  if ($null -eq $asset) {
    continue
  }

  $assetName = [string]$asset.name
  if ([string]::IsNullOrWhiteSpace($assetName)) {
    Write-Warning "Skipping a release asset with an empty name on $ReleaseTag."
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

  if ($matchedByName -or $matchedByPattern) {
    $selectedByName[$assetName] = [pscustomobject]@{
      assetName = $assetName
      assetSize = [long]$asset.size
    }
  }
}

$selectedAssets = @(
  $selectedByName.GetEnumerator() |
    ForEach-Object { $_.Value } |
    Sort-Object -Property @{ Expression = { [long]$_.assetSize } }, @{ Expression = { [string]$_.assetName } }
)

if ($selectedAssets.Count -eq 0) {
  throw "No release assets matched the requested names/patterns on $ReleaseTag."
}

Write-Host "Selected $($selectedAssets.Count) release asset(s) from $ReleaseTag."

$oversizedAssets = @($selectedAssets | Where-Object { [long]$_.assetSize -ge $MaxPackageBytes })
if ($oversizedAssets.Count -gt 0) {
  $details = $oversizedAssets | ForEach-Object { "$($_.assetName)=$($_.assetSize)" }
  throw "One or more assets exceed MaxPackageBytes '$MaxPackageBytes': $($details -join ', ')"
}

$downloadRoot = [IO.Path]::Combine([IO.Path]::GetTempPath(), ("jyppx-release-package-assets-{0}" -f [Guid]::NewGuid().ToString("N")))
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null

$published = New-Object System.Collections.Generic.List[object]
try {
  foreach ($asset in $selectedAssets) {
    $assetName = [string]$asset.assetName
    if ([string]::IsNullOrWhiteSpace($assetName)) {
      throw "A selected release asset has an empty name. Refusing to publish an ambiguous package asset."
    }

    $assetSize = [long]$asset.assetSize
    $packagePath = [IO.Path]::Combine($downloadRoot, $assetName)

    for ($attempt = 1; $attempt -le $DownloadAttempts; $attempt++) {
      if (Test-Path -LiteralPath $packagePath -PathType Leaf) {
        Remove-Item -LiteralPath $packagePath -Force
      }

      Write-Host "Downloading release asset attempt ${attempt}/${DownloadAttempts}: $assetName"
      gh release download $ReleaseTag --repo $Repository --dir $downloadRoot --pattern $assetName --clobber
      if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
        $actualSize = (Get-Item -LiteralPath $packagePath).Length
        if ($actualSize -eq $assetSize) {
          break
        }

        Write-Warning "Downloaded asset '$assetName' has size $actualSize, expected $assetSize."
      }

      if ($attempt -eq $DownloadAttempts) {
        throw "Failed to download and verify release asset '$assetName' after $DownloadAttempts attempts."
      }

      Start-Sleep -Seconds $RetryDelaySeconds
    }

    Write-Host "Publishing release asset to GitHub Packages: $assetName"
    pwsh -NoProfile -File ([IO.Path]::Combine($RepositoryRoot, "eng", "Push-NuGetPackages.ps1")) `
      -PackageRoot $downloadRoot `
      -PackagePattern $assetName `
      -Source $PackageSource `
      -ApiKeyEnvironmentVariable $PackageTokenEnvironmentVariable `
      -TimeoutSeconds 3600 `
      -MaxAttempts 3 `
      -RetryDelaySeconds $RetryDelaySeconds
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to publish release asset '$assetName' to GitHub Packages."
    }

    $published.Add([pscustomobject]@{
        name = $assetName
        size = $assetSize
      }) | Out-Null

    Remove-Item -LiteralPath $packagePath -Force -ErrorAction SilentlyContinue
  }
}
finally {
  if (Test-Path -LiteralPath $downloadRoot) {
    Remove-Item -LiteralPath $downloadRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$reportRoot = [IO.Path]::Combine($RepositoryRoot, "artifacts", "release-package-publish")
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$safeTag = $ReleaseTag -replace '[^A-Za-z0-9._-]', '-'
$reportPath = [IO.Path]::Combine($reportRoot, "github-packages-$safeTag.json")
$report = [ordered]@{
  releaseTag = [string]$ReleaseTag
  repository = [string]$Repository
  packageSource = [string]$PackageSource
  publishedCount = [int]$published.Count
  publishedAssets = @($published.ToArray())
}

$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "Published $($published.Count) release assets to GitHub Packages."
Write-Host "Report written to $reportPath"
