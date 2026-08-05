[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$PackageOwner,
  [ValidateSet("auto", "user", "org")]
  [string]$PackageOwnerKind = "auto",
  [string]$ManagedPackageId = "JYPPX.TensorRT.CSharp.API",
  [string[]]$ManagedExtensionPackageId = @(),
  [string]$ManagedVersion,
  [string]$ReleaseTag,
  [string[]]$RuntimeReleaseTag = @(),
  [AllowEmptyString()]
  [string]$NuGetApiKeyAvailable,
  [switch]$RequireNuGetApiKey,
  [switch]$RequireManagedGitHubPackages,
  [switch]$RequireManagedNuGetOrg,
  [switch]$RequireManagedReleaseAsset,
  [switch]$RequireRuntimeGitHubPackagesCoverage,
  [switch]$CheckFailedWorkflowRuns,
  [int]$FailedWorkflowRunLimit = 50,
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

if ([string]::IsNullOrWhiteSpace($ManagedVersion) -and -not [string]::IsNullOrWhiteSpace($ReleaseTag)) {
  $ManagedVersion = $ReleaseTag.TrimStart("v")
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Add-Check {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [bool]$Passed,
    [string]$Detail = ""
  )

  $script:checks.Add([pscustomobject]@{
      name = $Name
      passed = $Passed
      detail = $Detail
    }) | Out-Null
}

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

function Invoke-GhJson {
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
          output = ""
          stderr = $stderr
        }
      }

      throw "gh $($Arguments -join ' ') failed. $stderr"
    }

    return [pscustomobject]@{
      success = $true
      output = ($output -join "`n")
      stderr = $stderr
    }
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
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
  $result = Invoke-GhJson -Arguments @("api", $endpoint, "--paginate", "--jq", ".[] | [.id,.name,.created_at,.updated_at] | @tsv") -AllowFailure
  if (-not $result.success) {
    if ($result.stderr -match "404|Package not found|Not Found") {
      return [pscustomobject]@{
        exists = $false
        ownerKind = $OwnerKind
        versions = @()
      }
    }

    throw "Failed to query GitHub Packages for '$PackageId' under $OwnerKind '$Owner'. $($result.stderr)"
  }

  $versions = @(
    foreach ($line in ($result.output -split "`n")) {
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

  [pscustomobject]@{
    exists = $true
    ownerKind = $OwnerKind
    versions = $versions
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

function Get-NuGetOrgVersions {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  $normalizedId = $PackageId.ToLowerInvariant()
  $uri = "https://api.nuget.org/v3-flatcontainer/$normalizedId/index.json"
  try {
    $response = Invoke-RestMethod -Uri $uri -UseBasicParsing
    return @($response.versions)
  }
  catch {
    if ($_.Exception.Message -match "404|Not Found") {
      return @()
    }

    throw
  }
}

$checks = New-Object System.Collections.Generic.List[object]
$runtimeReleaseTags = @(Expand-TokenList -Values $RuntimeReleaseTag)

if (-not [string]::IsNullOrWhiteSpace($NuGetApiKeyAvailable)) {
  $hasNuGetApiKey = $NuGetApiKeyAvailable -in @("1", "true", "True", "TRUE", "yes", "Yes", "YES")
  $requirement = if ($RequireNuGetApiKey.IsPresent) { "required" } else { "optional" }
  Add-Check -Name "repository secret NUGET_API_KEY availability" -Passed ($hasNuGetApiKey -or -not $RequireNuGetApiKey.IsPresent) -Detail "required=$requirement found=$hasNuGetApiKey source=input"
}
else {
  $secretResult = Invoke-GhJson -Arguments @("secret", "list", "--repo", $Repository) -AllowFailure
  if ($secretResult.success) {
    $secretNames = @(
      $secretResult.output -split "`n" |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { ($_ -split "\s+")[0] }
    )
    $hasNuGetApiKey = $secretNames -contains "NUGET_API_KEY"
    $requirement = if ($RequireNuGetApiKey.IsPresent) { "required" } else { "optional" }
    Add-Check -Name "repository secret NUGET_API_KEY availability" -Passed ($hasNuGetApiKey -or -not $RequireNuGetApiKey.IsPresent) -Detail "required=$requirement found=$hasNuGetApiKey source=gh-secret-list"
  }
  else {
    Add-Check -Name "repository secret list is readable" -Passed $false -Detail $secretResult.stderr
  }
}

$managedPackageIds = @(Expand-TokenList -Values @($ManagedPackageId, $ManagedExtensionPackageId))
if (-not [string]::IsNullOrWhiteSpace($ManagedVersion)) {
  foreach ($packageId in $managedPackageIds) {
    $githubVersions = Resolve-GitHubPackageVersions -Owner $PackageOwner -OwnerKind $PackageOwnerKind -PackageId $packageId
    $githubVersionExists = @($githubVersions.versions | Where-Object { [string]$_.name -eq $ManagedVersion }).Count -gt 0
    Add-Check `
      -Name "managed bundle package version exists in GitHub Packages: $packageId" `
      -Passed ($githubVersionExists -or -not $RequireManagedGitHubPackages.IsPresent) `
      -Detail "$packageId $ManagedVersion ownerKind=$($githubVersions.ownerKind) exists=$githubVersionExists"

    $nugetOrgVersions = @(Get-NuGetOrgVersions -PackageId $packageId)
    $nugetOrgVersionExists = $nugetOrgVersions -contains $ManagedVersion
    Add-Check `
      -Name "managed bundle package version exists on nuget.org: $packageId" `
      -Passed ($nugetOrgVersionExists -or -not $RequireManagedNuGetOrg.IsPresent) `
      -Detail "$packageId $ManagedVersion exists=$nugetOrgVersionExists"
  }
}

if (-not [string]::IsNullOrWhiteSpace($ReleaseTag)) {
  $releaseResult = Invoke-GhJson -Arguments @("release", "view", $ReleaseTag, "--repo", $Repository, "--json", "assets,url", "--jq", "@json") -AllowFailure
  if ($releaseResult.success) {
    $release = $releaseResult.output | ConvertFrom-Json
    $assetNames = @($release.assets | ForEach-Object { [string]$_.name })
    foreach ($packageId in $managedPackageIds) {
      $expectedManagedAsset = if ([string]::IsNullOrWhiteSpace($ManagedVersion)) { "" } else { "$packageId.$ManagedVersion.nupkg" }
      $hasManagedReleaseAsset = -not [string]::IsNullOrWhiteSpace($expectedManagedAsset) -and ($assetNames -contains $expectedManagedAsset)
      Add-Check `
        -Name "managed bundle Release asset exists: $packageId" `
        -Passed ($hasManagedReleaseAsset -or -not $RequireManagedReleaseAsset.IsPresent) `
        -Detail "$ReleaseTag asset=$expectedManagedAsset exists=$hasManagedReleaseAsset assetCount=$($assetNames.Count)"
    }
  }
  else {
    Add-Check -Name "managed package release exists" -Passed (-not $RequireManagedReleaseAsset.IsPresent) -Detail "$ReleaseTag not found"
  }
}

foreach ($tag in $runtimeReleaseTags) {
  if ($RequireRuntimeGitHubPackagesCoverage.IsPresent) {
    & (Join-Path $RepositoryRoot "eng\Test-GitHubPackagesCoverage.ps1") -ReleaseTag $tag -Repository $Repository -PackageOwner $PackageOwner -PackageOwnerKind $PackageOwnerKind
    Add-Check -Name "runtime release assets exist in GitHub Packages" -Passed $true -Detail $tag
  }
  else {
    $releaseResult = Invoke-GhJson -Arguments @("release", "view", $tag, "--repo", $Repository, "--json", "assets", "--jq", "@json") -AllowFailure
    $exists = $releaseResult.success
    $assetCount = if ($exists) { @((($releaseResult.output | ConvertFrom-Json).assets)).Count } else { 0 }
    Add-Check -Name "runtime release exists" -Passed $exists -Detail "$tag assetCount=$assetCount"
  }
}

if ($CheckFailedWorkflowRuns.IsPresent) {
  $failedRunResult = Invoke-GhJson -Arguments @("run", "list", "--repo", $Repository, "--status", "failure", "--limit", ([string]$FailedWorkflowRunLimit), "--json", "databaseId,name,workflowName,status,conclusion,createdAt,url", "--jq", "@json") -AllowFailure
  if ($failedRunResult.success) {
    $failedRuns = @($failedRunResult.output | ConvertFrom-Json)
    Add-Check -Name "no failed workflow runs in inspected window" -Passed ($failedRuns.Count -eq 0) -Detail "failedRuns=$($failedRuns.Count) limit=$FailedWorkflowRunLimit"
  }
  else {
    Add-Check -Name "failed workflow run list is readable" -Passed $false -Detail $failedRunResult.stderr
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release-publication"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$safeVersion = if ([string]::IsNullOrWhiteSpace($ManagedVersion)) { "current" } else { $ManagedVersion -replace '[^A-Za-z0-9._-]', '-' }
$jsonPath = Join-Path $outputRoot "release-publication-state-$safeVersion.json"
$markdownPath = Join-Path $outputRoot "release-publication-state-$safeVersion.md"

$failed = @($checks | Where-Object { -not $_.passed })
[pscustomobject]@{
  repository = $Repository
  packageOwner = $PackageOwner
  packageOwnerKind = $PackageOwnerKind
  managedPackageId = $ManagedPackageId
  managedExtensionPackageId = @($ManagedExtensionPackageId)[0]
  managedExtensionPackageIds = @($ManagedExtensionPackageId)
  managedPackageIds = $managedPackageIds
  managedVersion = $ManagedVersion
  releaseTag = $ReleaseTag
  runtimeReleaseTags = @($runtimeReleaseTags)
  failedCount = $failed.Count
  checks = @($checks.ToArray())
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Release Publication State")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("| Check | Passed | Detail |")
$lines.Add("| --- | --- | --- |")
foreach ($check in $checks) {
  $detail = ([string]$check.detail).Replace("|", "\|")
  $lines.Add("| $($check.name) | $($check.passed) | $detail |")
}
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Checks: $($checks.Count)")
$lines.Add("- Failed checks: $($failed.Count)")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release publication state written to $jsonPath"
Write-Host "Release publication state written to $markdownPath"

if ($failed.Count -gt 0) {
  $message = "Release publication state has $($failed.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
