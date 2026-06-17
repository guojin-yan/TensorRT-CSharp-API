[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$PackageOwner,
  [ValidateSet("auto", "user", "org")]
  [string]$PackageOwnerKind = "auto",
  [string[]]$ExpectedReleaseTag = @("v4.0.6156", "v4.0.6167", "v4.0.6169", "v4.0.6170"),
  [string]$InventoryJsonPath,
  [switch]$RequireInfrastructureBlockedTargetsPublished,
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

function Get-DependencyCombination {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package
  )

  $tensorRt = ([string]$Package.tensorRtVersion) -replace '^([0-9]+\.[0-9]+).*', '$1'
  $cuda = [string]$Package.cudaVersion
  $cudnn = ([string]$Package.cudnnVersion) -replace '^([0-9]+\.[0-9]+).*', '$1'
  "trt$tensorRt-cuda$cuda-cudnn$cudnn"
}

function Get-PublicationTarget {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package
  )

  if ([string]$Package.platform -eq "windows") {
    return [string]$Package.rid
  }

  "linux-$($Package.architecture).$($Package.linuxDistro)$($Package.linuxDistroVersion)"
}

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

function New-TargetDefinition {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Target,
    [Parameter(Mandatory = $true)]
    [string]$Requirement,
    [Parameter(Mandatory = $true)]
    [string]$Notes,
    [Parameter(Mandatory = $true)]
    [object[]]$Packages
  )

  $combos = @($Packages | ForEach-Object { Get-DependencyCombination -Package $_ } | Sort-Object -Unique)
  [pscustomobject]@{
    target = $Target
    requirement = $Requirement
    notes = $Notes
    expectedDependencyCombinations = @($combos)
    expectedPackageKeys = @($Packages | Select-Object -ExpandProperty key)
  }
}

if ([string]::IsNullOrWhiteSpace($InventoryJsonPath)) {
  $InventoryJsonPath = Join-Path $RepositoryRoot "artifacts\publication-inventory\github-publication-inventory.json"
}

if (-not (Test-Path -LiteralPath $InventoryJsonPath -PathType Leaf)) {
  $inventoryArguments = @(
    "-NoProfile",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-GitHubPublicationInventory.ps1"),
    "-Repository",
    $Repository,
    "-PackageOwner",
    $PackageOwner,
    "-PackageOwnerKind",
    $PackageOwnerKind,
    "-ExpectedReleaseTag",
    ((Expand-TokenList -Values $ExpectedReleaseTag) -join ","),
    "-RepositoryRoot",
    $RepositoryRoot
  )

  & pwsh @inventoryArguments
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate GitHub publication inventory."
  }
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$packages = @($manifest.packages)
$inventory = Get-Content -LiteralPath $InventoryJsonPath -Raw -Encoding utf8 | ConvertFrom-Json

$windowsPackages = @($packages | Where-Object { [string]$_.platform -eq "windows" })
$linuxUbuntu22Packages = @($packages | Where-Object { [string]$_.platform -eq "linux" -and [string]$_.architecture -eq "x64" -and [string]$_.linuxDistro -eq "ubuntu" -and [string]$_.linuxDistroVersion -eq "22.04" })
$linuxUbuntu24Packages = @($packages | Where-Object { [string]$_.platform -eq "linux" -and [string]$_.architecture -eq "x64" -and [string]$_.linuxDistro -eq "ubuntu" -and [string]$_.linuxDistroVersion -eq "24.04" })
$linuxUbuntu20Packages = @($packages | Where-Object { [string]$_.platform -eq "linux" -and [string]$_.architecture -eq "x64" -and [string]$_.linuxDistro -eq "ubuntu" -and [string]$_.linuxDistroVersion -eq "20.04" })

$targetDefinitions = @(
  New-TargetDefinition `
    -Target "win-x64" `
    -Requirement "published-required" `
    -Notes "Windows x64 is the primary self-hosted runtime publication target and must publish all six dependency combinations." `
    -Packages $windowsPackages
  New-TargetDefinition `
    -Target "linux-x64.ubuntu22.04" `
    -Requirement "published-required" `
    -Notes "Ubuntu 22.04 x64 is the hosted Linux publication target for all six dependency combinations." `
    -Packages $linuxUbuntu22Packages
  New-TargetDefinition `
    -Target "linux-x64.ubuntu24.04" `
    -Requirement "published-required" `
    -Notes "Ubuntu 24.04 x64 is hosted for the modern combinations that NVIDIA publishes for that distro." `
    -Packages $linuxUbuntu24Packages
  New-TargetDefinition `
    -Target "linux-x64.ubuntu20.04" `
    -Requirement "published-required" `
    -Notes "Ubuntu 20.04 x64 runs through the hosted Ubuntu 20.04 container lane and must publish its modeled legacy dependency combinations." `
    -Packages $linuxUbuntu20Packages
)

$runtimeMatrixRows = @($inventory.runtimeMatrix)
$packageVersionRows = @($inventory.packageVersions)
$runtimePackageVersionRows = @(
  foreach ($row in $packageVersionRows) {
    $identity = ConvertFrom-RuntimePackageId -PackageId ([string]$row.packageId)
    if ($null -ne $identity) {
      [pscustomobject]@{
        packageId = [string]$row.packageId
        version = [string]$row.version
        repository = [string]$row.repository
        expected = [bool]$row.expected
        target = [string]$identity.target
        dependencyCombination = [string]$identity.dependencyCombination
        component = [string]$identity.component
      }
    }
  }
)

$targetResults = New-Object System.Collections.Generic.List[object]
$failures = New-Object System.Collections.Generic.List[string]

foreach ($definition in $targetDefinitions) {
  $comboResults = New-Object System.Collections.Generic.List[object]
  foreach ($combo in @($definition.expectedDependencyCombinations)) {
    $matchingMatrixRows = @($runtimeMatrixRows | Where-Object { [string]$_.target -eq [string]$definition.target -and [string]$_.dependencyCombination -eq [string]$combo })
    $publishedRows = @(
      $matchingMatrixRows |
        Where-Object {
          [int]$_.releaseAssetCount -gt 0 -and
          [int]$_.githubPackageVersionCount -gt 0 -and
          @($_.missingGitHubPackageComponents).Count -eq 0 -and
          @($_.unexpectedGitHubPackageComponents).Count -eq 0
        }
    )

    $comboPackageRows = @($runtimePackageVersionRows | Where-Object { [string]$_.target -eq [string]$definition.target -and [string]$_.dependencyCombination -eq [string]$combo })
    $repositoryMismatches = @($comboPackageRows | Where-Object { [string]$_.repository -ne $Repository })
    $isPublished = $publishedRows.Count -gt 0
    $repositoryAssociated = $isPublished -and $comboPackageRows.Count -gt 0 -and $repositoryMismatches.Count -eq 0

    $comboResults.Add([pscustomobject]@{
        dependencyCombination = [string]$combo
        published = $isPublished
        repositoryAssociated = $repositoryAssociated
        versions = @($matchingMatrixRows | Select-Object -ExpandProperty version -Unique)
        releaseAssetComponentCount = [int](($matchingMatrixRows | Measure-Object -Property releaseAssetCount -Sum).Sum)
        githubPackageComponentCount = [int](($matchingMatrixRows | Measure-Object -Property githubPackageVersionCount -Sum).Sum)
        releaseComponents = @($matchingMatrixRows | ForEach-Object { $_.releaseComponents } | Sort-Object -Unique)
        githubPackageComponents = @($matchingMatrixRows | ForEach-Object { $_.githubPackageComponents } | Sort-Object -Unique)
        packageRows = @($comboPackageRows)
      }) | Out-Null
  }

  $publishedComboResults = @($comboResults | Where-Object { $_.published -and $_.repositoryAssociated })
  $missingComboResults = @($comboResults | Where-Object { -not ($_.published -and $_.repositoryAssociated) })
  $coverageState = if ($publishedComboResults.Count -eq $definition.expectedDependencyCombinations.Count -and $definition.expectedDependencyCombinations.Count -gt 0) {
    "complete"
  }
  elseif ($publishedComboResults.Count -gt 0) {
    "partial"
  }
  else {
    "not-published"
  }

  $targetPassed = $true
  if ([string]$definition.requirement -eq "published-required") {
    $targetPassed = $coverageState -eq "complete"
  }
  elseif ($RequireInfrastructureBlockedTargetsPublished.IsPresent -and [string]$definition.requirement -eq "infrastructure-blocked") {
    $targetPassed = $coverageState -eq "complete"
  }

  if (-not $targetPassed) {
    $failures.Add("$($definition.target): requirement=$($definition.requirement) state=$coverageState missing=$((@($missingComboResults | ForEach-Object { $_.dependencyCombination })) -join ',')") | Out-Null
  }

  $targetResults.Add([pscustomobject]@{
      target = [string]$definition.target
      requirement = [string]$definition.requirement
      coverageState = $coverageState
      passed = $targetPassed
      expectedCombinationCount = @($definition.expectedDependencyCombinations).Count
      publishedCombinationCount = $publishedComboResults.Count
      expectedDependencyCombinations = @($definition.expectedDependencyCombinations)
      publishedDependencyCombinations = @($publishedComboResults | ForEach-Object { $_.dependencyCombination })
      missingDependencyCombinations = @($missingComboResults | ForEach-Object { $_.dependencyCombination })
      expectedPackageKeys = @($definition.expectedPackageKeys)
      notes = [string]$definition.notes
      combinations = @($comboResults.ToArray())
    }) | Out-Null
}

$futureTargets = @(
  [pscustomobject]@{
    target = "linux-arm64-sbsa"
    requirement = "future-separate-package-line"
    status = "not-modeled"
    requiredEvidence = @(
      "distro/architecture-qualified runtime package IDs",
      "arm64/SBSA runner labels or runner pool",
      "official NVIDIA arm64/SBSA dependency plan",
      "package consumer validation evidence"
    )
  }
  [pscustomobject]@{
    target = "linux-jetson-l4t"
    requirement = "future-separate-package-line"
    status = "not-modeled"
    requiredEvidence = @(
      "Jetson/L4T release identity in runtime package IDs",
      "Jetson board or L4T image validation strategy",
      "JetPack/L4T-compatible NVIDIA dependency source",
      "package consumer validation evidence"
    )
  }
  [pscustomobject]@{
    target = "non-ubuntu-linux"
    requirement = "future-separate-package-line"
    status = "not-modeled"
    requiredEvidence = @(
      "distribution/version-qualified runtime package IDs",
      "runner image or labels for the target distribution",
      "official NVIDIA dependency source for that distribution",
      "package consumer validation evidence"
    )
  }
)

$manifestFuturePackages = @(
  $packages | Where-Object {
    [string]$_.platform -eq "linux" -and
    ([string]$_.architecture -ne "x64" -or [string]$_.linuxDistro -ne "ubuntu")
  }
)
$publishedUnexpectedFutureRows = @(
  $runtimeMatrixRows | Where-Object {
    [string]$_.target -notin @("win-x64", "linux-x64.ubuntu20.04", "linux-x64.ubuntu22.04", "linux-x64.ubuntu24.04")
  }
)

if ($manifestFuturePackages.Count -gt 0) {
  $failures.Add("Future Linux package lines are present in the manifest before target coverage rules were added: $((@($manifestFuturePackages | ForEach-Object { $_.key })) -join ', ')") | Out-Null
}

if ($publishedUnexpectedFutureRows.Count -gt 0) {
  $failures.Add("Unexpected future/non-modeled runtime package rows are published: $((@($publishedUnexpectedFutureRows | ForEach-Object { "$($_.target)/$($_.dependencyCombination)" })) -join ', ')") | Out-Null
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-publication-target-coverage"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "runtime-publication-target-coverage.json"
$markdownPath = Join-Path $outputRoot "runtime-publication-target-coverage.md"

[pscustomobject]@{
  repository = $Repository
  inventoryJsonPath = $InventoryJsonPath
  failedCount = $failures.Count
  targets = @($targetResults.ToArray())
  futureTargets = @($futureTargets)
  manifestFuturePackageKeys = @($manifestFuturePackages | Select-Object -ExpandProperty key)
  publishedUnexpectedFutureRows = @($publishedUnexpectedFutureRows)
  failures = @($failures.ToArray())
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Runtime Publication Target Coverage")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("| Target | Requirement | State | Expected combos | Published combos | Passed |")
$lines.Add("| --- | --- | --- | ---: | ---: | --- |")
foreach ($target in $targetResults) {
  $lines.Add("| " + $codeQuote + $target.target + $codeQuote + " | " + $codeQuote + $target.requirement + $codeQuote + " | " + $codeQuote + $target.coverageState + $codeQuote + " | $($target.expectedCombinationCount) | $($target.publishedCombinationCount) | $($target.passed) |")
}

$lines.Add("")
$lines.Add("## Target Details")
foreach ($target in $targetResults) {
  $lines.Add("")
  $lines.Add("### " + $target.target)
  $lines.Add("")
  $lines.Add("- requirement: " + $codeQuote + $target.requirement + $codeQuote)
  $lines.Add("- state: " + $codeQuote + $target.coverageState + $codeQuote)
  $lines.Add("- notes: $($target.notes)")
  $lines.Add("- expected combinations: " + (($target.expectedDependencyCombinations | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  $lines.Add("- published combinations: " + (($target.publishedDependencyCombinations | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  if (@($target.missingDependencyCombinations).Count -gt 0) {
    $lines.Add("- missing or not yet required combinations: " + (($target.missingDependencyCombinations | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  }
}

$lines.Add("")
$lines.Add("## Future Separate Package Lines")
foreach ($futureTarget in $futureTargets) {
  $lines.Add("- " + $codeQuote + $futureTarget.target + $codeQuote + ": " + $codeQuote + $futureTarget.status + $codeQuote + "; " + (($futureTarget.requiredEvidence | ForEach-Object { $_ }) -join "; "))
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Failed checks: $($failures.Count)")
$lines.Add("- Manifest future package keys: $($manifestFuturePackages.Count)")
$lines.Add("- Published unexpected future rows: $($publishedUnexpectedFutureRows.Count)")

if ($failures.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Failures")
  foreach ($failure in $failures) {
    $lines.Add("- $failure")
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime publication target coverage written to $jsonPath"
Write-Host "Runtime publication target coverage written to $markdownPath"

if ($failures.Count -gt 0) {
  $message = "Runtime publication target coverage has $($failures.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
