[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$Ref = "TensorRtSharp4.0",
  [string]$SuggestedVersion,
  [string]$InventoryJsonPath,
  [string]$RuntimePublicationTargetCoverageJsonPath,
  [string]$ReleaseReadinessJsonPath,
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

if ([string]::IsNullOrWhiteSpace($Ref)) {
  $Ref = $env:GITHUB_REF_NAME
}

if ([string]::IsNullOrWhiteSpace($Ref)) {
  $Ref = "TensorRtSharp4.0"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Read-JsonFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [switch]$Optional
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    if ($Optional.IsPresent) {
      return $null
    }

    throw "Required JSON file was not found: $Path"
  }

  Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-JsonMapText {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.IDictionary]$Map
  )

  if ($Map.Count -eq 0) {
    return "{}"
  }

  $Map | ConvertTo-Json -Compress -Depth 8
}

function ConvertTo-ReleaseConfigJsonText {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.IDictionary]$Config
  )

  if ($Config.Count -eq 0) {
    return "{}"
  }

  $Config | ConvertTo-Json -Compress -Depth 12
}

function ConvertTo-QuotedCommandArgument {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  if ($Value -notmatch "[\s`"']") {
    return $Value
  }

  "'" + ($Value -replace "'", "''") + "'"
}

function Format-CommandLine {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Parts
  )

  ($Parts | ForEach-Object { ConvertTo-QuotedCommandArgument -Value $_ }) -join " "
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

function Get-LinuxCatalogTargetName {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Target
  )

  "$($Target.linuxDistro)$($Target.linuxDistroVersion)-$($Target.architecture)-$($Target.runnerMode)"
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

function Get-HighestPatchVersion {
  param(
    [object[]]$Rows
  )

  $patches = @(
    foreach ($row in @($Rows)) {
      $version = [string]$row.version
      if ($version -match '^4\.0\.(?<patch>[0-9]+)$') {
        [int]$Matches["patch"]
      }
    }
  )

  if ($patches.Count -eq 0) {
    return $null
  }

  ($patches | Measure-Object -Maximum).Maximum
}

function Get-RoleFromComponent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Component
  )

  switch -Regex ($Component) {
    '^CudaCudnn$' { return "cuda-cudnn" }
    '^TensorRt' { return "tensorrt" }
    '^Bridge$' { return "bridge" }
    '^Base$' { return "collection" }
    default { return "other" }
  }
}

function Get-ComponentPinsForRuntime {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package,
    [object[]]$ReleaseAssets
  )

  $basePackageId = [string]$Package.packageId
  $roleRows = @(
    foreach ($asset in @($ReleaseAssets)) {
      $packageId = [string]$asset.packageId
      if ([string]::IsNullOrWhiteSpace($packageId)) {
        continue
      }

      if ($packageId -ne $basePackageId -and -not $packageId.StartsWith("$basePackageId.", [System.StringComparison]::OrdinalIgnoreCase)) {
        continue
      }

      $identity = ConvertFrom-RuntimePackageId -PackageId $packageId
      if ($null -eq $identity) {
        continue
      }

      [pscustomobject]@{
        role = Get-RoleFromComponent -Component ([string]$identity.component)
        component = [string]$identity.component
        packageId = $packageId
        version = [string]$asset.version
        releaseTag = [string]$asset.releaseTag
        assetName = [string]$asset.assetName
      }
    }
  )

  [pscustomobject]@{
    runtimeKey = [string]$Package.key
    packageId = $basePackageId
    target = Get-PublicationTarget -Package $Package
    dependencyCombination = Get-DependencyCombination -Package $Package
    cudaCudnn = @($roleRows | Where-Object { $_.role -eq "cuda-cudnn" })
    tensorRt = @($roleRows | Where-Object { $_.role -eq "tensorrt" })
    bridge = @($roleRows | Where-Object { $_.role -eq "bridge" })
    collection = @($roleRows | Where-Object { $_.role -eq "collection" })
  }
}

function Add-ExactPinIfSingle {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.Specialized.OrderedDictionary]$Map,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey,
    [object[]]$Rows,
    [Parameter(Mandatory = $true)]
    [string]$PropertyName
  )

  $values = @(
    $Rows |
      ForEach-Object { [string]$_.$PropertyName } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
      Sort-Object -Unique
  )

  if ($values.Count -eq 1) {
    $Map[$RuntimeKey] = $values[0]
  }
}

function New-TargetCommandPlan {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string]$Description,
    [Parameter(Mandatory = $true)]
    [string[]]$CommandParts,
    [string]$WhenToUse,
    [string[]]$Notes = @()
  )

  [pscustomobject]@{
    name = $Name
    description = $Description
    whenToUse = $WhenToUse
    command = Format-CommandLine -Parts $CommandParts
    commandParts = @($CommandParts)
    notes = @($Notes)
  }
}

if ([string]::IsNullOrWhiteSpace($InventoryJsonPath)) {
  $InventoryJsonPath = Join-Path $RepositoryRoot "artifacts\publication-inventory\github-publication-inventory.json"
}

if ([string]::IsNullOrWhiteSpace($RuntimePublicationTargetCoverageJsonPath)) {
  $RuntimePublicationTargetCoverageJsonPath = Join-Path $RepositoryRoot "artifacts\runtime-publication-target-coverage\runtime-publication-target-coverage.json"
}

if ([string]::IsNullOrWhiteSpace($ReleaseReadinessJsonPath)) {
  $ReleaseReadinessJsonPath = Join-Path $RepositoryRoot "artifacts\release-readiness\release-readiness.json"
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$linuxTargetCatalogPath = Join-Path $RepositoryRoot "pack\runtime\linux-runtime-targets.manifest.json"
$manifest = Read-JsonFile -Path $manifestPath
$linuxTargetCatalog = Read-JsonFile -Path $linuxTargetCatalogPath
$inventory = Read-JsonFile -Path $InventoryJsonPath -Optional
$targetCoverage = Read-JsonFile -Path $RuntimePublicationTargetCoverageJsonPath -Optional
$releaseReadiness = Read-JsonFile -Path $ReleaseReadinessJsonPath -Optional

$packages = @($manifest.packages)
$releaseAssets = if ($null -ne $inventory) { @($inventory.releaseAssets) } else { @() }
$packageVersionRows = if ($null -ne $inventory) { @($inventory.packageVersions) } else { @() }
$runtimeMatrixRows = if ($null -ne $inventory) { @($inventory.runtimeMatrix) } else { @() }

if ([string]::IsNullOrWhiteSpace($SuggestedVersion)) {
  $highestPatch = Get-HighestPatchVersion -Rows @($releaseAssets + $packageVersionRows)
  $SuggestedVersion = if ($null -eq $highestPatch) { "4.0.<next>" } else { "4.0.$([int]$highestPatch + 1)" }
}

$runtimePins = @(
  foreach ($package in $packages) {
    Get-ComponentPinsForRuntime -Package $package -ReleaseAssets $releaseAssets
  }
)

$windowsCudaCudnnVersionMap = [ordered]@{}
$windowsTensorRtVersionMap = [ordered]@{}
$windowsCudaCudnnReleaseTagMap = [ordered]@{}
$windowsTensorRtReleaseTagMap = [ordered]@{}
$linuxCudaCudnnVersionMap = [ordered]@{}
$linuxTensorRtVersionMap = [ordered]@{}
$linuxCudaCudnnReleaseTagMap = [ordered]@{}
$linuxTensorRtReleaseTagMap = [ordered]@{}

foreach ($pin in $runtimePins) {
  $package = $packages | Where-Object { [string]$_.key -eq [string]$pin.runtimeKey } | Select-Object -First 1
  if ($null -eq $package) {
    continue
  }

  if ([string]$package.platform -eq "windows") {
    Add-ExactPinIfSingle -Map $windowsCudaCudnnVersionMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.cudaCudnn) -PropertyName "version"
    Add-ExactPinIfSingle -Map $windowsTensorRtVersionMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.tensorRt) -PropertyName "version"
    Add-ExactPinIfSingle -Map $windowsCudaCudnnReleaseTagMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.cudaCudnn) -PropertyName "releaseTag"
    Add-ExactPinIfSingle -Map $windowsTensorRtReleaseTagMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.tensorRt) -PropertyName "releaseTag"
  }
  elseif ([string]$package.platform -eq "linux") {
    Add-ExactPinIfSingle -Map $linuxCudaCudnnVersionMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.cudaCudnn) -PropertyName "version"
    Add-ExactPinIfSingle -Map $linuxTensorRtVersionMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.tensorRt) -PropertyName "version"
    Add-ExactPinIfSingle -Map $linuxCudaCudnnReleaseTagMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.cudaCudnn) -PropertyName "releaseTag"
    Add-ExactPinIfSingle -Map $linuxTensorRtReleaseTagMap -RuntimeKey ([string]$pin.runtimeKey) -Rows @($pin.tensorRt) -PropertyName "releaseTag"
  }
}

$targetCoverageRows = if ($null -ne $targetCoverage) { @($targetCoverage.targets) } else { @() }
$publishedTargets = @(
  foreach ($target in $targetCoverageRows) {
    [pscustomobject]@{
      target = [string]$target.target
      requirement = [string]$target.requirement
      coverageState = [string]$target.coverageState
      passed = [bool]$target.passed
      expectedCombinationCount = [int]$target.expectedCombinationCount
      publishedCombinationCount = [int]$target.publishedCombinationCount
      expectedDependencyCombinations = @($target.expectedDependencyCombinations | ForEach-Object { [string]$_ })
      publishedDependencyCombinations = @($target.publishedDependencyCombinations | ForEach-Object { [string]$_ })
      missingDependencyCombinations = @($target.missingDependencyCombinations | ForEach-Object { [string]$_ })
      notes = [string]$target.notes
    }
  }
)

$linuxModeledTargets = @(
  foreach ($target in @($linuxTargetCatalog.targets)) {
    $expectedCombinations = @($target.expectedCombinations | ForEach-Object { [string]$_ })
    if ($expectedCombinations -contains "all") {
      $expectedCombinations = @($linuxTargetCatalog.dependencyCombinations | ForEach-Object { [string]$_ })
    }

    $publicationTarget = "linux-$($target.architecture).$($target.linuxDistro)$($target.linuxDistroVersion)"
    $coverageRow = $publishedTargets | Where-Object { [string]$_.target -eq $publicationTarget } | Select-Object -First 1
    [pscustomobject]@{
      catalogTarget = Get-LinuxCatalogTargetName -Target $target
      publicationTarget = $publicationTarget
      keySetAliases = @($target.keySetAliases | ForEach-Object { [string]$_ })
      runnerMode = [string]$target.runnerMode
      runnerLabels = @(
        $packages |
          Where-Object {
            [string]$_.platform -eq "linux" -and
            [string]$_.architecture -eq [string]$target.architecture -and
            [string]$_.linuxDistro -eq [string]$target.linuxDistro -and
            [string]$_.linuxDistroVersion -eq [string]$target.linuxDistroVersion
          } |
          Select-Object -First 1 |
          ForEach-Object { @($_.runnerLabels | ForEach-Object { [string]$_ }) }
      )
      publicationRequirement = [string]$target.publicationRequirement
      status = [string]$target.status
      expectedCombinations = @($expectedCombinations)
      coverageState = if ($coverageRow) { [string]$coverageRow.coverageState } else { "not-audited" }
      publishedCombinationCount = if ($coverageRow) { [int]$coverageRow.publishedCombinationCount } else { 0 }
      missingDependencyCombinations = if ($coverageRow) { @($coverageRow.missingDependencyCombinations) } else { @($expectedCombinations) }
      notes = [string]$target.notes
    }
  }
)

$futureTargets = @(
  foreach ($futureTarget in @($linuxTargetCatalog.futureTargets)) {
    [pscustomobject]@{
      target = [string]$futureTarget.target
      status = [string]$futureTarget.status
      runnerMode = [string]$futureTarget.runnerMode
      architecture = [string]$futureTarget.architecture
      packageIdentityRule = [string]$futureTarget.packageIdentityRule
      requiredEvidenceItems = @($futureTarget.requiredEvidenceItems | ForEach-Object { [string]$_ })
    }
  }
)

$readinessRows = if ($null -ne $releaseReadiness) { @($releaseReadiness.readiness) } else { @() }
$notReady = @($readinessRows | Where-Object { -not [bool]$_.ready })
$runnerRows = if ($null -ne $releaseReadiness) { @($releaseReadiness.runners) } else { @() }

$dispatchableTargets = New-Object System.Collections.Generic.List[object]
$blockedTargets = New-Object System.Collections.Generic.List[object]
foreach ($target in $publishedTargets) {
  if ([string]$target.requirement -eq "published-required" -and [string]$target.coverageState -ne "complete") {
    $dispatchableTargets.Add([pscustomobject]@{
        target = [string]$target.target
        reason = "Published-required target is not complete."
        missingDependencyCombinations = @($target.missingDependencyCombinations)
      }) | Out-Null
  }
  elseif ([string]$target.requirement -eq "infrastructure-blocked") {
    $blockedTargets.Add([pscustomobject]@{
        target = [string]$target.target
        reason = [string]$target.notes
        missingDependencyCombinations = @($target.missingDependencyCombinations)
      }) | Out-Null
  }
}

$runtimeReleaseTags = if ($null -ne $inventory) { @($inventory.expectedReleaseTags | Where-Object { [string]$_ -match '^v4\.0\.' }) } else { @() }
$remoteReleaseTags = if ($null -ne $inventory) { @($inventory.remoteReleaseTags | ForEach-Object { [string]$_ }) } else { @() }
$managedPackageRows = @($packageVersionRows | Where-Object { [string]$_.packageId -eq "JYPPX.TensorRT.CSharp.API" } | Sort-Object version)
$latestManagedPackage = $managedPackageRows | Select-Object -Last 1

$windowsReleaseConfig = [ordered]@{
  windows_cuda_cudnn_package_version_map = ConvertTo-JsonMapText -Map $windowsCudaCudnnVersionMap
  windows_tensorrt_package_version_map = ConvertTo-JsonMapText -Map $windowsTensorRtVersionMap
  windows_cuda_cudnn_package_release_tag_map = ConvertTo-JsonMapText -Map $windowsCudaCudnnReleaseTagMap
  windows_tensorrt_package_release_tag_map = ConvertTo-JsonMapText -Map $windowsTensorRtReleaseTagMap
  windows_include_meta_package = $true
}

$linuxReleaseConfig = [ordered]@{
  linux_cuda_cudnn_package_version_map = ConvertTo-JsonMapText -Map $linuxCudaCudnnVersionMap
  linux_tensorrt_package_version_map = ConvertTo-JsonMapText -Map $linuxTensorRtVersionMap
  linux_cuda_cudnn_package_release_tag_map = ConvertTo-JsonMapText -Map $linuxCudaCudnnReleaseTagMap
  linux_tensorrt_package_release_tag_map = ConvertTo-JsonMapText -Map $linuxTensorRtReleaseTagMap
  linux_include_meta_package = $true
}

$routineReleaseConfig = [ordered]@{}
foreach ($entry in $windowsReleaseConfig.GetEnumerator()) { $routineReleaseConfig[$entry.Key] = $entry.Value }
foreach ($entry in $linuxReleaseConfig.GetEnumerator()) { $routineReleaseConfig[$entry.Key] = $entry.Value }
$routineReleaseConfigJson = ConvertTo-ReleaseConfigJsonText -Config $routineReleaseConfig

$linuxUbuntu20ReleaseConfig = [ordered]@{}
foreach ($entry in $linuxReleaseConfig.GetEnumerator()) { $linuxUbuntu20ReleaseConfig[$entry.Key] = $entry.Value }
$linuxUbuntu20ReleaseConfig["linux_ubuntu20_runtime_key_set"] = "hosted-container-ubuntu20"
$linuxUbuntu20ReleaseConfigJson = ConvertTo-ReleaseConfigJsonText -Config $linuxUbuntu20ReleaseConfig

$commands = New-Object System.Collections.Generic.List[object]
$commands.Add((New-TargetCommandPlan `
      -Name "managed-package-and-docs" `
      -Description "Publish the managed C# package and API docs without republishing runtime dependency packages." `
      -WhenToUse "Use when only C# API, docs, or managed packaging changed." `
      -CommandParts @(
        "pwsh", "-NoProfile", "-File", ".\eng\Invoke-RemoteReleaseBundle.ps1",
        "-Repository", $Repository,
        "-Ref", $Ref,
        "-Version", $SuggestedVersion,
        "-PublishManagedToGitHubPackages", "true",
        "-PublishManagedToNuGet", "true",
        "-RunDocsRelease", "true",
        "-PublishRuntimeToGitHubPackages", "false"
      ) `
      -Notes @(
        "Requires repository secret NUGET_API_KEY before nuget.org publication can succeed.",
        "Runtime CudaCudnn and TensorRt packages are intentionally not republished."
      ))) | Out-Null

$commands.Add((New-TargetCommandPlan `
      -Name "routine-bridge-and-collection-refresh" `
      -Description "Publish managed package plus Windows and hosted Linux Bridge/collection packages while pinning already-published CUDA/cuDNN and TensorRT packages." `
      -WhenToUse "Use when the local C ABI bridge or wrapper code changed but NVIDIA dependency sets did not change." `
      -CommandParts @(
        "pwsh", "-NoProfile", "-File", ".\eng\Invoke-RemoteReleaseBundle.ps1",
        "-Repository", $Repository,
        "-Ref", $Ref,
        "-Version", $SuggestedVersion,
        "-RuntimeVersion", $SuggestedVersion,
        "-RunWindowsRuntimePackaging",
        "-WindowsSplitPackageRoles", "bridge,collection",
        "-RunLinuxRuntimePackaging",
        "-LinuxRuntimeKeySet", "hosted-all",
        "-LinuxSplitPackageRoles", "bridge,collection",
        "-PublishManagedToGitHubPackages", "true",
        "-PublishManagedToNuGet", "false",
        "-PublishRuntimeToGitHubPackages", "true",
        "-AttachRuntimeToGitHubRelease", "true",
        "-RunLinuxSmoke", "false",
        "-WindowsCudaCudnnPackageVersionMap", (ConvertTo-JsonMapText -Map $windowsCudaCudnnVersionMap),
        "-WindowsTensorRtPackageVersionMap", (ConvertTo-JsonMapText -Map $windowsTensorRtVersionMap),
        "-WindowsCudaCudnnPackageReleaseTagMap", (ConvertTo-JsonMapText -Map $windowsCudaCudnnReleaseTagMap),
        "-WindowsTensorRtPackageReleaseTagMap", (ConvertTo-JsonMapText -Map $windowsTensorRtReleaseTagMap),
        "-LinuxCudaCudnnPackageVersionMap", (ConvertTo-JsonMapText -Map $linuxCudaCudnnVersionMap),
        "-LinuxTensorRtPackageVersionMap", (ConvertTo-JsonMapText -Map $linuxTensorRtVersionMap),
        "-LinuxCudaCudnnPackageReleaseTagMap", (ConvertTo-JsonMapText -Map $linuxCudaCudnnReleaseTagMap),
        "-LinuxTensorRtPackageReleaseTagMap", (ConvertTo-JsonMapText -Map $linuxTensorRtReleaseTagMap),
        "-LinuxIncludeMetaPackage",
        "-WindowsIncludeMetaPackage"
      ) `
      -Notes @(
        "This avoids republishing stable CudaCudnn and TensorRt packages.",
        "Hosted Linux currently means Ubuntu 22.04 x64 plus the modeled Ubuntu 24.04 x64 package line.",
        "The same settings can also be passed as release_config_json: $routineReleaseConfigJson"
      ))) | Out-Null

$commands.Add((New-TargetCommandPlan `
      -Name "ubuntu20-hosted-container" `
      -Description "Publish Ubuntu 20.04 x64 packages through the hosted Ubuntu 20.04 container lane." `
      -WhenToUse "Use for a new Ubuntu 20.04 dependency refresh or bridge/collection refresh after the initial hosted-container package line has already been published." `
      -CommandParts @(
        "pwsh", "-NoProfile", "-File", ".\eng\Invoke-RemoteReleaseBundle.ps1",
        "-Repository", $Repository,
        "-Ref", $Ref,
        "-Version", $SuggestedVersion,
        "-RuntimeVersion", $SuggestedVersion,
        "-RunLinuxUbuntu20RuntimePackaging",
        "-LinuxRunnerMode", "hosted-container",
        "-LinuxUbuntu20RuntimeKeySet", "hosted-container-ubuntu20",
        "-LinuxSplitPackageRoles", "all",
        "-PublishManagedToGitHubPackages", "false",
        "-PublishRuntimeToGitHubPackages", "true",
        "-AttachRuntimeToGitHubRelease", "true",
        "-RunLinuxSmoke", "false"
      ) `
      -Notes @(
        "This lane runs on GitHub-hosted infrastructure with an Ubuntu 20.04 job container, not a repository self-hosted runner.",
        "Bridge-only refresh for this lane can use release_config_json: $linuxUbuntu20ReleaseConfigJson"
      ))) | Out-Null

$pinMapSummary = [pscustomobject]@{
  windows = [pscustomobject]@{
    cudaCudnnPackageVersionMap = $windowsCudaCudnnVersionMap
    tensorRtPackageVersionMap = $windowsTensorRtVersionMap
    cudaCudnnPackageReleaseTagMap = $windowsCudaCudnnReleaseTagMap
    tensorRtPackageReleaseTagMap = $windowsTensorRtReleaseTagMap
  }
  linux = [pscustomobject]@{
    cudaCudnnPackageVersionMap = $linuxCudaCudnnVersionMap
    tensorRtPackageVersionMap = $linuxTensorRtVersionMap
    cudaCudnnPackageReleaseTagMap = $linuxCudaCudnnReleaseTagMap
    tensorRtPackageReleaseTagMap = $linuxTensorRtReleaseTagMap
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-release-plan"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "runtime-release-plan.json"
$markdownPath = Join-Path $outputRoot "runtime-release-plan.md"

$plan = [pscustomobject]@{
  repository = $Repository
  ref = $Ref
  generatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  suggestedVersion = $SuggestedVersion
  manifestPath = $manifestPath
  linuxTargetCatalogPath = $linuxTargetCatalogPath
  inventoryJsonPath = $InventoryJsonPath
  runtimePublicationTargetCoverageJsonPath = $RuntimePublicationTargetCoverageJsonPath
  releaseReadinessJsonPath = $ReleaseReadinessJsonPath
  inputsAvailable = [pscustomobject]@{
    inventory = $null -ne $inventory
    runtimePublicationTargetCoverage = $null -ne $targetCoverage
    releaseReadiness = $null -ne $releaseReadiness
  }
  managedPackage = [pscustomobject]@{
    latestGitHubPackagesVersion = if ($latestManagedPackage) { [string]$latestManagedPackage.version } else { "" }
    knownVersions = @($managedPackageRows | ForEach-Object { [string]$_.version })
  }
  releaseInventory = [pscustomobject]@{
    expectedReleaseTags = @($runtimeReleaseTags)
    remoteReleaseTags = @($remoteReleaseTags)
    unexpectedReleaseTags = if ($null -ne $inventory) { @($inventory.unexpectedReleaseTags) } else { @() }
    missingReleaseTags = if ($null -ne $inventory) { @($inventory.missingReleaseTags) } else { @() }
    expectedPackageVersionCount = if ($null -ne $inventory) { [int]$inventory.expectedPackageVersionCount } else { 0 }
    actualPackageVersionCount = if ($null -ne $inventory) { [int]$inventory.actualPackageVersionCount } else { 0 }
  }
  publishedTargets = @($publishedTargets)
  linuxModeledTargets = @($linuxModeledTargets)
  dispatchableNextTargets = @($dispatchableTargets.ToArray())
  blockedTargets = @($blockedTargets.ToArray())
  futureTargets = @($futureTargets)
  readiness = [pscustomobject]@{
    readyCount = if ($null -ne $releaseReadiness) { [int]$releaseReadiness.readyCount } else { 0 }
    notReadyCount = if ($null -ne $releaseReadiness) { [int]$releaseReadiness.notReadyCount } else { 0 }
    notReady = @($notReady)
    runners = @($runnerRows)
  }
  stableDependencyPinMaps = $pinMapSummary
  commands = @($commands.ToArray())
} 

$plan | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$codeQuote = [string][char]96
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Runtime Release Plan")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("Ref: " + $codeQuote + $Ref + $codeQuote)
$lines.Add("Suggested next version: " + $codeQuote + $SuggestedVersion + $codeQuote)
$lines.Add("")
$lines.Add("## Audit Inputs")
$lines.Add("")
$lines.Add("| Input | Available | Path |")
$lines.Add("| --- | --- | --- |")
$lines.Add("| Publication inventory | $($null -ne $inventory) | " + $codeQuote + $InventoryJsonPath + $codeQuote + " |")
$lines.Add("| Runtime target coverage | $($null -ne $targetCoverage) | " + $codeQuote + $RuntimePublicationTargetCoverageJsonPath + $codeQuote + " |")
$lines.Add("| Release readiness | $($null -ne $releaseReadiness) | " + $codeQuote + $ReleaseReadinessJsonPath + $codeQuote + " |")

$lines.Add("")
$lines.Add("## Published Targets")
$lines.Add("")
$lines.Add("| Target | Requirement | State | Expected combos | Published combos | Missing combos |")
$lines.Add("| --- | --- | --- | ---: | ---: | --- |")
foreach ($target in $publishedTargets) {
  $missing = if (@($target.missingDependencyCombinations).Count -gt 0) { (@($target.missingDependencyCombinations) | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", " } else { "" }
  $lines.Add("| " + $codeQuote + $target.target + $codeQuote + " | " + $codeQuote + $target.requirement + $codeQuote + " | " + $codeQuote + $target.coverageState + $codeQuote + " | $($target.expectedCombinationCount) | $($target.publishedCombinationCount) | $missing |")
}

$lines.Add("")
$lines.Add("## Linux Target Lines")
$lines.Add("")
$lines.Add("| Catalog target | Publication target | Runner mode | Requirement | State | Expected combos | Published combos |")
$lines.Add("| --- | --- | --- | --- | --- | ---: | ---: |")
foreach ($target in $linuxModeledTargets) {
  $lines.Add("| " + $codeQuote + $target.catalogTarget + $codeQuote + " | " + $codeQuote + $target.publicationTarget + $codeQuote + " | " + $codeQuote + $target.runnerMode + $codeQuote + " | " + $codeQuote + $target.publicationRequirement + $codeQuote + " | " + $codeQuote + $target.coverageState + $codeQuote + " | $(@($target.expectedCombinations).Count) | $($target.publishedCombinationCount) |")
}

$lines.Add("")
$lines.Add("## Stable Dependency Pin Maps")
$lines.Add("")
$lines.Add("Use these maps when publishing only " + $codeQuote + "bridge,collection" + $codeQuote + " so CUDA/cuDNN/TensorRT packages are not republished.")
$lines.Add("")
$lines.Add("| Platform | CUDA/cuDNN versions | TensorRT versions | CUDA/cuDNN release tags | TensorRT release tags |")
$lines.Add("| --- | ---: | ---: | ---: | ---: |")
$lines.Add("| Windows | $($windowsCudaCudnnVersionMap.Count) | $($windowsTensorRtVersionMap.Count) | $($windowsCudaCudnnReleaseTagMap.Count) | $($windowsTensorRtReleaseTagMap.Count) |")
$lines.Add("| Linux | $($linuxCudaCudnnVersionMap.Count) | $($linuxTensorRtVersionMap.Count) | $($linuxCudaCudnnReleaseTagMap.Count) | $($linuxTensorRtReleaseTagMap.Count) |")
$lines.Add("")
$lines.Add("Windows CUDA/cuDNN version map:")
$lines.Add("")
$lines.Add('```json')
$lines.Add((ConvertTo-JsonMapText -Map $windowsCudaCudnnVersionMap))
$lines.Add('```')
$lines.Add("")
$lines.Add("Windows TensorRT version map:")
$lines.Add("")
$lines.Add('```json')
$lines.Add((ConvertTo-JsonMapText -Map $windowsTensorRtVersionMap))
$lines.Add('```')
$lines.Add("")
$lines.Add("Linux CUDA/cuDNN version map:")
$lines.Add("")
$lines.Add('```json')
$lines.Add((ConvertTo-JsonMapText -Map $linuxCudaCudnnVersionMap))
$lines.Add('```')
$lines.Add("")
$lines.Add("Linux TensorRT version map:")
$lines.Add("")
$lines.Add('```json')
$lines.Add((ConvertTo-JsonMapText -Map $linuxTensorRtVersionMap))
$lines.Add('```')

$lines.Add("")
$lines.Add("## Suggested Commands")
foreach ($command in @($commands.ToArray())) {
  $lines.Add("")
  $lines.Add("### " + [string]$command.name)
  $lines.Add("")
  $lines.Add([string]$command.description)
  $lines.Add("")
  $lines.Add("When to use: " + [string]$command.whenToUse)
  $lines.Add("")
  $lines.Add('```powershell')
  $lines.Add([string]$command.command)
  $lines.Add('```')
  foreach ($note in @($command.notes)) {
    $lines.Add("- $([string]$note)")
  }
}

$lines.Add("")
$lines.Add("## Blocked And Future Targets")
if ($blockedTargets.Count -eq 0 -and $futureTargets.Count -eq 0 -and $notReady.Count -eq 0) {
  $lines.Add("")
  $lines.Add("No blocked or future target evidence was reported.")
}
else {
  foreach ($target in @($blockedTargets.ToArray())) {
    $lines.Add("- " + $codeQuote + [string]$target.target + $codeQuote + ": " + [string]$target.reason)
  }

  foreach ($item in @($notReady)) {
    $lines.Add("- " + $codeQuote + [string]$item.area + $codeQuote + ": " + [string]$item.detail)
  }

  foreach ($futureTarget in @($futureTargets)) {
    $lines.Add("- " + $codeQuote + [string]$futureTarget.target + $codeQuote + ": " + [string]$futureTarget.packageIdentityRule)
  }
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Dispatchable missing published-required targets: $($dispatchableTargets.Count)")
$lines.Add("- Infrastructure-blocked targets: $($blockedTargets.Count)")
$lines.Add("- Future separate package lines: $($futureTargets.Count)")
$lines.Add("- Readiness not-ready checks: $($notReady.Count)")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime release plan written to $jsonPath"
Write-Host "Runtime release plan written to $markdownPath"
