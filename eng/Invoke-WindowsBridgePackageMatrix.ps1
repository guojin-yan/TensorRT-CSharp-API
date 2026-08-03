[CmdletBinding()]
param(
  [string[]]$SourceRuntimeKey = @(),
  [string]$Version = "4.0.0",
  [string]$Configuration = "Release",
  [switch]$SkipPack,
  [switch]$SkipManagedPack,
  [switch]$SkipConsumerValidation,
  [switch]$RunDependencyProbe,
  [string]$ManagedPackageDirectory,
  [string]$ReportDirectory,
  [string]$ConsumerOutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
  $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
elseif (-not [IO.Path]::IsPathRooted($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ManagedPackageDirectory))
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\bridge-package-matrix"
}
elseif (-not [IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

if ([string]::IsNullOrWhiteSpace($ConsumerOutputRoot)) {
  $ConsumerOutputRoot = Join-Path $RepositoryRoot "build-out\bridge-package-matrix"
}
elseif (-not [IO.Path]::IsPathRooted($ConsumerOutputRoot)) {
  $ConsumerOutputRoot = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ConsumerOutputRoot))
}

$powerShellCommand = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$policyPath = Join-Path $RepositoryRoot "pack\external-vendor-runtime-policy.json"
$packageOutputRoot = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version

function Expand-KeyList {
  param([string[]]$Values)

  @(
    foreach ($value in @($Values)) {
      foreach ($part in @(([string]$value) -split "[,;]")) {
        $trimmed = $part.Trim()
        if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
          $trimmed
        }
      }
    }
  ) | Sort-Object -Unique
}

function Invoke-CheckedCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$ArgumentList
  )

  Write-Host "> $FilePath $($ArgumentList -join ' ')"
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
  }
}

function Get-PackageInventory {
  param(
    [Parameter(Mandatory = $true)]
    [object]$BridgePackage,
    [Parameter(Mandatory = $true)]
    [string]$PackagePath
  )

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead($PackagePath)
  try {
    $nativeEntries = @($archive.Entries | Where-Object { $_.FullName -match '^runtimes/[^/]+/native/[^/]+$' })
    $vendorEntries = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $archive.Entries) {
      $fileName = [IO.Path]::GetFileName($entry.FullName)
      if ([string]::IsNullOrWhiteSpace($fileName)) {
        continue
      }
      foreach ($pattern in @($policy.forbiddenNativeFileNamePatterns)) {
        if ($fileName -match [string]$pattern) {
          $vendorEntries.Add($entry.FullName)
          break
        }
      }
    }

    $expectedNativeEntry = "runtimes/$($BridgePackage.rid)/native/$([string]($BridgePackage.assets | Select-Object -First 1))"
    return [pscustomobject]@{
      packagePath = $PackagePath
      packageFileName = [IO.Path]::GetFileName($PackagePath)
      packageSha256 = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
      packageSizeBytes = (Get-Item -LiteralPath $PackagePath).Length
      nativeEntryCount = $nativeEntries.Count
      nativeEntries = @($nativeEntries.FullName)
      expectedNativeEntry = $expectedNativeEntry
      expectedNativeEntryPresent = @($nativeEntries.FullName) -contains $expectedNativeEntry
      vendorRuntimeEntryCount = $vendorEntries.Count
      vendorRuntimeEntries = @($vendorEntries.ToArray())
    }
  }
  finally {
    $archive.Dispose()
  }
}

$windowsBridgePackages = @(
  $splitManifest.packages |
    Where-Object { [string]$_.role -eq "bridge" -and [string]$_.platform -eq "windows" } |
    Sort-Object sourceRuntimeKey
)
if ($windowsBridgePackages.Count -eq 0) {
  throw "No Windows bridge packages were found in $splitManifestPath."
}

$requestedKeys = @(Expand-KeyList -Values $SourceRuntimeKey)
if ($requestedKeys.Count -eq 0) {
  $requestedKeys = @($windowsBridgePackages.sourceRuntimeKey)
}

$selectedPackages = New-Object System.Collections.Generic.List[object]
foreach ($key in $requestedKeys) {
  $matches = @($windowsBridgePackages | Where-Object { [string]$_.sourceRuntimeKey -eq $key })
  if ($matches.Count -ne 1) {
    throw "Runtime key '$key' must resolve to exactly one Windows bridge package; found $($matches.Count)."
  }
  $selectedPackages.Add($matches[0])
}

if (-not $SkipPack.IsPresent -and -not $SkipManagedPack.IsPresent) {
  Invoke-CheckedCommand -FilePath "dotnet" -ArgumentList @(
    "pack",
    (Join-Path $RepositoryRoot "pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj"),
    "-c",
    $Configuration,
    "-o",
    $ManagedPackageDirectory,
    "-p:JYPPXPackageVersion=$resolvedVersion"
  )
  Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-ManagedPackageContent.ps1")
  )
}

foreach ($bridgePackage in $selectedPackages) {
  $key = [string]$bridgePackage.sourceRuntimeKey
  if (-not $SkipPack.IsPresent) {
    Invoke-CheckedCommand -FilePath $powerShellCommand -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      (Join-Path $RepositoryRoot "eng\Invoke-LocalSplitRuntimePackage.ps1"),
      "-SourceRuntimeKey",
      $key,
      "-Version",
      $resolvedVersion,
      "-Configuration",
      $Configuration,
      "-SplitPackageRole",
      "bridge",
      "-SkipManagedPack",
      "-SkipConsumerValidation"
    )
  }
}

$packageFiles = New-Object System.Collections.Generic.List[string]
$expectedPackageIds = New-Object System.Collections.Generic.List[string]
$inventories = @{}
foreach ($bridgePackage in $selectedPackages) {
  $key = [string]$bridgePackage.sourceRuntimeKey
  $packageDirectory = Join-Path $packageOutputRoot $key
  $matches = @(Get-ChildItem -LiteralPath $packageDirectory -Filter *.nupkg -File -ErrorAction SilentlyContinue)
  if ($matches.Count -ne 1) {
    throw "Expected exactly one bridge package under '$packageDirectory', found $($matches.Count)."
  }

  $packageFiles.Add($matches[0].FullName)
  $expectedPackageIds.Add([string]$bridgePackage.packageId)
  $inventories[$key] = Get-PackageInventory -BridgePackage $bridgePackage -PackagePath $matches[0].FullName
}

& (Join-Path $RepositoryRoot "eng\Test-ExternalVendorRuntimePackagePolicy.ps1") `
  -PackagePath @($packageFiles.ToArray()) `
  -ExpectedPackageId @($expectedPackageIds.ToArray()) `
  -ExpectedPackageVersion $resolvedVersion `
  -RequireExactPackageSet `
  -RepositoryRoot $RepositoryRoot | Out-Host

$rows = New-Object System.Collections.Generic.List[object]
foreach ($bridgePackage in $selectedPackages) {
  $key = [string]$bridgePackage.sourceRuntimeKey
  $keyReportDirectory = Join-Path $ReportDirectory $key
  $consumerReport = $null
  if (-not $SkipConsumerValidation.IsPresent) {
    $consumerArguments = @{
      SourceRuntimeKey = $key
      ManagedPackageDirectory = $ManagedPackageDirectory
      BridgePackageDirectory = (Join-Path $packageOutputRoot $key)
      ReportDirectory = $keyReportDirectory
      OutputRoot = (Join-Path $ConsumerOutputRoot $key)
      RepositoryRoot = $RepositoryRoot
    }
    if (-not $RunDependencyProbe.IsPresent) {
      $consumerArguments.SkipProbe = $true
    }

    & (Join-Path $RepositoryRoot "eng\Test-BridgePackageConsumer.ps1") @consumerArguments
    $consumerReportPath = Join-Path $keyReportDirectory "bridge-package-consumer-validation-summary.json"
    $consumerReport = Get-Content -LiteralPath $consumerReportPath -Raw -Encoding utf8 | ConvertFrom-Json
  }

  $inventory = $inventories[$key]
  $consumerPassed = $SkipConsumerValidation.IsPresent -or [bool]$consumerReport.PackageConsumerValidationSucceeded
  $packagePassed = $inventory.nativeEntryCount -eq 1 -and
    $inventory.expectedNativeEntryPresent -and
    $inventory.vendorRuntimeEntryCount -eq 0
  $rows.Add([pscustomobject]@{
    sourceRuntimeKey = $key
    bridgePackageKey = [string]$bridgePackage.key
    bridgePackageId = [string]$bridgePackage.packageId
    packageVersion = $resolvedVersion
    packageFileName = $inventory.packageFileName
    packageSha256 = $inventory.packageSha256
    packageSizeBytes = $inventory.packageSizeBytes
    nativeEntryCount = $inventory.nativeEntryCount
    nativeEntries = @($inventory.nativeEntries)
    expectedNativeEntryPresent = $inventory.expectedNativeEntryPresent
    vendorRuntimeEntryCount = $inventory.vendorRuntimeEntryCount
    packagePolicyPassed = $packagePassed
    consumerValidationStatus = if ($SkipConsumerValidation.IsPresent) { "not-requested" } else { "passed" }
    consumerValidationPassed = $consumerPassed
    dependencyProbeStatus = if ($SkipConsumerValidation.IsPresent) { "not-requested" } else { [string]$consumerReport.ProbeResult }
    nativeDependencyStatus = if ($SkipConsumerValidation.IsPresent) { "not-requested" } else { [string]$consumerReport.NativeDependencyStatus }
    reportDirectory = $keyReportDirectory
  })
}

$gitCommit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()
$allPassed = @($rows | Where-Object { -not $_.packagePolicyPassed -or -not $_.consumerValidationPassed }).Count -eq 0
$summary = [ordered]@{
  schemaVersion = 1
  evidenceKind = "local-windows-bridge-package-matrix"
  generatedAtUtc = [DateTime]::UtcNow.ToString("o")
  repositoryCommit = $gitCommit
  packageVersion = $resolvedVersion
  runtimeKeyCount = $rows.Count
  packagePolicyPassed = $true
  vendorRuntimeBundled = $false
  consumerValidationRequested = -not $SkipConsumerValidation.IsPresent
  dependencyProbeRequested = $RunDependencyProbe.IsPresent -and -not $SkipConsumerValidation.IsPresent
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  canPublishPublicly = $false
  publicationExecuted = $false
  allPassed = $allPassed
  rows = @($rows.ToArray())
}

New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
$jsonPath = Join-Path $ReportDirectory "windows-bridge-package-matrix.json"
$markdownPath = Join-Path $ReportDirectory "windows-bridge-package-matrix.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Windows Bridge Package Matrix")
$lines.Add("")
$lines.Add("| Runtime key | Package | SHA256 | Native entries | Vendor runtime entries | Consumer | Dependency probe |")
$lines.Add("| --- | --- | --- | ---: | ---: | --- | --- |")
foreach ($row in $rows) {
  $lines.Add("| $($row.sourceRuntimeKey) | ``$($row.bridgePackageId) $($row.packageVersion)`` | ``$($row.packageSha256)`` | $($row.nativeEntryCount) | $($row.vendorRuntimeEntryCount) | $($row.consumerValidationStatus) | $($row.dependencyProbeStatus) |")
}
$lines.Add("")
$lines.Add("- all passed: $allPassed")
$lines.Add("- vendor runtime bundled: False")
$lines.Add("- runtime execution proof: False")
$lines.Add("- package-consumer runtime proof: False")
$lines.Add("- can publish publicly: False")
$lines.Add("- publication executed: False")
$lines.Add("")
$lines.Add("Generated by ``eng/Invoke-WindowsBridgePackageMatrix.ps1``. This is local candidate evidence only.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Windows bridge package matrix: $($rows.Count) row(s), allPassed=$allPassed"
Write-Host "JSON: $jsonPath"
Write-Host "Markdown: $markdownPath"
if (-not $allPassed) {
  throw "Windows bridge package matrix validation failed."
}
