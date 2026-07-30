[CmdletBinding()]
param(
  [string[]]$PackagePath = @(),
  [switch]$StaticOnly,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$policyPath = Join-Path $RepositoryRoot "pack\external-vendor-runtime-policy.json"
if (-not (Test-Path -LiteralPath $policyPath -PathType Leaf)) {
  throw "External vendor runtime policy was not found: $policyPath"
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$failures = New-Object System.Collections.Generic.List[string]
$inspectedPackages = New-Object System.Collections.Generic.List[object]

function Add-Failure {
  param([string]$Message)
  $failures.Add($Message)
}

function Test-TextContains {
  param([string]$Path, [string]$Text)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    Add-Failure "Required policy surface is missing: $Path"
    return $false
  }

  return (Get-Content -LiteralPath $Path -Raw -Encoding utf8).IndexOf($Text, [StringComparison]::Ordinal) -ge 0
}

if ([int]$policy.schemaVersion -ne 1) {
  Add-Failure "Unsupported policy schemaVersion '$($policy.schemaVersion)'."
}
if (@($policy.allowedPackageKinds).Count -ne 2 -or @($policy.allowedPackageKinds) -notcontains "managed" -or @($policy.allowedPackageKinds) -notcontains "bridge") {
  Add-Failure "allowedPackageKinds must contain only managed and bridge."
}

$fullRuntimeProps = Join-Path $RepositoryRoot "pack\runtime\Directory.Build.props"
if (-not (Test-TextContains -Path $fullRuntimeProps -Text "<IsPackable>false</IsPackable>")) {
  Add-Failure "Full-runtime projects must be disabled with IsPackable=false."
}

$splitRuntimeProps = Join-Path $RepositoryRoot "pack\runtime-split\Directory.Build.props"
if (-not (Test-TextContains -Path $splitRuntimeProps -Text "<IsPackable>false</IsPackable>")) {
  Add-Failure "Split runtime projects must be non-packable by default."
}

$splitProjects = @(Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split") -Filter *.csproj -File -Recurse)
foreach ($project in $splitProjects) {
  $text = Get-Content -LiteralPath $project.FullName -Raw -Encoding utf8
  $isBridge = $project.BaseName.EndsWith(".Bridge", [StringComparison]::Ordinal)
  $optsIn = $text.IndexOf("<JYPPXPackageKind>bridge</JYPPXPackageKind>", [StringComparison]::Ordinal) -ge 0 -and
    $text.IndexOf("<IsPackable>true</IsPackable>", [StringComparison]::Ordinal) -ge 0

  if ($isBridge -and -not $optsIn) {
    Add-Failure "Bridge project must explicitly opt in with JYPPXPackageKind=bridge and IsPackable=true: $($project.FullName)"
  }
  if (-not $isBridge -and ($text.IndexOf("<IsPackable>true</IsPackable>", [StringComparison]::Ordinal) -ge 0 -or $text.IndexOf("<JYPPXPackageKind>bridge</JYPPXPackageKind>", [StringComparison]::Ordinal) -ge 0)) {
    Add-Failure "Retired vendor/meta project must not opt in to packing: $($project.FullName)"
  }
}

$splitScript = Join-Path $RepositoryRoot "eng\Invoke-LocalSplitRuntimePackage.ps1"
if (-not (Test-TextContains -Path $splitScript -Text '[string[]]$SplitPackageRole = @("bridge")')) {
  Add-Failure "Invoke-LocalSplitRuntimePackage.ps1 must default to the bridge role."
}
if (-not (Test-TextContains -Path $splitScript -Text "Only the 'bridge' split package role is allowed")) {
  Add-Failure "Invoke-LocalSplitRuntimePackage.ps1 must reject retired roles explicitly."
}

$fullRuntimeScript = Join-Path $RepositoryRoot "eng\Invoke-LocalRuntimePackage.ps1"
if (-not (Test-TextContains -Path $fullRuntimeScript -Text "Full-runtime packaging is retired")) {
  Add-Failure "Invoke-LocalRuntimePackage.ps1 must fail closed after full-runtime retirement."
}

foreach ($workflowName in @("package-managed.yml", "package-source.yml", "runtime-windows.yml", "runtime-linux.yml", "release-bundle.yml", "release-quality-gate.yml")) {
  $workflowPath = Join-Path $RepositoryRoot ".github\workflows\$workflowName"
  if (-not (Test-TextContains -Path $workflowPath -Text "Test-ExternalVendorRuntimePackagePolicy.ps1")) {
    Add-Failure "Workflow '$workflowName' must run the external vendor runtime policy gate."
  }
}

foreach ($workflowName in @("runtime-windows.yml", "runtime-linux.yml")) {
  $workflowPath = Join-Path $RepositoryRoot ".github\workflows\$workflowName"
  $workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding utf8
  if ($workflowText -match 'default:\s+all') {
    Add-Failure "Workflow '$workflowName' must not default to all split roles."
  }
  if ($workflowText -match "if:\s*\$\{\{\s*inputs\.runtime_delivery_mode\s*!=\s*'split'") {
    Add-Failure "Workflow '$workflowName' still has an activatable full-runtime job."
  }
  if ($workflowText.IndexOf("split_package_roles must be 'bridge'", [StringComparison]::Ordinal) -lt 0) {
    Add-Failure "Workflow '$workflowName' must fail closed when a non-bridge role is requested."
  }
}

function Expand-PackagePaths {
  param([string[]]$Values)

  $result = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    $candidate = if ([IO.Path]::IsPathRooted($value)) { $value } else { Join-Path $RepositoryRoot $value }
    if (Test-Path -LiteralPath $candidate -PathType Container) {
      Get-ChildItem -LiteralPath $candidate -Filter *.nupkg -File -Recurse | ForEach-Object { $result.Add($_.FullName) }
    }
    elseif (Test-Path -LiteralPath $candidate -PathType Leaf) {
      $result.Add((Resolve-Path -LiteralPath $candidate).Path)
    }
    else {
      Add-Failure "Package path does not exist: $candidate"
    }
  }

  return @($result | Sort-Object -Unique)
}

if (-not $StaticOnly.IsPresent) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  foreach ($packageFile in @(Expand-PackagePaths -Values $PackagePath)) {
    $archive = [IO.Compression.ZipFile]::OpenRead($packageFile)
    try {
      $nuspecEntry = @($archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) }) | Select-Object -First 1
      if ($null -eq $nuspecEntry) {
        Add-Failure "Package has no nuspec: $packageFile"
        continue
      }

      $reader = [IO.StreamReader]::new($nuspecEntry.Open())
      try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
      $packageId = [string]$nuspec.package.metadata.id
      $kind = if (@($policy.managedPackageIds) -contains $packageId) {
        "managed"
      }
      elseif ($packageId -match [string]$policy.bridgePackageIdPattern) {
        "bridge"
      }
      else {
        "forbidden"
      }

      if ($kind -eq "forbidden") {
        Add-Failure "Package id is not allowed by policy: $packageId ($packageFile)"
      }

      $nativeEntries = New-Object System.Collections.Generic.List[string]
      foreach ($entry in $archive.Entries) {
        $fileName = [IO.Path]::GetFileName($entry.FullName)
        if ([string]::IsNullOrWhiteSpace($fileName)) {
          continue
        }

        foreach ($pattern in @($policy.forbiddenNativeFileNamePatterns)) {
          if ($fileName -match [string]$pattern) {
            Add-Failure "Package '$packageId' contains forbidden NVIDIA runtime binary '$($entry.FullName)'."
          }
        }

        if ($entry.FullName -match '^runtimes/[^/]+/native/(.+)$') {
          $nativeEntries.Add($fileName)
        }
      }

      if ($kind -eq "managed" -and $nativeEntries.Count -gt 0) {
        Add-Failure "Managed package '$packageId' must not contain runtimes/*/native assets."
      }
      if ($kind -eq "bridge") {
        if ($nativeEntries.Count -ne 1 -or @($policy.allowedBridgeNativeFileNames) -notcontains $nativeEntries[0]) {
          Add-Failure "Bridge package '$packageId' must contain exactly one approved project-owned bridge binary. Found: $($nativeEntries -join ', ')"
        }
      }

      $inspectedPackages.Add([pscustomobject]@{
        path = $packageFile
        packageId = $packageId
        kind = $kind
        nativeEntries = @($nativeEntries.ToArray())
      })
    }
    finally {
      $archive.Dispose()
    }
  }
}

$result = [pscustomobject]@{
  policyId = [string]$policy.policyId
  policyPath = $policyPath
  staticOnly = $StaticOnly.IsPresent
  inspectedPackageCount = $inspectedPackages.Count
  inspectedPackages = @($inspectedPackages.ToArray())
  failureCount = $failures.Count
  failures = @($failures.ToArray())
  passed = $failures.Count -eq 0
}

$result | ConvertTo-Json -Depth 8
if ($failures.Count -gt 0) {
  throw "External vendor runtime package policy failed with $($failures.Count) finding(s): $($failures -join ' | ')"
}
