[CmdletBinding()]
param(
  [string[]]$ArtifactPath = @(),
  [switch]$StaticOnly,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$policyPath = Join-Path $RepositoryRoot "pack\publication-license-policy.json"
if (-not (Test-Path -LiteralPath $policyPath -PathType Leaf)) {
  throw "Publication license policy was not found: $policyPath"
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$failures = New-Object System.Collections.Generic.List[string]
$inspectedArtifacts = New-Object System.Collections.Generic.List[object]

function Add-Failure {
  param([string]$Message)
  $failures.Add($Message)
}

function Test-TextContains {
  param([string]$Path, [string]$Text)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    Add-Failure "Required publication surface is missing: $Path"
    return $false
  }

  return (Get-Content -LiteralPath $Path -Raw -Encoding utf8).IndexOf($Text, [StringComparison]::Ordinal) -ge 0
}

function Test-GatePrecedesFirstPublicationCommand {
  param([string]$Path, [string]$GateName)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return
  }

  $content = Get-Content -LiteralPath $Path -Raw -Encoding utf8
  $gateIndex = $content.IndexOf($GateName, [StringComparison]::Ordinal)
  $publicationIndexes = @(
    $content.IndexOf("gh release create", [StringComparison]::OrdinalIgnoreCase),
    $content.IndexOf("gh release upload", [StringComparison]::OrdinalIgnoreCase)
  ) | Where-Object { $_ -ge 0 }
  if ($publicationIndexes.Count -gt 0 -and ($gateIndex -lt 0 -or $gateIndex -gt ($publicationIndexes | Measure-Object -Minimum).Minimum)) {
    Add-Failure "Publication license gate must precede the first direct Release command in '$Path'."
  }
}

function Test-LicenseValueAllowed {
  param([string]$Value)

  $trimmed = if ($null -eq $Value) { "" } else { $Value.Trim() }
  foreach ($disallowed in @($policy.disallowedLicenseValues)) {
    if ([string]::Equals($trimmed, [string]$disallowed, [StringComparison]::OrdinalIgnoreCase)) {
      return $false
    }
  }

  return -not [string]::IsNullOrWhiteSpace($trimmed)
}

if ([int]$policy.schemaVersion -ne 1) {
  Add-Failure "Unsupported publication license policy schemaVersion '$($policy.schemaVersion)'."
}
if (-not [bool]$policy.publicationRequiresDeclaredLicense) {
  Add-Failure "publicationRequiresDeclaredLicense must remain true."
}
if (-not [bool]$policy.dryRunMayProceedWithoutDeclaredLicense) {
  Add-Failure "dryRunMayProceedWithoutDeclaredLicense must remain true so license selection stays an Owner publication decision."
}

$acceptedLicenseTypes = @($policy.acceptedPackageLicenseTypes | ForEach-Object { ([string]$_).ToLowerInvariant() })
if ($acceptedLicenseTypes.Count -ne 2 -or $acceptedLicenseTypes -notcontains "expression" -or $acceptedLicenseTypes -notcontains "file") {
  Add-Failure "acceptedPackageLicenseTypes must contain only expression and file."
}
if (@($policy.sourceArchiveLicenseFileNames).Count -eq 0) {
  Add-Failure "sourceArchiveLicenseFileNames must not be empty."
}

$ownerDecisionState = ([string]$policy.ownerDecisionState).Trim().ToLowerInvariant()
$selectedPackageLicenseType = ([string]$policy.selectedPackageLicense.type).Trim().ToLowerInvariant()
$selectedPackageLicenseValue = ([string]$policy.selectedPackageLicense.value).Trim()
$selectedSourceLicenseFileName = ([string]$policy.selectedSourceArchiveLicenseFileName).Trim()
if ($ownerDecisionState -notin @("required", "approved")) {
  Add-Failure "ownerDecisionState must be required or approved."
}
if ($ownerDecisionState -eq "approved") {
  if ($acceptedLicenseTypes -notcontains $selectedPackageLicenseType) {
    Add-Failure "An approved Owner decision must select package license type expression or file."
  }
  if (-not (Test-LicenseValueAllowed -Value $selectedPackageLicenseValue)) {
    Add-Failure "An approved Owner decision must select a non-placeholder package license value."
  }
  if ([string]::IsNullOrWhiteSpace($selectedSourceLicenseFileName) -or
      @($policy.sourceArchiveLicenseFileNames) -notcontains $selectedSourceLicenseFileName -or
      [IO.Path]::GetFileName($selectedSourceLicenseFileName) -ne $selectedSourceLicenseFileName) {
    Add-Failure "An approved Owner decision must select one allowed root source archive license file name."
  }
}

$gateName = "Test-PublicationLicenseReadiness.ps1"
$pushScriptPath = Join-Path $RepositoryRoot "eng\Push-NuGetPackages.ps1"
if (-not (Test-TextContains -Path $pushScriptPath -Text $gateName)) {
  Add-Failure "Push-NuGetPackages.ps1 must run the publication license gate before any package push."
}

foreach ($workflowName in @("package-managed.yml", "package-source.yml", "runtime-windows.yml", "runtime-linux.yml", "release-bundle.yml")) {
  $workflowPath = Join-Path $RepositoryRoot ".github\workflows\$workflowName"
  if (-not (Test-TextContains -Path $workflowPath -Text $gateName)) {
    Add-Failure "Workflow '$workflowName' must run the publication license gate before a direct Release upload."
  }
  Test-GatePrecedesFirstPublicationCommand -Path $workflowPath -GateName $gateName
}

function Expand-ArtifactPaths {
  param([string[]]$Values)

  $result = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    $candidate = if ([IO.Path]::IsPathRooted($value)) { $value } else { Join-Path $RepositoryRoot $value }
    if (Test-Path -LiteralPath $candidate -PathType Container) {
      Get-ChildItem -LiteralPath $candidate -File -Recurse | Where-Object {
        $_.Extension.Equals(".nupkg", [StringComparison]::OrdinalIgnoreCase) -or
        $_.Name -like "TensorRtSharp4.0-source-*.zip"
      } | ForEach-Object { $result.Add($_.FullName) }
    }
    elseif (Test-Path -LiteralPath $candidate -PathType Leaf) {
      $file = Get-Item -LiteralPath $candidate
      if ($file.Extension.Equals(".nupkg", [StringComparison]::OrdinalIgnoreCase) -or
          $file.Name -like "TensorRtSharp4.0-source-*.zip") {
        $result.Add($file.FullName)
      }
      else {
        Add-Failure "Unsupported publication artifact: $($file.FullName)"
      }
    }
    else {
      Add-Failure "Publication artifact path does not exist: $candidate"
    }
  }

  return @($result | Sort-Object -Unique)
}

if (-not $StaticOnly.IsPresent) {
  if ($ownerDecisionState -ne "approved") {
    Add-Failure "Owner license decision remains required; publication artifacts cannot be approved yet."
  }

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $resolvedArtifacts = @(Expand-ArtifactPaths -Values $ArtifactPath)
  if ($resolvedArtifacts.Count -eq 0) {
    Add-Failure "At least one nupkg or tracked source archive is required for publication license validation."
  }

  foreach ($artifact in $resolvedArtifacts) {
    $file = Get-Item -LiteralPath $artifact
    $archive = [IO.Compression.ZipFile]::OpenRead($file.FullName)
    try {
      if ($file.Extension.Equals(".nupkg", [StringComparison]::OrdinalIgnoreCase)) {
        $nuspecEntries = @($archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) })
        if ($nuspecEntries.Count -ne 1) {
          Add-Failure "Package must contain exactly one nuspec: $($file.FullName)"
          continue
        }

        $reader = [IO.StreamReader]::new($nuspecEntries[0].Open(), [Text.Encoding]::UTF8)
        try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $metadata = $nuspec.SelectSingleNode("//*[local-name()='metadata']")
        $licenseNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='license']")
        $packageId = [string]$metadata.id
        $packageVersion = [string]$metadata.version
        $licenseType = if ($null -eq $licenseNode) { "" } else { ([string]$licenseNode.GetAttribute("type")).Trim().ToLowerInvariant() }
        $licenseValue = if ($null -eq $licenseNode) { "" } else { ([string]$licenseNode.InnerText).Trim() }
        $licenseEntry = ""

        if ($null -eq $licenseNode) {
          Add-Failure "Package '$packageId' has no nuspec license metadata. Owner must select a license before publication."
        }
        elseif ($acceptedLicenseTypes -notcontains $licenseType) {
          Add-Failure "Package '$packageId' uses unsupported license metadata type '$licenseType'."
        }
        elseif (-not (Test-LicenseValueAllowed -Value $licenseValue)) {
          Add-Failure "Package '$packageId' has an empty or placeholder license value '$licenseValue'."
        }
        elseif ($licenseType -eq "file") {
          $normalizedLicensePath = $licenseValue.Replace("\", "/")
          $segments = @($normalizedLicensePath.Split("/") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
          if ($licenseValue.Contains("\") -or
              $normalizedLicensePath.StartsWith("/", [StringComparison]::Ordinal) -or
              [IO.Path]::IsPathRooted($licenseValue) -or
              $segments -contains "." -or
              $segments -contains ".." -or
              $normalizedLicensePath.Contains(":")) {
            Add-Failure "Package '$packageId' has an unsafe license file path '$licenseValue'."
          }
          else {
            $entry = @($archive.Entries | Where-Object { [string]::Equals($_.FullName, $normalizedLicensePath, [StringComparison]::OrdinalIgnoreCase) }) | Select-Object -First 1
            if ($null -eq $entry -or $entry.Length -le 0) {
              Add-Failure "Package '$packageId' license file '$licenseValue' is missing or empty."
            }
            else {
              $licenseEntry = $entry.FullName
            }
          }
        }

        if ($ownerDecisionState -eq "approved" -and
            (-not [string]::Equals($licenseType, $selectedPackageLicenseType, [StringComparison]::Ordinal) -or
             -not [string]::Equals($licenseValue, $selectedPackageLicenseValue, [StringComparison]::Ordinal))) {
          Add-Failure "Package '$packageId' license '${licenseType}:$licenseValue' does not match the Owner-selected license '${selectedPackageLicenseType}:$selectedPackageLicenseValue'."
        }

        $inspectedArtifacts.Add([pscustomobject]@{
          artifactType = "nupkg"
          path = $file.FullName
          packageId = $packageId
          packageVersion = $packageVersion
          licenseType = $licenseType
          licenseValue = $licenseValue
          licenseEntry = $licenseEntry
        })
      }
      else {
        $allowedNames = @($policy.sourceArchiveLicenseFileNames)
        $licenseEntries = @($archive.Entries | Where-Object {
          $entryName = $_.FullName
          $trimmed = $entryName.Trim("/")
          $segments = @($trimmed.Split("/") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
          $isSafePath = -not $entryName.Contains("\") -and
            -not $entryName.StartsWith("/", [StringComparison]::Ordinal) -and
            -not $entryName.Contains(":") -and
            $segments.Count -ge 1 -and
            $segments.Count -le 2 -and
            $segments -notcontains "." -and
            $segments -notcontains ".."
          $isSafePath -and $allowedNames -contains $segments[-1] -and $_.Length -gt 0
        })

        if ($licenseEntries.Count -eq 0) {
          Add-Failure "Source archive '$($file.Name)' has no non-empty root license file. Owner must select a license before publication."
        }
        elseif ($ownerDecisionState -eq "approved" -and
                -not ($licenseEntries | Where-Object {
                  $entryName = $_.FullName.Trim("/")
                  $entrySegments = @($entryName.Split("/") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
                  [string]::Equals($entrySegments[-1], $selectedSourceLicenseFileName, [StringComparison]::OrdinalIgnoreCase)
                })) {
          Add-Failure "Source archive '$($file.Name)' does not contain the Owner-selected root license file '$selectedSourceLicenseFileName'."
        }

        $inspectedArtifacts.Add([pscustomobject]@{
          artifactType = "source-archive"
          path = $file.FullName
          packageId = ""
          packageVersion = ""
          licenseType = "file"
          licenseValue = if ($licenseEntries.Count -gt 0) { [IO.Path]::GetFileName($licenseEntries[0].FullName) } else { "" }
          licenseEntry = if ($licenseEntries.Count -gt 0) { $licenseEntries[0].FullName } else { "" }
        })
      }
    }
    finally {
      $archive.Dispose()
    }
  }
}

$result = [pscustomobject]@{
  policyId = [string]$policy.policyId
  policyPath = $policyPath
  ownerDecisionState = [string]$policy.ownerDecisionState
  selectedPackageLicenseType = $selectedPackageLicenseType
  selectedPackageLicenseValue = $selectedPackageLicenseValue
  selectedSourceArchiveLicenseFileName = $selectedSourceLicenseFileName
  staticOnly = $StaticOnly.IsPresent
  inspectedArtifactCount = $inspectedArtifacts.Count
  inspectedArtifacts = @($inspectedArtifacts.ToArray())
  failureCount = $failures.Count
  failures = @($failures.ToArray())
  passed = $failures.Count -eq 0
  performsPublish = $false
}

$result | ConvertTo-Json -Depth 8
if ($failures.Count -gt 0) {
  throw "Publication license readiness failed with $($failures.Count) finding(s): $($failures -join ' | ')"
}
