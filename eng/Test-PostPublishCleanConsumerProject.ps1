[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectPath,
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path (Get-Location).Path $Path
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-FullPathOrEmpty {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  try {
    return [System.IO.Path]::GetFullPath($Path)
  }
  catch {
    return ""
  }
}

function Split-PackageSourceValues {
  param([AllowNull()][string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return @()
  }

  return @($Value.Split([char[]]@(';'), [System.StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { $_.Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Test-LocalPackageSourceValue {
  param([AllowNull()][string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }

  $candidate = $Value.Trim()
  if ($candidate.StartsWith("http://", [System.StringComparison]::OrdinalIgnoreCase) -or
    $candidate.StartsWith("https://", [System.StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }

  if ($candidate.StartsWith("file://", [System.StringComparison]::OrdinalIgnoreCase) -or
    $candidate.IndexOf(".nupkg", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $candidate.IndexOf('$(MSBuildProjectDirectory)', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $candidate.IndexOf('$(SolutionDir)', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $candidate.IndexOf('$(RepositoryRoot)', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    [IO.Path]::IsPathRooted($candidate) -or
    $candidate.StartsWith(".", [System.StringComparison]::Ordinal)) {
    return $true
  }

  return $candidate.IndexOf("\", [System.StringComparison]::Ordinal) -ge 0 -or
    $candidate.IndexOf("/", [System.StringComparison]::Ordinal) -ge 0
}

function Test-LocalNupkgReference {
  param([AllowNull()][string]$Value)

  return -not [string]::IsNullOrWhiteSpace($Value) -and
    [string]$Value -match "(?i)\.nupkg"
}

$resolvedProjectPath = Resolve-InputPath -Path $ProjectPath
$projectExists = Test-Path -LiteralPath $resolvedProjectPath -PathType Leaf
$projectExtensionReady = $resolvedProjectPath.EndsWith(".csproj", [System.StringComparison]::OrdinalIgnoreCase)

$repositoryRootFull = Get-FullPathOrEmpty -Path $RepositoryRoot
$projectFull = Get-FullPathOrEmpty -Path $resolvedProjectPath
$projectDirectoryFull = if ([string]::IsNullOrWhiteSpace($projectFull)) { "" } else { [System.IO.Path]::GetDirectoryName($projectFull) }
$projectOutsideRepository = $false
if (-not [string]::IsNullOrWhiteSpace($repositoryRootFull) -and -not [string]::IsNullOrWhiteSpace($projectDirectoryFull)) {
  $repoTrimmed = $repositoryRootFull.TrimEnd('\', '/')
  $projectTrimmed = $projectDirectoryFull.TrimEnd('\', '/')
  $projectOutsideRepository = -not [string]::Equals($projectTrimmed, $repoTrimmed, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not $projectTrimmed.StartsWith($repoTrimmed + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not $projectTrimmed.StartsWith($repoTrimmed + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

$packageReferences = @()
$projectReferences = @()
$restoreSourceProperties = @()
$nugetConfigPackageSources = @()
$localPackageSources = @()
$localNupkgPackageReferences = @()
$targetTensorRtReferenceFound = $false
$xmlParseError = ""

if ($projectExists) {
  try {
    [xml]$projectXml = Get-Content -LiteralPath $resolvedProjectPath -Raw -Encoding utf8

    $restoreSourcePropertyNames = @(
      "RestoreSources",
      "RestoreAdditionalProjectSources",
      "RestoreFallbackFolders",
      "RestoreAdditionalProjectFallbackFolders"
    )
    foreach ($propertyGroup in @($projectXml.Project.PropertyGroup)) {
      if ($null -eq $propertyGroup) { continue }

      foreach ($child in @($propertyGroup.ChildNodes)) {
        if ($null -eq $child -or $restoreSourcePropertyNames -notcontains $child.Name) { continue }

        $rawValue = [string]$child.InnerText
        foreach ($sourceValue in Split-PackageSourceValues -Value $rawValue) {
          $isLocalSource = Test-LocalPackageSourceValue -Value $sourceValue
          $item = [pscustomobject]@{
            sourceKind = "project-property"
            propertyName = $child.Name
            value = $sourceValue
            isLocalPackageSource = $isLocalSource
          }
          $restoreSourceProperties += $item
          if ($isLocalSource) {
            $localPackageSources += $item
          }
        }
      }
    }

    $packageReferenceNodes = @($projectXml.Project.ItemGroup.PackageReference)
    foreach ($node in $packageReferenceNodes) {
      if ($null -eq $node) { continue }
      $include = [string]$node.Include
      if ([string]::IsNullOrWhiteSpace($include)) {
        $include = [string]$node.Update
      }
      $version = [string]$node.Version
      if ([string]::IsNullOrWhiteSpace($version) -and $node.PSObject.Properties.Name -contains "Version") {
        $version = [string]$node.Version
      }
      if (-not [string]::IsNullOrWhiteSpace($include)) {
        $isLocalNupkgReference = Test-LocalNupkgReference -Value $version
        $packageReference = [pscustomobject]@{
          include = $include
          version = $version
          localNupkgReference = $isLocalNupkgReference
        }
        $packageReferences += $packageReference
        if ($isLocalNupkgReference) {
          $localNupkgPackageReferences += $packageReference
        }

        if ($include.IndexOf("JYPPX.TensorRtSharp", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $include.IndexOf("TensorRtSharp", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $include.IndexOf("TensorRT", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
          $targetTensorRtReferenceFound = $true
        }
      }
    }

    $projectReferenceNodes = @($projectXml.Project.ItemGroup.ProjectReference)
    foreach ($node in $projectReferenceNodes) {
      if ($null -eq $node) { continue }
      $include = [string]$node.Include
      if (-not [string]::IsNullOrWhiteSpace($include)) {
        $projectReferences += [pscustomobject]@{ include = $include }
      }
    }
  }
  catch {
    $xmlParseError = $_.Exception.Message
  }
}

$nugetConfigPaths = @()
if (-not [string]::IsNullOrWhiteSpace($projectDirectoryFull)) {
  $directory = [System.IO.DirectoryInfo]::new($projectDirectoryFull)
  while ($null -ne $directory) {
    $nugetConfigPath = Join-Path $directory.FullName "NuGet.config"
    if (Test-Path -LiteralPath $nugetConfigPath -PathType Leaf) {
      $nugetConfigPaths += $nugetConfigPath
      try {
        [xml]$nugetConfigXml = Get-Content -LiteralPath $nugetConfigPath -Raw -Encoding utf8
        foreach ($node in @($nugetConfigXml.configuration.packageSources.add)) {
          if ($null -eq $node) { continue }
          $key = [string]$node.key
          $value = [string]$node.value
          $isLocalSource = Test-LocalPackageSourceValue -Value $value
          $item = [pscustomobject]@{
            sourceKind = "nuget-config"
            configPath = $nugetConfigPath
            key = $key
            value = $value
            isLocalPackageSource = $isLocalSource
          }
          $nugetConfigPackageSources += $item
          if ($isLocalSource) {
            $localPackageSources += $item
          }
        }
      }
      catch {
        $item = [pscustomobject]@{
          sourceKind = "nuget-config"
          configPath = $nugetConfigPath
          key = "parse-error"
          value = $_.Exception.Message
          isLocalPackageSource = $true
        }
        $nugetConfigPackageSources += $item
        $localPackageSources += $item
      }
    }

    $directory = $directory.Parent
  }
}

$packageReferenceCount = @($packageReferences).Count
$projectReferenceCount = @($projectReferences).Count
$localPackageSourceCount = @($localPackageSources).Count
$localNupkgPackageReferenceCount = @($localNupkgPackageReferences).Count
$hasPackageReference = $packageReferenceCount -gt 0
$hasProjectReference = $projectReferenceCount -gt 0
$hasLocalPackageSource = $localPackageSourceCount -gt 0
$hasLocalNupkgPackageReference = $localNupkgPackageReferenceCount -gt 0
$xmlParsed = $projectExists -and [string]::IsNullOrWhiteSpace($xmlParseError)
$scanPassed = $projectExists -and $projectExtensionReady -and $xmlParsed -and $projectOutsideRepository -and $hasPackageReference -and $targetTensorRtReferenceFound -and -not $hasProjectReference -and -not $hasLocalPackageSource -and -not $hasLocalNupkgPackageReference
$scanState = if ($scanPassed) { "clean-consumer-project-scan-passed" } elseif (-not $projectExists) { "missing-project" } elseif (-not $projectOutsideRepository) { "blocked-project-inside-repository" } elseif ($hasProjectReference) { "blocked-project-reference-present" } elseif ($hasLocalNupkgPackageReference) { "blocked-local-nupkg-reference-present" } elseif ($hasLocalPackageSource) { "blocked-local-package-source-present" } elseif (-not $targetTensorRtReferenceFound) { "blocked-target-package-reference-missing" } else { "blocked-clean-consumer-project-scan" }

$checks = @(
  [pscustomobject]@{ id = "project-exists"; passed = $projectExists; ownerAction = "Provide a clean consumer .csproj path."; boundary = "A missing project cannot prove post-publish package consumption." }
  [pscustomobject]@{ id = "project-extension"; passed = $projectExtensionReady; ownerAction = "Point ProjectPath to a .csproj file."; boundary = "Folder-only evidence is not enough for clean consumer proof." }
  [pscustomobject]@{ id = "xml-parsed"; passed = $xmlParsed; ownerAction = "Fix the clean consumer project XML so it can be audited."; boundary = "Unparseable project files cannot prove PackageReference usage." }
  [pscustomobject]@{ id = "outside-repository"; passed = $projectOutsideRepository; ownerAction = "Create the clean consumer outside the source repository tree."; boundary = "Repository-local projects can hide ProjectReference or build-output coupling." }
  [pscustomobject]@{ id = "package-reference-present"; passed = $hasPackageReference; ownerAction = "Restore the published package via PackageReference."; boundary = "No PackageReference means the channel package was not consumed." }
  [pscustomobject]@{ id = "target-tensorrt-reference-present"; passed = $targetTensorRtReferenceFound; ownerAction = "Add PackageReference for JYPPX.TensorRtSharp or its runtime package."; boundary = "A clean consumer must consume the release package under verification." }
  [pscustomobject]@{ id = "no-project-reference"; passed = (-not $hasProjectReference); ownerAction = "Remove ProjectReference entries from the clean consumer."; boundary = "ProjectReference bypasses published package verification." }
  [pscustomobject]@{ id = "no-local-package-source"; passed = (-not $hasLocalPackageSource); ownerAction = "Remove RestoreSources, fallback folders, or NuGet.config packageSources that point to local files, repository artifacts, or local feeds."; boundary = "Local package sources can substitute unpublished artifacts for public-channel packages." }
  [pscustomobject]@{ id = "no-local-nupkg-reference"; passed = (-not $hasLocalNupkgPackageReference); ownerAction = "Use a versioned PackageReference from the release channel, not a local .nupkg path."; boundary = "A direct .nupkg path bypasses post-publish package verification." }
)

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-project-scan.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-project-scan.md"

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "post-publish-clean-consumer-project-scan"
  projectPath = $ProjectPath
  resolvedProjectPath = $resolvedProjectPath
  repositoryRoot = $RepositoryRoot
  scanState = $scanState
  scanPassed = $scanPassed
  canCloseReleaseIssue = $false
  isPostPublishVerificationProof = $false
  ownerActionRequired = -not $scanPassed
  projectExists = $projectExists
  projectExtensionReady = $projectExtensionReady
  xmlParsed = $xmlParsed
  xmlParseError = $xmlParseError
  projectOutsideRepository = $projectOutsideRepository
  packageReferenceCount = $packageReferenceCount
  projectReferenceCount = $projectReferenceCount
  localPackageSourceCount = $localPackageSourceCount
  localNupkgPackageReferenceCount = $localNupkgPackageReferenceCount
  hasLocalPackageSource = $hasLocalPackageSource
  hasLocalNupkgPackageReference = $hasLocalNupkgPackageReference
  targetTensorRtReferenceFound = $targetTensorRtReferenceFound
  packageReferences = @($packageReferences)
  projectReferences = @($projectReferences)
  restoreSourceProperties = @($restoreSourceProperties)
  nugetConfigPaths = @($nugetConfigPaths)
  nugetConfigPackageSources = @($nugetConfigPackageSources)
  localPackageSources = @($localPackageSources)
  localNupkgPackageReferences = @($localNupkgPackageReferences)
  checks = $checks
  ownerActionSummary = @(
    "Use this scan only as clean consumer project evidence; it is not post-publish proof by itself.",
    "The clean consumer project must live outside the source repository tree.",
    "The clean consumer must use PackageReference for the released TensorRtSharp package and no ProjectReference.",
    "The clean consumer must not use local NuGet package sources, repository artifact folders, fallback folders, or direct .nupkg references.",
    "Release issue close still requires real post-publish record validation with logs, SHA256, host metadata, runtime smoke, and -FailOnNotProof."
  )
  nonSubstituteProofKinds = @(
    "clean consumer scan only",
    "local feed",
    "local package source",
    "local nupkg",
    "repository artifact package source",
    "ProjectReference",
    "template",
    "draft",
    "dependency-probe-only"
  )
}

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Clean Consumer Project Scan")
$lines.Add("")
$lines.Add("- scan state: ``$scanState``")
$lines.Add("- scan passed: ``$scanPassed``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("- post-publish verification proof: ``False``")
$lines.Add("- project path: ``$ProjectPath``")
$lines.Add("- resolved project path: ``$resolvedProjectPath``")
$lines.Add("- repository root: ``$RepositoryRoot``")
$lines.Add("- package references: ``$packageReferenceCount``")
$lines.Add("- project references: ``$projectReferenceCount``")
$lines.Add("- local package sources: ``$localPackageSourceCount``")
$lines.Add("- local .nupkg references: ``$localNupkgPackageReferenceCount``")
$lines.Add("- target TensorRtSharp reference found: ``$targetTensorRtReferenceFound``")
$lines.Add("")
$lines.Add("## Checks")
$lines.Add("")
$lines.Add("| ID | Passed | Owner action | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($check in $checks) {
  $lines.Add("| ``$($check.id)`` | ``$($check.passed)`` | $(ConvertTo-MarkdownCell $check.ownerAction) | $(ConvertTo-MarkdownCell $check.boundary) |")
}
$lines.Add("")
$lines.Add("## Package References")
$lines.Add("")
if ($packageReferenceCount -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($reference in $packageReferences) {
    $lines.Add("- ``$($reference.include)`` ``$($reference.version)``")
  }
}
$lines.Add("")
$lines.Add("## Project References")
$lines.Add("")
if ($projectReferenceCount -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($reference in $projectReferences) {
    $lines.Add("- ``$($reference.include)``")
  }
}
$lines.Add("")
$lines.Add("## Local Package Sources")
$lines.Add("")
if ($localPackageSourceCount -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($source in $localPackageSources) {
    $sourceLabel = if ($source.PSObject.Properties.Name -contains "propertyName") { $source.propertyName } elseif ($source.PSObject.Properties.Name -contains "key") { $source.key } else { $source.sourceKind }
    $lines.Add("- ``$($source.sourceKind)`` ``$sourceLabel`` ``$($source.value)``")
  }
}
$lines.Add("")
$lines.Add("## Local Nupkg References")
$lines.Add("")
if ($localNupkgPackageReferenceCount -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($reference in $localNupkgPackageReferences) {
    $lines.Add("- ``$($reference.include)`` ``$($reference.version)``")
  }
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $record.ownerActionSummary) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Non-Substitute Proof Kinds")
$lines.Add("")
foreach ($item in $record.nonSubstituteProofKinds) {
  $lines.Add("- ``$item``")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer project scan written to $jsonPath"
Write-Host "ScanState=$scanState ScanPassed=$scanPassed CanCloseReleaseIssue=False"
