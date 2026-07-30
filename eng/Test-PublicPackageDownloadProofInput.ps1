[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-download-proof-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [bool]$DefaultValue
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) {
    return [bool]$value
  }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) {
    return $parsed
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Test-FileHashMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$Sha256
  )

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-ValueInSet {
  param(
    [AllowNull()][object]$Value,
    [string[]]$AllowedValues
  )

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  foreach ($allowedValue in $AllowedValues) {
    if ($text.Equals($allowedValue, [StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }

  return $false
}

function Test-SourceUrlIsPublicCandidate {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) {
    return $false
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or
      $text.StartsWith("\\", [StringComparison]::Ordinal) -or
      $text.StartsWith("./", [StringComparison]::Ordinal) -or
      $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $false
  }

  if ($text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains("local", [StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }

  return $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)
}

function Test-NuGetPackagePageUrl {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  return -not (Test-IsPlaceholder -Value $text) -and $text.StartsWith("https://www.nuget.org/packages/", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PublicDownloadUrl {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) { return $false }
  if (-not $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  if ($text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -or
      $text.Contains("local", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  return $true
}

function Test-GitHubPublicUrl {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  return -not (Test-IsPlaceholder -Value $text) -and $text.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PositiveInt64 {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = [Int64]::MinValue
  return [Int64]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -gt 0
}

function Test-FileSizeMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$ExpectedSize
  )

  if ((Test-IsPlaceholder -Value $Path) -or -not (Test-PositiveInt64 -Value $ExpectedSize)) { return $false }
  $resolvedPath = Resolve-InputPath -Path ([string]$Path)
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $false }
  $expected = [Int64]([string]$ExpectedSize)
  return ([IO.FileInfo]::new($resolvedPath)).Length -eq $expected
}

function Get-ForbiddenSubstituteFindings {
  param([string[]]$Values)

  $findings = New-Object System.Collections.Generic.List[string]
  $text = ($Values -join "`n")
  foreach ($pattern in @(
      @{ id = "local-feed"; regex = '(?i)local\s+feed|local-feed|file://' },
      @{ id = "project-reference"; regex = '(?i)projectreference|project\s+reference' },
      @{ id = "direct-nupkg"; regex = '(?i)direct\s+\.?nupkg|direct-nupkg' },
      @{ id = "package-managed-dry-run"; regex = '(?i)package-managed-dry-run' },
      @{ id = "manual-approval"; regex = '(?i)manual\s+approval' },
      @{ id = "queued-workflow"; regex = '(?i)queued\s+(github\s+actions\s+)?workflow|queued\s+github\s+actions\s+run' },
      @{ id = "missing-runner"; regex = '(?i)missing\s+(self-hosted\s+)?runner' },
      @{ id = "dashboard-only"; regex = '(?i)dashboard-only|dashboard\s+only' },
      @{ id = "artifact-only"; regex = '(?i)artifact-only|artifact\s+only' },
      @{ id = "local-dotnet-test"; regex = '(?i)local\s+dotnet\s+test' },
      @{ id = "sidecar-only"; regex = '(?i)sidecar-only|sidecar\s+only' }
    )) {
    if ($text -match $pattern.regex) { $findings.Add([string]$pattern.id) | Out-Null }
  }

  return @($findings.ToArray() | Select-Object -Unique)
}

function Test-DownloadedPathIsPublicDownloadCandidate {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$DryRunPath
  )

  $pathText = [string]$Path
  $dryRunText = [string]$DryRunPath
  if (Test-IsPlaceholder -Value $pathText) {
    return $false
  }

  if ($pathText.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("\artifacts\", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("/artifacts/", [StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }

  if (-not (Test-IsPlaceholder -Value $dryRunText)) {
    try {
      $candidateFullPath = [IO.Path]::GetFullPath((Resolve-InputPath -Path $pathText))
      $dryRunFullPath = [IO.Path]::GetFullPath((Resolve-InputPath -Path $dryRunText))
      if ($candidateFullPath.Equals($dryRunFullPath, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
      }
    }
    catch {
      if ($pathText.Equals($dryRunText, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
      }
    }
  }

  return $pathText.EndsWith(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

$publicPackageDownloadProofRequiredFields = @(
  "managedPackageId",
  "managedPackageVersion",
  "managedPackagePageUrl",
  "managedPackageDownloadUrl",
  "downloadedManagedNupkgPath",
  "downloadedManagedNupkgSha256",
  "downloadedManagedNupkgSizeBytes",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "runtimePackagePageUrl",
  "runtimePackageDownloadUrl",
  "downloadedRuntimeNupkgPath",
  "downloadedRuntimeNupkgSha256",
  "downloadedRuntimeNupkgSizeBytes",
  "publicPackageSourceKind",
  "publicPackageSourceUrl",
  "downloadCommand",
  "downloadedAtUtc",
  "capturedAtUtc",
  "ownerName",
  "ownerReviewer",
  "sourceGitHubActionsRunEvidenceReady",
  "sourceOwnerPublicPublishResultReady",
  "sourceWorkflowRunLogSha256",
  "sourceArtifactManifestSha256",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256",
  "githubReleaseUrl",
  "githubReleaseAssetUrl",
  "githubReleaseAssetDownloadedPath",
  "githubReleaseAssetSha256",
  "githubReleaseAssetSizeBytes"
)

$publicPackageDownloadProofRejectedSubstitutes = @(
  "local-feed-restore",
  "direct-local-nupkg",
  "project-reference",
  "repo-internal-consumer",
  "package-managed-dry-run-artifact",
  "github-actions-artifact-only",
  "dashboard-only",
  "queued-workflow-only",
  "missing-runner",
  "local-dotnet-test-only",
  "sidecar-only-report"
)

$publicPackageDownloadProofSourceReadinessSignals = @(
  "sourceGitHubActionsRunEvidenceReady",
  "sourceWorkflowRunLogSha256",
  "sourceArtifactManifestSha256",
  "sourceOwnerPublicPublishResultReady",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256"
)

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public package download proof input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true
$usesPublishToken = Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true
$canPublishPublicly = Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true
$canPublishGitHubPackages = Get-BoolPropertyOrDefault -Object $record -Name "canPublishGitHubPackages" -DefaultValue $true
$canCloseReleaseIssue = Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true
$canClaimRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimRuntimeProof" -DefaultValue $true
$canClaimPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true
$isPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true
$isPostPublishProof = Get-BoolPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true
$isPublishedPackageProof = Get-BoolPropertyOrDefault -Object $record -Name "isPublishedPackageProof" -DefaultValue $false
$ownerAuthorizationState = [string](Get-PropertyOrDefault -Object $record -Name "ownerAuthorizationState" -DefaultValue "")
$sourceKind = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSourceKind" -DefaultValue "")
$sourceUrl = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSourceUrl" -DefaultValue "")
$managedPageUrl = [string](Get-PropertyOrDefault -Object $record -Name "managedPackagePageUrl" -DefaultValue "")
$managedDownloadUrl = [string](Get-PropertyOrDefault -Object $record -Name "managedPackageDownloadUrl" -DefaultValue "")
$runtimePageUrl = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackagePageUrl" -DefaultValue "")
$runtimeDownloadUrl = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageDownloadUrl" -DefaultValue "")
$githubReleaseUrl = [string](Get-PropertyOrDefault -Object $record -Name "githubReleaseUrl" -DefaultValue "")
$githubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $record -Name "githubReleaseAssetUrl" -DefaultValue "")
$githubReleaseAssetPath = [string](Get-PropertyOrDefault -Object $record -Name "githubReleaseAssetDownloadedPath" -DefaultValue "")
$githubReleaseAssetSha = [string](Get-PropertyOrDefault -Object $record -Name "githubReleaseAssetSha256" -DefaultValue "")
$githubReleaseAssetSize = [string](Get-PropertyOrDefault -Object $record -Name "githubReleaseAssetSizeBytes" -DefaultValue "")
$managedPath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgPath" -DefaultValue "")
$runtimePath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
$managedSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgSha256" -DefaultValue "")
$runtimeSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
$managedSize = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgSizeBytes" -DefaultValue "")
$runtimeSize = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgSizeBytes" -DefaultValue "")
$dryRunPath = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunArtifactPath" -DefaultValue "")
$dryRunSha = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunManagedNupkgSha256" -DefaultValue "")
$sourceGitHubActionsRunEvidenceReady = Get-BoolPropertyOrDefault -Object $record -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false
$sourceOwnerPublicPublishResultReady = Get-BoolPropertyOrDefault -Object $record -Name "sourceOwnerPublicPublishResultReady" -DefaultValue $false
$sourceWorkflowRunLogSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceWorkflowRunLogSha256" -DefaultValue "")
$sourceArtifactManifestSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceArtifactManifestSha256" -DefaultValue "")
$sourceOwnerPublicPackageUrl = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
$sourceOwnerPublicPackageVersion = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
$sourceOwnerPublicPackageSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
$capturedAtUtc = [string](Get-PropertyOrDefault -Object $record -Name "capturedAtUtc" -DefaultValue "")
$ownerReviewer = [string](Get-PropertyOrDefault -Object $record -Name "ownerReviewer" -DefaultValue "")
$forbiddenFindings = Get-ForbiddenSubstituteFindings -Values @(
  $sourceKind, $sourceUrl, $managedPageUrl, $managedDownloadUrl, $runtimePageUrl, $runtimeDownloadUrl,
  $githubReleaseUrl, $githubReleaseAssetUrl, $githubReleaseAssetPath, $managedPath, $runtimePath,
  [string](Get-PropertyOrDefault -Object $record -Name "downloadCommand" -DefaultValue ""),
  $ownerName, $ownerReviewer
)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "public-package-download-proof-input") -Severity "blocker" -Detail "recordKind must be public-package-download-proof-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $usesPublishToken -and -not $canPublishPublicly -and -not $canPublishGitHubPackages -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Public package download proof input validation must not publish, use token, approve publication, or close issues.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-claims-false" -Passed (-not $canClaimRuntimeProof -and -not $canClaimPackageConsumerRuntimeProof -and -not $isPackageConsumerRuntimeProof -and -not $isPostPublishProof) -Severity "blocker" -Detail "Download proof input is not runtime proof, package-consumer runtime proof, or post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-authorization-not-default-true" -Passed (-not $ownerAuthorizationState.Equals("true", [StringComparison]::OrdinalIgnoreCase) -and -not $ownerAuthorizationState.Equals("authorized", [StringComparison]::OrdinalIgnoreCase) -and -not $ownerAuthorizationState.Equals("approved", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "ownerAuthorizationState must not default to true/authorized/approved.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-kind-valid" -Passed (Test-ValueInSet -Value $sourceKind -AllowedValues @("nuget.org", "github-packages")) -Severity "action-required" -Detail "publicPackageSourceKind must be nuget.org or github-packages.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-url-public-not-local" -Passed (Test-SourceUrlIsPublicCandidate -Value $sourceUrl) -Severity "action-required" -Detail "publicPackageSourceUrl must be an HTTPS package source, not a local path, artifacts path, direct .nupkg, template placeholder, or dry-run artifact.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-page-url-nuget" -Passed (Test-NuGetPackagePageUrl -Value $managedPageUrl) -Severity "action-required" -Detail "managedPackagePageUrl must be a nuget.org package page.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-download-url-public" -Passed (Test-PublicDownloadUrl -Value $managedDownloadUrl) -Severity "action-required" -Detail "managedPackageDownloadUrl must be an HTTPS public download URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-page-url-nuget" -Passed (Test-NuGetPackagePageUrl -Value $runtimePageUrl) -Severity "action-required" -Detail "runtimePackagePageUrl must be a nuget.org package page.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-download-url-public" -Passed (Test-PublicDownloadUrl -Value $runtimeDownloadUrl) -Severity "action-required" -Detail "runtimePackageDownloadUrl must be an HTTPS public download URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-url-public" -Passed (Test-GitHubPublicUrl -Value $githubReleaseUrl) -Severity "action-required" -Detail "githubReleaseUrl must be a GitHub release URL for the managed plus bridge-only asset route.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-url-public" -Passed (Test-GitHubPublicUrl -Value $githubReleaseAssetUrl) -Severity "action-required" -Detail "githubReleaseAssetUrl must identify a managed or bridge-only public asset.")) | Out-Null
$items.Add((New-ValidationItem -Id "download-command-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "downloadCommand" -DefaultValue ""))) -Severity "action-required" -Detail "downloadCommand must capture the exact public package download command.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "downloadedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "downloadedAtUtc must be a real parseable DateTimeOffset.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-name-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerName" -DefaultValue ""))) -Severity "action-required" -Detail "ownerName must be filled by the owner collecting real public package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-reviewer-present" -Passed (-not (Test-IsPlaceholder -Value $ownerReviewer)) -Severity "action-required" -Detail "ownerReviewer must identify the person reviewing public download proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "captured-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $capturedAtUtc) -Severity "action-required" -Detail "capturedAtUtc must be parseable.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-version-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "managedPackageVersion" -DefaultValue ""))) -Severity "action-required" -Detail "managedPackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-version-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "runtimePackageVersion" -DefaultValue ""))) -Severity "action-required" -Detail "runtimePackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-github-actions-run-evidence-ready" -Passed $sourceGitHubActionsRunEvidenceReady -Severity "action-required" -Detail "Public download proof must link to ready GitHub Actions run evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-publish-result-ready" -Passed $sourceOwnerPublicPublishResultReady -Severity "action-required" -Detail "Public download proof must link to a ready Owner public publish result.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-workflow-log-sha256" -Passed (Test-Sha256Format -Value $sourceWorkflowRunLogSha256) -Severity "action-required" -Detail "Source workflow run log SHA256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifact-manifest-sha256" -Passed (Test-Sha256Format -Value $sourceArtifactManifestSha256) -Severity "action-required" -Detail "Source artifact manifest SHA256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-url-match" -Passed (-not (Test-IsPlaceholder -Value $sourceOwnerPublicPackageUrl) -and $managedPageUrl.Equals($sourceOwnerPublicPackageUrl, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "managedPackagePageUrl must match the Owner public publish result package URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-version-match" -Passed (-not (Test-IsPlaceholder -Value $sourceOwnerPublicPackageVersion) -and ([string](Get-PropertyOrDefault -Object $record -Name "managedPackageVersion" -DefaultValue "")).Equals($sourceOwnerPublicPackageVersion, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "managedPackageVersion must match the Owner public publish result version.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-sha256-match" -Passed (Test-Sha256Format -Value $sourceOwnerPublicPackageSha256) -Severity "action-required" -Detail "Owner public publish result SHA256 must be present for cross-checking.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $managedPath -DryRunPath $dryRunPath) -Severity "action-required" -Detail "downloadedManagedNupkgPath must be a real downloaded .nupkg path and not a dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $runtimePath -DryRunPath "") -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must be a real downloaded .nupkg path and not an artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-size-positive" -Passed (Test-PositiveInt64 -Value $managedSize) -Severity "action-required" -Detail "downloadedManagedNupkgSizeBytes must be a positive integer.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-size-positive" -Passed (Test-PositiveInt64 -Value $runtimeSize) -Severity "action-required" -Detail "downloadedRuntimeNupkgSizeBytes must be a positive integer.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-sha256-format" -Passed (Test-Sha256Format -Value $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-sha256-format" -Passed (Test-Sha256Format -Value $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-hash-match" -Passed (Test-FileHashMatches -Path $managedPath -Sha256 $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgPath must exist and match downloadedManagedNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-hash-match" -Passed (Test-FileHashMatches -Path $runtimePath -Sha256 $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must exist and match downloadedRuntimeNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-size-match" -Passed (Test-FileSizeMatches -Path $managedPath -ExpectedSize $managedSize) -Severity "action-required" -Detail "Downloaded managed package size must match downloadedManagedNupkgSizeBytes.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-size-match" -Passed (Test-FileSizeMatches -Path $runtimePath -ExpectedSize $runtimeSize) -Severity "action-required" -Detail "Downloaded runtime package size must match downloadedRuntimeNupkgSizeBytes.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-sha256" -Passed (Test-Sha256Format -Value $githubReleaseAssetSha) -Severity "action-required" -Detail "githubReleaseAssetSha256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-size-positive" -Passed (Test-PositiveInt64 -Value $githubReleaseAssetSize) -Severity "action-required" -Detail "githubReleaseAssetSizeBytes must be positive.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-sha-not-substituted" -Passed (-not (Test-Sha256Format -Value $dryRunSha) -or -not $managedSha.Equals($dryRunSha, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub Actions dry-run SHA256 must not be reused as downloaded public package SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-absent" -Passed ($forbiddenFindings.Count -eq 0) -Severity "blocker" -Detail $(if ($forbiddenFindings.Count -eq 0) { "No local feed, direct nupkg, dry-run, dashboard-only, artifact-only, ProjectReference, manual approval, queued workflow, or local test substitute was detected." } else { "Forbidden substitute(s): $($forbiddenFindings -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "template-default-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "validationState" -DefaultValue "") -ne "ready" -and -not $isPublishedPackageProof) -Severity "blocker" -Detail "Template/default input must stay blocked until owner supplies real download evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-download-required-field-contract" -Passed ($publicPackageDownloadProofRequiredFields.Count -ge 16) -Severity "blocker" -Detail "Public package download proof must expose a stable required-field contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-download-rejected-substitute-contract" -Passed ($publicPackageDownloadProofRejectedSubstitutes.Count -ge 10) -Severity "blocker" -Detail "Public package download proof must expose a stable rejected-substitute contract.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-package-download-proof-input"
}
elseif ($failedActionRequired.Count -eq 0) {
  "public-package-download-proof-input-ready"
}
else {
  "blocked-public-package-download-proof-required"
}

$validation = [pscustomobject]@{
  recordKind = "public-package-download-proof-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  publicPackageDownloadProofReady = ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0)
  sourceUrlPublicNotLocal = (@($items | Where-Object { $_.id -eq "source-url-public-not-local" -and $_.passed }).Count -eq 1)
  downloadedManagedNupkgSha256Ready = (@($items | Where-Object { $_.id -eq "downloaded-managed-sha256-format" -and $_.passed }).Count -eq 1)
  downloadedManagedNupkgHashMatches = (@($items | Where-Object { $_.id -eq "downloaded-managed-hash-match" -and $_.passed }).Count -eq 1)
  dryRunShaNotSubstituted = (@($items | Where-Object { $_.id -eq "dry-run-sha-not-substituted" -and $_.passed }).Count -eq 1)
  sourceGitHubActionsRunEvidenceReady = $sourceGitHubActionsRunEvidenceReady
  sourceOwnerPublicPublishResultReady = $sourceOwnerPublicPublishResultReady
  publicPackageDownloadProofRequiredFields = @($publicPackageDownloadProofRequiredFields)
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFields.Count
  publicPackageDownloadProofRejectedSubstitutes = @($publicPackageDownloadProofRejectedSubstitutes)
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstitutes.Count
  publicPackageDownloadProofSourceReadinessSignals = @($publicPackageDownloadProofSourceReadinessSignals)
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignals.Count
  managedPackagePageUrl = $managedPageUrl
  managedPackageDownloadUrl = $managedDownloadUrl
  runtimePackagePageUrl = $runtimePageUrl
  runtimePackageDownloadUrl = $runtimeDownloadUrl
  githubReleaseUrl = $githubReleaseUrl
  githubReleaseAssetUrl = $githubReleaseAssetUrl
  forbiddenSubstituteFindings = @($forbiddenFindings)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates real public package download input only. It rejects local paths, direct .nupkg shortcuts, dry-run artifacts, and dry-run hashes as substitute proof."
}

$jsonPath = Join-Path $OutputRoot "public-package-download-proof-input-validation.json"
$markdownPath = Join-Path $OutputRoot "public-package-download-proof-input-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Public Package Download Proof Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| publicPackageDownloadProofReady | ``$($validation.publicPackageDownloadProofReady)`` |
| publicPackageDownloadProofRequiredFieldCount | ``$($validation.publicPackageDownloadProofRequiredFieldCount)`` |
| publicPackageDownloadProofRejectedSubstituteCount | ``$($validation.publicPackageDownloadProofRejectedSubstituteCount)`` |
| publicPackageDownloadProofSourceReadinessSignalCount | ``$($validation.publicPackageDownloadProofSourceReadinessSignalCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canClaimPackageConsumerRuntimeProof | ``$($validation.canClaimPackageConsumerRuntimeProof)`` |
| isPackageConsumerRuntimeProof | ``$($validation.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public package download proof input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Public package download proof input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False"
