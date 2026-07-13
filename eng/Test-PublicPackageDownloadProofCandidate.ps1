[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-download-proof-candidate.json",
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
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
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
  if ($value -is [bool]) { return [bool]$value }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }

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

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) { return $false }
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-ValueInSet {
  param(
    [AllowNull()][object]$Value,
    [string[]]$AllowedValues
  )

  if (Test-IsPlaceholder -Value $Value) { return $false }
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
  if (Test-IsPlaceholder -Value $text) { return $false }

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

function Test-DownloadedPathIsPublicDownloadCandidate {
  param([AllowNull()][object]$Path)

  $pathText = [string]$Path
  if (Test-IsPlaceholder -Value $pathText) { return $false }

  if ($pathText.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("\artifacts\", [StringComparison]::OrdinalIgnoreCase) -or
      $pathText.Contains("/artifacts/", [StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }

  return $pathText.EndsWith(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Test-FileSizeMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$ExpectedSize
  )

  if ((Test-IsPlaceholder -Value $Path) -or -not (Test-PositiveInt64 -Value $ExpectedSize)) { return $false }
  $resolvedPath = Resolve-RepositoryPath -Path ([string]$Path)
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $false }
  $expected = [Int64]([string]$ExpectedSize)
  return ([IO.FileInfo]::new($resolvedPath)).Length -eq $expected
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

  $resolvedPath = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-PublicPackageDownloadProofCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 16
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
$candidateItemCount = [int](Get-PropertyOrDefault -Object $record -Name "candidateItemCount" -DefaultValue 0)
$readyCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyCandidateCount" -DefaultValue 0)
$blockedCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCandidateCount" -DefaultValue 0)
$sourceInputValidationState = [string](Get-PropertyOrDefault -Object $record -Name "sourceInputValidationState" -DefaultValue "")
$sourcePublicPackageDownloadProofReady = [bool](Get-PropertyOrDefault -Object $record -Name "sourcePublicPackageDownloadProofReady" -DefaultValue $false)
$candidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $false)
$proofCandidateReady = [bool](Get-PropertyOrDefault -Object $record -Name "proofCandidateReady" -DefaultValue $false)

$managedPackageId = [string](Get-PropertyOrDefault -Object $record -Name "managedPackageId" -DefaultValue "")
$managedPackageVersion = [string](Get-PropertyOrDefault -Object $record -Name "managedPackageVersion" -DefaultValue "")
$runtimePackageId = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageId" -DefaultValue "")
$runtimePackageVersion = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageVersion" -DefaultValue "")
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
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
$downloadCommand = [string](Get-PropertyOrDefault -Object $record -Name "downloadCommand" -DefaultValue "")
$downloadedAtUtc = [string](Get-PropertyOrDefault -Object $record -Name "downloadedAtUtc" -DefaultValue "")
$capturedAtUtc = [string](Get-PropertyOrDefault -Object $record -Name "capturedAtUtc" -DefaultValue "")
$ownerName = [string](Get-PropertyOrDefault -Object $record -Name "ownerName" -DefaultValue "")
$ownerReviewer = [string](Get-PropertyOrDefault -Object $record -Name "ownerReviewer" -DefaultValue "")
$sourceGitHubActionsRunEvidenceReady = Get-BoolPropertyOrDefault -Object $record -Name "sourceGitHubActionsRunEvidenceReady" -DefaultValue $false
$sourceOwnerPublicPublishResultReady = Get-BoolPropertyOrDefault -Object $record -Name "sourceOwnerPublicPublishResultReady" -DefaultValue $false
$sourceWorkflowRunLogSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceWorkflowRunLogSha256" -DefaultValue "")
$sourceArtifactManifestSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceArtifactManifestSha256" -DefaultValue "")
$sourceOwnerPublicPackageUrl = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
$sourceOwnerPublicPackageVersion = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
$sourceOwnerPublicPackageSha256 = [string](Get-PropertyOrDefault -Object $record -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
$preReleaseReadinessMatrixPath = [string](Get-PropertyOrDefault -Object $record -Name "preReleaseReadinessMatrixPath" -DefaultValue "artifacts\final-release\pre-release-package-proof-readiness-matrix.json")
$preReleaseReadinessMatrixState = [string](Get-PropertyOrDefault -Object $record -Name "preReleaseReadinessMatrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseReadinessBlockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "preReleaseReadinessBlockedLaneCount" -DefaultValue 999)
$preReleasePublicPackageDownloadLaneReady = Get-BoolPropertyOrDefault -Object $record -Name "preReleasePublicPackageDownloadLaneReady" -DefaultValue $false
$preReleaseCanPromotePublicProof = Get-BoolPropertyOrDefault -Object $record -Name "preReleaseCanPromotePublicProof" -DefaultValue $false
$projectionForbiddenFindings = @((Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteFindings" -DefaultValue @()))
$forbiddenFindings = Get-ForbiddenSubstituteFindings -Values @(
  $sourceKind, $sourceUrl, $managedPageUrl, $managedDownloadUrl, $runtimePageUrl, $runtimeDownloadUrl,
  $githubReleaseUrl, $githubReleaseAssetUrl, $githubReleaseAssetPath, $managedPath, $runtimePath,
  $downloadCommand, $ownerName, $ownerReviewer, ($projectionForbiddenFindings -join " ")
)

$performsPublish = Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true
$usesPublishToken = Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true
$canPublishPublicly = Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true
$canPublishGitHubPackages = Get-BoolPropertyOrDefault -Object $record -Name "canPublishGitHubPackages" -DefaultValue $true
$canCloseReleaseIssue = Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true
$canClaimRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimRuntimeProof" -DefaultValue $true
$canClaimPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true
$canPromoteRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true
$isRuntimeExecutionProof = Get-BoolPropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true
$isPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true
$isPostPublishProof = Get-BoolPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true
$isReleaseCloseProof = Get-BoolPropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true
$isGitHubActionsProof = Get-BoolPropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $true

$sourceReady = $sourceInputValidationState -eq "public-package-download-proof-input-ready" -and $sourcePublicPackageDownloadProofReady
$candidatePromotionGateReady = $sourceReady -and $preReleaseCanPromotePublicProof
$candidateCountsConsistent = if ($candidatePromotionGateReady) {
  $candidateReady -and $proofCandidateReady -and $candidateItemCount -eq 1 -and $readyCandidateCount -eq 1 -and $blockedCandidateCount -eq 0
}
else {
  -not $candidateReady -and -not $proofCandidateReady -and $candidateItemCount -eq 0 -and $readyCandidateCount -eq 0 -and $blockedCandidateCount -eq 1
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "public-package-download-proof-candidate") -Severity "blocker" -Detail "recordKind must be public-package-download-proof-candidate.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-state-valid" -Passed (@("public-package-download-proof-candidate-imported", "blocked-public-package-download-proof-required") -contains $candidateState) -Severity "blocker" -Detail "candidateState must be imported or blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-counts-consistent" -Passed $candidateCountsConsistent -Severity "blocker" -Detail "Candidate counts and proofCandidateReady must align with source input validation readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $usesPublishToken -and -not $canPublishPublicly -and -not $canPublishGitHubPackages -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Candidate validation must not publish, use tokens, approve publication, or close issues.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-claims-false" -Passed (-not $canClaimRuntimeProof -and -not $canClaimPackageConsumerRuntimeProof -and -not $canPromoteRuntimeProof -and -not $isRuntimeExecutionProof -and -not $isPackageConsumerRuntimeProof -and -not $isPostPublishProof -and -not $isReleaseCloseProof -and -not $isGitHubActionsProof) -Severity "blocker" -Detail "Candidate is not runtime proof, package-consumer runtime proof, post-publish proof, release close proof, or GitHub Actions proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($raw.IndexOf("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("cannot close", [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Candidate must document non-proof/non-side-effect boundary.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-input-ready" -Passed $sourceReady -Severity "action-required" -Detail "Source public package download input validation must be public-package-download-proof-input-ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-public-package-download-lane-ready" -Passed $preReleasePublicPackageDownloadLaneReady -Severity "action-required" -Detail "Pre-release readiness matrix public-package-download lane must be ready before candidate can promote public proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-can-promote-public-proof" -Passed $preReleaseCanPromotePublicProof -Severity "action-required" -Detail "Pre-release readiness matrix must explicitly set canPromotePublicProof for the public-package-download lane.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-id-present" -Passed ($managedPackageId -eq "JYPPX.TensorRT.CSharp.API") -Severity "action-required" -Detail "Managed package id must match the public managed package.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-version-present" -Passed (-not (Test-IsPlaceholder -Value $managedPackageVersion)) -Severity "action-required" -Detail "managedPackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-id-present" -Passed ($runtimePackageId.StartsWith("JYPPX.TensorRT.CSharp.API.runtime.", [StringComparison]::Ordinal)) -Severity "action-required" -Detail "Runtime package id must be a public runtime package id.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-version-present" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageVersion)) -Severity "action-required" -Detail "runtimePackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-key-present" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey)) -Severity "action-required" -Detail "runtimePackageKey must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-kind-valid" -Passed (Test-ValueInSet -Value $sourceKind -AllowedValues @("nuget.org", "github-packages")) -Severity "action-required" -Detail "publicPackageSourceKind must be nuget.org or github-packages.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-url-public-not-local" -Passed (Test-SourceUrlIsPublicCandidate -Value $sourceUrl) -Severity "action-required" -Detail "publicPackageSourceUrl must be an HTTPS package source, not local/dry-run/direct .nupkg.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-page-url-nuget" -Passed (Test-NuGetPackagePageUrl -Value $managedPageUrl) -Severity "action-required" -Detail "managedPackagePageUrl must be a nuget.org package page.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-download-url-public" -Passed (Test-PublicDownloadUrl -Value $managedDownloadUrl) -Severity "action-required" -Detail "managedPackageDownloadUrl must be an HTTPS public download URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-page-url-nuget" -Passed (Test-NuGetPackagePageUrl -Value $runtimePageUrl) -Severity "action-required" -Detail "runtimePackagePageUrl must be a nuget.org package page.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-download-url-public" -Passed (Test-PublicDownloadUrl -Value $runtimeDownloadUrl) -Severity "action-required" -Detail "runtimePackageDownloadUrl must be an HTTPS public download URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-url-public" -Passed (Test-GitHubPublicUrl -Value $githubReleaseUrl) -Severity "action-required" -Detail "githubReleaseUrl must be a GitHub release URL for the full dependency package route.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-url-public" -Passed (Test-GitHubPublicUrl -Value $githubReleaseAssetUrl) -Severity "action-required" -Detail "githubReleaseAssetUrl must be a GitHub release asset URL for the full dependency package route.")) | Out-Null
$items.Add((New-ValidationItem -Id "download-command-present" -Passed (-not (Test-IsPlaceholder -Value $downloadCommand)) -Severity "action-required" -Detail "downloadCommand must capture the exact public package download command.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $downloadedAtUtc) -Severity "action-required" -Detail "downloadedAtUtc must be parseable.")) | Out-Null
$items.Add((New-ValidationItem -Id "captured-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $capturedAtUtc) -Severity "action-required" -Detail "capturedAtUtc must be parseable.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-name-present" -Passed (-not (Test-IsPlaceholder -Value $ownerName)) -Severity "action-required" -Detail "ownerName must be filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-reviewer-present" -Passed (-not (Test-IsPlaceholder -Value $ownerReviewer)) -Severity "action-required" -Detail "ownerReviewer must identify the person reviewing public download proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-github-actions-run-evidence-ready" -Passed $sourceGitHubActionsRunEvidenceReady -Severity "action-required" -Detail "Candidate must link to ready GitHub Actions run evidence validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-publish-result-ready" -Passed $sourceOwnerPublicPublishResultReady -Severity "action-required" -Detail "Candidate must link to a ready Owner public publish result validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-workflow-log-sha256" -Passed (Test-Sha256Format -Value $sourceWorkflowRunLogSha256) -Severity "action-required" -Detail "Source workflow run log SHA256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifact-manifest-sha256" -Passed (Test-Sha256Format -Value $sourceArtifactManifestSha256) -Severity "action-required" -Detail "Source artifact manifest SHA256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-url-match" -Passed (-not (Test-IsPlaceholder -Value $sourceOwnerPublicPackageUrl) -and $managedPageUrl.Equals($sourceOwnerPublicPackageUrl, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "managedPackagePageUrl must match the Owner public publish result package URL.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-version-match" -Passed (-not (Test-IsPlaceholder -Value $sourceOwnerPublicPackageVersion) -and $managedPackageVersion.Equals($sourceOwnerPublicPackageVersion, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "managedPackageVersion must match the Owner public publish result version.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-owner-public-package-sha256-match" -Passed (Test-Sha256Format -Value $sourceOwnerPublicPackageSha256) -Severity "action-required" -Detail "Owner public publish result SHA256 must be present for cross-checking.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $managedPath) -Severity "action-required" -Detail "downloadedManagedNupkgPath must be a downloaded .nupkg, not dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $runtimePath) -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must be a downloaded .nupkg, not dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-size-positive" -Passed (Test-PositiveInt64 -Value $managedSize) -Severity "action-required" -Detail "downloadedManagedNupkgSizeBytes must be a positive integer.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-size-positive" -Passed (Test-PositiveInt64 -Value $runtimeSize) -Severity "action-required" -Detail "downloadedRuntimeNupkgSizeBytes must be a positive integer.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-sha256-format" -Passed (Test-Sha256Format -Value $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgSha256 must be SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-sha256-format" -Passed (Test-Sha256Format -Value $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgSha256 must be SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-hash-match" -Passed (Test-FileHashMatches -Path $managedPath -Sha256 $managedSha) -Severity "action-required" -Detail "Downloaded managed package hash must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-hash-match" -Passed (Test-FileHashMatches -Path $runtimePath -Sha256 $runtimeSha) -Severity "action-required" -Detail "Downloaded runtime package hash must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-size-match" -Passed (Test-FileSizeMatches -Path $managedPath -ExpectedSize $managedSize) -Severity "action-required" -Detail "Downloaded managed package size must match downloadedManagedNupkgSizeBytes.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-size-match" -Passed (Test-FileSizeMatches -Path $runtimePath -ExpectedSize $runtimeSize) -Severity "action-required" -Detail "Downloaded runtime package size must match downloadedRuntimeNupkgSizeBytes.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-sha256" -Passed (Test-Sha256Format -Value $githubReleaseAssetSha) -Severity "action-required" -Detail "githubReleaseAssetSha256 must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-size-positive" -Passed (Test-PositiveInt64 -Value $githubReleaseAssetSize) -Severity "action-required" -Detail "githubReleaseAssetSizeBytes must be positive.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-hash-match" -Passed (Test-FileHashMatches -Path $githubReleaseAssetPath -Sha256 $githubReleaseAssetSha) -Severity "action-required" -Detail "GitHub release asset downloaded path must exist and match githubReleaseAssetSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "github-release-asset-size-match" -Passed (Test-FileSizeMatches -Path $githubReleaseAssetPath -ExpectedSize $githubReleaseAssetSize) -Severity "action-required" -Detail "GitHub release asset downloaded path size must match githubReleaseAssetSizeBytes.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-absent" -Passed ($forbiddenFindings.Count -eq 0) -Severity "blocker" -Detail $(if ($forbiddenFindings.Count -eq 0) { "No local feed, direct nupkg, dry-run, dashboard-only, artifact-only, ProjectReference, manual approval, queued workflow, missing runner, sidecar-only, or local test substitute was detected." } else { "Forbidden substitute(s): $($forbiddenFindings -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-download-required-field-contract" -Passed ($publicPackageDownloadProofRequiredFields.Count -ge 16) -Severity "blocker" -Detail "Public package download proof candidate must expose a stable required-field contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-download-rejected-substitute-contract" -Passed ($publicPackageDownloadProofRejectedSubstitutes.Count -ge 10) -Severity "blocker" -Detail "Public package download proof candidate must expose a stable rejected-substitute contract.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-package-download-proof-candidate"
}
elseif ($failedActionRequired.Count -eq 0) {
  "public-package-download-proof-candidate-ready"
}
else {
  "blocked-public-package-download-proof-required"
}

$validation = [pscustomobject]@{
  recordKind = "public-package-download-proof-candidate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateState = $candidateState
  sourceInputValidationState = $sourceInputValidationState
  sourceGitHubActionsRunEvidenceReady = $sourceGitHubActionsRunEvidenceReady
  sourceOwnerPublicPublishResultReady = $sourceOwnerPublicPublishResultReady
  sourceWorkflowRunLogSha256 = $sourceWorkflowRunLogSha256
  sourceArtifactManifestSha256 = $sourceArtifactManifestSha256
  sourceOwnerPublicPackageUrl = $sourceOwnerPublicPackageUrl
  sourceOwnerPublicPackageVersion = $sourceOwnerPublicPackageVersion
  sourceOwnerPublicPackageSha256 = $sourceOwnerPublicPackageSha256
  preReleaseReadinessMatrixPath = $preReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessBlockedLaneCount = $preReleaseReadinessBlockedLaneCount
  preReleasePublicPackageDownloadLaneReady = $preReleasePublicPackageDownloadLaneReady
  preReleaseCanPromotePublicProof = $preReleaseCanPromotePublicProof
  managedPackagePageUrl = $managedPageUrl
  managedPackageDownloadUrl = $managedDownloadUrl
  runtimePackagePageUrl = $runtimePageUrl
  runtimePackageDownloadUrl = $runtimeDownloadUrl
  githubReleaseUrl = $githubReleaseUrl
  githubReleaseAssetUrl = $githubReleaseAssetUrl
  githubReleaseAssetSha256 = $githubReleaseAssetSha
  forbiddenSubstituteFindings = @($forbiddenFindings)
  publicPackageDownloadProofRequiredFields = @($publicPackageDownloadProofRequiredFields)
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFields.Count
  publicPackageDownloadProofRejectedSubstitutes = @($publicPackageDownloadProofRejectedSubstitutes)
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstitutes.Count
  publicPackageDownloadProofSourceReadinessSignals = @($publicPackageDownloadProofSourceReadinessSignals)
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignals.Count
  candidateItemCount = $candidateItemCount
  readyCandidateCount = $readyCandidateCount
  blockedCandidateCount = $blockedCandidateCount
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  publicPackageDownloadProofCandidateReady = ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0)
  proofCandidateReady = ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Public package download proof candidate validation only. It is not runtime proof, not package-consumer runtime proof, not post-publish proof, not publish approval, not release close approval, not GitHub Actions proof, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "public-package-download-proof-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "public-package-download-proof-candidate-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Public Package Download Proof Candidate Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateState | ``$($validation.candidateState)`` |
| sourceInputValidationState | ``$($validation.sourceInputValidationState)`` |
| sourceGitHubActionsRunEvidenceReady | ``$($validation.sourceGitHubActionsRunEvidenceReady)`` |
| sourceOwnerPublicPublishResultReady | ``$($validation.sourceOwnerPublicPublishResultReady)`` |
| preReleaseReadinessMatrixPath | ``$($validation.preReleaseReadinessMatrixPath)`` |
| preReleaseReadinessMatrixState | ``$($validation.preReleaseReadinessMatrixState)`` |
| preReleaseReadinessBlockedLaneCount | ``$($validation.preReleaseReadinessBlockedLaneCount)`` |
| preReleasePublicPackageDownloadLaneReady | ``$($validation.preReleasePublicPackageDownloadLaneReady)`` |
| preReleaseCanPromotePublicProof | ``$($validation.preReleaseCanPromotePublicProof)`` |
| managedPackagePageUrl | ``$($validation.managedPackagePageUrl)`` |
| managedPackageDownloadUrl | ``$($validation.managedPackageDownloadUrl)`` |
| runtimePackagePageUrl | ``$($validation.runtimePackagePageUrl)`` |
| runtimePackageDownloadUrl | ``$($validation.runtimePackageDownloadUrl)`` |
| githubReleaseUrl | ``$($validation.githubReleaseUrl)`` |
| githubReleaseAssetUrl | ``$($validation.githubReleaseAssetUrl)`` |
| candidateItemCount | ``$($validation.candidateItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| publicPackageDownloadProofRequiredFieldCount | ``$($validation.publicPackageDownloadProofRequiredFieldCount)`` |
| publicPackageDownloadProofRejectedSubstituteCount | ``$($validation.publicPackageDownloadProofRejectedSubstituteCount)`` |
| publicPackageDownloadProofSourceReadinessSignalCount | ``$($validation.publicPackageDownloadProofSourceReadinessSignalCount)`` |
| publicPackageDownloadProofCandidateReady | ``$($validation.publicPackageDownloadProofCandidateReady)`` |
| proofCandidateReady | ``$($validation.proofCandidateReady)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| isRuntimeExecutionProof | ``$($validation.isRuntimeExecutionProof)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |
| isReleaseCloseProof | ``$($validation.isReleaseCloseProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public package download proof candidate validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Public package download proof candidate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
