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
$managedPath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgPath" -DefaultValue "")
$runtimePath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
$managedSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgSha256" -DefaultValue "")
$runtimeSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
$dryRunPath = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunArtifactPath" -DefaultValue "")
$dryRunSha = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunManagedNupkgSha256" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "public-package-download-proof-input") -Severity "blocker" -Detail "recordKind must be public-package-download-proof-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $usesPublishToken -and -not $canPublishPublicly -and -not $canPublishGitHubPackages -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Public package download proof input validation must not publish, use token, approve publication, or close issues.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-claims-false" -Passed (-not $canClaimRuntimeProof -and -not $canClaimPackageConsumerRuntimeProof -and -not $isPackageConsumerRuntimeProof -and -not $isPostPublishProof) -Severity "blocker" -Detail "Download proof input is not runtime proof, package-consumer runtime proof, or post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-authorization-not-default-true" -Passed (-not $ownerAuthorizationState.Equals("true", [StringComparison]::OrdinalIgnoreCase) -and -not $ownerAuthorizationState.Equals("authorized", [StringComparison]::OrdinalIgnoreCase) -and -not $ownerAuthorizationState.Equals("approved", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "ownerAuthorizationState must not default to true/authorized/approved.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-kind-valid" -Passed (Test-ValueInSet -Value $sourceKind -AllowedValues @("nuget.org", "github-packages", "private-feed")) -Severity "action-required" -Detail "publicPackageSourceKind must be nuget.org, github-packages, or private-feed.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-url-public-not-local" -Passed (Test-SourceUrlIsPublicCandidate -Value $sourceUrl) -Severity "action-required" -Detail "publicPackageSourceUrl must be an HTTPS package source, not a local path, artifacts path, direct .nupkg, template placeholder, or dry-run artifact.")) | Out-Null
$items.Add((New-ValidationItem -Id "download-command-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "downloadCommand" -DefaultValue ""))) -Severity "action-required" -Detail "downloadCommand must capture the exact public package download command.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "downloadedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "downloadedAtUtc must be a real parseable DateTimeOffset.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-name-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerName" -DefaultValue ""))) -Severity "action-required" -Detail "ownerName must be filled by the owner collecting real public package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-version-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "managedPackageVersion" -DefaultValue ""))) -Severity "action-required" -Detail "managedPackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-version-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "runtimePackageVersion" -DefaultValue ""))) -Severity "action-required" -Detail "runtimePackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $managedPath -DryRunPath $dryRunPath) -Severity "action-required" -Detail "downloadedManagedNupkgPath must be a real downloaded .nupkg path and not a dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $runtimePath -DryRunPath "") -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must be a real downloaded .nupkg path and not an artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-sha256-format" -Passed (Test-Sha256Format -Value $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-sha256-format" -Passed (Test-Sha256Format -Value $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgSha256 must be a 64-character SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-hash-match" -Passed (Test-FileHashMatches -Path $managedPath -Sha256 $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgPath must exist and match downloadedManagedNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-hash-match" -Passed (Test-FileHashMatches -Path $runtimePath -Sha256 $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must exist and match downloadedRuntimeNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-sha-not-substituted" -Passed (-not (Test-Sha256Format -Value $dryRunSha) -or -not $managedSha.Equals($dryRunSha, [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub Actions dry-run SHA256 must not be reused as downloaded public package SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "template-default-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "validationState" -DefaultValue "") -ne "ready" -and -not $isPublishedPackageProof) -Severity "blocker" -Detail "Template/default input must stay blocked until owner supplies real download evidence.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
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
