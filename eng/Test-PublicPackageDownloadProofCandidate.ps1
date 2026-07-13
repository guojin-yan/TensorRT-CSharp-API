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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

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
$managedPath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgPath" -DefaultValue "")
$runtimePath = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgPath" -DefaultValue "")
$managedSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgSha256" -DefaultValue "")
$runtimeSha = [string](Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")
$downloadCommand = [string](Get-PropertyOrDefault -Object $record -Name "downloadCommand" -DefaultValue "")
$downloadedAtUtc = [string](Get-PropertyOrDefault -Object $record -Name "downloadedAtUtc" -DefaultValue "")
$ownerName = [string](Get-PropertyOrDefault -Object $record -Name "ownerName" -DefaultValue "")

$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$usesPublishToken = [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canPublishGitHubPackages = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishGitHubPackages" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$canClaimRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canClaimRuntimeProof" -DefaultValue $true)
$canClaimPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true)
$canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)
$isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)
$isPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true)
$isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)
$isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)
$isGitHubActionsProof = [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $true)

$sourceReady = $sourceInputValidationState -eq "public-package-download-proof-input-ready" -and $sourcePublicPackageDownloadProofReady
$candidateCountsConsistent = if ($sourceReady) {
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
$items.Add((New-ValidationItem -Id "managed-package-id-present" -Passed ($managedPackageId -eq "JYPPX.TensorRT.CSharp.API") -Severity "action-required" -Detail "Managed package id must match the public managed package.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-version-present" -Passed (-not (Test-IsPlaceholder -Value $managedPackageVersion)) -Severity "action-required" -Detail "managedPackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-id-present" -Passed ($runtimePackageId.StartsWith("JYPPX.TensorRT.CSharp.API.runtime.", [StringComparison]::Ordinal)) -Severity "action-required" -Detail "Runtime package id must be a public runtime package id.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-version-present" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageVersion)) -Severity "action-required" -Detail "runtimePackageVersion must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-key-present" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey)) -Severity "action-required" -Detail "runtimePackageKey must be real.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-kind-valid" -Passed (Test-ValueInSet -Value $sourceKind -AllowedValues @("nuget.org", "github-packages", "private-feed")) -Severity "action-required" -Detail "publicPackageSourceKind must be nuget.org, github-packages, or private-feed.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-url-public-not-local" -Passed (Test-SourceUrlIsPublicCandidate -Value $sourceUrl) -Severity "action-required" -Detail "publicPackageSourceUrl must be an HTTPS package source, not local/dry-run/direct .nupkg.")) | Out-Null
$items.Add((New-ValidationItem -Id "download-command-present" -Passed (-not (Test-IsPlaceholder -Value $downloadCommand)) -Severity "action-required" -Detail "downloadCommand must capture the exact public package download command.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $downloadedAtUtc) -Severity "action-required" -Detail "downloadedAtUtc must be parseable.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-name-present" -Passed (-not (Test-IsPlaceholder -Value $ownerName)) -Severity "action-required" -Detail "ownerName must be filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $managedPath) -Severity "action-required" -Detail "downloadedManagedNupkgPath must be a downloaded .nupkg, not dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-path-public-download" -Passed (Test-DownloadedPathIsPublicDownloadCandidate -Path $runtimePath) -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must be a downloaded .nupkg, not dry-run/artifacts path.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-sha256-format" -Passed (Test-Sha256Format -Value $managedSha) -Severity "action-required" -Detail "downloadedManagedNupkgSha256 must be SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-sha256-format" -Passed (Test-Sha256Format -Value $runtimeSha) -Severity "action-required" -Detail "downloadedRuntimeNupkgSha256 must be SHA256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-managed-hash-match" -Passed (Test-FileHashMatches -Path $managedPath -Sha256 $managedSha) -Severity "action-required" -Detail "Downloaded managed package hash must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-hash-match" -Passed (Test-FileHashMatches -Path $runtimePath -Sha256 $runtimeSha) -Severity "action-required" -Detail "Downloaded runtime package hash must match.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
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
| candidateItemCount | ``$($validation.candidateItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
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
