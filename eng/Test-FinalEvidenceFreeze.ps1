[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-evidence-freeze.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final evidence freeze not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$bundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "bundlePerformsPublish" -DefaultValue $true)
$bundleCanPublish = [bool](Get-PropertyOrDefault -Object $record -Name "bundleCanPublishPublicly" -DefaultValue $true)
$bundleCanClose = [bool](Get-PropertyOrDefault -Object $record -Name "bundleCanCloseReleaseIssue" -DefaultValue $true)
$freezeArtifacts = @((Get-PropertyOrDefault -Object $record -Name "freezeArtifacts" -DefaultValue @()))
$missingSourceArtifactCount = [int](Get-PropertyOrDefault -Object $record -Name "missingSourceArtifactCount" -DefaultValue -1)
$missingHashCount = [int](Get-PropertyOrDefault -Object $record -Name "missingHashCount" -DefaultValue -1)
$packageEvidenceClassificationContract = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "packageEvidenceClassificationContract" -DefaultValue @()))
$classificationIds = @($packageEvidenceClassificationContract | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$acceptedPublicPackageEvidenceKindIds = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "acceptedPublicPackageEvidenceKindIds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$rejectedPackageEvidenceKindIds = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "rejectedPackageEvidenceKindIds" -DefaultValue @()) | ForEach-Object { [string]$_ })
$requiredAcceptedKinds = @("public-package-download-proof", "post-publish-clean-consumer-proof")
$requiredRejectedKinds = @("local-feed-proof", "direct-nupkg-proof", "project-reference-proof")
$missingAcceptedKinds = @($requiredAcceptedKinds | Where-Object { $acceptedPublicPackageEvidenceKindIds -notcontains $_ -or $classificationIds -notcontains $_ })
$missingRejectedKinds = @($requiredRejectedKinds | Where-Object { $rejectedPackageEvidenceKindIds -notcontains $_ -or $classificationIds -notcontains $_ })
$invalidClassificationRows = @($packageEvidenceClassificationContract | Where-Object {
  $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  $validator = [string](Get-PropertyOrDefault -Object $_ -Name "requiredValidator" -DefaultValue "")
  $accepted = [bool](Get-PropertyOrDefault -Object $_ -Name "acceptedAsPublicProof" -DefaultValue $false)
  [string]::IsNullOrWhiteSpace($id) -or
    [string]::IsNullOrWhiteSpace($validator) -or
    ($requiredRejectedKinds -contains $id -and $accepted) -or
    ($requiredAcceptedKinds -contains $id -and -not $accepted)
})

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "final-evidence-freeze") -Severity "blocker" -Detail "recordKind must be final-evidence-freeze.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Final evidence freeze must not publish, approve publication, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "bundle-side-effects-still-false" -Passed (-not $bundlePerformsPublish -and -not $bundleCanPublish -and -not $bundleCanClose) -Severity "blocker" -Detail "Freeze must not mask a bundle that claims publish or close permission.")) | Out-Null
$items.Add((New-ValidationItem -Id "freeze-artifacts-present" -Passed ($freezeArtifacts.Count -ge 8) -Severity "blocker" -Detail "Freeze must include all required source artifact hashes.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-missing-source-artifacts" -Passed ($missingSourceArtifactCount -eq 0) -Severity "blocker" -Detail "All freeze source artifacts must exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-missing-hashes" -Passed ($missingHashCount -eq 0) -Severity "blocker" -Detail "All freeze source artifacts must have SHA256 values.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-evidence-classification-contract" -Passed ($packageEvidenceClassificationContract.Count -ge 5 -and $invalidClassificationRows.Count -eq 0) -Severity "blocker" -Detail "Freeze must classify local feed, direct nupkg, ProjectReference, public download proof, and post-publish clean consumer proof with strict validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "rejected-package-substitutes" -Passed ($missingRejectedKinds.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "rejectedPackageEvidenceKindCount" -DefaultValue 0) -ge 3) -Severity "blocker" -Detail "Freeze must reject local feed, direct nupkg, and ProjectReference as public/package-consumer proof substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "accepted-public-package-proof-kinds" -Passed ($missingAcceptedKinds.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "acceptedPublicPackageEvidenceKindCount" -DefaultValue 0) -ge 2) -Severity "blocker" -Detail "Freeze must identify public package download proof and post-publish clean consumer proof as the only accepted public package proof categories.")) | Out-Null

foreach ($artifact in $freezeArtifacts) {
  $id = [string](Get-PropertyOrDefault -Object $artifact -Name "id" -DefaultValue "unknown")
  $relativePath = [string](Get-PropertyOrDefault -Object $artifact -Name "path" -DefaultValue "")
  $expectedHash = [string](Get-PropertyOrDefault -Object $artifact -Name "sha256" -DefaultValue "")
  $resolvedPath = Resolve-RepositoryPath -Path $relativePath
  $hashMatches = $false
  if ((Test-Path -LiteralPath $resolvedPath -PathType Leaf) -and $expectedHash -match "^[0-9a-fA-F]{64}$") {
    $actualHash = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
    $hashMatches = $actualHash.Equals($expectedHash, [StringComparison]::OrdinalIgnoreCase)
  }

  $items.Add((New-ValidationItem -Id "hash-$id" -Passed $hashMatches -Severity "blocker" -Detail "$relativePath must exist and match the frozen SHA256.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "final-evidence-freeze-valid-blocked-owner-action-required"
}
else {
  "blocked-final-evidence-freeze-invalid"
}

$validation = [pscustomobject]@{
  recordKind = "final-evidence-freeze-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  freezeState = [string](Get-PropertyOrDefault -Object $record -Name "freezeState" -DefaultValue "")
  isValidFreeze = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  sourceArtifactCount = $freezeArtifacts.Count
  missingSourceArtifactCount = $missingSourceArtifactCount
  missingHashCount = $missingHashCount
  packageEvidenceClassificationContractCount = $packageEvidenceClassificationContract.Count
  rejectedPackageEvidenceKindCount = $rejectedPackageEvidenceKindIds.Count
  acceptedPublicPackageEvidenceKindCount = $acceptedPublicPackageEvidenceKindIds.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates final evidence freeze integrity only. It cannot publish packages, approve publication, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "final-evidence-freeze-validation.json"
$markdownPath = Join-Path $OutputRoot "final-evidence-freeze-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Final Evidence Freeze Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| freezeState | ``$($validation.freezeState)`` |
| isValidFreeze | ``$($validation.isValidFreeze)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| sourceArtifactCount | ``$($validation.sourceArtifactCount)`` |
| packageEvidenceClassificationContractCount | ``$($validation.packageEvidenceClassificationContractCount)`` |
| rejectedPackageEvidenceKindCount | ``$($validation.rejectedPackageEvidenceKindCount)`` |
| acceptedPublicPackageEvidenceKindCount | ``$($validation.acceptedPublicPackageEvidenceKindCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final evidence freeze validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final evidence freeze validation failed with $($failedBlockers.Count) blocker(s)."
}
