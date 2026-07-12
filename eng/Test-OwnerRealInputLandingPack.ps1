[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-input-landing-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner real input landing pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$expectedBlockers = @(
  "owner-authorization",
  "package-consumer-runtime",
  "linux-runner-proof",
  "real-model-runtime",
  "post-publish-verification"
)
$requiredForbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "dry-run",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "build-only",
  "parse-only",
  "sidecar-only",
  "template"
)

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$landingState = [string](Get-PropertyOrDefault -Object $record -Name "landingState" -DefaultValue "")
$blockers = @((Get-PropertyOrDefault -Object $record -Name "blockers" -DefaultValue @()))
$blockerIds = Convert-ToStringArray ($blockers | ForEach-Object { Get-PropertyOrDefault -Object $_ -Name "blockerId" -DefaultValue "" })
$allForbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-real-input-landing-pack") -Severity "blocker" -Detail "recordKind must be owner-real-input-landing-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "landing-state" -Passed ($landingState -eq "blocked-owner-real-input-required") -Severity "blocker" -Detail "Landing pack must remain blocked until real Owner input exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "five-blockers" -Passed ($blockers.Count -eq 5 -and [int](Get-PropertyOrDefault -Object $record -Name "blockerCount" -DefaultValue 0) -eq 5 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedBlockerCount" -DefaultValue 0) -eq 5) -Severity "blocker" -Detail "Landing pack must expose exactly five final blockers.")) | Out-Null

foreach ($expected in $expectedBlockers) {
  $items.Add((New-ValidationItem -Id "blocker-$expected-present" -Passed ($blockerIds -contains $expected) -Severity "blocker" -Detail "Expected final blocker $expected must be present.")) | Out-Null
}

foreach ($marker in $requiredForbiddenSubstitutes) {
  $items.Add((New-ValidationItem -Id "forbidden-$marker-present" -Passed ($allForbiddenSubstitutes -contains $marker) -Severity "blocker" -Detail "Forbidden substitute marker $marker must be present.")) | Out-Null
}

$flagCheck = (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and
  (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed $flagCheck -Severity "blocker" -Detail "Landing pack must not publish, promote proof, or close release.")) | Out-Null

$boundaryCheck = $boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not publish approval") -and $boundary.Contains("not release close approval") -and $boundary.Contains("not package push")
$items.Add((New-ValidationItem -Id "top-level-boundary" -Passed $boundaryCheck -Severity "blocker" -Detail "Landing pack boundary must state all non-proof limits.")) | Out-Null

foreach ($blocker in $blockers) {
  $id = [string](Get-PropertyOrDefault -Object $blocker -Name "blockerId" -DefaultValue "")
  $requiredOwnerInputFiles = Convert-ToStringArray (Get-PropertyOrDefault -Object $blocker -Name "requiredOwnerInputFiles" -DefaultValue @())
  $requiredFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $blocker -Name "requiredFields" -DefaultValue @())
  $strictValidator = [string](Get-PropertyOrDefault -Object $blocker -Name "strictValidator" -DefaultValue "")
  $forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $blocker -Name "forbiddenSubstitutes" -DefaultValue @())
  $promotionBlockedUntil = [string](Get-PropertyOrDefault -Object $blocker -Name "promotionBlockedUntil" -DefaultValue "")
  $nonProofBoundary = [string](Get-PropertyOrDefault -Object $blocker -Name "nonProofBoundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "blocker-$id-required-files" -Passed ($requiredOwnerInputFiles.Count -ge 2) -Severity "blocker" -Detail "Blocker $id must list required owner input files.")) | Out-Null
  $items.Add((New-ValidationItem -Id "blocker-$id-required-fields" -Passed ($requiredFields.Count -ge 8) -Severity "blocker" -Detail "Blocker $id must list concrete required fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "blocker-$id-strict-validator" -Passed ($strictValidator -like "pwsh*Test-*.ps1*") -Severity "blocker" -Detail "Blocker $id must name a strict validator command.")) | Out-Null
  if ($id -eq "package-consumer-runtime") {
    $items.Add((New-ValidationItem -Id "blocker-$id-package-consumer-strong-gate" -Passed ($strictValidator.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictValidator.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $strictValidator.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $strictValidator.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Package consumer runtime blocker must require Strict, existing logs, and FailOnNotProof.")) | Out-Null
  }
  if ($id -eq "post-publish-verification") {
    $items.Add((New-ValidationItem -Id "blocker-$id-post-publish-strong-gate" -Passed ($strictValidator.Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictValidator.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $strictValidator.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Post-publish verification blocker must require existing logs and FailOnNotProof.")) | Out-Null
  }
  $items.Add((New-ValidationItem -Id "blocker-$id-promotion-blocked" -Passed ($promotionBlockedUntil -match "Owner|validator|真实") -Severity "blocker" -Detail "Blocker $id must state promotionBlockedUntil.")) | Out-Null
  $items.Add((New-ValidationItem -Id "blocker-$id-non-proof-boundary" -Passed ($nonProofBoundary.Contains("not runtime proof") -and $nonProofBoundary.Contains("not package push")) -Severity "blocker" -Detail "Blocker $id must state non-proof boundary.")) | Out-Null
  foreach ($marker in $requiredForbiddenSubstitutes) {
    $items.Add((New-ValidationItem -Id "blocker-$id-forbidden-$marker" -Passed ($forbiddenSubstitutes -contains $marker) -Severity "blocker" -Detail "Blocker $id must include forbidden substitute $marker.")) | Out-Null
  }
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockers -eq 0) { "blocked-owner-real-input-required" } else { "owner-real-input-landing-pack-invalid" }

$validation = [ordered]@{
  recordKind = "owner-real-input-landing-pack-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  landingState = $landingState
  blockerCount = $blockers.Count
  expectedBlockerCount = 5
  requiredForbiddenSubstituteCount = $requiredForbiddenSubstitutes.Count
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = 5
  findingCount = $failedItems.Count
  findings = @($failedItems)
  validationItems = @($items.ToArray())
  performsPublish = $false
  notExecutedByAutomation = $true
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real input landing pack validation checks shape and non-substitute boundaries only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-input-landing-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-input-landing-pack-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$($item.id)`` | ``$($item.severity)`` | ``$($item.passed)`` | $($item.detail.Replace("|", "\|")) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Real Input Landing Pack Validation",
  "",
  "- validationState: $validationState",
  "- failedBlockerCount: $failedBlockers",
  "- failedActionRequiredCount: 5",
  "- canPublishPublicly: False",
  "- canCloseReleaseIssue: False",
  "",
  "| Item | Severity | Passed | Detail |",
  "|---|---|---:|---|",
  @($rows)
)

Write-Host "ValidationState=$validationState FailedBlockerCount=$failedBlockers"
if ($Strict -and $failedBlockers -gt 0) {
  throw "Owner real input landing pack validation failed with $failedBlockers blocker(s)."
}
