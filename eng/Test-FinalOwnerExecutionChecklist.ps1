[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-checklist.json",
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

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
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
  throw "Final owner execution checklist not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$steps = @((Get-PropertyOrDefault -Object $record -Name "executionSteps" -DefaultValue @()))
$stepIds = Convert-ToStringArray ($steps | ForEach-Object { Get-PropertyOrDefault -Object $_ -Name "stepId" -DefaultValue "" })
$requiredCapture = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredCapture" -DefaultValue @())
$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$dualPackageRoutes = @((Get-PropertyOrDefault -Object $record -Name "dualPackageRoutes" -DefaultValue @()))
$dualPackageRouteOwnerActions = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "dualPackageRouteOwnerActions" -DefaultValue @())
$dualPackageExternalProofMissingReasons = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "dualPackageExternalProofMissingReasons" -DefaultValue @())
$dualPackagePostPublishProofMissingReasons = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "dualPackagePostPublishProofMissingReasons" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-checklist") -Severity "blocker" -Detail "recordKind must be final-owner-execution-checklist.")) | Out-Null
$items.Add((New-ValidationItem -Id "checklist-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "") -eq "blocked-final-owner-execution-checklist-real-owner-input-required") -Severity "blocker" -Detail "Checklist must remain blocked until real Owner input is imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "six-steps" -Passed ($steps.Count -eq 6 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Checklist must expose five blocker steps plus final close validation.")) | Out-Null

foreach ($expected in @("01-owner-authorization", "02-clean-external-package-consumer", "03-linux-runner-proof", "04-real-model-runtime", "05-post-publish-verification", "06-final-close-validation")) {
  $items.Add((New-ValidationItem -Id "step-$expected-present" -Passed ($stepIds -contains $expected) -Severity "blocker" -Detail "Step $expected must be present.")) | Out-Null
}

foreach ($capture in @("stdout", "stderr", "log", "hash", "SHA256", "exitCode", "host identity")) {
  $items.Add((New-ValidationItem -Id "capture-$capture-present" -Passed ($requiredCapture -contains $capture) -Severity "blocker" -Detail "Required capture $capture must be present.")) | Out-Null
}

$commands = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "commands" -DefaultValue @())
$strictValidators = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "strictValidators" -DefaultValue @())
$containsNuGetPushCommand = @($commands | Where-Object { $_ -match "dotnet\s+nuget\s+push" }).Count -gt 0
$items.Add((New-ValidationItem -Id "does-not-contain-nuget-push-command" -Passed (-not $containsNuGetPushCommand) -Severity "blocker" -Detail "Checklist commands must not execute or prescribe dotnet nuget push automation.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-consumer-proof-strong-gate" -Passed (@($strictValidators | Where-Object { $_.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Package consumer proof validator must require Strict, existing logs, and FailOnNotProof.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-strong-gate" -Passed (@($strictValidators | Where-Object { $_.Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Post-publish proof validator must require existing logs and FailOnNotProof.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Checklist must not publish, promote proof, or close release.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-source-artifact" -Passed ($sourceArtifacts -contains "artifacts/final-release/dual-package-publish-preflight-matrix.json") -Severity "blocker" -Detail "Checklist must cite the dual-package publish preflight matrix source artifact.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-route-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "dualPackageRouteCount" -DefaultValue 0) -eq 2 -and $dualPackageRoutes.Count -eq 2) -Severity "blocker" -Detail "Checklist must surface both NuGet managed and GitHub Packages bridge-only routes; vendor runtime package routes are forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-owner-actions" -Passed (($dualPackageRouteOwnerActions -contains "owner-authorize-public-nuget-publish-and-import-clean-external-consumer-proof") -and ($dualPackageRouteOwnerActions -contains "owner-authorize-github-packages-publish-and-import-credentialed-clean-runtime-proof")) -Severity "blocker" -Detail "Checklist must surface next owner actions for both publish routes.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-external-proof-gaps" -Passed (($dualPackageExternalProofMissingReasons -contains "public-package-download-and-clean-consumer-runtime-proof-missing") -and ($dualPackageExternalProofMissingReasons -contains "github-packages-restore-source-runtime-dll-resolution-clean-smoke-missing")) -Severity "blocker" -Detail "Checklist must carry external proof missing reasons from the dual-package matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-post-publish-proof-gaps" -Passed (($dualPackagePostPublishProofMissingReasons -contains "post-publish-clean-consumer-proof-missing") -and ($dualPackagePostPublishProofMissingReasons -contains "post-publish-github-packages-clean-consumer-proof-missing")) -Severity "blocker" -Detail "Checklist must carry post-publish proof missing reasons from the dual-package matrix.")) | Out-Null
$unsafeDualPackageRoutes = @($dualPackageRoutes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishGitHubPackages" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true) -or [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsSubstituteProof" -DefaultValue $true) })
$items.Add((New-ValidationItem -Id "dual-package-routes-non-proof" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "dualPackageAcceptsSubstituteProof" -DefaultValue $true)) -and $unsafeDualPackageRoutes.Count -eq 0) -Severity "blocker" -Detail "Dual-package route summaries must not publish, claim package consumer proof, or accept substitute proof.")) | Out-Null

foreach ($step in $steps) {
  $id = [string](Get-PropertyOrDefault -Object $step -Name "stepId" -DefaultValue "")
  $fields = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "requiredBackfillFields" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "expectedArtifacts" -DefaultValue @())
  $strictValidator = [string](Get-PropertyOrDefault -Object $step -Name "strictValidator" -DefaultValue "")
  $stepCapture = Convert-ToStringArray (Get-PropertyOrDefault -Object $step -Name "requiredCapture" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $step -Name "boundary" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "step-$id-fields" -Passed ($fields.Count -ge 7) -Severity "blocker" -Detail "Step $id must list concrete backfill fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-artifacts" -Passed ($expectedArtifacts.Count -ge 1) -Severity "blocker" -Detail "Step $id must list expected artifacts.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-validator" -Passed ($strictValidator -like "pwsh*Test-*.ps1*") -Severity "blocker" -Detail "Step $id must list a strict validator.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-capture" -Passed (@("stdout", "stderr", "log", "SHA256", "exitCode", "host identity") | Where-Object { $stepCapture -contains $_ } | Measure-Object).Count -ge 6 -Severity "blocker" -Detail "Step $id must require stdout/stderr/log/SHA256/exitCode/host identity capture.")) | Out-Null
  $items.Add((New-ValidationItem -Id "step-$id-boundary" -Passed ($boundary.Contains("not runtime proof") -and $boundary.Contains("not package push")) -Severity "blocker" -Detail "Step $id must state non-proof boundary.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockers -eq 0) { "blocked-final-owner-execution-checklist-real-owner-input-required" } else { "final-owner-execution-checklist-invalid" }

$validation = [ordered]@{
  recordKind = "final-owner-execution-checklist-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  checklistState = [string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "")
  stepCount = $steps.Count
  blockedStepCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = 6
  findingCount = $failedItems.Count
  findings = @($failedItems)
  validationItems = @($items.ToArray())
  dualPackageRouteCount = $dualPackageRoutes.Count
  dualPackageOwnerActionCount = $dualPackageRouteOwnerActions.Count
  dualPackageExternalProofMissingReasonCount = $dualPackageExternalProofMissingReasons.Count
  dualPackagePostPublishProofMissingReasonCount = $dualPackagePostPublishProofMissingReasons.Count
  performsPublish = $false
  notExecutedByAutomation = $true
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final owner execution checklist validation checks owner commands and required capture fields only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-checklist-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-checklist-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 18)
$rows = foreach ($item in $items) {
  "| ``$($item.id)`` | ``$($item.severity)`` | ``$($item.passed)`` | $($item.detail.Replace("|", "\|")) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Owner Execution Checklist Validation",
  "",
  "- validationState: $validationState",
  "- failedBlockerCount: $failedBlockers",
  "- failedActionRequiredCount: 6",
  "- dualPackageRouteCount: $($dualPackageRoutes.Count)",
  "- dualPackageOwnerActionCount: $($dualPackageRouteOwnerActions.Count)",
  "- canPublishPublicly: False",
  "- canCloseReleaseIssue: False",
  "",
  "| Item | Severity | Passed | Detail |",
  "|---|---|---:|---|",
  @($rows)
)

Write-Host "ValidationState=$validationState FailedBlockerCount=$failedBlockers"
if ($Strict -and $failedBlockers -gt 0) {
  throw "Final owner execution checklist validation failed with $failedBlockers blocker(s)."
}
