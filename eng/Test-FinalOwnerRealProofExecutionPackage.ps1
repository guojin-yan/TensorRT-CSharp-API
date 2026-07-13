[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-real-proof-execution-package.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("InputPath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerRealProofExecutionPackage.ps1") -RepositoryRoot $RepositoryRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordText = $record | ConvertTo-Json -Depth 20
$steps = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "ownerExecutionSequence" -DefaultValue @()))
$stepIds = @($steps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$forbidden = @((Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })

$recordKindOk = [string](Get-PropertyOrDefault $record "recordKind" "") -eq "final-owner-real-proof-execution-package"
$blockedNonProofOk = [string](Get-PropertyOrDefault $record "packageState" "") -eq "blocked-final-owner-real-proof-execution-required" -and -not [bool](Get-PropertyOrDefault $record "passed" $true) -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false)
$nonProofFlagsOk = -not [bool](Get-PropertyOrDefault $record "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $record "performsRuntimeExecution" $true) -and -not [bool](Get-PropertyOrDefault $record "canPromoteRuntimeProof" $true) -and -not [bool](Get-PropertyOrDefault $record "canPublishPublicly" $true) -and -not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isPostPublishProof" $true) -and -not [bool](Get-PropertyOrDefault $record "isReleaseCloseProof" $true)
$requiredStepsOk = @(@(
    "create-repository-external-clean-consumer-workspace",
    "restore-from-real-public-package-source",
    "build-clean-consumer",
    "run-clean-consumer-smoke",
    "collect-logs-and-native-assets",
    "compute-sha256-manifest",
    "fill-external-clean-consumer-owner-input",
    "run-external-clean-consumer-import-and-strict-validator",
    "download-public-published-packages",
    "fill-post-publish-owner-input",
    "run-post-publish-import-and-strict-validator",
    "fill-rollback-review",
    "fill-final-close-decision",
    "refresh-release-evidence-classification-and-close-readiness"
  ) | Where-Object { $stepIds -notcontains $_ }).Count -eq 0 -and $steps.Count -ge 14
$forbiddenOk = @(@("local feed", "ProjectReference", "direct .nupkg", "build-only", "template", "candidate", "command pack", "failedBlockerCount=0", "pre-publish smoke reused as post-publish proof") | Where-Object { $forbidden -notcontains $_ }).Count -eq 0
$commandContentOk = $recordText.Contains("dotnet restore", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("dotnet build", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("dotnet run", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("Get-FileHash", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("Import-ExternalCleanConsumerExecutionResult.ps1", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("Import-PostPublishCleanConsumerProofResult.ps1", [StringComparison]::OrdinalIgnoreCase) -and $recordText.Contains("Test-FinalOwnerRealProofConvergenceGate.ps1", [StringComparison]::OrdinalIgnoreCase)
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$boundaryOk = $boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" $recordKindOk "blocker" "recordKind must match final-owner-real-proof-execution-package.")) | Out-Null
$items.Add((New-ValidationItem "blocked-non-proof" $blockedNonProofOk "blocker" "Execution package must remain blocked, failed, and owner-action-required.")) | Out-Null
$items.Add((New-ValidationItem "non-proof-flags" $nonProofFlagsOk "blocker" "Execution package cannot publish, execute smoke, close, or claim proof.")) | Out-Null
$items.Add((New-ValidationItem "required-steps" $requiredStepsOk "blocker" "Execution sequence must include all Owner proof convergence steps.")) | Out-Null
$items.Add((New-ValidationItem "forbidden-substitutes" $forbiddenOk "blocker" "Forbidden substitutes must be called out explicitly.")) | Out-Null
$items.Add((New-ValidationItem "command-content" $commandContentOk "blocker" "Command content must route through strict import/validator and evidence refresh commands.")) | Out-Null
$items.Add((New-ValidationItem "boundary" $boundaryOk "blocker" "Boundary must preserve all non-proof classifications.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "final-owner-real-proof-execution-package-ready-non-proof" } else { "blocked-final-owner-real-proof-execution-package-invalid" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-real-proof-execution-package-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  executionStepCount = $steps.Count
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($validationItems)
  boundary = "Validation checks the final Owner execution package shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-real-proof-execution-package-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-real-proof-execution-package-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Owner Real Proof Execution Package Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- executionStepCount: ``$($steps.Count)``",
  "- passed: ``False``",
  "",
  "| ID | Passed | Severity | Detail |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "FinalOwnerRealProofExecutionPackageValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final Owner real proof execution package validation failed."
}
