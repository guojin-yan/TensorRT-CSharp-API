[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-release-execution-package.json",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Test-ContainsAny {
  param(
    [string[]]$Values,
    [string[]]$Needles
  )

  $joined = ($Values -join "`n")
  foreach ($needle in $Needles) {
    if ($joined.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
      return $false
    }
  }

  return $true
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner release execution package not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$packageState = [string](Get-PropertyOrDefault -Object $record -Name "packageState" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$requiresHumanOwner = [bool](Get-PropertyOrDefault -Object $record -Name "requiresHumanOwner" -DefaultValue $false)
$executionSteps = @((Get-PropertyOrDefault -Object $record -Name "executionSteps" -DefaultValue @()))
$manualPublishPlaceholders = @((Get-PropertyOrDefault -Object $record -Name "manualPublishPlaceholders" -DefaultValue @()))
$mustNotSubstitute = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "mustNotSubstitute" -DefaultValue @())
$requiredOwnerInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputs" -DefaultValue @())
$validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "validatorCommands" -DefaultValue @())
$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$oneScreenReleaseHoldChecklist = @((Get-PropertyOrDefault -Object $record -Name "oneScreenReleaseHoldChecklist" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-release-execution-package") -Severity "blocker" -Detail "recordKind must be owner-release-execution-package.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Owner release execution package must not publish, approve publication, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "human-owner-required" -Passed $requiresHumanOwner -Severity "blocker" -Detail "requiresHumanOwner must remain true.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-package-state" -Passed ($packageState -eq "blocked-real-proof-required") -Severity "action-required" -Detail "packageState should remain blocked-real-proof-required until real public-channel proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-steps-present" -Passed ($executionSteps.Count -ge 5) -Severity "action-required" -Detail "Execution package must include multiple owner execution steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "one-screen-release-hold-present" -Passed ($oneScreenReleaseHoldChecklist.Count -ge 5) -Severity "action-required" -Detail "oneScreenReleaseHoldChecklist should summarize the remaining owner blockers.")) | Out-Null
$items.Add((New-ValidationItem -Id "manual-publish-placeholders-present" -Passed ($manualPublishPlaceholders.Count -ge 1) -Severity "action-required" -Detail "Manual publish placeholders must be present as guidance only.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-owner-inputs-present" -Passed ($requiredOwnerInputs.Count -ge 5) -Severity "action-required" -Detail "requiredOwnerInputs must enumerate real owner evidence fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "validator-commands-present" -Passed ($validatorCommands.Count -ge 5) -Severity "action-required" -Detail "validatorCommands must include strict validation commands.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts-present" -Passed ($sourceArtifacts.Count -ge 5) -Severity "action-required" -Detail "sourceArtifacts must point to owner handoff and release evidence inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-substitute-boundaries" -Passed (Test-ContainsAny -Values $mustNotSubstitute -Needles @("template", "local feed", "ProjectReference", "dependency-probe", "blocked-by-cuda-driver")) -Severity "action-required" -Detail "mustNotSubstitute must explicitly reject template/local feed/ProjectReference/dependency-probe/blocked-by-cuda-driver proof substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-validator-command" -Passed (Test-ContainsAny -Values $validatorCommands -Needles @("Test-PostPublishVerificationRecord.ps1", "-FailOnNotProof")) -Severity "action-required" -Detail "Post-publish proof validation command must be included.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-close-validator-command" -Passed (Test-ContainsAny -Values $validatorCommands -Needles @("Test-ReleaseIssueCloseRecord.ps1", "-FailOnNotCloseReady")) -Severity "action-required" -Detail "Release issue close strict validator command must be included.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-execution-package-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-real-owner-execution-proof-required"
}
else {
  "invalid-owner-execution-package"
}

$validation = [pscustomobject]@{
  recordKind = "owner-release-execution-package-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  packageState = $packageState
  isValidExecutionPackageShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  executionStepCount = $executionSteps.Count
  oneScreenReleaseHoldChecklistCount = $oneScreenReleaseHoldChecklist.Count
  manualPublishPlaceholderCount = $manualPublishPlaceholders.Count
  requiredOwnerInputCount = $requiredOwnerInputs.Count
  validatorCommandCount = $validatorCommands.Count
  sourceArtifactCount = $sourceArtifacts.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner execution guidance only. It cannot publish packages, approve publication, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-release-execution-package-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-release-execution-package-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner Release Execution Package Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| packageState | ``$($validation.packageState)`` |
| isValidExecutionPackageShape | ``$($validation.isValidExecutionPackageShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| executionStepCount | ``$($validation.executionStepCount)`` |
| oneScreenReleaseHoldChecklistCount | ``$($validation.oneScreenReleaseHoldChecklistCount)`` |
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

Write-Host "Owner release execution package validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner release execution package validation failed with $($failedBlockers.Count) blocker(s)."
}
