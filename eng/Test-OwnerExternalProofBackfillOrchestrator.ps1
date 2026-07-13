[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-backfill-orchestrator.json",
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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
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

$resolvedInputPath = if ([System.IO.Path]::IsPathRooted($InputPath)) {
  $InputPath
}
else {
  Join-Path $RepositoryRoot $InputPath
}

if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner external proof backfill orchestrator not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$requiredLineIds = @(
  "owner-authorization",
  "package-consumer-runtime",
  "linux-runner-proof",
  "real-model-runtime",
  "post-publish-verification",
  "release-issue-close-record"
)

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$orchestratorState = [string](Get-PropertyOrDefault -Object $record -Name "orchestratorState" -DefaultValue "")
$backfillLines = @(Get-PropertyOrDefault -Object $record -Name "backfillLines" -DefaultValue @())
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-external-proof-backfill-orchestrator") -Severity "blocker" -Detail "recordKind must be owner-external-proof-backfill-orchestrator.")) | Out-Null
$items.Add((New-ValidationItem -Id "orchestrator-state" -Passed ($orchestratorState -eq "blocked-owner-external-proof-required") -Severity "blocker" -Detail "orchestratorState must remain blocked-owner-external-proof-required.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Orchestrator must not publish, approve public publication, or close the release issue.")) | Out-Null

if ($Strict) {
  $ids = @($backfillLines | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
  foreach ($id in $requiredLineIds) {
    $items.Add((New-ValidationItem -Id "line-$id-present" -Passed ($ids -contains $id) -Severity "blocker" -Detail "Backfill line '$id' must exist.")) | Out-Null
  }

  $items.Add((New-ValidationItem -Id "line-count" -Passed ($backfillLines.Count -eq 6) -Severity "blocker" -Detail "Strict orchestrator validation expects exactly six backfill lines.")) | Out-Null
}

foreach ($line in $backfillLines) {
  $id = [string](Get-PropertyOrDefault -Object $line -Name "id" -DefaultValue "")
  $state = [string](Get-PropertyOrDefault -Object $line -Name "backfillState" -DefaultValue "")
  $canPromoteProof = [bool](Get-PropertyOrDefault -Object $line -Name "canPromoteProof" -DefaultValue $true)
  $targetProofRecordPath = [string](Get-PropertyOrDefault -Object $line -Name "targetProofRecordPath" -DefaultValue "")
  $strictValidationCommand = [string](Get-PropertyOrDefault -Object $line -Name "strictValidationCommand" -DefaultValue "")

  $items.Add((New-ValidationItem -Id "line-$id-blocked" -Passed ($state -eq "blocked-owner-external-proof-required" -and -not $canPromoteProof) -Severity "blocker" -Detail "Line '$id' must remain blocked and non-promotable.")) | Out-Null
  $items.Add((New-ValidationItem -Id "line-$id-target-record" -Passed (-not [string]::IsNullOrWhiteSpace($targetProofRecordPath)) -Severity "blocker" -Detail "Line '$id' must expose targetProofRecordPath.")) | Out-Null
  $items.Add((New-ValidationItem -Id "line-$id-strict-validator" -Passed (-not [string]::IsNullOrWhiteSpace($strictValidationCommand)) -Severity "blocker" -Detail "Line '$id' must expose strictValidationCommand.")) | Out-Null
}

$packageConsumer = $backfillLines | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "package-consumer-runtime" } | Select-Object -First 1
if ($null -ne $packageConsumer) {
  $rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $packageConsumer -Name "requiredRealInputRules" -DefaultValue @())
  foreach ($rule in @("cleanExternalConsumerIdentity", "noProjectReference", "noLocalFeedAsPublicProof", "managedNupkgSha256", "runtimeNupkgSha256", "runtimePackageKeyMatches", "compatibleHostMetadata", "smokeCommandIncludesRuntimePackageKey", "smokeLogPath", "smokeLogSha256")) {
    $items.Add((New-ValidationItem -Id "package-consumer-rule-$rule" -Passed ($rules -contains $rule) -Severity "blocker" -Detail "package-consumer-runtime must require '$rule'.")) | Out-Null
  }

  $commands = Convert-ToStringArray (Get-PropertyOrDefault -Object $packageConsumer -Name "nextOwnerCommands" -DefaultValue @())
  $items.Add((New-ValidationItem -Id "package-consumer-external-root" -Passed (($commands -join " ") -match "outside this repository") -Severity "blocker" -Detail "package-consumer-runtime must require a clean external consumer outside the repository.")) | Out-Null
}

$closeRecord = $backfillLines | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "release-issue-close-record" } | Select-Object -First 1
if ($null -ne $closeRecord) {
  $rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $closeRecord -Name "requiredRealInputRules" -DefaultValue @())
  foreach ($rule in @("releaseEvidenceBundleSha256", "releaseClosePreflightPathAndHash", "staleClaimsAuditPathAndHash", "postPublishProofValidationPathAndHash", "rollbackPlan", "ownerFinalCloseDecision", "strictCloseValidatorCommand")) {
    $items.Add((New-ValidationItem -Id "close-record-rule-$rule" -Passed ($rules -contains $rule) -Severity "blocker" -Detail "release-issue-close-record must require '$rule'.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) {
  "valid-orchestrator-blocked-guidance"
}
else {
  "invalid-orchestrator"
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-external-proof-backfill-orchestrator-validation"
  inputPath = $InputPath
  validationState = $validationState
  isValidOrchestrator = ($validationState -eq "valid-orchestrator-blocked-guidance")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  strict = [bool]$Strict
  backfillLineCount = $backfillLines.Count
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  boundary = "This validator checks owner external proof backfill guidance only. It does not collect proof, publish packages, approve public publication, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-external-proof-backfill-orchestrator-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-external-proof-backfill-orchestrator-validation.md"

$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner External Proof Backfill Orchestrator Validation")
$lines.Add("")
$lines.Add("- generated: ``$($summary.generatedAtUtc)``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- valid orchestrator: ``$($summary.isValidOrchestrator)``")
$lines.Add("- backfill line count: ``$($summary.backfillLineCount)``")
$lines.Add("- failed blocker count: ``$($summary.failedBlockerCount)``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- can publish publicly: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("|---|---:|---|---|")
foreach ($item in $items) {
  $detail = ([string]$item.detail).Replace("|", "\|")
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $detail |")
}
$lines.Add("")
$lines.Add($summary.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof backfill orchestrator validation written to $jsonPath"
Write-Host "Owner external proof backfill orchestrator validation written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict -and $validationState -ne "valid-orchestrator-blocked-guidance") {
  Write-Error "Owner external proof backfill orchestrator strict validation failed. ValidationState=$validationState"
  exit 1
}
