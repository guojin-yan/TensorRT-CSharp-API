[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\external-clean-consumer-execution-result-import.json",
  [string]$CandidatePath = "artifacts\final-release\external-clean-consumer-execution-result-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("ImportPath", "CandidatePath", "OutputRoot")) {
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf) -or -not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-ExternalCleanConsumerExecutionResult.ps1") -RepositoryRoot $RepositoryRoot
}

$import = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$proofReady = [bool](Get-PropertyOrDefault -Object $import -Name "proofCandidateReady" -DefaultValue $false)

$recordKindOk = [string](Get-PropertyOrDefault $import "recordKind" "") -eq "external-clean-consumer-execution-result-import" -and [string](Get-PropertyOrDefault $candidate "recordKind" "") -eq "external-clean-consumer-execution-result-candidate"
$defaultBlockedOk = ([string](Get-PropertyOrDefault $import "importState" "")).Contains("blocked", [StringComparison]::OrdinalIgnoreCase) -or $proofReady
$nonProofDefaultOk = -not $proofReady -and -not [bool](Get-PropertyOrDefault $import "canPromoteRuntimeProof" $true) -and -not [bool](Get-PropertyOrDefault $import "isRuntimeExecutionProof" $true) -and [bool](Get-PropertyOrDefault $import "ownerActionRequired" $false)
$noPublishCloseOk = -not [bool](Get-PropertyOrDefault $import "performsPublish" $true) -and -not [bool](Get-PropertyOrDefault $import "canPublishPublicly" $true) -and -not [bool](Get-PropertyOrDefault $import "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $import "isPostPublishProof" $true)
$findingsPresentByDefaultOk = [int](Get-PropertyOrDefault $import "failedActionRequiredCount" 0) -gt 0 -or $proofReady
$boundary = [string](Get-PropertyOrDefault $import "boundary" "")
$boundaryOk = $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed $recordKindOk -Severity "blocker" -Detail "Import and candidate recordKind values must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "default-blocked" -Passed $defaultBlockedOk -Severity "blocker" -Detail "Default import must remain blocked unless real proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-default" -Passed $nonProofDefaultOk -Severity "blocker" -Detail "Default import must be owner-action-required and non-proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-close" -Passed $noPublishCloseOk -Severity "blocker" -Detail "Import must never publish, close, or claim post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "findings-present-by-default" -Passed $findingsPresentByDefaultOk -Severity "blocker" -Detail "Default/template import must report action-required findings.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed $boundaryOk -Severity "blocker" -Detail "Boundary must preserve non-proof classification.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "external-clean-consumer-execution-result-validation-ready" } else { "blocked-external-clean-consumer-execution-result-validation-invalid" }

$validation = [pscustomobject]@{
  recordKind = "external-clean-consumer-execution-result-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  proofCandidateReady = $proofReady
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $proofReady
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $proofReady
  isPackageConsumerRuntimeProof = $proofReady
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($validationItems)
  boundary = "Validation checks External CleanConsumer result import only. Default/template state is blocked and non-proof; it is not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "external-clean-consumer-execution-result-validation.json"
$markdownPath = Join-Path $OutputRoot "external-clean-consumer-execution-result-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}
Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# External CleanConsumer Execution Result Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- proofCandidateReady: ``$proofReady``",
  "",
  "| ID | Passed | Severity | Detail |",
  "|---|---:|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "ExternalCleanConsumerExecutionResultValidationState=$validationState FailedBlockers=$($failedBlockers.Count) ProofCandidateReady=$proofReady"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "External CleanConsumer execution result validation failed."
}
if ($FailOnNotProof.IsPresent -and -not $proofReady) {
  throw "External CleanConsumer execution result is not proof-ready."
}
