[CmdletBinding()]
param(
  [string]$RuntimeProofInputValidationPath = "artifacts\final-release\runtime-proof-execution-input-record-validation.json",
  [string]$OwnerRuntimeProofResultInputValidationPath = "artifacts\final-release\owner-runtime-proof-result-input-validation.json",
  [string]$OwnerRuntimeProofRunbookValidationPath = "artifacts\final-release\owner-runtime-proof-execution-runbook-validation.json",
  [string]$ReleaseCloseStrictBridgeValidationPath = "artifacts\final-release\release-close-strict-validation-bridge-validation.json",
  [string]$RuntimeProofInputPath = "artifacts\final-release\runtime-proof-execution-input-record.json",
  [string]$OwnerRuntimeProofResultInputPath = "artifacts\final-release\owner-runtime-proof-result-input.template.json",
  [string]$OwnerRuntimeProofRunbookPath = "artifacts\final-release\owner-runtime-proof-execution-runbook.json",
  [string]$ReleaseCloseStrictBridgePath = "artifacts\final-release\release-close-strict-validation-bridge.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolved = Resolve-InputPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ValidationDetailsForLane {
  param(
    [AllowNull()][object[]]$Items,
    [string]$ExecutionInputId,
    [string]$Category
  )

  @($Items | Where-Object {
    -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $true) -and
    ([string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")).IndexOf($ExecutionInputId, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    ([string](Get-PropertyOrDefault -Object $_ -Name "category" -DefaultValue "") -eq $Category -or [string]::IsNullOrWhiteSpace($Category))
  } | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "detail" -DefaultValue "") })
}

$runtimeInput = Read-JsonOrNull $RuntimeProofInputPath
$resultInput = Read-JsonOrNull $OwnerRuntimeProofResultInputPath
$runbook = Read-JsonOrNull $OwnerRuntimeProofRunbookPath
$bridge = Read-JsonOrNull $ReleaseCloseStrictBridgePath
$runtimeInputValidation = Read-JsonOrNull $RuntimeProofInputValidationPath
$resultInputValidation = Read-JsonOrNull $OwnerRuntimeProofResultInputValidationPath
$runbookValidation = Read-JsonOrNull $OwnerRuntimeProofRunbookValidationPath
$bridgeValidation = Read-JsonOrNull $ReleaseCloseStrictBridgeValidationPath

$executionInputs = @(Get-PropertyOrDefault -Object $runtimeInput -Name "executionInputs" -DefaultValue @())
$resultInputs = @(Get-PropertyOrDefault -Object $resultInput -Name "resultInputs" -DefaultValue @())
$runbookItems = @(Get-PropertyOrDefault -Object $runbook -Name "runbookItems" -DefaultValue @())
$bridgeItems = @(Get-PropertyOrDefault -Object $bridge -Name "bridgeItems" -DefaultValue @())
$resultValidationItems = @(Get-PropertyOrDefault -Object $resultInputValidation -Name "validationItems" -DefaultValue @())

$laneItems = foreach ($executionInput in $executionInputs) {
  $executionInputId = [string](Get-PropertyOrDefault -Object $executionInput -Name "executionInputId" -DefaultValue "unknown-execution-input")
  $candidateId = [string](Get-PropertyOrDefault -Object $executionInput -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $executionInput -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $result = @($resultInputs | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "executionInputId" -DefaultValue "") -eq $executionInputId } | Select-Object -First 1)[0]
  $runbookItem = @($runbookItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "executionInputId" -DefaultValue "") -eq $executionInputId } | Select-Object -First 1)[0]
  $bridgeItem = @($bridgeItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "proofLane" -DefaultValue "") -eq $proofLane } | Select-Object -First 1)[0]
  $realEvidenceMissing = @(Get-PropertyOrDefault -Object $result -Name "missingResultFields" -DefaultValue @())
  $substituteDetails = @(Get-ValidationDetailsForLane -Items $resultValidationItems -ExecutionInputId $executionInputId -Category "substitute-blocker")
  $requiredNextCommand = if ($null -ne $runbookItem) {
    $sequence = @(Get-PropertyOrDefault -Object $runbookItem -Name "commandSequence" -DefaultValue @())
    if ($sequence.Count -gt 0) { [string](Get-PropertyOrDefault -Object $sequence[0] -Name "command" -DefaultValue "Fill owner runtime proof result input and rerun validators.") }
    else { "Fill owner runtime proof result input and rerun validators." }
  }
  else {
    "Generate owner runtime proof execution runbook before collecting proof."
  }
  $validatorCommand = if ($null -ne $runbookItem) {
    $commands = @(Get-PropertyOrDefault -Object $runbookItem -Name "validatorCommands" -DefaultValue @())
    if ($commands.Count -gt 0) { [string]$commands[0] } else { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRuntimeProofResultInput.ps1 -Strict" }
  }
  else {
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRuntimeProofResultInput.ps1 -Strict"
  }
  $bridgeBlockedReasons = @(Get-PropertyOrDefault -Object $bridgeItem -Name "blockedReasons" -DefaultValue @())

  [pscustomobject]@{
    laneItemId = "$executionInputId-lane-dry-run"
    executionInputId = $executionInputId
    candidateId = $candidateId
    proofLane = $proofLane
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $executionInput -Name "runtimePackageKey" -DefaultValue "")
    laneState = "blocked-runtime-proof-lane-real-evidence-required"
    runtimeInputValidationState = [string](Get-PropertyOrDefault -Object $runtimeInputValidation -Name "validationState" -DefaultValue "missing-runtime-proof-execution-input-record-validation")
    ownerResultInputValidationState = [string](Get-PropertyOrDefault -Object $resultInputValidation -Name "validationState" -DefaultValue "missing-owner-runtime-proof-result-input-validation")
    ownerRunbookValidationState = [string](Get-PropertyOrDefault -Object $runbookValidation -Name "validationState" -DefaultValue "missing-owner-runtime-proof-execution-runbook-validation")
    releaseCloseBridgeValidationState = [string](Get-PropertyOrDefault -Object $bridgeValidation -Name "validationState" -DefaultValue "missing-release-close-strict-validation-bridge-validation")
    realEvidenceFieldsMissing = @($realEvidenceMissing)
    missingRealEvidenceFieldCount = @($realEvidenceMissing).Count
    substituteBlockers = @($substituteDetails)
    substituteBlockerCount = @($substituteDetails).Count
    bridgeBlockedReasons = @($bridgeBlockedReasons)
    bridgeBlockedReasonCount = @($bridgeBlockedReasons).Count
    requiredNextCommand = $requiredNextCommand
    validatorCommand = $validatorCommand
    closeImpact = "Release close remains blocked until this lane has real external proof inputs, existing logs, matching SHA256 values, owner review, passing validators, post-publish proof, and strict close validation."
    readyForRuntimeProof = $false
    canPromoteRuntimeProof = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$blockedLaneCount = @($laneItems | Where-Object { [string]$_.laneState -eq "blocked-runtime-proof-lane-real-evidence-required" }).Count
$readyLaneCount = @($laneItems | Where-Object { [bool]$_.readyForRuntimeProof }).Count
$missingFieldCount = ($laneItems | ForEach-Object { [int]$_.missingRealEvidenceFieldCount } | Measure-Object -Sum).Sum
$substituteBlockerCount = ($laneItems | ForEach-Object { [int]$_.substituteBlockerCount } | Measure-Object -Sum).Sum

$summary = [pscustomobject]@{
  recordKind = "runtime-proof-lane-dry-run-summary"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dryRunState = "blocked-runtime-proof-real-evidence-required"
  laneItemCount = @($laneItems).Count
  blockedLaneItemCount = $blockedLaneCount
  readyLaneItemCount = $readyLaneCount
  missingRealEvidenceFieldCount = [int]$missingFieldCount
  substituteBlockerCount = [int]$substituteBlockerCount
  laneItems = @($laneItems)
  sourceArtifacts = @(
    "artifacts/final-release/runtime-proof-execution-input-record-validation.json",
    "artifacts/final-release/owner-runtime-proof-result-input-validation.json",
    "artifacts/final-release/owner-runtime-proof-execution-runbook-validation.json",
    "artifacts/final-release/release-close-strict-validation-bridge-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This dry-run summary aggregates lane gaps only. It cannot substitute real runtime proof, package publish, post-publish verification, rollback approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "runtime-proof-lane-dry-run-summary.json"
$markdownPath = Join-Path $OutputRoot "runtime-proof-lane-dry-run-summary.md"
$summary | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Proof Lane Dry-Run Summary")
$lines.Add("")
$lines.Add("`runtime-proof-lane-dry-run-summary` 聚合每条 proof lane 的真实证据缺口、替代 proof blocker、下一步命令和 close impact。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| dryRunState | ``$(ConvertTo-MarkdownCell $summary.dryRunState)`` |")
$lines.Add("| laneItemCount | ``$($summary.laneItemCount)`` |")
$lines.Add("| blockedLaneItemCount | ``$($summary.blockedLaneItemCount)`` |")
$lines.Add("| readyLaneItemCount | ``$($summary.readyLaneItemCount)`` |")
$lines.Add("| missingRealEvidenceFieldCount | ``$($summary.missingRealEvidenceFieldCount)`` |")
$lines.Add("| substituteBlockerCount | ``$($summary.substituteBlockerCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($summary.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($summary.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Lanes")
$lines.Add("")
$lines.Add("| Lane | State | Missing Fields | Substitute Blockers | Bridge Reasons | Next Command |")
$lines.Add("| --- | --- | ---: | ---: | ---: | --- |")
foreach ($item in $laneItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.laneState) | ``$($item.missingRealEvidenceFieldCount)`` | ``$($item.substituteBlockerCount)`` | ``$($item.bridgeBlockedReasonCount)`` | $(ConvertTo-MarkdownCell $item.requiredNextCommand) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($summary.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime proof lane dry-run summary written to $jsonPath"
Write-Host "Runtime proof lane dry-run summary markdown written to $markdownPath"
Write-Host "DryRunState=$($summary.dryRunState) Lanes=$($summary.laneItemCount) Blocked=$($summary.blockedLaneItemCount) MissingFields=$($summary.missingRealEvidenceFieldCount)"
