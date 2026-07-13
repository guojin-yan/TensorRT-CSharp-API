[CmdletBinding()]
param(
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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function Get-FailingValidationItemIds {
  param(
    [AllowNull()][object]$Validation,
    [string]$Severity
  )

  return @(
    Get-PropertyOrDefault -Object $Validation -Name "validationItems" -DefaultValue @() |
      Where-Object {
        -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and
        [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq $Severity
      } |
      ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  )
}

function New-WorkItem {
  param(
    [string]$Id,
    [string]$Phase,
    [AllowNull()][object]$Validation,
    [AllowNull()][object]$Source,
    [string]$SourceArtifact,
    [string]$ValidatorCommand,
    [string[]]$RequiredOwnerActions,
    [string]$Boundary
  )

  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedBlockerCount" -DefaultValue 0)
  $failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedActionRequiredCount" -DefaultValue 0)
  $validationState = [string](Get-PropertyOrDefault -Object $Validation -Name "validationState" -DefaultValue "missing-validation")
  $sourceState = [string](Get-PropertyOrDefault -Object $Source -Name "candidateState" -DefaultValue (
      [string](Get-PropertyOrDefault -Object $Source -Name "proofState" -DefaultValue (
          [string](Get-PropertyOrDefault -Object $Source -Name "recordKind" -DefaultValue "missing-source")
        ))
    ))
  $blocked = $failedBlockerCount -gt 0 -or $failedActionRequiredCount -gt 0 -or $validationState.StartsWith("blocked", [StringComparison]::OrdinalIgnoreCase) -or $validationState -eq "template-only"

  [pscustomobject]@{
    id = $Id
    phase = $Phase
    sourceArtifact = $SourceArtifact
    sourceState = $sourceState
    validationState = $validationState
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    failingBlockerIds = @(Get-FailingValidationItemIds -Validation $Validation -Severity "blocker")
    failingActionRequiredIds = @(Get-FailingValidationItemIds -Validation $Validation -Severity "action-required")
    validatorCommand = $ValidatorCommand
    requiredOwnerActions = @($RequiredOwnerActions)
    blocked = $blocked
    readyForProofReview = (-not $blocked)
    boundary = $Boundary
  }
}

$ownerInputValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$ownerInputTemplate = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json"
$candidate = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-candidate.json"
$candidateValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-candidate-validation.json"
$record = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record.json"
$recordValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record-validation.json"
$scaffold = Read-JsonOrNull "artifacts\final-release\package-consumer-external-smoke-scaffold.json"

$workItems = @(
  New-WorkItem `
    -Id "owner-input-clean-consumer-fields" `
    -Phase "owner-input" `
    -Validation $ownerInputValidation `
    -Source $ownerInputTemplate `
    -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Provide a real cleanExternalConsumerRoot outside the repository.",
      "Provide a real external consumerProjectPath with no ProjectReference, local feed, or direct .nupkg reference.",
      "Provide public package source, package paths, SHA256 values, compatible host metadata, smoke command, stdout/stderr summaries, and smoke log hash."
    ) `
    -Boundary "Owner input fields are not proof until candidate and record validators pass with real clean external consumer smoke evidence."
  New-WorkItem `
    -Id "candidate-clean-runtime-smoke" `
    -Phase "candidate" `
    -Validation $candidateValidation `
    -Source $candidate `
    -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Run restore/build/dependency probe/runtime smoke in the external clean consumer.",
      "Capture managed/runtime package SHA256, smoke log path/SHA256, runtime package key, host OS/architecture, CUDA driver/runtime, and TensorRT version.",
      "Refresh the candidate after replacing placeholder/local feed/ProjectReference evidence with real public-package evidence."
    ) `
    -Boundary "Candidate metadata is an owner input surface only; it cannot become runtime proof without validator-passing smoke evidence."
  New-WorkItem `
    -Id "strict-proof-record" `
    -Phase "proof-record" `
    -Validation $recordValidation `
    -Source $record `
    -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-record.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof" `
    -RequiredOwnerActions @(
      "Project the owner input and candidate into package-consumer-runtime-proof-record.json.",
      "Run the strict proof validator and keep proofClassification=package-consumer-runtime only when real clean external smoke proof passes.",
      "Bridge the passing record into external-runtime-proof-record.json only after the strict validator promotes it."
    ) `
    -Boundary "The record remains template-only/non-proof until strict validation promotes it with real clean external consumer smoke."
  New-WorkItem `
    -Id "external-smoke-scaffold" `
    -Phase "scaffold" `
    -Validation $null `
    -Source $scaffold `
    -SourceArtifact "artifacts/final-release/package-consumer-external-smoke-scaffold.json" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerExternalSmokeScaffold.ps1 -OutputRoot <outside-repo-path>" `
    -RequiredOwnerActions @(
      "Create or refresh the external clean consumer project outside this repository.",
      "Use the scaffold only as a runnable project shape; do not treat scaffold generation as proof.",
      "Run the generated consumer on a compatible CUDA/TensorRT host and feed its real logs and hashes into owner input."
    ) `
    -Boundary "The scaffold is not proof; it only helps the owner create a clean consumer for real smoke execution."
)

$blockedWorkItems = @($workItems | Where-Object { [bool]$_.blocked })
$readyWorkItems = @($workItems | Where-Object { -not [bool]$_.blocked })
$failedBlockerCount = @($workItems | ForEach-Object { [int]$_.failedBlockerCount } | Measure-Object -Sum).Sum
$failedActionRequiredCount = @($workItems | ForEach-Object { [int]$_.failedActionRequiredCount } | Measure-Object -Sum).Sum
if ($null -eq $failedBlockerCount) { $failedBlockerCount = 0 }
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }

$ownerActionRequiredCount = [int]$failedActionRequiredCount + @($workItems | Where-Object { $_.id -eq "external-smoke-scaffold" }).Count
$worklistState = if ($blockedWorkItems.Count -eq 0 -and $failedActionRequiredCount -eq 0) {
  "ready-for-real-package-consumer-proof-review"
}
else {
  "blocked-real-package-consumer-runtime-proof-required"
}

$recordOut = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-worklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  worklistState = $worklistState
  workItemCount = $workItems.Count
  blockedWorkItemCount = $blockedWorkItems.Count
  readyWorkItemCount = $readyWorkItems.Count
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  ownerActionRequiredCount = [int]$ownerActionRequiredCount
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  workItems = @($workItems)
  sourceArtifacts = @(
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-candidate.json",
    "artifacts/final-release/package-consumer-runtime-proof-candidate-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-record.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
    "artifacts/final-release/package-consumer-external-smoke-scaffold.json"
  )
  boundary = "This worklist aggregates owner actions for package-consumer-runtime proof only. It is not proof, not publication approval, not release close approval, and not a substitute for validator-passing clean external consumer runtime smoke."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-runtime-proof-worklist.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-runtime-proof-worklist.md"

$recordOut | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Package Consumer Runtime Proof Worklist")
$lines.Add("")
$lines.Add("`package-consumer-runtime-proof-worklist` 汇总 owner input、candidate、strict proof record 和 external smoke scaffold 四段状态，给 Owner 一份可执行的 package-consumer-runtime proof 收敛清单。")
$lines.Add("")
$lines.Add("它不是 proof，不执行发布，不关闭 release issue，也不能替代真实 clean external consumer runtime smoke。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| worklistState | ``$(ConvertTo-MarkdownCell $recordOut.worklistState)`` |")
$lines.Add("| workItemCount | ``$($recordOut.workItemCount)`` |")
$lines.Add("| blockedWorkItemCount | ``$($recordOut.blockedWorkItemCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($recordOut.failedActionRequiredCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("")
$lines.Add("## Work Items")
$lines.Add("")
$lines.Add("| Id | Phase | State | Failed Action Required | Validator |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $workItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.id) | $(ConvertTo-MarkdownCell $item.phase) | $(ConvertTo-MarkdownCell $item.validationState) | ``$($item.failedActionRequiredCount)`` | ``$(ConvertTo-MarkdownCell $item.validatorCommand)`` |")
}
$lines.Add("")
$lines.Add("## Owner Actions")
$lines.Add("")
foreach ($item in $workItems) {
  $lines.Add("### $(ConvertTo-MarkdownCell $item.id)")
  foreach ($action in @($item.requiredOwnerActions)) {
    $lines.Add("- $(ConvertTo-MarkdownCell $action)")
  }
  $lines.Add("")
}
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof worklist written to $jsonPath"
Write-Host "Package consumer runtime proof worklist markdown written to $markdownPath"
Write-Host "WorklistState=$worklistState WorkItems=$($workItems.Count) Blocked=$($blockedWorkItems.Count) FailedActionRequired=$failedActionRequiredCount"
