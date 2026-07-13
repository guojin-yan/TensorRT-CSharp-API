[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22"
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

function Test-StateBlocked {
  param(
    [string]$State,
    [bool]$Promoted,
    [int]$FailedBlockerCount,
    [int]$FailedActionRequiredCount,
    [int]$FailedValidationItemCount
  )

  if ($Promoted) {
    return $false
  }

  if ($FailedBlockerCount -gt 0 -or $FailedActionRequiredCount -gt 0 -or $FailedValidationItemCount -gt 0) {
    return $true
  }

  if ([string]::IsNullOrWhiteSpace($State)) {
    return $true
  }

  return $State.StartsWith("blocked", [StringComparison]::OrdinalIgnoreCase) -or
    $State.StartsWith("incomplete", [StringComparison]::OrdinalIgnoreCase) -or
    $State.StartsWith("missing", [StringComparison]::OrdinalIgnoreCase) -or
    $State -eq "template-only" -or
    $State -eq "owner-action-required"
}

function New-WorkItem {
  param(
    [string]$Id,
    [string]$Phase,
    [string]$ProofKind,
    [AllowNull()][object]$Validation,
    [AllowNull()][object]$Source,
    [string[]]$SourceArtifacts,
    [string]$StateProperty,
    [string]$PromoteProperty,
    [string]$ValidatorCommand,
    [string[]]$RequiredOwnerActions,
    [string]$Boundary
  )

  $sourceState = [string](Get-PropertyOrDefault -Object $Source -Name $StateProperty -DefaultValue (
      [string](Get-PropertyOrDefault -Object $Validation -Name $StateProperty -DefaultValue (
          [string](Get-PropertyOrDefault -Object $Validation -Name "validationState" -DefaultValue "missing-source")
        ))
    ))
  $validationState = [string](Get-PropertyOrDefault -Object $Validation -Name "validationState" -DefaultValue $sourceState)
  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedBlockerCount" -DefaultValue 0)
  $failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedActionRequiredCount" -DefaultValue 0)
  $failedValidationItemCount = [int](Get-PropertyOrDefault -Object $Validation -Name "failedValidationItemCount" -DefaultValue 0)
  $promoted = [bool](Get-PropertyOrDefault -Object $Validation -Name $PromoteProperty -DefaultValue ([bool](Get-PropertyOrDefault -Object $Source -Name $PromoteProperty -DefaultValue $false)))
  $blocked = Test-StateBlocked -State $validationState -Promoted $promoted -FailedBlockerCount $failedBlockerCount -FailedActionRequiredCount $failedActionRequiredCount -FailedValidationItemCount $failedValidationItemCount

  [pscustomobject]@{
    id = $Id
    phase = $Phase
    proofKind = $ProofKind
    sourceArtifacts = @($SourceArtifacts)
    sourceState = $sourceState
    validationState = $validationState
    promoteProperty = $PromoteProperty
    promoted = $promoted
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    failedValidationItemCount = $failedValidationItemCount
    failingBlockerIds = @(Get-FailingValidationItemIds -Validation $Validation -Severity "blocker")
    failingActionRequiredIds = @(Get-FailingValidationItemIds -Validation $Validation -Severity "action-required")
    validatorCommand = $ValidatorCommand
    requiredOwnerActions = @($RequiredOwnerActions)
    blocked = $blocked
    readyForReleaseCloseReview = (-not $blocked)
    boundary = $Boundary
  }
}

$packageConsumerWorklist = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-worklist.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$realModelHandoff = Read-JsonOrNull "artifacts\user-acceptance\real-model-owner-handoff.json"
$ownerInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-owner-input-validation.json"
$candidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$finalDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$workItems = @(
  New-WorkItem `
    -Id "package-consumer-runtime-proof" `
    -Phase "runtime-proof" `
    -ProofKind "package-consumer-runtime" `
    -Validation $packageConsumerWorklist `
    -Source $packageConsumerWorklist `
    -SourceArtifacts @("artifacts/final-release/package-consumer-runtime-proof-worklist.json", "artifacts/final-release/package-consumer-runtime-proof-worklist.md") `
    -StateProperty "worklistState" `
    -PromoteProperty "canPromoteRuntimeProof" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof" `
    -RequiredOwnerActions @(
      "Complete clean external package consumer runtime smoke outside the repository.",
      "Replace owner input placeholders, local feed, ProjectReference, and direct .nupkg evidence with real public package evidence.",
      "Refresh package-consumer-runtime-proof-record.json and promote only after strict validation passes."
    ) `
    -Boundary "Package-consumer worklists, scaffolds, owner inputs, and candidates are not runtime proof until the strict package-consumer proof record promotes real clean external smoke."
  New-WorkItem `
    -Id "post-publish-verification-proof" `
    -Phase "post-publish" `
    -ProofKind "post-publish-verification" `
    -Validation $postPublishValidation `
    -Source $postPublishValidation `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "isPostPublishVerificationProof" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -FailOnNotProof" `
    -RequiredOwnerActions @(
      "Publish through the real owner-approved channel before collecting post-publish evidence.",
      "Record package URLs, package hashes, clean consumer identity, host metadata, commands, stdout/stderr summaries, and log SHA256 values.",
      "Refresh post-publish-verification-record.json and validation before release issue close."
    ) `
    -Boundary "Post-publish verification cannot be synthesized before real channel publish and clean consumer runtime smoke evidence."
  New-WorkItem `
    -Id "linux-runner-proof" `
    -Phase "platform-proof" `
    -ProofKind "linux-runner-proof" `
    -Validation $linuxRunnerValidation `
    -Source $linuxRunnerValidation `
    -SourceArtifacts @("artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "isRealLinuxRunnerProof" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidence.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Run the Linux package workflow on a real linux-x64 compatible host.",
      "Capture configure/build/package/consumer-copy command evidence and native asset listing.",
      "Keep Windows handoff and template-only records out of the proof bucket."
    ) `
    -Boundary "Linux runner templates and Windows-side handoff records are not Linux runner proof."
  New-WorkItem `
    -Id "real-model-runtime-proof" `
    -Phase "sample-runtime" `
    -ProofKind "real-model-runtime" `
    -Validation $sampleRunValidation `
    -Source $realModelHandoff `
    -SourceArtifacts @("artifacts/user-acceptance/sample-run-evidence-record-validation.json", "artifacts/user-acceptance/real-model-owner-handoff.json") `
    -StateProperty "validationState" `
    -PromoteProperty "canPromoteRealModelRuntime" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -FailOnNotProof" `
    -RequiredOwnerActions @(
      "Acquire real model assets and sidecars approved by the sample owner.",
      "Run the sample against real inputs and capture runtime log/hash evidence.",
      "Promote only when proofClassification=real-model-runtime and validation allows promotion."
    ) `
    -Boundary "Template-only, build-only, dependency-probe-only, or sidecar-only sample evidence cannot replace real-model-runtime proof."
  New-WorkItem `
    -Id "release-issue-close-owner-input" `
    -Phase "release-close-input" `
    -ProofKind "owner-input" `
    -Validation $ownerInputValidation `
    -Source $ownerInputValidation `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-owner-input-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "canCloseReleaseIssue" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Provide real post-publish proof validation state, rollback plan, rollback owner, rollback trigger, final close decision, release issue id, and URL.",
      "Capture release evidence bundle SHA256, release close preflight SHA256, stale claims audit SHA256, and post-publish validation SHA256.",
      "Do not interpret owner input shape validity as close approval."
    ) `
    -Boundary "Release close owner input is only an input contract; it cannot close the release issue without real proof and strict close validation."
  New-WorkItem `
    -Id "release-issue-close-candidate" `
    -Phase "release-close-candidate" `
    -ProofKind "candidate-real-proof" `
    -Validation $candidateValidation `
    -Source $candidateValidation `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-candidate-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "canCloseReleaseIssue" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Replace placeholder rollback and owner decision values with real owner-approved values.",
      "Reference validator-passing post-publish proof and current release evidence bundle hash.",
      "Refresh candidate validation before strict close record projection."
    ) `
    -Boundary "Release close candidates are not release-close proof and cannot bypass the strict close record validator."
  New-WorkItem `
    -Id "release-issue-final-close-decision" `
    -Phase "owner-final-decision" `
    -ProofKind "owner-final-close-decision" `
    -Validation $finalDecisionValidation `
    -Source $finalDecisionValidation `
    -SourceArtifacts @("artifacts/final-release/release-issue-final-close-decision-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "canCloseReleaseIssue" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict" `
    -RequiredOwnerActions @(
      "Record owner name, timestamp UTC, release issue id/URL, final close decision, rollback review, runtime smoke confirmation, and log/hash review confirmation.",
      "Run final close decision validation after real proof records are present.",
      "Keep the final close decision blocked until all runtime and post-publish proof gates pass."
    ) `
    -Boundary "Final close decision validation is owner input validation only until real proof records and strict close validation pass."
  New-WorkItem `
    -Id "strict-close-validator" `
    -Phase "strict-release-close" `
    -ProofKind "strict-close-validation" `
    -Validation $strictCloseValidation `
    -Source $strictCloseValidation `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-validation.json") `
    -StateProperty "validationState" `
    -PromoteProperty "canPromoteReleaseIssueCloseRecord" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" `
    -RequiredOwnerActions @(
      "Project all real proof and owner decision inputs into release-issue-close-record.json.",
      "Verify post-publish proof, preflight, stale claims audit, evidence bundle SHA256, rollback plan, and owner final decision.",
      "Only close the release issue after -FailOnNotCloseReady passes."
    ) `
    -Boundary "The strict close validator is the final gate; template-only close records and blocked validation cannot close the release issue."
)

$blockedWorkItems = @($workItems | Where-Object { [bool]$_.blocked })
$readyWorkItems = @($workItems | Where-Object { -not [bool]$_.blocked })
$failedBlockerCount = @($workItems | ForEach-Object { [int]$_.failedBlockerCount } | Measure-Object -Sum).Sum
$failedActionRequiredCount = @($workItems | ForEach-Object { [int]$_.failedActionRequiredCount } | Measure-Object -Sum).Sum
$failedValidationItemCount = @($workItems | ForEach-Object { [int]$_.failedValidationItemCount } | Measure-Object -Sum).Sum
if ($null -eq $failedBlockerCount) { $failedBlockerCount = 0 }
if ($null -eq $failedActionRequiredCount) { $failedActionRequiredCount = 0 }
if ($null -eq $failedValidationItemCount) { $failedValidationItemCount = 0 }

$worklistState = if ($blockedWorkItems.Count -eq 0) {
  "ready-for-strict-release-close-review"
}
else {
  "blocked-release-close-real-proof-required"
}

$sourceArtifacts = @(
  "artifacts/final-release/package-consumer-runtime-proof-worklist.json",
  "artifacts/final-release/package-consumer-runtime-proof-worklist.md",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/real-model-owner-handoff.json",
  "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
  "artifacts/final-release/release-issue-close-record-candidate-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/release-issue-close-record-validation.json"
)

$recordOut = [pscustomobject]@{
  recordKind = "release-close-proof-worklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  worklistState = $worklistState
  workItemCount = $workItems.Count
  blockedWorkItemCount = $blockedWorkItems.Count
  readyWorkItemCount = $readyWorkItems.Count
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  failedValidationItemCount = [int]$failedValidationItemCount
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  workItems = @($workItems)
  sourceArtifacts = $sourceArtifacts
  boundary = "This worklist aggregates release-close proof blockers only. It is not runtime proof, not post-publish verification proof, not publication approval, not release-close approval, and not a substitute for owner-filled real proof records plus Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady."
}

$jsonPath = Join-Path $OutputRoot "release-close-proof-worklist.json"
$markdownPath = Join-Path $OutputRoot "release-close-proof-worklist.md"

$recordOut | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Proof Worklist")
$lines.Add("")
$lines.Add("`release-close-proof-worklist` 汇总 release issue 最终关闭前仍缺失的真实 proof、Owner 输入和 strict validator 阻断项。")
$lines.Add("")
$lines.Add("它不是 runtime proof，不执行发布，不批准公开发布，不关闭 release issue，也不能替代真实 post-publish verification、clean package-consumer runtime smoke、Linux runner proof、real-model runtime proof、Owner rollback/final decision 或 strict close validation。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| worklistState | ``$(ConvertTo-MarkdownCell $recordOut.worklistState)`` |")
$lines.Add("| workItemCount | ``$($recordOut.workItemCount)`` |")
$lines.Add("| blockedWorkItemCount | ``$($recordOut.blockedWorkItemCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($recordOut.failedActionRequiredCount)`` |")
$lines.Add("| failedValidationItemCount | ``$($recordOut.failedValidationItemCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Work Items")
$lines.Add("")
$lines.Add("| Id | Phase | Proof Kind | State | Failed Action Required | Validator |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($item in $workItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.id) | $(ConvertTo-MarkdownCell $item.phase) | $(ConvertTo-MarkdownCell $item.proofKind) | $(ConvertTo-MarkdownCell $item.validationState) | ``$($item.failedActionRequiredCount)`` | ``$(ConvertTo-MarkdownCell $item.validatorCommand)`` |")
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
$lines.Add("## Source Artifacts")
$lines.Add("")
foreach ($artifact in $sourceArtifacts) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $artifact)``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close proof worklist written to $jsonPath"
Write-Host "Release close proof worklist markdown written to $markdownPath"
Write-Host "WorklistState=$worklistState WorkItems=$($workItems.Count) Blocked=$($blockedWorkItems.Count) FailedActionRequired=$failedActionRequiredCount FailedValidationItems=$failedValidationItemCount"
