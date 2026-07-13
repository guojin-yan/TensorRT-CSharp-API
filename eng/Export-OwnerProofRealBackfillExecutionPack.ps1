[CmdletBinding()]
param(
  [string]$RepositoryRoot
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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-OwnerInputTask {
  param([object]$Field, [string]$StrictValidatorCommand, [string[]]$NonSubstituteKinds)

  $id = [string](Get-PropertyOrDefault -Object $Field -Name "id" -DefaultValue "unknown-owner-input")
  $value = [string](Get-PropertyOrDefault -Object $Field -Name "value" -DefaultValue "")
  $target = switch ($id) {
    "rollback-plan" { "artifacts/final-release/release-issue-final-close-decision.template.json" }
    "rollback-owner" { "artifacts/final-release/release-issue-final-close-decision.template.json" }
    "rollback-trigger" { "artifacts/final-release/release-issue-final-close-decision.template.json" }
    "owner-final-close-decision" { "artifacts/final-release/release-issue-final-close-decision.template.json" }
    "release-issue-id" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "release-issue-url" { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
    "public-channel-package-source" { "artifacts/final-release/post-publish-verification-record.input-draft.json" }
    "clean-consumer-runtime-smoke-log" { "artifacts/final-release/package-consumer-runtime-proof-candidate.json" }
    default { "artifacts/final-release/release-issue-close-record-owner-input.template.json" }
  }
  $firstCommand = switch ($id) {
    "public-channel-package-source" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecord.ps1" }
    "clean-consumer-runtime-smoke-log" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofCandidate.ps1" }
    "release-issue-id" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    "release-issue-url" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1" }
    default { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1" }
  }

  [pscustomobject]@{
    id = $id
    currentValue = $value
    targetArtifact = $target
    targetField = $id
    requiredEvidence = "Owner must replace placeholder with real, reviewed value and keep command/log/hash evidence when applicable."
    firstCommand = $firstCommand
    validatorCommand = $StrictValidatorCommand
    nonSubstituteKinds = $NonSubstituteKinds
    taskState = "blocked-owner-input-required"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

function New-RealProofTask {
  param([object]$Blocker, [string]$StrictValidatorCommand, [string[]]$NonSubstituteKinds)

  $id = [string](Get-PropertyOrDefault -Object $Blocker -Name "id" -DefaultValue "unknown-real-proof")
  $state = [string](Get-PropertyOrDefault -Object $Blocker -Name "state" -DefaultValue "missing-state")
  $requiredAction = [string](Get-PropertyOrDefault -Object $Blocker -Name "requiredAction" -DefaultValue "Provide real validator-passing proof.")
  $target = switch ($id) {
    "post-publish-verification" { "artifacts/final-release/post-publish-verification-record.json" }
    "final-close-decision" { "artifacts/final-release/release-issue-final-close-decision.template.json" }
    "release-close-candidate" { "artifacts/final-release/release-issue-close-record-candidate.json" }
    "overlay-candidate" { "artifacts/final-release/release-issue-close-record-overlay-candidate.json" }
    "owner-external-execution-result-backfill-kit" { "artifacts/final-release/owner-external-execution-result-backfill-kit.json" }
    default { "artifacts/final-release/$id.json" }
  }
  $firstCommand = switch ($id) {
    "post-publish-verification" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecord.ps1" }
    "final-close-decision" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1" }
    "release-close-candidate" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordCandidate.ps1" }
    "overlay-candidate" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOverlayCandidate.ps1" }
    "owner-external-execution-result-backfill-kit" { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalExecutionResultBackfillKit.ps1" }
    default { "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1" }
  }

  [pscustomobject]@{
    id = $id
    currentState = $state
    targetArtifact = $target
    requiredAction = $requiredAction
    requiredRealProof = "Validator-passing owner-filled record with real public-channel/package/runtime evidence where required."
    whyBlocked = "Current state is $state; candidate/guidance/hash-only evidence cannot close the proof gap."
    firstCommand = $firstCommand
    validatorCommand = $StrictValidatorCommand
    nonSubstituteKinds = $NonSubstituteKinds
    taskState = "blocked-real-proof-required"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

function New-HashCheckTask {
  param([object]$Line, [string]$StrictValidatorCommand)

  $id = [string](Get-PropertyOrDefault -Object $Line -Name "id" -DefaultValue "unknown-hash-line")
  [pscustomobject]@{
    id = $id
    path = [string](Get-PropertyOrDefault -Object $Line -Name "path" -DefaultValue "")
    expectedSha256 = [string](Get-PropertyOrDefault -Object $Line -Name "expectedSha256" -DefaultValue "")
    actualSha256 = [string](Get-PropertyOrDefault -Object $Line -Name "actualSha256" -DefaultValue "")
    sha256Matches = [bool](Get-PropertyOrDefault -Object $Line -Name "sha256Matches" -DefaultValue $false)
    currentState = [string](Get-PropertyOrDefault -Object $Line -Name "state" -DefaultValue "missing-state")
    blockerRule = "Hash mismatch is a blocker; hash match is local consistency only and is not owner/runtime/post-publish proof."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputCrossHashAudit.ps1"
    validatorCommand = $StrictValidatorCommand
    proofPromotable = $false
    taskState = "blocked-proof-not-promotable"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$strictCandidate = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate.json"
$strictCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate-validation.json"

if ($null -eq $strictCandidate) {
  throw "Missing strict candidate. Run Export-ReleaseCloseStrictRecordCandidate.ps1 first."
}

$nonSubstituteKinds = @(
  "template",
  "draft",
  "candidate",
  "scaffold",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "schema-only",
  "preflight-only",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "hash-match-only",
  "owner-guidance-only"
)

$strictValidatorCommand = [string](Get-PropertyOrDefault -Object $strictCandidate -Name "strictValidatorCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady")
$ownerInputTasks = @(Get-PropertyOrDefault -Object $strictCandidate -Name "requiredOwnerFields" -DefaultValue @() | ForEach-Object { New-OwnerInputTask -Field $_ -StrictValidatorCommand $strictValidatorCommand -NonSubstituteKinds $nonSubstituteKinds })
$realProofTasks = @(Get-PropertyOrDefault -Object $strictCandidate -Name "releaseCloseBlockers" -DefaultValue @() | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocker" -DefaultValue $false) } | ForEach-Object { New-RealProofTask -Blocker $_ -StrictValidatorCommand $strictValidatorCommand -NonSubstituteKinds $nonSubstituteKinds })
$hashCheckTasks = @(Get-PropertyOrDefault -Object $strictCandidate -Name "hashLines" -DefaultValue @() | ForEach-Object { New-HashCheckTask -Line $_ -StrictValidatorCommand $strictValidatorCommand })

$firstCommands = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInput.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecord.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalExecutionResultBackfillKit.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputCrossHashAudit.ps1"
)
$strictValidators = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofRealBackfillExecutionPack.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict",
  $strictValidatorCommand
)
$sourceArtifacts = @(
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/release-close-strict-record-candidate-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-evidence-freeze.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/release-issue-close-record-candidate-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/real-external-proof-overlay-pack-validation.json",
  "artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json",
  "artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json",
  "artifacts/final-release/owner-input-cross-hash-audit-validation.json"
)

$blockedTaskCount = @($ownerInputTasks + $realProofTasks + $hashCheckTasks | Where-Object { ([string]$_.taskState).StartsWith("blocked", [StringComparison]::Ordinal) }).Count
$pack = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-proof-real-backfill-execution-pack"
  packState = "blocked-owner-real-proof-backfill-required"
  strictCandidateState = [string](Get-PropertyOrDefault -Object $strictCandidate -Name "candidateState" -DefaultValue "missing-release-close-strict-record-candidate")
  strictCandidateValidationState = [string](Get-PropertyOrDefault -Object $strictCandidateValidation -Name "validationState" -DefaultValue "missing-release-close-strict-record-candidate-validation")
  strictCandidateMismatchedHashCount = [int](Get-PropertyOrDefault -Object $strictCandidate -Name "mismatchedHashCount" -DefaultValue -1)
  ownerInputTaskCount = $ownerInputTasks.Count
  realProofTaskCount = $realProofTasks.Count
  hashCheckTaskCount = $hashCheckTasks.Count
  blockedTaskCount = $blockedTaskCount
  ownerInputTasks = $ownerInputTasks
  realProofTasks = $realProofTasks
  hashCheckTasks = $hashCheckTasks
  firstCommands = $firstCommands
  strictValidators = $strictValidators
  nonSubstituteProofKinds = $nonSubstituteKinds
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Owner proof real backfill execution pack is an owner handoff only. It does not publish, upload, approve public release, close a release issue, or promote template/draft/candidate/hash-only records to proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-real-backfill-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-real-backfill-execution-pack.md"
$pack | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$ownerRows = $ownerInputTasks | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.taskState)`` | ``$($_.targetArtifact)`` | ``$($_.firstCommand)`` |"
}
$proofRows = $realProofTasks | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.currentState)`` | ``$($_.targetArtifact)`` | $($_.requiredAction.Replace("|", "\|")) |"
}
$hashRows = $hashCheckTasks | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.sha256Matches)`` | ``$($_.currentState)`` | ``$($_.path)`` |"
}
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $nonSubstituteKinds | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Proof Real Backfill Execution Pack

生成时间：$($pack.generatedAtUtc)

该执行包把 `release-close-strict-record-candidate` 的 owner input、real proof blocker 和 hash line 拆成 Owner 可执行任务。它只用于真实回填交接，不发布包、不批准公开发布、不关闭 release issue。

| 项目 | 当前值 |
|---|---|
| packState | ``$($pack.packState)`` |
| strictCandidateState | ``$($pack.strictCandidateState)`` |
| strictCandidateValidationState | ``$($pack.strictCandidateValidationState)`` |
| ownerInputTaskCount | ``$($pack.ownerInputTaskCount)`` |
| realProofTaskCount | ``$($pack.realProofTaskCount)`` |
| hashCheckTaskCount | ``$($pack.hashCheckTaskCount)`` |
| blockedTaskCount | ``$($pack.blockedTaskCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Owner Input Tasks

| ID | State | Target Artifact | First Command |
|---|---|---|---|
$($ownerRows -join "`r`n")

## Real Proof Tasks

| ID | Current State | Target Artifact | Required Action |
|---|---|---|---|
$($proofRows -join "`r`n")

## Hash Check Tasks

| ID | SHA256 Matches | Current State | Path |
|---|---:|---|---|
$($hashRows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($pack.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof real backfill execution pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "PackState=$($pack.packState) OwnerInputTasks=$($pack.ownerInputTaskCount) RealProofTasks=$($pack.realProofTaskCount) HashCheckTasks=$($pack.hashCheckTaskCount) BlockedTaskCount=$($pack.blockedTaskCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
