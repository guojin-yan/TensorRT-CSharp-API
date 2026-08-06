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

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ProofTrack {
  param(
    [string]$Id,
    [string]$ProofKind,
    [string]$CurrentState,
    [string[]]$RequiredOwnerInputs,
    [string[]]$RequiredCommands,
    [string[]]$RequiredLogs,
    [string[]]$RequiredHashes,
    [string[]]$ValidatorCommands,
    [string[]]$SourceArtifacts,
    [string[]]$ForbiddenSubstitutes,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    proofKind = $ProofKind
    currentState = $CurrentState
    trackState = "blocked-real-external-proof-required"
    requiredOwnerInputs = @($RequiredOwnerInputs)
    requiredCommands = @($RequiredCommands)
    requiredLogs = @($RequiredLogs)
    requiredHashes = @($RequiredHashes)
    validatorCommands = @($ValidatorCommands)
    sourceArtifacts = @($SourceArtifacts)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    performsPublish = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = $Boundary
  }
}

$releaseCloseWorklist = Read-JsonOrNull "artifacts\final-release\release-close-proof-worklist.json"
$packageConsumerWorklist = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-worklist.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$realModelHandoff = Read-JsonOrNull "artifacts\user-acceptance\real-model-owner-handoff.json"
$releaseCloseOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-owner-input-validation.json"
$finalCloseDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$releaseCloseWorklistState = [string](Get-PropertyOrDefault -Object $releaseCloseWorklist -Name "worklistState" -DefaultValue "missing-release-close-proof-worklist")
$packageConsumerState = [string](Get-PropertyOrDefault -Object $packageConsumerWorklist -Name "worklistState" -DefaultValue "missing-package-consumer-runtime-proof-worklist")
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$linuxRunnerState = [string](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$realModelHandoffState = [string](Get-PropertyOrDefault -Object $realModelHandoff -Name "handoffState" -DefaultValue "missing-real-model-owner-handoff")
$releaseCloseOwnerInputState = [string](Get-PropertyOrDefault -Object $releaseCloseOwnerInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-owner-input-validation")
$finalCloseDecisionState = [string](Get-PropertyOrDefault -Object $finalCloseDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")
$strictCloseState = [string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")

$globalForbiddenSubstitutes = @(
  "template",
  "draft",
  "candidate",
  "scaffold",
  "runbook",
  "worklist",
  "dashboard",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "DependencyProbe",
  "build-only",
  "hash-only audit",
  "Windows handoff for Linux proof",
  "blocked-by-cuda-driver"
)

$tracks = @(
  New-ProofTrack `
    -Id "package-consumer-runtime-proof-execution" `
    -ProofKind "package-consumer-runtime" `
    -CurrentState $packageConsumerState `
    -RequiredOwnerInputs @(
      "cleanExternalConsumerRoot outside this repository",
      "consumerProjectPath with no ProjectReference, local feed, or direct .nupkg reference",
      "public package source and package identities",
      "managed/runtime nupkg SHA256 values",
      "runtimePackageKey and compatible host metadata",
      "restore/build/dependency probe/runtime smoke command and stdout/stderr summaries"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofCandidate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1"
    ) `
    -RequiredLogs @("restore log", "build log", "dependency probe log", "runtime smoke log") `
    -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "restore log SHA256", "build log SHA256", "dependency probe log SHA256", "runtime smoke log SHA256") `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof"
    ) `
    -SourceArtifacts @("artifacts/final-release/package-consumer-runtime-proof-worklist.json", "artifacts/final-release/package-consumer-runtime-proof-record-validation.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "Only validator-passing clean external consumer runtime smoke against real public package inputs can promote package-consumer-runtime proof."
  New-ProofTrack `
    -Id "post-publish-verification-execution" `
    -ProofKind "post-publish-verification" `
    -CurrentState $postPublishState `
    -RequiredOwnerInputs @(
      "selected public channel and package URLs",
      "published managed/runtime package identity and SHA256",
      "clean consumer identity outside repository",
      "host OS/architecture/CUDA/TensorRT metadata",
      "restore, native asset listing, dependency probe, and runtime smoke commands",
      "stdout/stderr summaries and reviewed log hashes"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecord.ps1"
    ) `
    -RequiredLogs @("restore log", "native asset listing", "dependency probe log", "runtime smoke log") `
    -RequiredHashes @("managed nupkg SHA256", "runtime nupkg SHA256", "native asset listing SHA256", "dependency probe log SHA256", "runtime smoke log SHA256") `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -FailOnNotProof"
    ) `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-validation.json", "artifacts/final-release/post-publish-verification-record.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "Post-publish verification requires real channel publish evidence and clean consumer smoke after publication; pre-publish dry-runs cannot close this gap."
  New-ProofTrack `
    -Id "linux-runner-proof-execution" `
    -ProofKind "linux-runner-proof" `
    -CurrentState $linuxRunnerState `
    -RequiredOwnerInputs @(
      "real linux-x64 runner identity",
      "CUDA/TensorRT/cuDNN host metadata",
      "configure/build/package/consumer-copy commands",
      "runtime asset listing and command evidence"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxHandoffIndex.ps1",
      "cmake --preset linux-x64-trt11-cuda13-release",
      "cmake --build --preset linux-x64-trt11-cuda13-release --parallel"
    ) `
    -RequiredLogs @("cmake configure log", "cmake build log", "runtime asset collection log", "package consumer copy log") `
    -RequiredHashes @("native asset listing SHA256", "runtime nupkg SHA256", "runner evidence log SHA256") `
    -ValidatorCommands @("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidence.ps1 -Strict") `
    -SourceArtifacts @("artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "Linux proof must come from a real Linux runner; Windows handoff and template-only records remain non-proof."
  New-ProofTrack `
    -Id "real-model-runtime-proof-execution" `
    -ProofKind "real-model-runtime" `
    -CurrentState "$sampleRunState; handoff=$realModelHandoffState" `
    -RequiredOwnerInputs @(
      "approved real model asset and sidecar",
      "real input sample and expected output criteria",
      "runtime package key and host metadata",
      "sample execution command and result summary"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1",
      "dotnet run --project .\samples\ComputerVision\01.Classification\Classification.csproj -- --runtime-package-key <key>",
      "dotnet run --project .\applications\YoloVision\YoloVision.csproj -- --runtime-package-key <key>"
    ) `
    -RequiredLogs @("sample runtime log", "model acquisition log", "sidecar audit log") `
    -RequiredHashes @("model file SHA256", "input file SHA256", "sample runtime log SHA256", "sidecar SHA256") `
    -ValidatorCommands @("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -FailOnNotProof") `
    -SourceArtifacts @("artifacts/user-acceptance/sample-run-evidence-record-validation.json", "artifacts/user-acceptance/real-model-owner-handoff.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "Real-model proof requires real model assets, real input execution, and validator-passing runtime evidence; build-only and dependency-probe records are not enough."
  New-ProofTrack `
    -Id "release-close-owner-input-execution" `
    -ProofKind "release-close-owner-input" `
    -CurrentState "$releaseCloseOwnerInputState; finalDecision=$finalCloseDecisionState" `
    -RequiredOwnerInputs @(
      "release issue id and URL",
      "rollback plan, rollback owner, rollback trigger",
      "owner final close decision and decision timestamp UTC",
      "post-publish validation SHA256",
      "release evidence bundle SHA256",
      "release close preflight SHA256",
      "stale claims audit SHA256"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueFinalCloseDecisionTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordCandidate.ps1"
    ) `
    -RequiredLogs @("owner final close review log", "rollback review log", "strict close validator log") `
    -RequiredHashes @("release evidence bundle SHA256", "post-publish validation SHA256", "release close preflight SHA256", "stale claims audit SHA256") `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict"
    ) `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-owner-input-validation.json", "artifacts/final-release/release-issue-final-close-decision-validation.json", "artifacts/final-release/release-issue-close-record-candidate-validation.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "Owner input shape validity is not close approval; real proof hashes, rollback review, and final owner decision must be present."
  New-ProofTrack `
    -Id "strict-close-validation-execution" `
    -ProofKind "strict-close-validation" `
    -CurrentState $strictCloseState `
    -RequiredOwnerInputs @(
      "release-issue-close-record.json projected from real proof and owner inputs",
      "validator-passing post-publish proof",
      "current release evidence bundle hash",
      "rollback plan and final owner close decision"
    ) `
    -RequiredCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
    ) `
    -RequiredLogs @("strict close validator log") `
    -RequiredHashes @("release-issue-close-record SHA256", "release evidence bundle SHA256", "post-publish validation SHA256") `
    -ValidatorCommands @("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady") `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-validation.json", "artifacts/final-release/release-close-proof-worklist.json") `
    -ForbiddenSubstitutes $globalForbiddenSubstitutes `
    -Boundary "The strict close validator is the final close gate; template-only close records and blocked validation cannot close the release issue."
)

$blockedTrackCount = @($tracks | Where-Object { [string]$_.trackState -like "blocked*" }).Count
$sourceArtifacts = @(
  "artifacts/final-release/release-close-proof-worklist.json",
  "artifacts/final-release/package-consumer-runtime-proof-worklist.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/real-model-owner-handoff.json",
  "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/release-issue-close-record-validation.json"
)

$recordOut = [pscustomobject]@{
  recordKind = "real-external-proof-backfill-execution-bundle"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  bundleState = "blocked-real-external-proof-execution-required"
  releaseCloseProofWorklistState = $releaseCloseWorklistState
  trackCount = $tracks.Count
  blockedTrackCount = $blockedTrackCount
  proofTracks = @($tracks)
  sourceArtifacts = $sourceArtifacts
  forbiddenSubstitutes = $globalForbiddenSubstitutes
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This execution bundle organizes real external proof backfill work for the owner. It is not proof, not a package push, not public release approval, not post-publish verification, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-external-proof-backfill-execution-bundle.json"
$markdownPath = Join-Path $OutputRoot "real-external-proof-backfill-execution-bundle.md"
$recordOut | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real External Proof Backfill Execution Bundle")
$lines.Add("")
$lines.Add("`real-external-proof-backfill-execution-bundle` 把真实外部 proof 回填工作收束成 Owner 可执行任务包。它不是 proof，不发布包，不批准公开发布，也不关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| bundleState | ``$(ConvertTo-MarkdownCell $recordOut.bundleState)`` |")
$lines.Add("| releaseCloseProofWorklistState | ``$(ConvertTo-MarkdownCell $recordOut.releaseCloseProofWorklistState)`` |")
$lines.Add("| trackCount | ``$($recordOut.trackCount)`` |")
$lines.Add("| blockedTrackCount | ``$($recordOut.blockedTrackCount)`` |")
$lines.Add("| canPublishPublicly | ``$($recordOut.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Proof Tracks")
$lines.Add("")
$lines.Add("| Id | Proof Kind | Current State | Validators |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($track in $tracks) {
  $lines.Add("| $(ConvertTo-MarkdownCell $track.id) | $(ConvertTo-MarkdownCell $track.proofKind) | $(ConvertTo-MarkdownCell $track.currentState) | $(ConvertTo-MarkdownCell (($track.validatorCommands -join '; '))) |")
}
$lines.Add("")
$lines.Add("## Required Owner Inputs")
$lines.Add("")
foreach ($track in $tracks) {
  $lines.Add("### $(ConvertTo-MarkdownCell $track.id)")
  foreach ($input in @($track.requiredOwnerInputs)) { $lines.Add("- $(ConvertTo-MarkdownCell $input)") }
  $lines.Add("")
}
$lines.Add("## Forbidden Substitutes")
$lines.Add("")
foreach ($item in $globalForbiddenSubstitutes) { $lines.Add("- ``$(ConvertTo-MarkdownCell $item)``") }
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real external proof backfill execution bundle written to $jsonPath"
Write-Host "Real external proof backfill execution bundle markdown written to $markdownPath"
Write-Host "BundleState=$($recordOut.bundleState) Tracks=$($recordOut.trackCount) Blocked=$($recordOut.blockedTrackCount)"
