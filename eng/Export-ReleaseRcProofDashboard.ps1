[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
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

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", "<br>")
}

function New-RcProofBlocker {
  param(
    [string]$Id,
    [string]$ProofClass,
    [string]$State,
    [string]$RequiredValidator,
    [string[]]$RequiredOwnerInputs,
    [string[]]$SourceArtifacts,
    [string]$NextCommand,
    [string[]]$CannotUse,
    [string[]]$NonSubstituteProofKinds
  )

  [pscustomobject]@{
    id = $Id
    proofClass = $ProofClass
    state = $State
    ready = $false
    requiredValidator = $RequiredValidator
    requiredOwnerInputs = @($RequiredOwnerInputs)
    sourceArtifacts = @($SourceArtifacts)
    nextCommand = $NextCommand
    cannotUse = @($CannotUse)
    nonSubstituteProofKinds = @($NonSubstituteProofKinds)
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$runtimeMatrix = Read-JsonOrNull "artifacts\final-release\release-runtime-proof-execution-matrix.json"
$readiness = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$evidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$packageConsumerPack = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-execution-pack.json"
$packageConsumerValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-pack-validation.json"
$callbackPack = Read-JsonOrNull "artifacts\final-release\callback-runtime-proof-execution-pack.json"
$callbackValidation = Read-JsonOrNull "artifacts\final-release\callback-runtime-proof-execution-pack-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$ownerProofInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"

$matrixState = [string](Get-PropertyOrDefault -Object $runtimeMatrix -Name "matrixState" -DefaultValue "missing-release-runtime-proof-execution-matrix")
$readinessState = [string](Get-PropertyOrDefault -Object $readiness -Name "readinessState" -DefaultValue "missing-release-proof-readiness-snapshot")
$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$evidenceState = [string](Get-PropertyOrDefault -Object $evidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$packagePackState = [string](Get-PropertyOrDefault -Object $packageConsumerPack -Name "packState" -DefaultValue "missing-package-consumer-runtime-proof-execution-pack")
$packageValidationState = [string](Get-PropertyOrDefault -Object $packageConsumerValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-pack-validation")
$packageMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $packageConsumerPack -Name "missingOwnerInputCount" -DefaultValue -1)
$callbackPackState = [string](Get-PropertyOrDefault -Object $callbackPack -Name "packState" -DefaultValue "missing-callback-runtime-proof-execution-pack")
$callbackValidationState = [string](Get-PropertyOrDefault -Object $callbackValidation -Name "validationState" -DefaultValue "missing-callback-runtime-proof-execution-pack-validation")
$callbackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $callbackPack -Name "missingOwnerInputCount" -DefaultValue -1)
$ownerProofInputValidationState = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "validationState" -DefaultValue "missing-release-owner-proof-input-record-validation")
$ownerProofInputClassification = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$ownerProofInputCanPromote = [bool](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$ownerProofInputFailedItemCount = [int](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "failedValidationItemCount" -DefaultValue -1)

$nonSubstitutes = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "build-only",
  "parse-only",
  "sidecar-only",
  "dependency-probe-only",
  "Skipped=True",
  "blocked-by-cuda-driver",
  "managed-readiness",
  "managed-readiness-only",
  "callback-allocator-readiness-snapshot",
  "CallbackAllocatorReadinessSnapshot",
  "TensorRtCallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only"
)

$blockers = @(
  New-RcProofBlocker `
    -Id "owner-authorization" `
    -ProofClass "owner-authorization" `
    -State "ownerPackageState=$ownerPackageState; ownerProofInputValidationState=$ownerProofInputValidationState; ownerProofInputClassification=$ownerProofInputClassification; canPromoteOwnerProofInput=$ownerProofInputCanPromote; failedValidationItemCount=$ownerProofInputFailedItemCount; explicit owner authorization still required" `
    -RequiredValidator "Test-ReleaseOwnerProofInputRecord.ps1 + Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -RequiredOwnerInputs @("release-owner-proof-input-record.json", "owner name", "owner decision id", "approval timestamp", "target channel", "release issue URL", "selected channel URL", "managed/runtime package URLs and SHA256", "clean consumer logs and SHA256", "host metadata", "rollback plan", "approved proof bundle SHA256") `
    -SourceArtifacts @("artifacts/final-release/release-owner-proof-input-record-template.json", "artifacts/final-release/release-owner-proof-input-record-validation.json", "artifacts/final-release/owner-release-execution-package.json", "artifacts/final-release/release-owner-approval-input-validation.json", "artifacts/final-release/owner-authorized-publish-command-plan-validation.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1" `
    -CannotUse @("template", "draft", "dashboard", "owner guidance", "readiness snapshot", "schema-only") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -State "packState=$packagePackState; validationState=$packageValidationState; missingOwnerInputCount=$packageMissingOwnerInputCount" `
    -RequiredValidator "Test-PackageConsumerRuntimeProofPack.ps1 + Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredOwnerInputs @("release-owner-proof-input-record.json package/runtime evidence", "clean external consumer outside repository", "real package source", "managed/runtime nupkg SHA256", "native asset listing SHA256", "restore/build/runtime smoke logs", "runtime smoke log SHA256", "host metadata", "no ProjectReference") `
    -SourceArtifacts @("artifacts/final-release/package-consumer-runtime-proof-execution-pack.json", "artifacts/final-release/package-consumer-runtime-proof-pack-validation.json", "artifacts/final-release/external-runtime-proof-record.json", "artifacts/final-release/release-owner-proof-input-record-validation.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof" `
    -CannotUse @("local feed", "ProjectReference", "dependency-probe-only", "build-only", "blocked-by-cuda-driver", "managed-readiness", "schema-only") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "callback-runtime-proof" `
    -ProofClass "real-callback-runtime-proof" `
    -State "packState=$callbackPackState; validationState=$callbackValidationState; missingOwnerInputCount=$callbackMissingOwnerInputCount" `
    -RequiredValidator "Test-CallbackRuntimeProofExecutionPack.ps1 and real callback runtime validator with InvocationCount>0" `
    -RequiredOwnerInputs @("compatible CUDA/TensorRT host", "real package consumer root", "runtime smoke command", "runtime smoke log SHA256", "callback invocation marker", "InvocationCount>0", "IsRealCallbackRuntimeProof=true evidence") `
    -SourceArtifacts @("artifacts/final-release/callback-runtime-proof-execution-pack.json", "artifacts/final-release/callback-runtime-proof-execution-pack-validation.json", "docs/articles/zh-cn/real-callback-runtime-evidence-schema.md") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CallbackRuntimeProofExecutionPack.ps1" `
    -CannotUse @("TensorRtCallbackAllocatorReadinessSnapshot", "RuntimeEvidenceKind=managed-readiness", "precheck-only", "dry-run-only", "schema-only") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -State "linuxRuntimePackageKey=$LinuxRuntimePackageKey; real Linux runner proof missing" `
    -RequiredValidator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -RequiredOwnerInputs @("real Linux x64 CUDA/TensorRT host", "Linux runner log", "runtime package key", "host metadata", "log SHA256") `
    -SourceArtifacts @("artifacts/final-release/linux-runner-proof-execution-pack.json", "artifacts/final-release/linux-runner-proof-pack-validation.json", "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1" `
    -CannotUse @("Windows handoff", "template-only record", "blocked-by-cuda-driver") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -State "Classification/YoloVision real model runtime proof missing" `
    -RequiredValidator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -RequiredOwnerInputs @("real model assets", "license metadata", "input artifact SHA256", "output artifact SHA256", "sample runner log SHA256", "host metadata") `
    -SourceArtifacts @("artifacts/final-release/real-case-proof-execution-pack.json", "artifacts/final-release/real-case-evidence-record-template.json", "artifacts/user-acceptance/sample-run-evidence-record-validation.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1" `
    -CannotUse @("model candidate list", "build-only", "parse-only", "sidecar-only", "missing hashes") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "post-publish-verification" `
    -ProofClass "post-publish-verification" `
    -State "post-publish proof missing; publication has not been performed" `
    -RequiredValidator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredOwnerInputs @("release-owner-proof-input-record.json selected channel fields", "real published package URL", "downloaded package hashes", "clean external consumer", "post-publish runtime smoke log", "host metadata", "no ProjectReference") `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-record-template.json", "artifacts/final-release/post-publish-verification-validation.json", "artifacts/final-release/post-publish-clean-consumer-project-scan.json", "artifacts/final-release/release-owner-proof-input-record-validation.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof" `
    -CannotUse @("template", "draft", "local feed", "ProjectReference", "dependency-probe-only", "clean consumer scan only") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "stale-release-claims" `
    -ProofClass "claim-audit" `
    -State "findingCount=$staleFindingCount" `
    -RequiredValidator "Test-StaleReleaseClaims.ps1" `
    -RequiredOwnerInputs @("release-facing docs without premature ready/passed/published/complete claims", "stale-release-claims-audit.json with findingCount=0") `
    -SourceArtifacts @("artifacts/final-release/stale-release-claims-audit.json", "README.md", "README.zh-CN.md", "docs/articles/zh-cn") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1" `
    -CannotUse @("release-ready claim", "runtime proof complete claim", "NuGet published claim", "canPublishPublicly=true claim") `
    -NonSubstituteProofKinds $nonSubstitutes

  New-RcProofBlocker `
    -Id "release-close-preflight" `
    -ProofClass "release-close-preflight" `
    -State "matrixState=$matrixState; readinessState=$readinessState; preflightState=$preflightState; evidenceState=$evidenceState" `
    -RequiredValidator "Export-ReleaseClosePreflight.ps1" `
    -RequiredOwnerInputs @("all upstream real proof validators passed", "owner authorization present", "post-publish verification present") `
    -SourceArtifacts @("artifacts/final-release/release-runtime-proof-execution-matrix.json", "artifacts/final-release/release-proof-readiness-snapshot.json", "artifacts/final-release/release-close-preflight.json", "artifacts/final-release/release-evidence-bundle.json") `
    -NextCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1" `
    -CannotUse @("dashboard", "readiness snapshot", "guidance pack", "blocked preflight") `
    -NonSubstituteProofKinds $nonSubstitutes
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-rc-proof-dashboard"
  dashboardState = "blocked-real-proof-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  proofBlockerCount = $blockers.Count
  readyProofBlockerCount = 0
  blockedProofBlockerCount = $blockers.Count
  nonSubstituteProofKinds = $nonSubstitutes
  proofBlockers = $blockers
  sourceArtifacts = @($blockers | ForEach-Object { $_.sourceArtifacts } | Select-Object -Unique)
  boundary = "Release RC proof dashboard is owner guidance only. It does not publish, close release issues, or convert readiness, precheck, dry-run, schema-only, template, runbook, dashboard, or collection package evidence into proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-rc-proof-dashboard.json"
$markdownPath = Join-Path $artifactRoot "release-rc-proof-dashboard.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $blockers | ForEach-Object {
  $inputs = (@($_.requiredOwnerInputs) -join "<br>").Replace("|", "\|")
  $sources = (@($_.sourceArtifacts) -join "<br>").Replace("|", "\|")
  $cannotUse = (@($_.cannotUse) -join "<br>").Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredValidator) | $inputs | $sources | $(ConvertTo-MarkdownCell $_.nextCommand) | $cannotUse |"
}

$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release RC Proof Dashboard

生成时间：$($record.generatedAtUtc)

## Summary

- dashboard state: ``$($record.dashboardState)``
- performs publish: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- proof blocker count: ``$($record.proofBlockerCount)``
- blocked proof blocker count: ``$($record.blockedProofBlockerCount)``

## Proof Blockers

| ID | Proof class | Ready | State | Validator | Required owner inputs | Source artifacts | Next command | Cannot use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release RC proof dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DashboardState=$($record.dashboardState)"
Write-Output "ProofBlockerCount=$($record.proofBlockerCount)"
Write-Output "CanPublishPublicly=False"
