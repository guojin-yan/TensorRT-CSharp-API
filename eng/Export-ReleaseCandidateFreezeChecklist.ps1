[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function New-OwnerDecisionItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CurrentStatus,
    [string]$OwnerAction,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    currentStatus = $CurrentStatus
    ownerAction = $OwnerAction
    requiredEvidence = $RequiredEvidence
    state = "pending-owner-action"
    boundary = $Boundary
  }
}

function New-ChecklistStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Status,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    status = $Status
    boundary = $Boundary
  }
}

$summary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
if ($null -eq $summary) {
  throw "Missing artifacts/release/release-candidate-freeze-summary.json. Run Export-ReleaseCandidateFreezeSummary.ps1 before Export-ReleaseCandidateFreezeChecklist.ps1."
}

$freezeState = [string](Get-PropertyOrDefault -Object $summary -Name "freezeState" -DefaultValue "missing-freeze-summary")
$canPublish = [bool](Get-PropertyOrDefault -Object $summary -Name "canPublish" -DefaultValue $false)
$canPromote = [bool](Get-PropertyOrDefault -Object $summary -Name "canPromote" -DefaultValue $false)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $summary -Name "canCloseReleaseIssue" -DefaultValue $false)
$blockingItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "blockingItemCount" -DefaultValue -1)
$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $summary -Name "releaseEvidenceBundleState" -DefaultValue "missing-release-evidence-bundle")
$releasePackageProofState = [string](Get-PropertyOrDefault -Object $summary -Name "releasePackageProofState" -DefaultValue "missing-release-package-proof-bundle")
$ownerApprovalStatus = [string](Get-PropertyOrDefault -Object $summary -Name "ownerApprovalInputValidationStatus" -DefaultValue "missing-owner-approval-input-validation")
$ownerDecisionState = [string](Get-PropertyOrDefault -Object $summary -Name "ownerDecisionState" -DefaultValue "missing-owner-decision-record")
$publishChecklistState = [string](Get-PropertyOrDefault -Object $summary -Name "publishExecutionChecklistState" -DefaultValue "missing-release-publish-execution-checklist")
$promotionIssueState = [string](Get-PropertyOrDefault -Object $summary -Name "promotionIssueState" -DefaultValue "missing-release-promotion-issue-record")
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $summary -Name "runtimeProofStatus" -DefaultValue "missing-runtime-proof")
$realExternalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $summary -Name "realExternalRuntimeProofReady" -DefaultValue $false)
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofState" -DefaultValue "missing")
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $summary -Name "externalRuntimeProofClassification" -DefaultValue "missing-proof-classification")
$postPublishState = [string](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationState" -DefaultValue "missing-post-publish-verification")
$realPostPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $summary -Name "realPostPublishVerificationReady" -DefaultValue $false)
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $summary -Name "postPublishCommandsReady" -DefaultValue $false)
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $summary -Name "postPublishStdoutStderrSummaryReady" -DefaultValue $false)
$postPublishCleanConsumerProjectScanState = [string](Get-PropertyOrDefault -Object $summary -Name "postPublishCleanConsumerProjectScanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")
$postPublishVerificationInputDraftKind = [string](Get-PropertyOrDefault -Object $summary -Name "postPublishVerificationInputDraftKind" -DefaultValue "missing-post-publish-verification-record-input-draft")
$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $summary -Name "releaseClosePreflightState" -DefaultValue "missing-release-close-preflight")
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "releaseClosePreflightFailedItemCount" -DefaultValue -1)
$staleFindingCount = [int](Get-PropertyOrDefault -Object $summary -Name "staleReleaseClaimsFindingCount" -DefaultValue -1)

$ownerDecisionItems = @(
  New-OwnerDecisionItem -Id "release-candidate-freeze-summary" -Title "Freeze summary" -CurrentStatus ("freezeState=$freezeState; blockingItemCount=$blockingItemCount") -OwnerAction "Review the freeze summary before authorizing any channel action." -RequiredEvidence "artifacts/release/release-candidate-freeze-summary.json" -Boundary "Freeze summary is review evidence, not publish execution."
  New-OwnerDecisionItem -Id "release-evidence-bundle" -Title "Release evidence bundle" -CurrentStatus ("bundleState=$releaseEvidenceBundleState") -OwnerAction "Resolve incomplete source evidence before publish authorization." -RequiredEvidence "artifacts/final-release/release-evidence-bundle.json" -Boundary "Evidence aggregation is not owner approval."
  New-OwnerDecisionItem -Id "release-package-proof-bundle" -Title "Release package proof bundle" -CurrentStatus ("proofState=$releasePackageProofState") -OwnerAction "Confirm local package proof and note that public package proof is still separate." -RequiredEvidence "artifacts/final-release/release-package-proof-bundle.json" -Boundary "Local package proof bundle is not public package proof."
  New-OwnerDecisionItem -Id "owner-approval-input" -Title "Owner approval input" -CurrentStatus ("validationStatus=$ownerApprovalStatus") -OwnerAction "Fill and validate a non-template owner approval input record." -RequiredEvidence "artifacts/final-release/release-owner-approval-input-record.json" -Boundary "Template-only owner input cannot approve publication."
  New-OwnerDecisionItem -Id "owner-decision-record" -Title "Owner decision record" -CurrentStatus ("recordState=$ownerDecisionState") -OwnerAction "Record final owner decision after proof and channel checks." -RequiredEvidence "artifacts/final-release/release-owner-decision-record.json" -Boundary "Generated pending decision is not authorization."
  New-OwnerDecisionItem -Id "external-runtime-proof" -Title "Compatible-host external runtime proof" -CurrentStatus ("state=$externalRuntimeProofState; classification=$externalRuntimeProofClassification; ready=$realExternalRuntimeProofReady; runtimeProofStatus=$runtimeProofStatus") -OwnerAction "Run package-consumer smoke on a compatible CUDA host and attach a validated real external-runtime-proof-record.json." -RequiredEvidence "artifacts/final-release/external-runtime-proof-record.json" -Boundary "blocked-by-cuda-driver, template-only, draft-only, runbook, collection bundle, and dependency-probe-only are not smoke passed."
  New-OwnerDecisionItem -Id "publish-execution-checklist" -Title "Publish execution checklist" -CurrentStatus ("executionState=$publishChecklistState; canPublish=$canPublish") -OwnerAction "Only after owner authorization, use channel placeholders manually." -RequiredEvidence "artifacts/final-release/release-publish-execution-checklist.json" -Boundary "The checklist does not execute dotnet nuget push or uploads."
  New-OwnerDecisionItem -Id "promotion-issue-record" -Title "Promotion issue record" -CurrentStatus ("promotionState=$promotionIssueState; canPromote=$canPromote") -OwnerAction "Use the issue record as owner-facing review material." -RequiredEvidence "artifacts/final-release/release-promotion-issue-record.json" -Boundary "Promotion issue generation is not approval."
  New-OwnerDecisionItem -Id "post-publish-verification" -Title "Post-publish verification" -CurrentStatus ("state=$postPublishState; ready=$realPostPublishVerificationReady; commandsReady=$postPublishCommandsReady; stdoutStderrSummaryReady=$postPublishStdoutStderrSummaryReady; canCloseReleaseIssue=$canCloseReleaseIssue") -OwnerAction "After real channel publish, run clean consumer restore/build/smoke and fill post-publish-verification-record.json." -RequiredEvidence "artifacts/final-release/post-publish-verification-record.json" -Boundary "Post-publish template-only or missing stdout/stderr summary cannot close release issue."
  New-OwnerDecisionItem -Id "post-publish-clean-consumer-project-scan" -Title "Post-publish clean consumer project scan" -CurrentStatus ("scanState=$postPublishCleanConsumerProjectScanState") -OwnerAction "Run against the real external clean consumer project after authorized publication." -RequiredEvidence "artifacts/final-release/post-publish-clean-consumer-project-scan.json" -Boundary "Clean consumer scan is helper evidence only and cannot close the release issue."
  New-OwnerDecisionItem -Id "post-publish-verification-input-draft" -Title "Post-publish verification input draft" -CurrentStatus ("recordKind=$postPublishVerificationInputDraftKind") -OwnerAction "Use only as fill guidance for the real post-publish verification record." -RequiredEvidence "artifacts/final-release/post-publish-verification-record.input-draft.json" -Boundary "Input draft is helper evidence only and cannot close the release issue."
  New-OwnerDecisionItem -Id "release-close-preflight" -Title "Release close preflight" -CurrentStatus ("preflightState=$releaseClosePreflightState; failedItemCount=$releaseClosePreflightFailedItemCount") -OwnerAction "Rerun after real external runtime proof, owner authorization, and post-publish proof are attached." -RequiredEvidence "artifacts/final-release/release-close-preflight.json" -Boundary "Release close preflight aggregates proof gaps and is not proof."
  New-OwnerDecisionItem -Id "stale-release-claims" -Title "Stale release claims audit" -CurrentStatus ("findingCount=$staleFindingCount") -OwnerAction "Keep release-facing docs and generated issue text free of stale proof claims." -RequiredEvidence "artifacts/final-release/stale-release-claims-audit.json" -Boundary "Text audit is a guardrail, not publication approval."
)

$publishPreflight = @(
  New-ChecklistStep -Id "no-publish-before-owner-authorization" -Title "Confirm owner authorization exists" -Status "pending" -Boundary "No script in this stage performs publish."
  New-ChecklistStep -Id "runtime-key" -Title "Confirm runtime package key" -Status $RuntimePackageKey -Boundary "Smoke command must include --runtime-package-key $RuntimePackageKey."
  New-ChecklistStep -Id "compatible-host-proof" -Title "Confirm compatible-host runtime proof" -Status ("ready=$realExternalRuntimeProofReady") -Boundary "blocked-by-cuda-driver is not smoke passed."
  New-ChecklistStep -Id "nvidia-redistribution" -Title "Confirm NVIDIA redistribution rights" -Status "pending-owner-legal-review" -Boundary "Do not publicly publish NVIDIA runtime components without approval."
  New-ChecklistStep -Id "package-hashes" -Title "Confirm package hashes" -Status "pending-owner-review" -Boundary "Package identity and SHA256 hashes must be auditable."
)

$postPublishActions = @(
  New-ChecklistStep -Id "clean-consumer" -Title "Create clean consumer" -Status "pending-after-publish" -Boundary "Existing repo build is not clean consumer proof."
  New-ChecklistStep -Id "restore-from-channel" -Title "Restore from selected channel" -Status "pending-after-publish" -Boundary "ProjectReference or local bin output is not channel proof."
  New-ChecklistStep -Id "smoke-with-runtime-key" -Title "Run smoke with runtime package key" -Status "pending-after-publish" -Boundary "Smoke command must include --runtime-package-key $RuntimePackageKey."
  New-ChecklistStep -Id "stdout-stderr-summary" -Title "Capture stdout/stderr summary" -Status "pending-after-publish" -Boundary "Raw logs need summarized stdout/stderr for issue review."
  New-ChecklistStep -Id "run-clean-consumer-scan" -Title "Run clean consumer scan" -Status "pending-after-publish" -Boundary "Scan output is helper evidence only."
  New-ChecklistStep -Id "fill-input-draft" -Title "Fill post-publish input draft" -Status "pending-after-publish" -Boundary "Input draft does not replace the real verification record."
  New-ChecklistStep -Id "validate-post-publish-record" -Title "Validate post-publish record" -Status "pending-after-publish" -Boundary "Only the validator can promote the record to close-ready."
  New-ChecklistStep -Id "rerun-release-close-preflight" -Title "Rerun release close preflight" -Status "pending-after-publish" -Boundary "Close preflight must remain blocked until real proof gates pass."
)

$publishPlaceholders = @(
  [pscustomobject]@{ id = "nuget-org"; command = "dotnet nuget push <package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json"; performsPublish = $false; boundary = "Placeholder only; owner must run manually after authorization." }
  [pscustomobject]@{ id = "github-packages"; command = "dotnet nuget push <package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>"; performsPublish = $false; boundary = "Placeholder only; owner must run manually after authorization." }
  [pscustomobject]@{ id = "github-release-assets"; command = "gh release upload <tag> <package>.nupkg <package>.sha256"; performsPublish = $false; boundary = "Placeholder only; owner must run manually after authorization." }
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-candidate-freeze-checklist"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  freezeState = $freezeState
  blockingItemCount = $blockingItemCount
  canPublish = $canPublish
  canPromote = $canPromote
  canCloseReleaseIssue = $canCloseReleaseIssue
  performsPublish = $false
  requiresHumanOwner = $true
  ownerDecisionItems = $ownerDecisionItems
  publishPreflight = $publishPreflight
  publishPlaceholders = $publishPlaceholders
  postPublishActions = $postPublishActions
  sourceEvidence = @(
    "artifacts/release/release-candidate-freeze-summary.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/release-publish-execution-checklist.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
    "artifacts/final-release/post-publish-verification-record.input-draft.json",
    "artifacts/final-release/release-close-preflight.json"
  )
  safetyNotes = @(
    "Release candidate freeze is not publish.",
    "Owner approval and owner decision are not proof by themselves.",
    "No NuGet, GitHub Packages, GitHub Release, delete, delist, or withdraw action is executed by this checklist.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Collection packages are copyable owner guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "Post-publish clean consumer scan and input draft are helper artifacts only; they cannot close the release issue.",
    "Release close preflight aggregates real-proof gaps but cannot substitute external runtime proof, owner authorization, or post-publish verification proof.",
    "Post-publish verification can close a release issue only after real channel publication and clean consumer smoke."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-candidate-freeze-checklist.json"
$markdownPath = Join-Path $outputRoot "release-candidate-freeze-checklist.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Candidate Freeze Checklist")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Freeze state: ``$freezeState``")
$lines.Add("")
$lines.Add("Can publish: ``$canPublish``")
$lines.Add("")
$lines.Add("Can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("")
$lines.Add("This checklist is owner-facing review material. It does not publish packages, upload assets, delete, delist, or withdraw anything.")
$lines.Add("")
$lines.Add("## Owner Decision Items")
$lines.Add("")
$lines.Add("| ID | Current status | Owner action |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $ownerDecisionItems) {
  $lines.Add("| ``$($item.id)`` | $($item.currentStatus.Replace("|", "\|")) | $($item.ownerAction.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Publish Placeholders")
$lines.Add("")
$lines.Add("These commands are text placeholders only. They are not executed by this script.")
$lines.Add("")
$lines.Add('```powershell')
foreach ($placeholder in $publishPlaceholders) {
  $lines.Add($placeholder.command)
}
$lines.Add('```')
$lines.Add("")
$lines.Add("## Post-publish Actions")
$lines.Add("")
$lines.Add("| ID | Status | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $postPublishActions) {
  $lines.Add("| ``$($item.id)`` | ``$($item.status)`` | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate freeze checklist written to $jsonPath"
Write-Host "Release candidate freeze checklist written to $markdownPath"
