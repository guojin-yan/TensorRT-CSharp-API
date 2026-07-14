[CmdletBinding()]
param(
  [string]$ReleaseEvidenceBundlePath,
  [string]$OutputDirectory,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($ReleaseEvidenceBundlePath)) {
  $ReleaseEvidenceBundlePath = Join-Path $RepositoryRoot "artifacts\final-release\release-evidence-bundle.json"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

function Write-Utf8FileWithRetry {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject,
    [int]$MaxAttempts = 8,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $content = @($InputObject) -join [Environment]::NewLine
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($LiteralPath)), [Guid]::NewGuid().ToString("N"))
  [IO.File]::WriteAllText($tempPath, $content + [Environment]::NewLine, $script:utf8)

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Move-Item -LiteralPath $tempPath -Destination $LiteralPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) {
        throw
      }

      Start-Sleep -Milliseconds $DelayMilliseconds
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $MaxAttempts) {
        throw
      }

      Start-Sleep -Milliseconds $DelayMilliseconds
    }
  }
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
    return $Object.$Name
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

function Test-TextContainsAny {
  param(
    [AllowNull()][string]$Text,
    [string[]]$Markers
  )

  if ([string]::IsNullOrWhiteSpace($Text)) {
    return $false
  }

  foreach ($marker in @($Markers)) {
    if ([string]::IsNullOrWhiteSpace($marker)) {
      continue
    }

    if ($Text.IndexOf([string]$marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  return $false
}

if (-not (Test-Path -LiteralPath $ReleaseEvidenceBundlePath -PathType Leaf)) {
  throw "Release evidence bundle not found: $ReleaseEvidenceBundlePath"
}

$bundle = Get-Content -LiteralPath $ReleaseEvidenceBundlePath -Raw -Encoding utf8 | ConvertFrom-Json
$evidenceItems = @((Get-PropertyOrDefault -Object $bundle -Name "evidenceItems" -DefaultValue @()))
$nonSubstituteProofKinds = @((Get-PropertyOrDefault -Object $bundle -Name "nonSubstituteProofKinds" -DefaultValue @()))

$designGateIds = @(
  "error-recorder-diagnostics-design-gate",
  "dimension-expression-snapshot-design-gate",
  "calibrator-metadata-design-gate",
  "runtime-deserialization-boundary-precheck",
  "runtime-deserialization-dependency-diagnostics"
)

$mustRemainFailedIds = @(
  "error-recorder-diagnostics-design-gate",
  "dimension-expression-snapshot-design-gate",
  "calibrator-metadata-design-gate",
  "runtime-deserialization-boundary-precheck",
  "runtime-deserialization-dependency-diagnostics",
  "runtime-proof-blocker-owner-action",
  "linux-runner-validation",
  "external-runtime-proof-validation",
  "external-runtime-proof-owner-handoff",
  "compatible-host-runtime-proof-runbook",
  "compatible-host-runtime-proof-collection-bundle",
  "external-runtime-proof-backfill-plan",
  "post-publish-verification-backfill-plan",
  "external-runtime-proof-collection-package",
  "post-publish-verification-collection-package",
  "post-publish-clean-consumer-project-scan",
  "post-publish-verification-input-draft",
  "post-publish-verification-owner-input",
  "post-publish-verification-record",
  "owner-release-execution-package-validation",
  "real-model-and-package-proof-input-package",
  "release-close-gap-dashboard",
  "compatible-host-proof-execution-pack",
  "owner-proof-backfill-execution-pack",
  "owner-proof-execution-handoff",
  "owner-external-proof-input-preflight",
  "owner-proof-input-repair-pack",
  "owner-proof-input-draft-pack",
  "owner-external-proof-backfill-orchestrator",
  "package-consumer-runtime-proof-owner-input",
  "package-consumer-runtime-proof-record",
  "package-consumer-external-smoke-scaffold",
  "package-consumer-runtime-proof-candidate",
  "release-issue-close-record-owner-input",
  "release-issue-close-record-candidate",
  "final-evidence-freeze",
  "release-issue-final-close-decision",
  "real-external-proof-overlay-pack",
  "release-issue-close-record-overlay-candidate",
  "owner-external-execution-result-backfill-kit",
  "owner-input-cross-hash-audit",
  "release-close-strict-record-candidate",
  "owner-proof-real-backfill-execution-pack",
  "release-issue-close-record-real-input-map",
  "owner-proof-real-input-convergence",
  "owner-input-contract-convergence",
  "release-close-final-owner-runbook",
  "runtime-proof-execution-input-record",
  "owner-runtime-proof-execution-runbook",
  "release-close-strict-validation-bridge",
  "owner-runtime-proof-result-input-template",
  "owner-runtime-proof-result-input-validation",
  "runtime-proof-lane-dry-run-summary",
  "release-close-strict-dry-run-summary",
  "owner-external-proof-execution-bundle",
  "owner-external-proof-execution-result-import",
  "real-external-proof-record-import-validator",
  "release-close-owner-input-bridge",
  "public-package-proof-owner-input",
  "post-publish-proof-owner-confirmation",
  "release-close-public-proof-bridge",
  "release-issue-close-final-owner-decision-audit",
  "final-post-publish-audit-pack",
  "release-docs-and-nuget-metadata-audit",
  "post-publish-user-verification-pack",
  "public-package-download-proof-owner-execution-pack",
  "release-candidate-final-freeze-manifest",
  "public-publish-owner-manual-command-handoff",
  "final-release-close-blocker-dashboard",
  "public-publish-result-owner-input",
  "public-publish-result-import",
  "post-publish-clean-consumer-result-convergence",
  "strict-close-ready-convergence-dashboard",
  "public-publish-final-owner-execution-pack",
  "owner-public-publish-execution-final-intake-pack",
  "final-real-proof-input-availability-sweep",
  "final-real-proof-import-and-close-candidate-pack",
  "owner-public-publish-authorization-input",
  "public-publish-result-authorization-convergence-gate",
  "public-publish-command-cross-check",
  "release-issue-close-owner-decision-input",
  "final-evidence-freeze-non-proof-audit",
  "final-prepublish-quality-freeze-dashboard",
  "owner-public-publish-execution-consistency-gate",
  "github-publish-and-ci-status-snapshot",
  "remote-ci-and-public-publish-proof-backfill-gate",
  "final-public-release-closure-bridge",
  "public-publish-real-result-owner-input-contract",
  "post-publish-clean-consumer-proof-record-contract",
  "release-issue-close-strict-owner-decision-import",
  "final-close-gate-convergence",
  "public-publish-real-result-record-draft",
  "post-publish-clean-consumer-proof-record-draft",
  "public-publish-forbidden-substitute-scan",
  "release-close-real-proof-import-bridge",
  "final-owner-close-readiness-checkpoint",
  "final-owner-strict-close-execution-order",
  "final-release-close-record-real-validator",
  "final-owner-release-close-record-projection",
  "final-release-close-hash-consistency-gate",
  "final-close-owner-approval-boundary-audit",
  "release-candidate-final-publishability-audit",
  "release-candidate-owner-action-roadmap",
  "release-candidate-non-substitute-final-scan",
  "release-candidate-final-owner-checklist",
  "public-release-owner-execution-package",
  "external-clean-consumer-proof-kit",
  "runtime-proof-compatible-host-kit",
  "post-publish-owner-verification-kit",
  "owner-public-release-execution-readiness-pack",
  "owner-external-real-proof-input-contract",
  "owner-external-real-proof-import-validator",
  "post-publish-clean-consumer-real-proof-gate",
  "runtime-compatible-host-real-proof-gate",
  "release-close-real-proof-readiness-gate",
  "release-candidate-real-proof-final-freeze",
  "owner-real-input-import-preflight",
  "public-package-hash-cross-check-gate",
  "real-owner-evidence-strict-validator-orchestration",
  "release-close-real-input-candidate-promotion-readiness",
  "final-quality-freeze-dashboard",
  "public-proof-claim-boundary-audit",
  "article-roadmap-30plus",
  "post-publish-docs-and-samples-final-landing-pack",
  "owner-real-publish-evidence-import-readiness-dashboard",
  "release-close-final-candidate-audit-pack",
  "cuda-device-initialization-local-smoke-classification",
  "clean-consumer-proof-execution-bundle",
  "clean-consumer-external-proof-closure-pack",
  "external-clean-consumer-execution-workspace-contract",
  "external-clean-consumer-owner-command-pack",
  "external-clean-consumer-execution-result-import",
  "external-clean-consumer-execution-result-candidate",
  "post-publish-clean-consumer-proof-result-import",
  "post-publish-clean-consumer-proof-result-candidate",
  "final-owner-real-proof-execution-package",
  "final-owner-real-proof-gap-matrix",
  "final-owner-real-proof-convergence-gate",
  "owner-real-proof-evidence-backfill-package",
  "owner-real-proof-staging-workspace-contract",
  "owner-real-proof-staging-workspace-import",
  "final-owner-rollback-review-import",
  "final-owner-close-decision-import",
  "final-owner-execution-one-screen-pack",
  "release-candidate-public-proof-final-audit",
  "final-owner-execution-input-skeleton",
  "final-owner-execution-blocker-ledger",
  "final-owner-execution-input-preflight",
  "final-owner-execution-real-input-template",
  "final-owner-execution-real-input-import",
  "final-owner-execution-real-input-candidate",
  "final-owner-execution-real-input-strict-preflight",
  "final-owner-execution-close-readiness-from-real-input",
  "owner-real-input-landing-pack",
  "final-owner-execution-checklist",
  "final-owner-execution-repair-checklist",
  "final-owner-execution-repair-input-skeleton",
  "final-owner-execution-owner-input-draft",
  "final-owner-execution-external-result-input-contract",
  "final-owner-execution-external-result-input-preflight",
  "final-owner-execution-external-result-candidate",
  "final-post-publish-clean-consumer-proof-record-contract",
  "final-post-publish-clean-consumer-proof-preflight",
  "final-post-publish-clean-consumer-proof-candidate",
  "final-release-close-owner-approval-contract",
  "final-release-close-owner-approval-preflight",
  "final-release-close-owner-approval-candidate",
  "final-public-publish-pre-execution-freeze",
  "final-public-publish-owner-action-worklist",
  "final-public-publish-command-dry-contract",
  "owner-public-publish-execution-result-input-contract",
  "owner-public-publish-execution-result-input-template",
  "owner-public-publish-execution-result-preflight",
  "owner-public-publish-execution-result-candidate",
  "post-publish-clean-consumer-real-proof-from-owner-result",
  "final-public-publish-acceptance-gate",
  "owner-real-evidence-end-to-end-release-gate",
  "owner-real-evidence-final-intake-checklist",
  "final-release-close-approval-real-input-from-owner-result",
  "real-proof-import-boundary-audit",
  "clean-consumer-runtime-proof-cross-check-gate",
  "post-publish-rollback-owner-decision-gate",
  "release-close-final-real-input-admission-pack",
  "owner-real-input-json-contract",
  "owner-real-input-json-import",
  "owner-real-input-hash-and-path-validator",
  "owner-real-input-forbidden-substitute-validator",
  "strict-close-real-input-dry-run",
  "strict-close-real-input-finding-report",
  "strict-close-owner-action-pack",
  "release-close-real-input-final-blocker-ledger",
  "final-owner-real-input-template-pack",
  "release-proof-readiness-snapshot",
  "owner-proof-input-readiness",
  "release-issue-close-record-validation",
  "release-candidate-final-evidence-freeze",
  "release-close-preflight",
  "deferred-safety-triage"
)

$requiredNonProofIds = @(
  "package-consumer-validation",
  "runtime-package-readiness",
  "error-recorder-diagnostics-design-gate",
  "dimension-expression-snapshot-design-gate",
  "calibrator-metadata-design-gate",
  "runtime-deserialization-boundary-precheck",
  "runtime-deserialization-dependency-diagnostics",
  "runtime-proof-blocker-owner-action",
  "linux-runner-validation",
  "external-runtime-proof-validation",
  "external-runtime-proof-owner-handoff",
  "compatible-host-runtime-proof-runbook",
  "compatible-host-runtime-proof-collection-bundle",
  "external-runtime-proof-backfill-plan",
  "post-publish-verification-backfill-plan",
  "external-runtime-proof-collection-package",
  "post-publish-verification-collection-package",
  "post-publish-clean-consumer-project-scan",
  "post-publish-verification-input-draft",
  "post-publish-verification-owner-input",
  "post-publish-verification-record",
  "owner-release-execution-package-validation",
  "real-model-and-package-proof-input-package",
  "release-close-gap-dashboard",
  "compatible-host-proof-execution-pack",
  "owner-proof-backfill-execution-pack",
  "owner-proof-execution-handoff",
  "owner-external-proof-input-preflight",
  "owner-proof-input-repair-pack",
  "owner-proof-input-draft-pack",
  "owner-external-proof-backfill-orchestrator",
  "package-consumer-runtime-proof-owner-input",
  "package-consumer-runtime-proof-record",
  "package-consumer-external-smoke-scaffold",
  "package-consumer-runtime-proof-candidate",
  "release-issue-close-record-owner-input",
  "release-issue-close-record-candidate",
  "final-evidence-freeze",
  "release-issue-final-close-decision",
  "real-external-proof-overlay-pack",
  "release-issue-close-record-overlay-candidate",
  "owner-external-execution-result-backfill-kit",
  "owner-input-cross-hash-audit",
  "release-close-strict-record-candidate",
  "owner-proof-real-backfill-execution-pack",
  "release-issue-close-record-real-input-map",
  "owner-proof-real-input-convergence",
  "owner-input-contract-convergence",
  "release-close-final-owner-runbook",
  "runtime-proof-execution-input-record",
  "owner-runtime-proof-execution-runbook",
  "release-close-strict-validation-bridge",
  "owner-runtime-proof-result-input-template",
  "owner-runtime-proof-result-input-validation",
  "runtime-proof-lane-dry-run-summary",
  "release-close-strict-dry-run-summary",
  "owner-external-proof-execution-bundle",
  "owner-external-proof-execution-result-import",
  "real-external-proof-record-import-validator",
  "release-close-owner-input-bridge",
  "public-package-proof-owner-input",
  "post-publish-proof-owner-confirmation",
  "release-close-public-proof-bridge",
  "release-issue-close-final-owner-decision-audit",
  "final-post-publish-audit-pack",
  "release-docs-and-nuget-metadata-audit",
  "post-publish-user-verification-pack",
  "public-package-download-proof-owner-execution-pack",
  "release-candidate-final-freeze-manifest",
  "public-publish-owner-manual-command-handoff",
  "final-release-close-blocker-dashboard",
  "public-publish-result-owner-input",
  "public-publish-result-import",
  "post-publish-clean-consumer-result-convergence",
  "strict-close-ready-convergence-dashboard",
  "public-publish-final-owner-execution-pack",
  "owner-public-publish-execution-final-intake-pack",
  "final-real-proof-input-availability-sweep",
  "final-real-proof-import-and-close-candidate-pack",
  "owner-public-publish-authorization-input",
  "public-publish-result-authorization-convergence-gate",
  "public-publish-command-cross-check",
  "release-issue-close-owner-decision-input",
  "final-evidence-freeze-non-proof-audit",
  "final-prepublish-quality-freeze-dashboard",
  "owner-public-publish-execution-consistency-gate",
  "github-publish-and-ci-status-snapshot",
  "remote-ci-and-public-publish-proof-backfill-gate",
  "final-public-release-closure-bridge",
  "public-publish-real-result-owner-input-contract",
  "post-publish-clean-consumer-proof-record-contract",
  "release-issue-close-strict-owner-decision-import",
  "final-close-gate-convergence",
  "public-publish-real-result-record-draft",
  "post-publish-clean-consumer-proof-record-draft",
  "public-publish-forbidden-substitute-scan",
  "release-close-real-proof-import-bridge",
  "final-owner-close-readiness-checkpoint",
  "final-owner-strict-close-execution-order",
  "final-release-close-record-real-validator",
  "final-owner-release-close-record-projection",
  "final-release-close-hash-consistency-gate",
  "final-close-owner-approval-boundary-audit",
  "release-candidate-final-publishability-audit",
  "release-candidate-owner-action-roadmap",
  "release-candidate-non-substitute-final-scan",
  "release-candidate-final-owner-checklist",
  "public-release-owner-execution-package",
  "external-clean-consumer-proof-kit",
  "runtime-proof-compatible-host-kit",
  "post-publish-owner-verification-kit",
  "owner-public-release-execution-readiness-pack",
  "owner-external-real-proof-input-contract",
  "owner-external-real-proof-import-validator",
  "post-publish-clean-consumer-real-proof-gate",
  "runtime-compatible-host-real-proof-gate",
  "release-close-real-proof-readiness-gate",
  "release-candidate-real-proof-final-freeze",
  "owner-real-input-import-preflight",
  "public-package-hash-cross-check-gate",
  "real-owner-evidence-strict-validator-orchestration",
  "release-close-real-input-candidate-promotion-readiness",
  "final-quality-freeze-dashboard",
  "public-proof-claim-boundary-audit",
  "article-roadmap-30plus",
  "post-publish-docs-and-samples-final-landing-pack",
  "owner-real-publish-evidence-import-readiness-dashboard",
  "release-close-final-candidate-audit-pack",
  "cuda-device-initialization-local-smoke-classification",
  "clean-consumer-proof-execution-bundle",
  "clean-consumer-external-proof-closure-pack",
  "external-clean-consumer-execution-workspace-contract",
  "external-clean-consumer-owner-command-pack",
  "external-clean-consumer-execution-result-import",
  "external-clean-consumer-execution-result-candidate",
  "post-publish-clean-consumer-proof-result-import",
  "post-publish-clean-consumer-proof-result-candidate",
  "final-owner-real-proof-execution-package",
  "final-owner-real-proof-gap-matrix",
  "final-owner-real-proof-convergence-gate",
  "owner-real-proof-evidence-backfill-package",
  "owner-real-proof-staging-workspace-contract",
  "owner-real-proof-staging-workspace-import",
  "final-owner-rollback-review-import",
  "final-owner-close-decision-import",
  "final-owner-execution-one-screen-pack",
  "release-candidate-public-proof-final-audit",
  "final-owner-execution-input-skeleton",
  "final-owner-execution-blocker-ledger",
  "final-owner-execution-input-preflight",
  "final-owner-execution-real-input-template",
  "final-owner-execution-real-input-import",
  "final-owner-execution-real-input-candidate",
  "final-owner-execution-real-input-strict-preflight",
  "final-owner-execution-close-readiness-from-real-input",
  "owner-real-input-landing-pack",
  "final-owner-execution-checklist",
  "final-owner-execution-repair-checklist",
  "final-owner-execution-repair-input-skeleton",
  "final-owner-execution-owner-input-draft",
  "final-owner-execution-external-result-input-contract",
  "final-owner-execution-external-result-input-preflight",
  "final-owner-execution-external-result-candidate",
  "final-post-publish-clean-consumer-proof-record-contract",
  "final-post-publish-clean-consumer-proof-preflight",
  "final-post-publish-clean-consumer-proof-candidate",
  "final-release-close-owner-approval-contract",
  "final-release-close-owner-approval-preflight",
  "final-release-close-owner-approval-candidate",
  "final-public-publish-pre-execution-freeze",
  "final-public-publish-owner-action-worklist",
  "final-public-publish-command-dry-contract",
  "owner-public-publish-execution-result-input-contract",
  "owner-public-publish-execution-result-input-template",
  "owner-public-publish-execution-result-preflight",
  "owner-public-publish-execution-result-candidate",
  "post-publish-clean-consumer-real-proof-from-owner-result",
  "final-public-publish-acceptance-gate",
  "owner-real-evidence-end-to-end-release-gate",
  "owner-real-evidence-final-intake-checklist",
  "final-release-close-approval-real-input-from-owner-result",
  "real-proof-import-boundary-audit",
  "clean-consumer-runtime-proof-cross-check-gate",
  "post-publish-rollback-owner-decision-gate",
  "release-close-final-real-input-admission-pack",
  "owner-real-input-json-contract",
  "owner-real-input-json-import",
  "owner-real-input-hash-and-path-validator",
  "owner-real-input-forbidden-substitute-validator",
  "strict-close-real-input-dry-run",
  "strict-close-real-input-finding-report",
  "strict-close-owner-action-pack",
  "release-close-real-input-final-blocker-ledger",
  "final-owner-real-input-template-pack",
  "release-proof-readiness-snapshot",
  "owner-proof-input-readiness",
  "release-owner-proof-input-record-validation",
  "release-issue-close-record-validation",
  "release-candidate-final-evidence-freeze",
  "release-close-preflight",
  "deferred-safety-triage"
)

$requiredNonSubstituteMarkers = @(
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
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "final quality freeze dashboard",
  "public proof claim boundary audit",
  "article roadmap",
  "clean-consumer-proof-execution-bundle",
  "clean-consumer-external-proof-closure-pack",
  "external clean consumer execution workspace contract",
  "external clean consumer owner command pack",
  "external clean consumer execution result import",
  "external clean consumer execution result candidate",
  "post-publish clean consumer proof result import",
  "post-publish clean consumer proof result candidate",
  "final owner real proof execution package",
  "final owner real proof gap matrix",
  "final owner real proof convergence gate",
  "owner real proof evidence backfill package",
  "owner real proof staging workspace contract",
  "owner real proof staging workspace import",
  "final owner rollback review import",
  "final owner close decision import",
  "repository-external workspace contract",
  "owner-copyable command guidance",
  "final owner execution one-screen pack",
  "owner one-screen execution guidance",
  "owner input gap table",
  "final public proof path",
  "release candidate public proof final audit",
  "final owner execution input skeleton",
  "final owner execution blocker ledger",
  "final owner execution input preflight",
  "external proof closure guidance",
  "owner-action closure pack",
  "owner-action-required",
  "host metadata without runtime smoke",
  "package hash without existing log validation",
  "native asset listing without runtime smoke",
  "pre-publish smoke reused as post-publish proof",
    "final owner execution real input template",
    "final owner execution real input import",
    "final owner execution real input candidate",
    "final owner execution real input strict preflight",
    "final owner execution close readiness from real input",
  "owner real input landing pack",
  "final owner execution checklist",
  "real proof import boundary audit",
  "final public publish acceptance gate",
  "public publish acceptance remains blocked until real Owner evidence",
  "owner real evidence end-to-end release gate",
  "end-to-end release gate remains blocked until real Owner evidence",
  "owner real evidence final intake checklist",
  "final intake checklist remains blocked until real Owner evidence",
  "post-publish docs and samples final landing pack",
  "owner real publish evidence import readiness dashboard",
  "release close final candidate audit pack",
  "owner public publish execution final intake pack",
  "final real proof input availability sweep",
  "final real proof import and close candidate pack",
  "design-gate-required planning input",
  "keep-deferred boundary disclosure"
)

$findings = New-Object System.Collections.Generic.List[object]
$auditedItems = New-Object System.Collections.Generic.List[object]

$bundleCanPublish = [bool](Get-PropertyOrDefault -Object $bundle -Name "canPublishPublicly" -DefaultValue $false)
$bundleCanClose = [bool](Get-PropertyOrDefault -Object $bundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$bundleIsRuntimeProof = [bool](Get-PropertyOrDefault -Object $bundle -Name "isRuntimeExecutionProof" -DefaultValue $true)
$bundleState = [string](Get-PropertyOrDefault -Object $bundle -Name "bundleState" -DefaultValue "missing-bundle-state")

if ($bundleCanPublish) {
  $findings.Add([pscustomobject]@{
      id = "bundle-can-publish-publicly"
      severity = "blocker"
      message = "release-evidence-bundle reports canPublishPublicly=true before real owner proof is present."
    })
}

if ($bundleCanClose) {
  $findings.Add([pscustomobject]@{
      id = "bundle-can-close-release-issue"
      severity = "blocker"
      message = "release-evidence-bundle reports canCloseReleaseIssue=true before real close proof is present."
    })
}

if ($bundleIsRuntimeProof) {
  $findings.Add([pscustomobject]@{
      id = "bundle-is-runtime-execution-proof"
      severity = "blocker"
      message = "release-evidence-bundle must not identify itself as runtime execution proof."
    })
}

$itemById = @{}
foreach ($item in $evidenceItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")
  if (-not [string]::IsNullOrWhiteSpace($id)) {
    $itemById[$id] = $item
  }
}

foreach ($requiredId in $requiredNonProofIds) {
  if (-not $itemById.ContainsKey($requiredId)) {
    $findings.Add([pscustomobject]@{
        id = "missing-required-non-proof-item"
        severity = "blocker"
        itemId = $requiredId
        message = "Required non-proof release-facing evidence item is missing from release-evidence-bundle."
      })
    continue
  }

  $item = $itemById[$requiredId]
  $state = [string](Get-PropertyOrDefault -Object $item -Name "state" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")
  $passed = [bool](Get-PropertyOrDefault -Object $item -Name "passed" -DefaultValue $false)
  $classificationText = "$requiredId $state $boundary"
  $isDesignGate = $designGateIds -contains $requiredId
  $hasNonProofBoundary = Test-TextContainsAny -Text $classificationText -Markers @(
    "not proof",
    "not runtime proof",
    "not runtime execution proof",
    "not publication approval",
    "not owner authorization",
    "not post-publish proof",
    "not Linux runner proof",
    "cannot promote",
    "cannot替代",
    "cannot close",
    "keeps",
    "remain deferred",
    "deferred",
    "template-only",
    "owner-action-required",
    "blocked",
    "design gate",
    "Design gate",
    "precheck",
    "dependency diagnostics",
    "dependency-probe",
    "scaffold",
    "handoff",
    "input",
    "audit",
    "runbook",
    "guidance",
    "planning input"
  )

  if (($mustRemainFailedIds -contains $requiredId) -and $passed) {
    $findings.Add([pscustomobject]@{
        id = "non-proof-item-marked-passed"
        severity = "blocker"
        itemId = $requiredId
        state = $state
        message = "Required non-proof release-facing evidence item is marked passed."
      })
  }

  if (-not $hasNonProofBoundary) {
    $findings.Add([pscustomobject]@{
        id = "missing-non-proof-boundary"
        severity = "blocker"
        itemId = $requiredId
        state = $state
        boundary = $boundary
        message = "Required non-proof item does not state a clear non-proof boundary."
      })
  }

  if ($isDesignGate -and -not (Test-TextContainsAny -Text $classificationText -Markers @("proof=False", "runtimeProofBlocked=True", "not runtime execution proof", "cannot promote runtime proof", "deferred"))) {
    $findings.Add([pscustomobject]@{
        id = "design-gate-boundary-not-explicit"
        severity = "blocker"
        itemId = $requiredId
        state = $state
        boundary = $boundary
        message = "Design/precheck item does not explicitly block runtime proof promotion."
      })
  }

  $auditedItems.Add([pscustomobject]@{
      id = $requiredId
      state = $state
      passed = $passed
      designGateOrPrecheck = $isDesignGate
      hasNonProofBoundary = $hasNonProofBoundary
      boundary = $boundary
    })
}

foreach ($marker in $requiredNonSubstituteMarkers) {
  if ($nonSubstituteProofKinds -notcontains $marker) {
    $findings.Add([pscustomobject]@{
        id = "missing-non-substitute-proof-marker"
        severity = "blocker"
        marker = $marker
        message = "release-evidence-bundle nonSubstituteProofKinds is missing a required marker."
      })
  }
}

$failedFindingCount = $findings.Count
$auditPassed = $failedFindingCount -eq 0 -and -not $bundleCanPublish -and -not $bundleCanClose -and -not $bundleIsRuntimeProof
$auditState = if ($auditPassed) { "classification-audit-passed-non-proof-boundaries-intact" } else { "classification-audit-blocked" }

$record = [pscustomobject]@{
  recordKind = "release-evidence-classification-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  releaseEvidenceBundlePath = $ReleaseEvidenceBundlePath
  bundleState = $bundleState
  auditState = $auditState
  auditPassed = $auditPassed
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  performsPublish = $false
  approvesPublicRelease = $false
  requiredNonProofItemCount = $requiredNonProofIds.Count
  auditedNonProofItemCount = $auditedItems.Count
  nonSubstituteProofKindCount = $nonSubstituteProofKinds.Count
  requiredNonSubstituteMarkerCount = $requiredNonSubstituteMarkers.Count
  findingCount = $failedFindingCount
  findings = @($findings.ToArray())
  auditedItems = @($auditedItems.ToArray())
  requiredNonSubstituteMarkers = @($requiredNonSubstituteMarkers)
  boundary = "This audit verifies release-facing classification only. It is not runtime proof, owner approval, publication, post-publish verification, release issue close approval, and not permission to delete deferred records."
}

$jsonPath = Join-Path $OutputDirectory "release-evidence-classification-audit.json"
$markdownPath = Join-Path $OutputDirectory "release-evidence-classification-audit.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Evidence Classification Audit")
$lines.Add("")
$lines.Add("`release-evidence-classification-audit` 只审计发布证据分类边界，确认 design gate、precheck、dependency diagnostics、template、draft、runbook、owner guidance、input package、scaffold、local-feed 和 deferred safety triage 没有被误晋级为 runtime proof。")
$lines.Add("")
$lines.Add("它不执行发布、不批准公开发布、不关闭 release issue，也不允许删除 deferred 记录。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| auditState | ``$(ConvertTo-MarkdownCell $record.auditState)`` |")
$lines.Add("| auditPassed | ``$($record.auditPassed)`` |")
$lines.Add("| bundleState | ``$(ConvertTo-MarkdownCell $record.bundleState)`` |")
$lines.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |")
$lines.Add("| requiredNonProofItemCount | ``$($record.requiredNonProofItemCount)`` |")
$lines.Add("| auditedNonProofItemCount | ``$($record.auditedNonProofItemCount)`` |")
$lines.Add("| findingCount | ``$($record.findingCount)`` |")
$lines.Add("")
$lines.Add("## Audited Non-Proof Items")
$lines.Add("")
$lines.Add("| Id | State | Passed | Boundary OK |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $auditedItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.id) | $(ConvertTo-MarkdownCell $item.state) | ``$($item.passed)`` | ``$($item.hasNonProofBoundary)`` |")
}
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($findings.Count -eq 0) {
  $lines.Add("- No classification findings. Non-proof boundaries remain intact.")
}
else {
  foreach ($finding in $findings) {
    $findingId = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $finding -Name "id" -DefaultValue "")
    $itemId = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $finding -Name "itemId" -DefaultValue "")
    $message = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $finding -Name "message" -DefaultValue "")
    $lines.Add("- ``$findingId`` $itemId $message")
  }
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $lines

Write-Host "Release evidence classification audit written: $jsonPath"
Write-Host "Release evidence classification audit markdown written: $markdownPath"
Write-Host "AuditState=$auditState FindingCount=$failedFindingCount"

if ($Strict -and -not $auditPassed) {
  throw "Release evidence classification audit failed with $failedFindingCount finding(s)."
}
