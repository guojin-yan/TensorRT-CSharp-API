[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function New-AuditLane {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$ValidatorCommand,
    [string]$OwnerAction
  )

  $exists = $null -ne $Record
  $state = if ($exists) { [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue "missing-$StateProperty") } else { "missing-artifact" }
  [pscustomobject]@{
    id = $Id
    title = $Title
    artifact = $Artifact
    artifactExists = $exists
    state = $state
    validatorCommand = $ValidatorCommand
    ownerAction = $OwnerAction
    performsPublish = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false) } else { $false }
    canPublishPublicly = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false) } else { $false }
    canCloseReleaseIssue = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false) } else { $false }
    isReleaseCloseProof = if ($exists) { [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false) } else { $false }
    boundary = "Readonly audit lane only. It cannot publish, upload, delete, delist, withdraw, deprecate, close release issue, or substitute real public package/PostPublish proof."
  }
}

function New-WorkflowAudit {
  param([System.IO.FileInfo]$File)

  $raw = Get-Content -LiteralPath $File.FullName -Raw -Encoding utf8
  $hasPublishTrigger = $raw.Contains("workflow_dispatch", [StringComparison]::OrdinalIgnoreCase)
  $hasPack = $raw.Contains("dotnet pack", [StringComparison]::OrdinalIgnoreCase) -or $raw.Contains("pack", [StringComparison]::OrdinalIgnoreCase)
  $hasNuGetPush = $raw.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -or $raw.Contains("nuget push", [StringComparison]::OrdinalIgnoreCase)
  $hasGitHubPackages = $raw.Contains("GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -or $raw.Contains("nuget.pkg.github.com", [StringComparison]::OrdinalIgnoreCase) -or $raw.Contains("packages:", [StringComparison]::OrdinalIgnoreCase)
  $hasSelfHosted = $raw.Contains("self-hosted", [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    fileName = $File.Name
    relativePath = ".github/workflows/$($File.Name)"
    hasWorkflowDispatch = $hasPublishTrigger
    hasPackCoverage = $hasPack
    hasNuGetPushCommand = $hasNuGetPush
    hasGitHubPackagesCoverage = $hasGitHubPackages
    mentionsSelfHostedRunner = $hasSelfHosted
    queuedWorkflowIsProof = $false
    missingSelfHostedRunnerIsProof = $false
    ownerActionRequired = $hasPublishTrigger -or $hasSelfHosted
    boundary = "Workflow metadata is readonly audit evidence only. A queued run or missing self-hosted runner is not publish proof, package proof, PostPublish proof, or close approval."
  }
}

$records = @{
  freeze = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-public-publish-pre-execution-freeze-validation.json"
  commandHandoff = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-publish-owner-manual-command-handoff-validation.json"
  ownerContract = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-input-contract.json"
  ownerPreflight = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-public-publish-execution-result-preflight.json"
  forbiddenScan = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-publish-forbidden-substitute-scan-validation.json"
  postPublishOwnerInput = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-owner-input-validation.json"
  postPublishRecord = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\post-publish-verification-validation.json"
  rollbackReview = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-rollback-review-validation.json"
  closeDecision = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-owner-close-decision-validation.json"
  strictClosure = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-validation.json"
  strictDashboard = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\release-close-strict-evidence-closure-dashboard-validation.json"
  acceptanceGate = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\final-public-publish-acceptance-gate.json"
}

$lanes = @(
  New-AuditLane -Id "pre-execution-freeze" -Title "Final public publish pre-execution freeze" -Artifact "artifacts/final-release/final-public-publish-pre-execution-freeze-validation.json" -Record $records.freeze -StateProperty "validationState" -ValidatorCommand "Export-FinalPublicPublishPreExecutionFreeze.ps1; Test-FinalPublicPublishPreExecutionFreeze.ps1 -Strict" -OwnerAction "Confirm release remains blocked before Owner real publish."
  New-AuditLane -Id "manual-command-handoff" -Title "Owner manual command handoff" -Artifact "artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json" -Record $records.commandHandoff -StateProperty "validationState" -ValidatorCommand "Export-PublicPublishOwnerManualCommandHandoff.ps1; Test-PublicPublishOwnerManualCommandHandoff.ps1 -Strict" -OwnerAction "Use copyable command placeholders only after Owner authorization; this audit never runs them."
  New-AuditLane -Id "owner-public-publish-contract" -Title "Owner public publish result contract" -Artifact "artifacts/final-release/owner-public-publish-execution-result-input-contract.json" -Record $records.ownerContract -StateProperty "contractState" -ValidatorCommand "Export-OwnerPublicPublishExecutionResultInputContract.ps1" -OwnerAction "Fill real public package result fields after Owner executes publish."
  New-AuditLane -Id "owner-public-publish-preflight" -Title "Owner public publish result preflight" -Artifact "artifacts/final-release/owner-public-publish-execution-result-preflight.json" -Record $records.ownerPreflight -StateProperty "preflightState" -ValidatorCommand "Test-OwnerPublicPublishExecutionResultPreflight.ps1 -Strict" -OwnerAction "Keep blocked until real public publish input is supplied."
  New-AuditLane -Id "forbidden-substitute-scan" -Title "Public publish forbidden substitute scan" -Artifact "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json" -Record $records.forbiddenScan -StateProperty "validationState" -ValidatorCommand "Export-PublicPublishForbiddenSubstituteScan.ps1; Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict" -OwnerAction "Rerun scan before accepting any imported Owner evidence."
  New-AuditLane -Id "post-publish-owner-input" -Title "PostPublish owner input validation" -Artifact "artifacts/final-release/post-publish-verification-owner-input-validation.json" -Record $records.postPublishOwnerInput -StateProperty "validationState" -ValidatorCommand "Test-PostPublishVerificationOwnerInput.ps1 -Strict" -OwnerAction "Owner fills public package URL/hash/log fields after publish."
  New-AuditLane -Id "post-publish-record" -Title "PostPublish record validation" -Artifact "artifacts/final-release/post-publish-verification-validation.json" -Record $records.postPublishRecord -StateProperty "validationState" -ValidatorCommand "Export-PostPublishVerificationRecordFromOwnerInput.ps1; Test-PostPublishVerificationRecord.ps1 -Strict" -OwnerAction "Promote only after clean consumer public package smoke evidence passes."
  New-AuditLane -Id "rollback-review" -Title "Final owner rollback review" -Artifact "artifacts/final-release/final-owner-rollback-review-validation.json" -Record $records.rollbackReview -StateProperty "validationState" -ValidatorCommand "Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict" -OwnerAction "Complete rollback review without executing delete/delist/withdraw/deprecate."
  New-AuditLane -Id "close-decision" -Title "Final owner close decision" -Artifact "artifacts/final-release/final-owner-close-decision-validation.json" -Record $records.closeDecision -StateProperty "validationState" -ValidatorCommand "Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict" -OwnerAction "Defer final close decision until real proof chain passes."
  New-AuditLane -Id "strict-closure" -Title "ReleaseClose strict evidence closure" -Artifact "artifacts/final-release/release-close-strict-evidence-closure-validation.json" -Record $records.strictClosure -StateProperty "validationState" -ValidatorCommand "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict" -OwnerAction "Use as final non-proof closure gate before Owner close approval."
  New-AuditLane -Id "strict-closure-dashboard" -Title "ReleaseClose strict evidence closure dashboard" -Artifact "artifacts/final-release/release-close-strict-evidence-closure-dashboard-validation.json" -Record $records.strictDashboard -StateProperty "validationState" -ValidatorCommand "Export-ReleaseCloseStrictEvidenceClosureDashboard.ps1; Test-ReleaseCloseStrictEvidenceClosureDashboard.ps1 -Strict" -OwnerAction "Use as missing materials dashboard only."
  New-AuditLane -Id "final-acceptance-gate" -Title "Final public publish acceptance gate" -Artifact "artifacts/final-release/final-public-publish-acceptance-gate.json" -Record $records.acceptanceGate -StateProperty "gateState" -ValidatorCommand "Test-FinalPublicPublishAcceptanceGate.ps1 -Strict" -OwnerAction "Do not close release while gate remains blocked."
)

$workflowDir = Join-Path $RepositoryRoot ".github\workflows"
$workflowAudits = @()
if (Test-Path -LiteralPath $workflowDir -PathType Container) {
  $workflowAudits = @(Get-ChildItem -LiteralPath $workflowDir -File -Include *.yml,*.yaml | Sort-Object Name | ForEach-Object { New-WorkflowAudit -File $_ })
}

$forbiddenNonProofSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dashboard",
  "dry-run",
  "manual approval",
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "sidecar-only",
  "TensorRtExec report"
)

$forbiddenReleaseActions = @("delete", "delist", "withdraw", "deprecate", "dotnet nuget delete", "nuget delete")
$unsafeLaneCount = @($lanes | Where-Object { [bool]$_.performsPublish -or [bool]$_.canPublishPublicly -or [bool]$_.canCloseReleaseIssue }).Count
$publishWorkflowCount = @($workflowAudits | Where-Object { [bool]$_.hasGitHubPackagesCoverage -or [bool]$_.hasNuGetPushCommand }).Count
$state = if ($unsafeLaneCount -eq 0) { "blocked-final-readonly-publish-audit-owner-action-required" } else { "failed-final-readonly-publish-audit-unsafe-side-effect" }

$record = [pscustomobject]@{
  recordKind = "final-readonly-publish-audit-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = $state
  auditLaneCount = $lanes.Count
  unsafeLaneCount = $unsafeLaneCount
  workflowAuditCount = $workflowAudits.Count
  packageWorkflowCount = $publishWorkflowCount
  workflows = @($workflowAudits)
  lanes = @($lanes)
  forbiddenNonProofSubstitutes = @($forbiddenNonProofSubstitutes)
  forbiddenReleaseActions = @($forbiddenReleaseActions)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  performsNuGetPublish = $false
  executesDeleteDelistWithdrawDeprecate = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  queuedWorkflowIsProof = $false
  missingSelfHostedRunnerIsProof = $false
  manualApprovalIsProof = $false
  dashboardIsProof = $false
  dryRunIsProof = $false
  boundary = "Final readonly publish audit pack only. It inspects owner-facing contracts, strict closure gates, and GitHub workflow metadata; it never runs dotnet nuget push, never publishes GitHub Packages, never deletes/delists/withdraws/deprecates packages, and never treats queued workflows, missing runners, manual approval, dashboards, dry-runs, local feeds, ProjectReference, direct .nupkg, sidecars, or TensorRtExec reports as proof."
}

$jsonPath = Join-Path $OutputRoot "final-readonly-publish-audit-pack.json"
$mdPath = Join-Path $OutputRoot "final-readonly-publish-audit-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Final Readonly Publish Audit Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- auditState: ``$state``") | Out-Null
$md.Add("- workflowAuditCount: ``$($workflowAudits.Count)``") | Out-Null
$md.Add("- packageWorkflowCount: ``$publishWorkflowCount``") | Out-Null
$md.Add("- performsPublish: ``False``") | Out-Null
$md.Add("- canCloseReleaseIssue: ``False``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Owner-Facing Lanes") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Lane | State | Artifact | Validator | Owner Action |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- |") | Out-Null
foreach ($lane in $lanes) {
  $md.Add("| $($lane.id) | $(ConvertTo-MarkdownCell $lane.state) | $(ConvertTo-MarkdownCell $lane.artifact) | ``$(ConvertTo-MarkdownCell $lane.validatorCommand)`` | $(ConvertTo-MarkdownCell $lane.ownerAction) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Required Owner Evidence") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Required Material | Required Hash / Check | Validator |") | Out-Null
$md.Add("| --- | --- | --- |") | Out-Null
$md.Add("| Owner public publish result | public package URL/version/SHA256 and command plan hash | ``Import-OwnerPublicPublishExecutionResultCandidate.ps1; Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict`` |") | Out-Null
$md.Add("| PostPublish proof | downloaded nupkg SHA256, clean consumer logs, native asset listing, dependency probe hash | ``Test-PostPublishVerificationOwnerInput.ps1 -Strict; Test-PostPublishVerificationRecord.ps1 -Strict`` |") | Out-Null
$md.Add("| Rollback review | rollback review artifact SHA256 and forbidden rollback/delete action confirmation | ``Import-FinalOwnerRollbackReview.ps1; Test-FinalOwnerRollbackReview.ps1 -Strict`` |") | Out-Null
$md.Add("| Close decision | release evidence bundle SHA256 and aligned public package version | ``Import-FinalOwnerCloseDecision.ps1; Test-FinalOwnerCloseDecision.ps1 -Strict`` |") | Out-Null
$md.Add("| Strict closure | strict closure/dashboard validation outputs and forbidden substitute scan hash | ``Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict`` |") | Out-Null
$md.Add("") | Out-Null
$md.Add("## GitHub Workflow Readonly Audit") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Workflow | Pack | NuGet Push Command | GitHub Packages | Self-hosted | Boundary |") | Out-Null
$md.Add("| --- | --- | --- | --- | --- | --- |") | Out-Null
foreach ($workflow in $workflowAudits) {
  $md.Add("| $($workflow.fileName) | $($workflow.hasPackCoverage) | $($workflow.hasNuGetPushCommand) | $($workflow.hasGitHubPackagesCoverage) | $($workflow.mentionsSelfHostedRunner) | $(ConvertTo-MarkdownCell $workflow.boundary) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Forbidden Non-Proof Substitutes") | Out-Null
$md.Add("") | Out-Null
foreach ($item in $forbiddenNonProofSubstitutes) { $md.Add("- ``$item``") | Out-Null }
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md

Write-Host "FinalReadonlyPublishAuditPackState=$state Lanes=$($lanes.Count) Workflows=$($workflowAudits.Count) PackageWorkflows=$publishWorkflowCount UnsafeLanes=$unsafeLaneCount"
