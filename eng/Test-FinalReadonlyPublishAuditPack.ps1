[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-readonly-publish-audit-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalReadonlyPublishAuditPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @(Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @())
$workflows = @(Get-PropertyOrDefault -Object $record -Name "workflows" -DefaultValue @())
$forbidden = @(Get-PropertyOrDefault -Object $record -Name "forbiddenNonProofSubstitutes" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredLaneIds = @(
  "pre-execution-freeze",
  "manual-command-handoff",
  "owner-public-publish-contract",
  "owner-public-publish-preflight",
  "forbidden-substitute-scan",
  "post-publish-owner-input",
  "post-publish-record",
  "rollback-review",
  "close-decision",
  "strict-closure",
  "strict-closure-dashboard",
  "final-acceptance-gate"
)
$requiredSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-readonly-publish-audit-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-non-proof" (([string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "")).Contains("blocked") -and -not [bool](Get-PropertyOrDefault -Object $record -Name "passed" -DefaultValue $true)) "blocker" "Audit pack must remain blocked/non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsNuGetPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "executesDeleteDelistWithdrawDeprecate" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Audit pack must not publish, delete-like, or close release issue.")) | Out-Null
$items.Add((New-OwnerValidationItem "workflow-audit-present" ($workflows.Count -ge 1 -and [int](Get-PropertyOrDefault -Object $record -Name "packageWorkflowCount" -DefaultValue 0) -ge 1) "blocker" "GitHub workflow readonly audit must include package/GitHub Packages coverage metadata.")) | Out-Null
$items.Add((New-OwnerValidationItem "workflow-non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "queuedWorkflowIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "missingSelfHostedRunnerIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "manualApprovalIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "dashboardIsProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "dryRunIsProof" -DefaultValue $true)) "blocker" "Workflow/dry-run/manual/dashboard states must remain non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "lane-count" ($lanes.Count -eq $requiredLaneIds.Count) "blocker" "Audit pack must include all owner-facing lanes.")) | Out-Null

foreach ($id in $requiredLaneIds) {
  $items.Add((New-OwnerValidationItem "lane-$id" (@($lanes | Where-Object { [string]$_.id -eq $id -and -not [bool]$_.performsPublish -and -not [bool]$_.canCloseReleaseIssue }).Count -eq 1) "blocker" "Missing or unsafe lane: $id")) | Out-Null
}
foreach ($substitute in $requiredSubstitutes) {
  $items.Add((New-OwnerValidationItem "substitute-$substitute" (@($forbidden | Where-Object { [string]$_ -eq $substitute }).Count -eq 1) "blocker" "Missing forbidden substitute: $substitute")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("never runs dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("never publishes GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("delete", [StringComparison]::OrdinalIgnoreCase) -ge 0
$items.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Boundary must state no publish/GitHub Packages/delete-like side effects.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "final-readonly-publish-audit-pack-validation-ready-non-proof" } else { "blocked-final-readonly-publish-audit-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "final-readonly-publish-audit-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  auditLaneCount = [int]$lanes.Count
  workflowAuditCount = [int]$workflows.Count
  packageWorkflowCount = [int](Get-PropertyOrDefault -Object $record -Name "packageWorkflowCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsGitHubPackagesPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseCloseProof = $false
  boundary = "Readonly audit validation only; not proof, not publish approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "final-readonly-publish-audit-pack-validation.json"
$mdPath = Join-Path $OutputRoot "final-readonly-publish-audit-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Final Readonly Publish Audit Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- auditLaneCount: ``$($validation.auditLaneCount)``",
  "- workflowAuditCount: ``$($validation.workflowAuditCount)``",
  "",
  $validation.boundary
)
Write-Host "FinalReadonlyPublishAuditPackValidationState=$state FailedBlockers=$failedBlockerCount Lanes=$($validation.auditLaneCount) Workflows=$($validation.workflowAuditCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Final readonly publish audit pack validation failed." }
