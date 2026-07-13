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

$guard = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-authorization-command-guard-pack.json"
$oneScreen = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-final-missing-action-one-screen-pack.json"
$repair = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json"
$proposal = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-source-patch-proposal-pack.json"
$readiness = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-source-patch-apply-readiness-pack.json"

if ($null -eq $guard) { & (Join-Path $RepositoryRoot "eng\Export-OwnerAuthorizationCommandGuardPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $guard = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-authorization-command-guard-pack.json" }
if ($null -eq $oneScreen) { & (Join-Path $RepositoryRoot "eng\Export-OwnerFinalMissingActionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $oneScreen = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-final-missing-action-one-screen-pack.json" }
if ($null -eq $repair) { & (Join-Path $RepositoryRoot "eng\Export-OwnerMissingRealPublishEvidenceRepairPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $repair = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\owner-missing-real-publish-evidence-repair-pack.json" }
if ($null -eq $proposal) { & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSourcePatchProposalPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $proposal = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-source-patch-proposal-pack.json" }
if ($null -eq $readiness) { & (Join-Path $RepositoryRoot "eng\Export-PublicArticleSourcePatchApplyReadinessPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot; $readiness = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts\final-release\public-article-source-patch-apply-readiness-pack.json" }

$commands = @(Get-PropertyOrDefault -Object $guard -Name "commands" -DefaultValue @())
$blockedCommands = @($commands | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "authorizationState" -DefaultValue "") -eq "blocked-until-owner-authorization-and-real-proof" })
$allowedReadonlyCommands = @($commands | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "authorizationState" -DefaultValue "") -eq "allowed-before-owner-authorization-readonly-non-proof" })
$realProofGaps = @(
  [pscustomobject]@{ id = "owner-public-publish-evidence"; count = [int](Get-PropertyOrDefault -Object $repair -Name "ownerPublishMissingFieldCount" -DefaultValue 0); ownerAction = "Fill the 178-field Owner public publish result from real NuGet/GitHub Packages evidence." },
  [pscustomobject]@{ id = "post-publish-owner-input"; count = [int](Get-PropertyOrDefault -Object $repair -Name "postPublishMissingFieldCount" -DefaultValue 0); ownerAction = "Fill PostPublish clean consumer evidence from a repository-external public-source consumer." },
  [pscustomobject]@{ id = "post-publish-strict-cross-check"; count = [int](Get-PropertyOrDefault -Object $repair -Name "failedPostPublishCrossCheckCount" -DefaultValue 0); ownerAction = "Repair URL/hash/no-substitute mismatches until strict cross-check passes." },
  [pscustomobject]@{ id = "article-source-patch-proposals"; count = [int](Get-PropertyOrDefault -Object $proposal -Name "proposalCount" -DefaultValue 0); ownerAction = "Review article patch proposals; do not publish claim text before proof gates pass." },
  [pscustomobject]@{ id = "article-source-patch-readiness"; count = [int](Get-PropertyOrDefault -Object $readiness -Name "readinessItemCount" -DefaultValue 0); ownerAction = "Use readiness mapping only after Owner approves source patch application." }
)

$record = [pscustomobject]@{
  recordKind = "owner-final-authorization-request-summary"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  requestState = "blocked-owner-final-authorization-required"
  blockedCommandCount = $blockedCommands.Count
  allowedReadonlyCommandCount = $allowedReadonlyCommands.Count
  missingActionCount = [int](Get-PropertyOrDefault -Object $oneScreen -Name "missingActionCount" -DefaultValue 0)
  realProofGapCount = $realProofGaps.Count
  articleProposalCount = [int](Get-PropertyOrDefault -Object $proposal -Name "proposalCount" -DefaultValue 0)
  articleReadinessItemCount = [int](Get-PropertyOrDefault -Object $readiness -Name "readinessItemCount" -DefaultValue 0)
  nowBlockedCommands = @($blockedCommands)
  allowedReadonlyNonProofCommands = @($allowedReadonlyCommands)
  realProofGaps = @($realProofGaps)
  ownerDecisionRequired = "Authorize real public publish and post-publish evidence collection, or keep all publish/article/release-close commands blocked."
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct nupkg", "dashboard", "dry-run", "manual approval only", "queued workflow", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner final authorization request summary only. It requests a human decision and never runs publish, workflow dispatch, article publication, destructive package actions, proof promotion, or release close commands."
}

$jsonPath = Join-Path $OutputRoot "owner-final-authorization-request-summary.json"
$mdPath = Join-Path $OutputRoot "owner-final-authorization-request-summary.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Owner Final Authorization Request Summary") | Out-Null
$md.Add("") | Out-Null
$md.Add("- requestState: ``$($record.requestState)``") | Out-Null
$md.Add("- blockedCommandCount: ``$($record.blockedCommandCount)``") | Out-Null
$md.Add("- allowedReadonlyCommandCount: ``$($record.allowedReadonlyCommandCount)``") | Out-Null
$md.Add("- missingActionCount: ``$($record.missingActionCount)``") | Out-Null
$md.Add("- articleProposalCount: ``$($record.articleProposalCount)``") | Out-Null
$md.Add("- articleReadinessItemCount: ``$($record.articleReadinessItemCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("## Now Blocked Commands") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Command | Group | Required Precondition |") | Out-Null
$md.Add("| --- | --- | --- |") | Out-Null
foreach ($command in $blockedCommands) {
  $md.Add("| ``$($command.id)`` | ``$($command.commandGroup)`` | $(ConvertTo-MarkdownCell $command.requiredPrecondition) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Allowed Readonly Non-Proof Commands") | Out-Null
$md.Add("") | Out-Null
$md.Add("| Command | Validator |") | Out-Null
$md.Add("| --- | --- |") | Out-Null
foreach ($command in $allowedReadonlyCommands) {
  $md.Add("| ``$($command.id)`` | ``$(ConvertTo-MarkdownCell $command.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "OwnerFinalAuthorizationRequestSummaryState=$($record.requestState) BlockedCommands=$($record.blockedCommandCount) AllowedReadonly=$($record.allowedReadonlyCommandCount) ArticleProposals=$($record.articleProposalCount)"
