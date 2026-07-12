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

function New-CommandGuard {
  param(
    [int]$Order,
    [string]$Id,
    [string]$CommandGroup,
    [string]$CommandText,
    [string]$AuthorizationState,
    [string]$RequiredPrecondition,
    [string]$ValidatorCommand,
    [string]$Risk
  )
  [pscustomobject]@{
    order = $Order
    id = $Id
    commandGroup = $CommandGroup
    commandText = $CommandText
    authorizationState = $AuthorizationState
    requiredPrecondition = $RequiredPrecondition
    validatorCommand = $ValidatorCommand
    risk = $Risk
    ownerActionRequired = $true
    blocked = $AuthorizationState -eq "blocked-until-owner-authorization-and-real-proof"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Command guard item only. It classifies whether a command is allowed before Owner authorization and never runs the command."
  }
}

$commands = @(
  New-CommandGuard 1 "nuget-org-push" "authorized-only-publish" "dotnet nuget push <package>.nupkg --source https://api.nuget.org/v3/index.json --api-key <OWNER_PROVIDED_KEY>" "blocked-until-owner-authorization-and-real-proof" "Explicit Owner authorization, package/version confirmation, rollback plan, and publish transcript capture path." "Export-FinalOwnerOneScreenExecutionManual.ps1; Test-FinalOwnerOneScreenExecutionManual.ps1 -Strict" "public-package-publish"
  New-CommandGuard 2 "github-packages-push" "authorized-only-publish" "dotnet nuget push <package>.nupkg --source github --api-key <OWNER_PROVIDED_TOKEN>" "blocked-until-owner-authorization-and-real-proof" "Explicit Owner authorization, GitHub Packages target confirmation, token scope confirmation, and transcript capture path." "Export-FinalOwnerOneScreenExecutionManual.ps1; Test-FinalOwnerOneScreenExecutionManual.ps1 -Strict" "public-package-publish"
  New-CommandGuard 3 "workflow-dispatch-publish" "authorized-only-workflow" "gh workflow run <publish-workflow> --ref TensorRtSharp4.0" "blocked-until-owner-authorization-and-real-proof" "Owner authorization plus workflow inputs proving public publish target and evidence capture." "Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1; Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1 -Strict" "queued-workflow-is-not-proof"
  New-CommandGuard 4 "release-close" "authorized-only-release-close" "gh issue close <release-issue> --comment <real-proof-summary>" "blocked-until-owner-authorization-and-real-proof" "Release close strict closure accepted with real publish, PostPublish, rollback/no-rollback, and no-substitute evidence." "Export-ReleaseCloseStrictEvidenceClosure.ps1; Test-ReleaseCloseStrictEvidenceClosure.ps1 -Strict" "release-close"
  New-CommandGuard 5 "article-publish" "authorized-only-article-publish" "publish docs/articles after proof gate clears" "blocked-until-owner-authorization-and-real-proof" "PostPublish article proof gate accepted and blocked claim review cleared or patched with real evidence." "Export-PostPublishArticleProofGate.ps1; Test-PostPublishArticleProofGate.ps1 -Strict" "public-claim-publish"
  New-CommandGuard 6 "readonly-dashboard-validation" "allowed-before-authorization" ".\eng\Export-FinalOwnerPublishAndArticleActionDashboard.ps1; .\eng\Test-FinalOwnerPublishAndArticleActionDashboard.ps1 -Strict" "allowed-before-owner-authorization-readonly-non-proof" "Readonly dashboard validation only; output remains non-proof." "Export-FinalOwnerPublishAndArticleActionDashboard.ps1; Test-FinalOwnerPublishAndArticleActionDashboard.ps1 -Strict" "readonly-non-proof"
  New-CommandGuard 7 "article-proposal-generation" "allowed-before-authorization" ".\eng\Export-PublicArticleSourcePatchProposalPack.ps1; .\eng\Test-PublicArticleSourcePatchProposalPack.ps1 -Strict" "allowed-before-owner-authorization-readonly-non-proof" "Artifact-only article proposal generation; no source article overwrite." "Export-PublicArticleSourcePatchProposalPack.ps1; Test-PublicArticleSourcePatchProposalPack.ps1 -Strict" "artifact-only-non-proof"
  New-CommandGuard 8 "owner-missing-action-summary" "allowed-before-authorization" ".\eng\Export-OwnerFinalMissingActionOneScreenPack.ps1; .\eng\Test-OwnerFinalMissingActionOneScreenPack.ps1 -Strict" "allowed-before-owner-authorization-readonly-non-proof" "Readonly final missing action summary only; no publish/close side effects." "Export-OwnerFinalMissingActionOneScreenPack.ps1; Test-OwnerFinalMissingActionOneScreenPack.ps1 -Strict" "readonly-non-proof"
)

$blocked = @($commands | Where-Object { [string]$_.authorizationState -eq "blocked-until-owner-authorization-and-real-proof" })
$allowed = @($commands | Where-Object { [string]$_.authorizationState -eq "allowed-before-owner-authorization-readonly-non-proof" })
$record = [pscustomobject]@{
  recordKind = "owner-authorization-command-guard-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  guardState = "blocked-owner-authorization-required-for-publish-close-and-article-release"
  commandCount = $commands.Count
  blockedCommandCount = $blocked.Count
  allowedReadonlyCommandCount = $allowed.Count
  commands = @($commands)
  ownerActionRequired = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner authorization command guard pack only. It enumerates allowed/blocked command classes and never runs publish, workflow dispatch, article publication, package deletion, package delisting, package withdrawal, deprecation, or release close commands."
}

$jsonPath = Join-Path $OutputRoot "owner-authorization-command-guard-pack.json"
$mdPath = Join-Path $OutputRoot "owner-authorization-command-guard-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Owner Authorization Command Guard Pack") | Out-Null
$md.Add("") | Out-Null
$md.Add("- guardState: ``$($record.guardState)``") | Out-Null
$md.Add("- commandCount: ``$($record.commandCount)``") | Out-Null
$md.Add("- blockedCommandCount: ``$($record.blockedCommandCount)``") | Out-Null
$md.Add("- allowedReadonlyCommandCount: ``$($record.allowedReadonlyCommandCount)``") | Out-Null
$md.Add("") | Out-Null
$md.Add("| # | Command Id | Group | Authorization State | Validator |") | Out-Null
$md.Add("| ---: | --- | --- | --- | --- |") | Out-Null
foreach ($command in $commands) {
  $md.Add("| $($command.order) | ``$($command.id)`` | ``$($command.commandGroup)`` | ``$($command.authorizationState)`` | ``$(ConvertTo-MarkdownCell $command.validatorCommand)`` |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Boundary") | Out-Null
$md.Add("") | Out-Null
$md.Add($record.boundary) | Out-Null
Write-Utf8File -LiteralPath $mdPath -InputObject $md
Write-Host "OwnerAuthorizationCommandGuardPackState=$($record.guardState) Commands=$($record.commandCount) Blocked=$($record.blockedCommandCount) AllowedReadonly=$($record.allowedReadonlyCommandCount)"
