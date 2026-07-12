[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-authorization-command-guard-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerAuthorizationCommandGuardPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$commands = @(Get-PropertyOrDefault -Object $record -Name "commands" -DefaultValue @())
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-authorization-command-guard-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "command-count" ($commands.Count -ge 8 -and [int](Get-PropertyOrDefault -Object $record -Name "commandCount" -DefaultValue 0) -eq $commands.Count) "blocker" "Command guard pack must include publish, workflow, release, article, and readonly command classes.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-publish-commands" ($text.Contains("dotnet nuget push") -and $text.Contains("nuget-org-push") -and $text.Contains("github-packages-push") -and $text.Contains("blocked-until-owner-authorization-and-real-proof")) "blocker" "Publish commands must stay blocked before Owner authorization and real proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "workflow-and-release-guards" ($text.Contains("workflow-dispatch-publish") -and $text.Contains("queued-workflow-is-not-proof") -and $text.Contains("release-close") -and $text.Contains("article-publish")) "blocker" "Workflow dispatch, release close, and article publish must be guarded.")) | Out-Null
$items.Add((New-OwnerValidationItem "readonly-allowed" ($text.Contains("allowed-before-owner-authorization-readonly-non-proof") -and $text.Contains("article-proposal-generation") -and $text.Contains("owner-missing-action-summary")) "blocker" "Readonly/artifact-only commands should remain allowed but non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-destructive-actions" ($text.Contains("deletion") -or $text.Contains("delisting") -or $text.Contains("withdrawal") -or $text.Contains("deprecation")) "blocker" "Guard boundary must forbid destructive package actions.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Command guard pack must not publish packages/articles or close release.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "owner-authorization-command-guard-pack-validation-ready-non-proof" } else { "blocked-owner-authorization-command-guard-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-authorization-command-guard-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  commandCount = [int](Get-PropertyOrDefault -Object $record -Name "commandCount" -DefaultValue 0)
  blockedCommandCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCommandCount" -DefaultValue 0)
  allowedReadonlyCommandCount = [int](Get-PropertyOrDefault -Object $record -Name "allowedReadonlyCommandCount" -DefaultValue 0)
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner authorization command guard pack validation only; not package publication, workflow dispatch, article publication, destructive package action, proof promotion, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-authorization-command-guard-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-authorization-command-guard-pack-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Authorization Command Guard Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- commandCount: ``$($validation.commandCount)``",
  "- blockedCommandCount: ``$($validation.blockedCommandCount)``",
  "- allowedReadonlyCommandCount: ``$($validation.allowedReadonlyCommandCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerAuthorizationCommandGuardPackValidationState=$state FailedBlockers=$failedBlockerCount Commands=$($validation.commandCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Owner authorization command guard pack validation failed." }
