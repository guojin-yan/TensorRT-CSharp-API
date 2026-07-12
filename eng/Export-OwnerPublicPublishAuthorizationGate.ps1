[CmdletBinding()]
param(
  [string]$ExecutionPackPath = "artifacts\final-release\public-publish-final-owner-execution-pack.json",
  [string]$AuthorizationInputPath = "artifacts\final-release\owner-public-publish-authorization-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$executionPack = Read-JsonOrNull -Path $ExecutionPackPath
$authorizationInput = Read-JsonOrNull -Path $AuthorizationInputPath

$authorizationPhrase = [string](Get-PropertyOrDefault -Object $authorizationInput -Name "explicitAuthorizationPhrase" -DefaultValue "")
$authorizedBy = [string](Get-PropertyOrDefault -Object $authorizationInput -Name "authorizedBy" -DefaultValue "")
$authorizedAtUtc = [string](Get-PropertyOrDefault -Object $authorizationInput -Name "authorizedAtUtc" -DefaultValue "")
$ownerAuthorized = [bool](Get-PropertyOrDefault -Object $authorizationInput -Name "ownerAuthorizedPublicPublish" -DefaultValue $false)
$expectedPhrase = "I AUTHORIZE REAL PUBLIC PACKAGE PUBLISH FOR TENSORRTSHARP 4.0"
$phraseMatches = $authorizationPhrase -ceq $expectedPhrase
$authorizationReady = $ownerAuthorized -and $phraseMatches -and
  -not [string]::IsNullOrWhiteSpace($authorizedBy) -and
  -not [string]::IsNullOrWhiteSpace($authorizedAtUtc)

$executionPackSafe = $null -ne $executionPack -and
  [bool](Get-PropertyOrDefault -Object $executionPack -Name "notExecutedByAutomation" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $executionPack -Name "ownerExecutionOnly" -DefaultValue $false) -and
  -not [bool](Get-PropertyOrDefault -Object $executionPack -Name "performsPublish" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $executionPack -Name "canPublishPublicly" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $executionPack -Name "canCloseReleaseIssue" -DefaultValue $true)

$manualPolicy = Get-PropertyOrDefault -Object $executionPack -Name "manualCommandPolicy" -DefaultValue $null
$materializedExecutableCommand = [string](Get-PropertyOrDefault -Object $manualPolicy -Name "materializedExecutableCommand" -DefaultValue "")
$materializedCommandBlocked = [string]::IsNullOrWhiteSpace($materializedExecutableCommand)

$forbiddenExecutableTokens = @(
  "dotnet nuget push",
  "nuget push",
  "gh release upload",
  "gh workflow run",
  "workflow_dispatch",
  "gh issue close",
  "gh release edit",
  "Publish-Module",
  "Invoke-WebRequest -Method Put"
)

$scanText = @(
  [string](Get-PropertyOrDefault -Object $executionPack -Name "safetyBoundary" -DefaultValue ""),
  [string](Get-PropertyOrDefault -Object $manualPolicy -Name "commandTemplateBoundary" -DefaultValue ""),
  $materializedExecutableCommand
) -join "`n"

$forbiddenHits = @($forbiddenExecutableTokens | Where-Object {
    $scanText.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    $_ -ne "dotnet nuget push"
  })

$commandTemplateTextAllowed = $scanText.IndexOf("must not run package push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
  $scanText.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0

$gateItems = @(
  [pscustomobject]@{
    id = "owner-authorization-present"
    ready = $authorizationReady
    status = if ($authorizationReady) { "ready" } else { "blocked-owner-explicit-authorization-required" }
    detail = "Requires ownerAuthorizedPublicPublish=true and exact phrase: $expectedPhrase."
  },
  [pscustomobject]@{
    id = "execution-pack-safe"
    ready = $executionPackSafe
    status = if ($executionPackSafe) { "ready" } else { "blocked-public-publish-final-owner-execution-pack-required" }
    detail = "Execution pack must remain owner-only, non-publish, non-proof, and non-close."
  },
  [pscustomobject]@{
    id = "no-materialized-executable-command"
    ready = $materializedCommandBlocked
    status = if ($materializedCommandBlocked) { "blocked-by-design-no-executable-command-materialized" } else { "invalid-executable-command-materialized" }
    detail = "Unauthorized state must not materialize dotnet nuget push, GitHub Packages push, workflow dispatch, or release close commands."
  },
  [pscustomobject]@{
    id = "forbidden-executable-token-scan"
    ready = $forbiddenHits.Count -eq 0 -and $commandTemplateTextAllowed
    status = if ($forbiddenHits.Count -eq 0 -and $commandTemplateTextAllowed) { "blocked-by-design-executable-token-scan-clean" } else { "invalid-forbidden-executable-token-found" }
    detail = "Forbidden executable hits: $($forbiddenHits -join ', '). Boundary text may mention dotnet nuget push only as a prohibited action."
  }
)

$failedBlockers = @($gateItems | Where-Object { $_.status.StartsWith("invalid-", [StringComparison]::OrdinalIgnoreCase) })
$blockedActionRequired = @($gateItems | Where-Object { -not [bool]$_.ready -and -not $_.status.StartsWith("invalid-", [StringComparison]::OrdinalIgnoreCase) })

$record = [pscustomobject]@{
  recordKind = "owner-public-publish-authorization-gate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  gateState = if ($failedBlockers.Count -gt 0) { "invalid-owner-public-publish-authorization-gate" } elseif ($authorizationReady -and $executionPackSafe -and $materializedCommandBlocked) { "owner-public-publish-authorized-for-manual-execution-review" } else { "blocked-owner-public-publish-authorization-required" }
  expectedAuthorizationPhrase = $expectedPhrase
  authorizationInputPath = Resolve-RepositoryPath -Path $AuthorizationInputPath
  executionPackPath = Resolve-RepositoryPath -Path $ExecutionPackPath
  ownerAuthorizedPublicPublish = $ownerAuthorized
  authorizationPhraseMatches = $phraseMatches
  authorizedBy = $authorizedBy
  authorizedAtUtc = $authorizedAtUtc
  materializedExecutableCommand = $materializedExecutableCommand
  gateItemCount = $gateItems.Count
  blockedActionRequiredCount = $blockedActionRequired.Count
  failedBlockerCount = $failedBlockers.Count
  gateItems = @($gateItems)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Authorization gate is a fail-closed preflight only. It never executes dotnet nuget push, never publishes GitHub Packages, never dispatches workflows, never closes a release, and never promotes local or dry-run evidence to proof."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-authorization-gate.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-authorization-gate.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.gateItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.status) | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Owner Public Publish Authorization Gate

| Item | Value |
|---|---|
| gateState | ``$($record.gateState)`` |
| ownerAuthorizedPublicPublish | ``$($record.ownerAuthorizedPublicPublish)`` |
| authorizationPhraseMatches | ``$($record.authorizationPhraseMatches)`` |
| blockedActionRequiredCount | ``$($record.blockedActionRequiredCount)`` |
| failedBlockerCount | ``$($record.failedBlockerCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Gate Items

| ID | Ready | Status | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner public publish authorization gate written to $jsonPath"
Write-Host "GateState=$($record.gateState) Blocked=$($record.blockedActionRequiredCount) FailedBlockers=$($record.failedBlockerCount)"
