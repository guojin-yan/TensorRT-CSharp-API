[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function New-AuthorizationField {
  param([string]$Name, [string]$Description, [AllowNull()][object]$Value = $null, [string[]]$ForbiddenSubstitutes = @())
  [pscustomobject]@{
    name = $Name
    description = $Description
    value = $Value
    required = $true
    ready = $false
    valueState = "owner-authorization-input-required"
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

$expectedPhrase = "I AUTHORIZE REAL PUBLIC PACKAGE PUBLISH FOR TENSORRTSHARP 4.0"
$forbidden = @("local feed", "ProjectReference", "direct .nupkg", "dry-run", "dashboard", "manual approval", "queued workflow", "missing runner", "sidecar-only", "TensorRtExec report")
$fields = @(
  New-AuthorizationField -Name "ownerAuthorizedPublicPublish" -Description "Must be true only after Owner explicitly authorizes real public package publish." -Value $false -ForbiddenSubstitutes $forbidden
  New-AuthorizationField -Name "explicitAuthorizationPhrase" -Description "Must exactly match the expected authorization phrase." -Value "<owner-fill exact phrase>" -ForbiddenSubstitutes @("manual approval", "template")
  New-AuthorizationField -Name "authorizedBy" -Description "Human Owner name who authorizes real publish." -Value "<owner-fill>"
  New-AuthorizationField -Name "authorizedByEmail" -Description "Human Owner email or account identity." -Value "<owner-fill>"
  New-AuthorizationField -Name "authorizedAtUtc" -Description "UTC timestamp for the authorization decision." -Value "<owner-fill ISO-8601 UTC>"
  New-AuthorizationField -Name "expectedCommit" -Description "Exact git commit authorized for publish." -Value "<owner-fill commit sha>"
  New-AuthorizationField -Name "managedPackageId" -Description "Managed package id authorized for publish." -Value "JYPPX.TensorRtSharp"
  New-AuthorizationField -Name "managedPackageVersion" -Description "Managed package version authorized for publish." -Value "<owner-fill version>"
  New-AuthorizationField -Name "managedPackageSha256" -Description "SHA256 of the managed nupkg authorized for publish." -Value "<owner-fill 64 hex>"
  New-AuthorizationField -Name "runtimePackageId" -Description "Runtime package id authorized for publish." -Value "<owner-fill package id>"
  New-AuthorizationField -Name "runtimePackageVersion" -Description "Runtime package version authorized for publish." -Value "<owner-fill version>"
  New-AuthorizationField -Name "runtimePackageSha256" -Description "SHA256 of the runtime nupkg authorized for publish." -Value "<owner-fill 64 hex>"
  New-AuthorizationField -Name "nugetPackageSource" -Description "NuGet package source authorized for publish." -Value "https://api.nuget.org/v3/index.json" -ForbiddenSubstitutes @("local feed")
  New-AuthorizationField -Name "githubPackagesSource" -Description "GitHub Packages source authorized for publish." -Value "https://nuget.pkg.github.com/guojin-yan/index.json" -ForbiddenSubstitutes @("local feed")
  New-AuthorizationField -Name "packageOutputRoot" -Description "Local package output root reviewed by Owner before publish." -Value "<owner-fill package output root>"
  New-AuthorizationField -Name "publishCommandPlanSha256" -Description "SHA256 of the reviewed publish command handoff plan." -Value "<owner-fill 64 hex>" -ForbiddenSubstitutes @("runbook without hash", "manual approval")
  New-AuthorizationField -Name "rollbackPlanSha256" -Description "SHA256 of reviewed rollback plan." -Value "<owner-fill 64 hex>"
  New-AuthorizationField -Name "noLocalFeedConfirmation" -Description "Owner confirmation that local feed is not used as public proof." -Value $false -ForbiddenSubstitutes @("local feed")
  New-AuthorizationField -Name "noProjectReferenceConfirmation" -Description "Owner confirmation that ProjectReference is not used as public proof." -Value $false -ForbiddenSubstitutes @("ProjectReference")
  New-AuthorizationField -Name "noDirectNupkgConfirmation" -Description "Owner confirmation that direct .nupkg path is not used as public proof." -Value $false -ForbiddenSubstitutes @("direct .nupkg")
)

$record = [pscustomobject]@{
  recordKind = "owner-public-publish-authorization-input-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  templateState = "blocked-owner-public-publish-authorization-input-required"
  expectedAuthorizationPhrase = $expectedPhrase
  requiredFieldCount = $fields.Count
  blockedRequiredFieldCount = $fields.Count
  readyRequiredFieldCount = 0
  ownerFields = @($fields)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Authorization input template is owner-fill input only. It does not publish, dispatch workflows, close releases, or promote proof."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-authorization-input.template.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-authorization-input.template.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $fields | ForEach-Object { "| $($_.name) | $($_.required) | $($_.ready) | $($_.description.Replace("|", "\|")) |" }
$markdown = @"
# Owner Public Publish Authorization Input Template

| Item | Value |
|---|---|
| templateState | ``$($record.templateState)`` |
| expectedAuthorizationPhrase | ``$($record.expectedAuthorizationPhrase)`` |
| requiredFieldCount | ``$($record.requiredFieldCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Fields

| Name | Required | Ready | Description |
|---|---:|---:|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner public publish authorization input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "TemplateState=$($record.templateState) RequiredFields=$($record.requiredFieldCount)"

