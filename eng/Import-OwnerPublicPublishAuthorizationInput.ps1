[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-public-publish-authorization-input.template.json",
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

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-OwnerFieldValue {
  param([AllowNull()][object]$Record, [string]$Name, [AllowNull()][object]$DefaultValue)
  $direct = Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue $null
  if ($null -ne $direct) { return $direct }
  $fields = @((Get-PropertyOrDefault -Object $Record -Name "ownerFields" -DefaultValue @()))
  foreach ($field in $fields) {
    if ([string](Get-PropertyOrDefault -Object $field -Name "name" -DefaultValue "") -eq $Name) {
      return Get-PropertyOrDefault -Object $field -Name "value" -DefaultValue $DefaultValue
    }
  }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner public publish authorization input not found: $resolvedInputPath"
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$expectedPhrase = "I AUTHORIZE REAL PUBLIC PACKAGE PUBLISH FOR TENSORRTSHARP 4.0"

$record = [pscustomobject]@{
  recordKind = "owner-public-publish-authorization-input-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = "blocked-owner-public-publish-authorization-input-required"
  sourceInputPath = $resolvedInputPath
  expectedAuthorizationPhrase = $expectedPhrase
  ownerAuthorizedPublicPublish = [bool](Get-OwnerFieldValue -Record $inputRecord -Name "ownerAuthorizedPublicPublish" -DefaultValue $false)
  explicitAuthorizationPhrase = [string](Get-OwnerFieldValue -Record $inputRecord -Name "explicitAuthorizationPhrase" -DefaultValue "")
  authorizedBy = [string](Get-OwnerFieldValue -Record $inputRecord -Name "authorizedBy" -DefaultValue "")
  authorizedByEmail = [string](Get-OwnerFieldValue -Record $inputRecord -Name "authorizedByEmail" -DefaultValue "")
  authorizedAtUtc = [string](Get-OwnerFieldValue -Record $inputRecord -Name "authorizedAtUtc" -DefaultValue "")
  expectedCommit = [string](Get-OwnerFieldValue -Record $inputRecord -Name "expectedCommit" -DefaultValue "")
  managedPackageId = [string](Get-OwnerFieldValue -Record $inputRecord -Name "managedPackageId" -DefaultValue "")
  managedPackageVersion = [string](Get-OwnerFieldValue -Record $inputRecord -Name "managedPackageVersion" -DefaultValue "")
  managedPackageSha256 = [string](Get-OwnerFieldValue -Record $inputRecord -Name "managedPackageSha256" -DefaultValue "")
  runtimePackageId = [string](Get-OwnerFieldValue -Record $inputRecord -Name "runtimePackageId" -DefaultValue "")
  runtimePackageVersion = [string](Get-OwnerFieldValue -Record $inputRecord -Name "runtimePackageVersion" -DefaultValue "")
  runtimePackageSha256 = [string](Get-OwnerFieldValue -Record $inputRecord -Name "runtimePackageSha256" -DefaultValue "")
  nugetPackageSource = [string](Get-OwnerFieldValue -Record $inputRecord -Name "nugetPackageSource" -DefaultValue "")
  githubPackagesSource = [string](Get-OwnerFieldValue -Record $inputRecord -Name "githubPackagesSource" -DefaultValue "")
  packageOutputRoot = [string](Get-OwnerFieldValue -Record $inputRecord -Name "packageOutputRoot" -DefaultValue "")
  publishCommandPlanSha256 = [string](Get-OwnerFieldValue -Record $inputRecord -Name "publishCommandPlanSha256" -DefaultValue "")
  rollbackPlanSha256 = [string](Get-OwnerFieldValue -Record $inputRecord -Name "rollbackPlanSha256" -DefaultValue "")
  noLocalFeedConfirmation = [bool](Get-OwnerFieldValue -Record $inputRecord -Name "noLocalFeedConfirmation" -DefaultValue $false)
  noProjectReferenceConfirmation = [bool](Get-OwnerFieldValue -Record $inputRecord -Name "noProjectReferenceConfirmation" -DefaultValue $false)
  noDirectNupkgConfirmation = [bool](Get-OwnerFieldValue -Record $inputRecord -Name "noDirectNupkgConfirmation" -DefaultValue $false)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Authorization input import copies Owner input only. It does not publish, dispatch workflows, close releases, or promote proof."
}

$record | Add-Member -NotePropertyName "authorizationPhraseMatches" -NotePropertyValue ($record.explicitAuthorizationPhrase -ceq $record.expectedAuthorizationPhrase)
$record | Add-Member -NotePropertyName "authorizationInputReady" -NotePropertyValue ($record.ownerAuthorizedPublicPublish -and $record.authorizationPhraseMatches -and -not [string]::IsNullOrWhiteSpace($record.authorizedBy) -and -not [string]::IsNullOrWhiteSpace($record.authorizedAtUtc))

$jsonPath = Join-Path $OutputRoot "owner-public-publish-authorization-input-import.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-authorization-input-import.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Owner Public Publish Authorization Input Import

| Item | Value |
|---|---|
| importState | ``$($record.importState)`` |
| ownerAuthorizedPublicPublish | ``$($record.ownerAuthorizedPublicPublish)`` |
| authorizationPhraseMatches | ``$($record.authorizationPhraseMatches)`` |
| authorizationInputReady | ``$($record.authorizationInputReady)`` |
| managedPackageId | ``$(ConvertTo-MarkdownCell $record.managedPackageId)`` |
| runtimePackageId | ``$(ConvertTo-MarkdownCell $record.runtimePackageId)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner public publish authorization input import written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ImportState=$($record.importState) AuthorizationInputReady=$($record.authorizationInputReady)"

