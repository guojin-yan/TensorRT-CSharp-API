[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-result-owner-input.template.json",
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public publish result owner input not found: $resolvedInputPath"
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json

$imported = [pscustomobject]@{
  recordKind = "public-publish-result-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = "blocked-public-publish-result-owner-input-required"
  ownerName = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerName" -DefaultValue "")
  ownerEmail = [string](Get-PropertyOrDefault -Object $inputRecord -Name "ownerEmail" -DefaultValue "")
  publishedAtUtc = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publishedAtUtc" -DefaultValue "")
  selectedChannel = [string](Get-PropertyOrDefault -Object $inputRecord -Name "selectedChannel" -DefaultValue "")
  nugetPackageSource = [string](Get-PropertyOrDefault -Object $inputRecord -Name "nugetPackageSource" -DefaultValue "")
  nugetPackageUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "nugetPackageUrl" -DefaultValue "")
  githubPackageUrl = [string](Get-PropertyOrDefault -Object $inputRecord -Name "githubPackageUrl" -DefaultValue "")
  githubRelease = Get-PropertyOrDefault -Object $inputRecord -Name "githubRelease" -DefaultValue $null
  packageId = [string](Get-PropertyOrDefault -Object $inputRecord -Name "packageId" -DefaultValue "")
  packageVersion = [string](Get-PropertyOrDefault -Object $inputRecord -Name "packageVersion" -DefaultValue "")
  managedNupkgPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedNupkgPath" -DefaultValue "")
  managedNupkgSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "managedNupkgSha256" -DefaultValue "")
  managedPackage = Get-PropertyOrDefault -Object $inputRecord -Name "managedPackage" -DefaultValue $null
  runtimePackageKey = [string](Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageKey" -DefaultValue "")
  runtimePackagePaths = @((Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackagePaths" -DefaultValue @()))
  runtimePackageSha256 = @((Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackageSha256" -DefaultValue @()))
  runtimePackage = Get-PropertyOrDefault -Object $inputRecord -Name "runtimePackage" -DefaultValue $null
  publishCommandTranscriptPath = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publishCommandTranscriptPath" -DefaultValue "")
  publishCommandTranscriptSha256 = [string](Get-PropertyOrDefault -Object $inputRecord -Name "publishCommandTranscriptSha256" -DefaultValue "")
  ownerReview = Get-PropertyOrDefault -Object $inputRecord -Name "ownerReview" -DefaultValue $null
  rollbackReview = Get-PropertyOrDefault -Object $inputRecord -Name "rollbackReview" -DefaultValue $null
  finalCloseDecision = Get-PropertyOrDefault -Object $inputRecord -Name "finalCloseDecision" -DefaultValue $null
  ownerReviewedPackageHash = [bool](Get-PropertyOrDefault -Object $inputRecord -Name "ownerReviewedPackageHash" -DefaultValue $false)
  ownerReviewedPublicUrl = [bool](Get-PropertyOrDefault -Object $inputRecord -Name "ownerReviewedPublicUrl" -DefaultValue $false)
  rollbackPlanReviewed = [bool](Get-PropertyOrDefault -Object $inputRecord -Name "rollbackPlanReviewed" -DefaultValue $false)
  sourceInputPath = $resolvedInputPath
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Public publish result import copies Owner-filled metadata only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "public-publish-result-import.json"
$markdownPath = Join-Path $artifactRoot "public-publish-result-import.md"
$imported | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$runtimeRows = for ($i = 0; $i -lt $imported.runtimePackagePaths.Count; $i++) {
  $hash = if ($i -lt $imported.runtimePackageSha256.Count) { $imported.runtimePackageSha256[$i] } else { "" }
  "| $(ConvertTo-MarkdownCell $imported.runtimePackagePaths[$i]) | $(ConvertTo-MarkdownCell $hash) |"
}

$markdown = @"
# Public Publish Result Import

生成时间：$($imported.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| importState | ``$($imported.importState)`` |
| selectedChannel | ``$($imported.selectedChannel)`` |
| packageId | ``$($imported.packageId)`` |
| packageVersion | ``$($imported.packageVersion)`` |
| notExecutedByAutomation | ``$($imported.notExecutedByAutomation)`` |
| performsPublish | ``$($imported.performsPublish)`` |
| canPublishPublicly | ``$($imported.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($imported.canCloseReleaseIssue)`` |

## Runtime Packages

| Path | SHA256 |
|---|---|
$($runtimeRows -join "`r`n")

## Boundary

$($imported.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish result import written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ImportState=$($imported.importState) PerformsPublish=False CanCloseReleaseIssue=False"
