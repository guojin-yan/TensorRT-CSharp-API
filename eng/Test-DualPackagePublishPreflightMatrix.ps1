[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\dual-package-publish-preflight-matrix.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Dual package publish preflight matrix not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$routes = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "routes" -DefaultValue @()))
$routeIds = @($routes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredRouteIds = @("nuget-small-bridge-core", "github-packages-bridge")
$ownerActions = @($routes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "nextOwnerAction" -DefaultValue "") })
$externalProofMissingReasons = @($routes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "externalProofMissingReason" -DefaultValue "") })
$postPublishProofMissingReasons = @($routes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "postPublishProofMissingReason" -DefaultValue "") })

$unsafeRoutes = @($routes | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "usesPublishToken" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishGitHubPackages" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsSubstituteProof" -DefaultValue $true) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "requiresOwnerAuthorization" -DefaultValue $false) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "externalProofRequired" -DefaultValue $false) -or
  -not [bool](Get-PropertyOrDefault -Object $_ -Name "postPublishProofRequired" -DefaultValue $false)
})

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "dual-package-publish-preflight-matrix") -Severity "blocker" -Detail "recordKind must be dual-package-publish-preflight-matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "route-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "routeCount" -DefaultValue 0) -eq 2 -and $routes.Count -eq 2) -Severity "blocker" -Detail "Matrix must include exactly two routes: managed API and project-owned bridge packages.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-routes" -Passed (@($requiredRouteIds | Where-Object { $routeIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Both required route ids must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-publish-non-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishGitHubPackages" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "acceptsSubstituteProof" -DefaultValue $true)) -Severity "blocker" -Detail "Matrix must not publish, use token, claim package consumer proof, close release, or accept substitute proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "route-non-publish-non-proof" -Passed ($unsafeRoutes.Count -eq 0) -Severity "blocker" -Detail "Each route must stay blocked, owner-authorized only, non-publish, non-proof, and no-substitute.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-actions" -Passed (($ownerActions -contains "owner-authorize-public-nuget-publish-and-import-clean-external-consumer-proof") -and ($ownerActions -contains "owner-authorize-github-bridge-publish-and-import-clean-runtime-proof")) -Severity "blocker" -Detail "Both route next owner actions must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "external-proof-gaps" -Passed (($externalProofMissingReasons -contains "public-package-download-and-clean-consumer-runtime-proof-missing") -and ($externalProofMissingReasons -contains "github-bridge-restore-external-dependency-resolution-clean-smoke-missing")) -Severity "blocker" -Detail "Both external proof missing reasons must be present.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-gaps" -Passed (($postPublishProofMissingReasons -contains "post-publish-clean-consumer-proof-missing") -and ($postPublishProofMissingReasons -contains "post-publish-github-bridge-clean-consumer-proof-missing")) -Severity "blocker" -Detail "Both post-publish proof missing reasons must be present.")) | Out-Null

foreach ($route in $routes) {
  $id = [string](Get-PropertyOrDefault -Object $route -Name "id" -DefaultValue "")
  $requirements = @(ConvertTo-Array (Get-PropertyOrDefault -Object $route -Name "evidenceRequirements" -DefaultValue @()))
  $requirementsNeedOwnerProof = @($requirements | Where-Object {
    [bool](Get-PropertyOrDefault -Object $_ -Name "satisfied" -DefaultValue $true) -and
    [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -ne "package-dry-run-pack-success"
  }).Count -eq 0
  $items.Add((New-ValidationItem -Id "requirements-$id-owner-proof" -Passed ($requirements.Count -ge 5 -and $requirementsNeedOwnerProof) -Severity "blocker" -Detail "Route $id must keep unsatisfied owner proof requirements visible.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-dual-package-publish-preflight-owner-proof-required" } else { "invalid-dual-package-publish-preflight-matrix" }

$validation = [pscustomobject]@{
  recordKind = "dual-package-publish-preflight-matrix-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  routeCount = $routes.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = if ($failedBlockers.Count -eq 0) { 1 } else { 0 }
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimPackageConsumerRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  acceptsSubstituteProof = $false
  boundary = "Validation checks dual package publish preflight shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "dual-package-publish-preflight-matrix-validation.json"
$markdownPath = Join-Path $OutputRoot "dual-package-publish-preflight-matrix-validation.md"
[System.IO.File]::WriteAllText($jsonPath, (($validation | ConvertTo-Json -Depth 12) + [Environment]::NewLine), $utf8)

$rows = foreach ($item in $items) {
  "| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $($item.detail.Replace("|", "\|")) |"
}

$markdown = @(
  "# Dual Package Publish Preflight Matrix Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| routeCount | ``$($validation.routeCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| canPublishPublicly | ``$($validation.canPublishPublicly)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Validation Items",
  "",
  "| ID | Passed | Severity | Detail |",
  "| --- | ---: | --- | --- |"
) + $rows + @(
  "",
  "## Boundary",
  "",
  $validation.boundary
)
[System.IO.File]::WriteAllText($markdownPath, (($markdown -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)

Write-Host "Dual package publish preflight matrix validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState Routes=$($routes.Count) FailedBlockers=$($failedBlockers.Count) CanPublish=False CanClose=False"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Dual package publish preflight matrix validation failed with $($failedBlockers.Count) blocker(s)."
}
