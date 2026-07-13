[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$InputPath,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot $OutputDirectory
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $OutputDirectory "owner-real-evidence-final-intake-checklist.json" }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-OwnerRealEvidenceFinalIntakeChecklist.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $record | ConvertTo-Json -Depth 32
$items = @((Get-PropertyOrDefault -Object $record -Name "checklistItems" -DefaultValue @()))
$nonSubstitutes = @((Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @()))
$requiredGroups = @("external-clean-consumer", "post-publish-clean-consumer", "public-package", "host-metadata", "package-metadata", "owner-governance", "owner-confirmations", "final-gates")
$requiredForbidden = @("local feed", "ProjectReference", "direct .nupkg", "pre-publish smoke", "template", "candidate", "dashboard", "runbook", "command pack", "build-only", "dependency-probe-only")

$validationItems = New-Object System.Collections.Generic.List[object]
$validationItems.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-real-evidence-final-intake-checklist") "blocker" "recordKind must match.")) | Out-Null
$validationItems.Add((New-ValidationItem "state-blocked" ([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "") -eq "blocked-owner-real-evidence-final-intake-required") "blocker" "Checklist must remain blocked until real Owner evidence is supplied.")) | Out-Null
$validationItems.Add((New-ValidationItem "item-count" ($items.Count -ge 15) "blocker" "Checklist must include all final intake evidence areas.")) | Out-Null
$validationItems.Add((New-ValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) "blocker" "Checklist must remain non-proof and non-publishing.")) | Out-Null
$validationItems.Add((New-ValidationItem "failed-blocker-count-is-not-proof" ([bool](Get-PropertyOrDefault -Object $record -Name "failedBlockerCountIsNotProof" -DefaultValue $false)) "blocker" "Checklist must state failedBlockerCount=0 is not proof.")) | Out-Null
$validationItems.Add((New-ValidationItem "boundary" ($raw.IndexOf("never executes dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("failedBlockerCount=0", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $raw.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0) "blocker" "Boundary must reject publishing and proof substitution.")) | Out-Null

foreach ($group in $requiredGroups) {
  $validationItems.Add((New-ValidationItem "group-$group" (@($items | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "group" -DefaultValue "") -eq $group }).Count -gt 0) "blocker" "Checklist must include group $group.")) | Out-Null
}

foreach ($marker in $requiredForbidden) {
  $validationItems.Add((New-ValidationItem "forbidden-$($marker.Replace(' ', '-').Replace('.', 'dot'))" ($nonSubstitutes -contains $marker) "blocker" "nonSubstituteProofKinds must include $marker.")) | Out-Null
}

$blockedItems = @($items | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validation = [ordered]@{
  recordKind = "owner-real-evidence-final-intake-checklist-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = if ($failedBlockers.Count -eq 0) { "owner-real-evidence-final-intake-checklist-ready-non-proof" } else { "failed-owner-real-evidence-final-intake-checklist" }
  checklistState = [string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "")
  checklistItemCount = $items.Count
  blockedChecklistItemCount = $blockedItems.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $blockedItems.Count
  failedBlockerCountIsNotProof = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($validationItems.ToArray())
  boundary = "Owner real evidence final intake checklist validation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not proof."
}

$jsonPath = Join-Path $OutputDirectory "owner-real-evidence-final-intake-checklist-validation.json"
$markdownPath = Join-Path $OutputDirectory "owner-real-evidence-final-intake-checklist-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 16)
Write-Utf8File -LiteralPath $markdownPath -InputObject @(
  "# Owner Real Evidence Final Intake Checklist Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- checklistItemCount: ``$($validation.checklistItemCount)``",
  "- blockedChecklistItemCount: ``$($validation.blockedChecklistItemCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "- failedBlockerCountIsNotProof: ``True``",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

Write-Host "OwnerRealEvidenceFinalIntakeChecklistValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner real evidence final intake checklist validation failed." }
