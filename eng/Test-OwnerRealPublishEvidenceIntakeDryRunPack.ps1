[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-publish-evidence-intake-dry-run-pack.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Get-PropertyOrDefault -Object $record -Name "intakeGroups" -DefaultValue @())
$fakeReadyCases = @(Get-PropertyOrDefault -Object $record -Name "fakeReadyCases" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredGroups = @("managed-runtime-public-package", "publish-command-and-transcripts", "owner-authorization", "source-runner-boundary", "forbidden-substitute-scan", "post-publish-clean-consumer")
$requiredSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dashboard", "dry-run", "manual approval", "queued GitHub Actions run", "missing self-hosted runner", "sidecar-only", "TensorRtExec report")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-real-publish-evidence-intake-dry-run-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-dry-run" (([string](Get-PropertyOrDefault -Object $record -Name "intakeState" -DefaultValue "")).Contains("blocked") -and [bool](Get-PropertyOrDefault -Object $record -Name "dryRunOnly" -DefaultValue $false)) "blocker" "Intake pack must stay blocked and dry-run only.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsGitHubPackagesPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "Dry-run pack must not publish or close release.")) | Out-Null
$items.Add((New-OwnerValidationItem "contract-field-count" ([int](Get-PropertyOrDefault -Object $record -Name "contractRequiredFieldCount" -DefaultValue 0) -ge 178) "blocker" "Owner public publish input contract must retain expanded required field count.")) | Out-Null
$items.Add((New-OwnerValidationItem "group-count" ($groups.Count -eq $requiredGroups.Count) "blocker" "Every intake group must be present.")) | Out-Null
$items.Add((New-OwnerValidationItem "contract-field-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "declaredButMissingInContractCount" -DefaultValue -1) -eq 0) "blocker" "Dry-run required fields must exist in the Owner public publish contract.")) | Out-Null
foreach ($id in $requiredGroups) {
  $items.Add((New-OwnerValidationItem "group-$id" (@($groups | Where-Object { [string]$_.id -eq $id -and -not [bool]$_.performsPublish -and -not [bool]$_.canCloseReleaseIssue -and [int]$_.requiredFieldCount -gt 0 }).Count -eq 1) "blocker" "Missing or unsafe intake group: $id")) | Out-Null
}
foreach ($substitute in $requiredSubstitutes) {
  $items.Add((New-OwnerValidationItem "fake-ready-$substitute" (@($fakeReadyCases | Where-Object { [string]$_.substitute -eq $substitute -and [bool]$_.fakeReadyBlocked -and -not [bool]$_.isProof }).Count -eq 1) "blocker" "Fake-ready substitute must remain blocked: $substitute")) | Out-Null
}
$boundaryOk = $boundary.IndexOf("never runs dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("never publishes GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0
$items.Add((New-OwnerValidationItem "boundary" $boundaryOk "blocker" "Boundary must forbid publish and fake-ready substitutes.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "owner-real-publish-evidence-intake-dry-run-pack-validation-ready-non-proof" } else { "blocked-owner-real-publish-evidence-intake-dry-run-pack-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-intake-dry-run-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  intakeGroupCount = [int]$groups.Count
  fakeReadyCaseCount = [int]$fakeReadyCases.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  dryRunOnly = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Dry-run intake validation only; not proof, not publish approval, not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-intake-dry-run-pack-validation.json"
$mdPath = Join-Path $OutputRoot "owner-real-publish-evidence-intake-dry-run-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Real Publish Evidence Intake Dry-Run Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- intakeGroupCount: ``$($validation.intakeGroupCount)``",
  "",
  $validation.boundary
)
Write-Host "OwnerRealPublishEvidenceIntakeDryRunPackValidationState=$state FailedBlockers=$failedBlockerCount Groups=$($validation.intakeGroupCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Owner real publish evidence intake dry-run pack validation failed." }
