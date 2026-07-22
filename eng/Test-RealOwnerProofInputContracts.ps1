[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-owner-proof-input-contracts.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInput = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInput -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Export-RealOwnerProofInputContracts.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null }
$record = Get-Content -LiteralPath $resolvedInput -Raw -Encoding utf8 | ConvertFrom-Json
$contracts = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "contracts" -DefaultValue @()))
$ids = @($contracts | ForEach-Object { [string]$_.id })
$requiredIds = @("public-package-url-hash", "external-clean-consumer-post-publish", "article-publication", "yolovision-real-model", "final-rollback-review", "final-close-decision", "github-ci-evidence", "release-evidence-bundle-hash-review", "classification-audit-hash-review")
$missingIds = @($requiredIds | Where-Object { $ids -notcontains $_ })
$duplicateIds = @($ids | Group-Object | Where-Object { $_.Count -gt 1 })
$missingScripts = @()
foreach ($contract in $contracts) {
  foreach ($scriptName in @([string]$contract.refreshScript, [string]$contract.validationScript)) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot "eng\$scriptName") -PathType Leaf)) { $missingScripts += $scriptName }
  }
}
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-owner-proof-input-contracts") "blocker" "recordKind must match."
  New-OwnerValidationItem "lane-coverage" ($contracts.Count -eq 9 -and $missingIds.Count -eq 0) "blocker" "Contracts must cover all nine Owner proof lanes."
  New-OwnerValidationItem "unique-lane-ids" ($duplicateIds.Count -eq 0) "blocker" "Lane IDs must be unique."
  New-OwnerValidationItem "field-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "requiredFieldCount" -DefaultValue 0) -ge 90 -and @($contracts | Where-Object { [int]$_.requiredFieldCount -lt 5 }).Count -eq 0) "blocker" "Each lane must expose concrete required fields."
  New-OwnerValidationItem "scripts-exist" ($missingScripts.Count -eq 0) "blocker" "Every lane refresh/validation script must exist."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) "blocker" "Contract manifest must remain non-proof."
  New-OwnerValidationItem "boundary" ($boundary.IndexOf("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $boundary.IndexOf("not package push", [StringComparison]::OrdinalIgnoreCase) -ge 0) "blocker" "Boundary must preserve non-proof status."
)
$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "real-owner-proof-input-contracts-validation-ready-non-proof" } else { "invalid-real-owner-proof-input-contracts" }
$validation = [pscustomobject]@{
  recordKind = "real-owner-proof-input-contracts-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  laneCount = $contracts.Count
  requiredFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredFieldCount" -DefaultValue 0)
  missingLaneCount = $missingIds.Count
  missingScriptCount = $missingScripts.Count
  failedBlockerCount = $failed.Count
  validationItems = @($items)
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Real Owner proof input contract validation checks schema and script linkage only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-input-contracts-validation.json") -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "real-owner-proof-input-contracts-validation.md") -InputObject @("# Real Owner Proof Input Contracts Validation", "", "- validationState: ``$state``", "- laneCount: ``$($validation.laneCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "RealOwnerProofInputContractsValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Real Owner proof input contracts validation failed." }
