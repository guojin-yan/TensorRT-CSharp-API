[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-staging-workspace-contract.json",
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
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) { & (Join-Path $RepositoryRoot "eng\Export-OwnerRealProofStagingWorkspaceContract.ps1") -RepositoryRoot $RepositoryRoot }
$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$files = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "requiredFiles" -DefaultValue @()))
$paths = @($files | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "relativePath" -DefaultValue "") })
$boundary = [string](Get-PropertyOrDefault $record "boundary" "")
$required = @(
  "public-package/nuget-managed-package.nupkg",
  "public-package/github-runtime-package.nupkg",
  "public-package/downloaded-packages.json",
  "public-package/download-transcript.log",
  "external-clean-consumer/consumer.csproj",
  "external-clean-consumer/package-source.json",
  "external-clean-consumer/resolved-packages.json",
  "external-clean-consumer/restore.log",
  "external-clean-consumer/build.log",
  "external-clean-consumer/smoke.stdout.log",
  "external-clean-consumer/smoke.stderr.log",
  "external-clean-consumer/merged-transcript.log",
  "external-clean-consumer/native-assets.json",
  "external-clean-consumer/host-metadata.json",
  "external-clean-consumer/package-metadata.json",
  "external-clean-consumer/no-local-substitute-confirmation.json",
  "yolovision/model.onnx",
  "yolovision/asset-manifest.json",
  "yolovision/output.json",
  "yolovision/runtime-transcript.log",
  "yolovision/real-model-execution-confirmation.json",
  "article-publication/article-proof-records.json",
  "article-publication/article-proof-manifest.json",
  "article-publication/screenshots.zip",
  "release-close/release-evidence-bundle.sha256",
  "release-close/classification-audit.sha256",
  "release-close/post-publish-proof.sha256",
  "release-close/rollback-review.json",
  "release-close/final-close-decision.json",
  "release-close/known-limitations.json"
)
$validationItems = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault $record "recordKind" "") -eq "owner-real-proof-staging-workspace-contract") "blocker" "recordKind must match."
  New-OwnerValidationItem "required-paths" (@($required | Where-Object { $paths -notcontains $_ }).Count -eq 0) "blocker" "Contract must include all required staging paths."
  New-OwnerValidationItem "lane-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0) -ge 5 -and [int](Get-PropertyOrDefault -Object $record -Name "requiredFileCount" -DefaultValue 0) -ge 37) "blocker" "Contract must cover public package, external consumer, YoloVision, article publication, and release-close lanes."
  New-OwnerValidationItem "sha-and-substitute-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "sha256RequiredFileCount" -DefaultValue 0) -ge 37 -and [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0) -ge 300) "blocker" "Contract must require hashes and enumerate forbidden substitutes broadly."
  New-OwnerValidationItem "blocked-non-proof" (-not [bool](Get-PropertyOrDefault $record "passed" $true) -and [bool](Get-PropertyOrDefault $record "ownerActionRequired" $false)) "blocker" "Contract must stay blocked/non-proof."
  New-OwnerValidationItem "non-proof-flags" (-not [bool](Get-PropertyOrDefault $record "isRuntimeExecutionProof" $true) -and -not [bool](Get-PropertyOrDefault $record "canCloseReleaseIssue" $true)) "blocker" "Contract cannot claim proof."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof status."
)
$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failed.Count -eq 0) { "owner-real-proof-staging-workspace-contract-ready-non-proof" } else { "blocked-owner-real-proof-staging-workspace-contract-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-contract-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failed.Count
  requiredFileCount = $files.Count
  laneCount = [int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0)
  sha256RequiredFileCount = [int](Get-PropertyOrDefault -Object $record -Name "sha256RequiredFileCount" -DefaultValue 0)
  forbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteCount" -DefaultValue 0)
  validationItems = @($validationItems)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation checks staging workspace contract shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-contract-validation.json"
$mdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-contract-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Owner Real Proof Staging Workspace Contract Validation", "", "- validationState: ``$state``", "- failedBlockerCount: ``$($failed.Count)``", "- requiredFileCount: ``$($files.Count)``", "", "## Boundary", "", $validation.boundary)
Write-Host "OwnerRealProofStagingWorkspaceContractValidationState=$state FailedBlockers=$($failed.Count)"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Owner real proof staging workspace contract validation failed." }
