[CmdletBinding()]
param(
  [string]$MatrixPath = "artifacts/interface-coverage/cross-task-reference-provenance-matrix.json",
  [string]$ContractPath = "samples/assets/cross-task-reference-provenance-contract.json",
  [string]$OutputPath = "artifacts/interface-coverage/cross-task-reference-provenance-validation.json",
  [string]$MarkdownPath = "artifacts/interface-coverage/cross-task-reference-provenance-validation.md",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else { $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot) }

$utf8 = [Text.UTF8Encoding]::new($false)
$checks = [Collections.Generic.List[object]]::new()

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-Sha256 {
  param([AllowEmptyString()][string]$Value)
  return $Value -match '^[0-9a-f]{64}$'
}

function Add-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [AllowEmptyString()][string]$Actual = ""
  )
  $checks.Add([pscustomobject][ordered]@{ id = $Id; passed = $Passed; actual = $Actual }) | Out-Null
}

$matrixFullPath = Resolve-RepositoryPath $MatrixPath
$contractFullPath = Resolve-RepositoryPath $ContractPath
foreach ($path in @($matrixFullPath, $contractFullPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required provenance artifact is missing: $path" }
}
$matrixText = Get-Content -LiteralPath $matrixFullPath -Raw -Encoding utf8
$matrix = $matrixText | ConvertFrom-Json -Depth 100
$contract = Get-Content -LiteralPath $contractFullPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
$rows = @($matrix.rows)
$expectedRowIds = @("classification", "yolo-det", "yolo-cls", "yolo-seg", "yolo-obb", "yolo-pose", "yolo-sem")
$expectedTasks = @("classification", "det", "cls", "seg", "obb", "pose", "sem")
$expectedCommonSectionIds = @("assetIdentity", "tensorContract", "executionIdentity", "referenceIdentity", "comparisonPolicy", "ownerDecision")
$expectedReuseFields = @("modelSha256", "inputTensorSha256", "preprocessContractSha256", "outputTensorContractSha256", "labelsSha256", "taskSemanticsSha256")

Add-Check "matrix-schema" ($matrix.schemaVersion -eq "cross-task-reference-provenance-matrix.v1") $matrix.schemaVersion
Add-Check "matrix-state" ($matrix.state -eq "owner-action-required") $matrix.state
Add-Check "matrix-counts" ([int]$matrix.rowCount -eq 7 -and [int]$matrix.readyRowCount -eq 0 -and [int]$matrix.ownerActionRequiredRowCount -eq 7 -and $rows.Count -eq 7) "$($matrix.rowCount)/$($matrix.readyRowCount)/$($matrix.ownerActionRequiredRowCount)/$($rows.Count)"
Add-Check "matrix-row-order" ((@($rows | ForEach-Object id) -join ",") -eq ($expectedRowIds -join ",")) (@($rows | ForEach-Object id) -join ",")
Add-Check "matrix-task-order" ((@($rows | ForEach-Object task) -join ",") -eq ($expectedTasks -join ",")) (@($rows | ForEach-Object task) -join ",")
Add-Check "matrix-path-free" ($matrixText -notmatch '(?i)[A-Z]:[\\/]') "absolute-windows-path-present=$($matrixText -match '(?i)[A-Z]:[\\/]')"

Add-Check "contract-schema" ($contract.schemaVersion -eq "cross-task-reference-provenance-contract.v1" -and $contract.recordKind -eq "cross-task-reference-provenance-contract") "$($contract.schemaVersion)/$($contract.recordKind)"
Add-Check "contract-state" ($contract.contractState -eq "owner-review-required") $contract.contractState
Add-Check "contract-common-sections" ((@($contract.commonSections | ForEach-Object id) -join ",") -eq ($expectedCommonSectionIds -join ",")) (@($contract.commonSections | ForEach-Object id) -join ",")
Add-Check "contract-common-fields" (@($contract.commonSections | Where-Object { @($_.requiredFields).Count -eq 0 }).Count -eq 0) "empty-sections=$(@($contract.commonSections | Where-Object { @($_.requiredFields).Count -eq 0 }).Count)"
Add-Check "contract-task-profiles" ((@($contract.taskProfiles | ForEach-Object id) -join ",") -eq ($expectedRowIds -join ",")) (@($contract.taskProfiles | ForEach-Object id) -join ",")
Add-Check "contract-task-semantics" (@($contract.taskProfiles | Where-Object { @($_.requiredSemanticFields).Count -lt 7 -or [string]::IsNullOrWhiteSpace($_.boundary) }).Count -eq 0) "invalid-profiles=$(@($contract.taskProfiles | Where-Object { @($_.requiredSemanticFields).Count -lt 7 -or [string]::IsNullOrWhiteSpace($_.boundary) }).Count)"
Add-Check "contract-reuse-fingerprints" ((@($contract.reuseFingerprintFields) -join ",") -eq ($expectedReuseFields -join ",")) (@($contract.reuseFingerprintFields) -join ",")
Add-Check "contract-reuse-rules" (@($contract.reuseRules).Count -eq 4 -and (@($contract.reuseRules) -join " ") -match "Owner" -and (@($contract.reuseRules) -join " ") -match "same-runtime") "count=$(@($contract.reuseRules).Count)"
Add-Check "contract-promotion-boundary" (-not [bool]$contract.promotionBoundary.independentFrameworkCandidateIsOwnerGolden -and -not [bool]$contract.promotionBoundary.localPackageConsumerIsPublicPackageProof -and -not [bool]$contract.promotionBoundary.canPublishPublicly -and -not [bool]$contract.promotionBoundary.canCloseReleaseIssue) "$($contract.promotionBoundary.independentFrameworkCandidateIsOwnerGolden)/$($contract.promotionBoundary.localPackageConsumerIsPublicPackageProof)/$($contract.promotionBoundary.canPublishPublicly)/$($contract.promotionBoundary.canCloseReleaseIssue)"
Add-Check "matrix-contract-cross-check" ($matrix.contract.path -eq "samples/assets/cross-task-reference-provenance-contract.json" -and $matrix.contract.sha256 -eq (Get-Sha256 $contractFullPath) -and $matrix.contract.schemaVersion -eq $contract.schemaVersion) "$($matrix.contract.path)/$($matrix.contract.sha256)/$($matrix.contract.schemaVersion)"

$sources = @($matrix.sources)
Add-Check "source-count" ($sources.Count -eq 4) "$($sources.Count)"
Add-Check "source-roles" ((@($sources | ForEach-Object role) -join ",") -eq "classification-manifest,yolovision-task-contract,yolovision-owner-input-template,independent-reference-candidate") (@($sources | ForEach-Object role) -join ",")
foreach ($source in $sources) {
  $sourcePath = Resolve-RepositoryPath ([string]$source.path)
  $actual = if (Test-Path -LiteralPath $sourcePath -PathType Leaf) { Get-Sha256 $sourcePath } else { "missing" }
  Add-Check "source-$($source.role)-hash" ($actual -eq [string]$source.sha256 -and (Test-Sha256 ([string]$source.sha256))) "$($source.path)/$actual"
}

foreach ($row in $rows) {
  $id = [string]$row.id
  $profile = @($contract.taskProfiles | Where-Object id -eq $id)[0]
  $common = @($row.commonFieldChecks)
  $semantics = @($row.taskSemanticChecks)
  $allFields = @($common + $semantics)
  $readyCount = @($allFields | Where-Object ready).Count
  $expectedSemanticIds = @($profile.requiredSemanticFields)

  Add-Check "$id-field-counts" ([int]$row.requiredFieldCount -eq $allFields.Count -and [int]$row.readyFieldCount -eq $readyCount -and [int]$row.missingFieldCount -eq ($allFields.Count - $readyCount) -and [int]$row.missingFieldCount -gt 0) "$($row.readyFieldCount)/$($row.requiredFieldCount)/$($row.missingFieldCount)"
  Add-Check "$id-common-fields" ($common.Count -eq 15 -and @($common | Group-Object id | Where-Object Count -ne 1).Count -eq 0) "count=$($common.Count)/duplicates=$(@($common | Group-Object id | Where-Object Count -ne 1).Count)"
  Add-Check "$id-semantic-fields" ((@($semantics | ForEach-Object id) -join ",") -eq ($expectedSemanticIds -join ",")) (@($semantics | ForEach-Object id) -join ",")
  Add-Check "$id-field-state-contract" (@($allFields | Where-Object { $_.valueState -notin @("recorded", "owner-action-required") -or [string]::IsNullOrWhiteSpace($_.source) }).Count -eq 0) "invalid=$(@($allFields | Where-Object { $_.valueState -notin @('recorded','owner-action-required') -or [string]::IsNullOrWhiteSpace($_.source) }).Count)"
  Add-Check "$id-reference-state" ($row.independentReferenceState -eq "not-captured-for-$($row.task)" -and -not [bool]$row.referenceReuseEligible) "$($row.independentReferenceState)/$($row.referenceReuseEligible)"
  Add-Check "$id-promotion-boundary" (-not [bool]$row.ownerReviewedGolden -and -not [bool]$row.canPromoteRealModelRuntime -and -not [bool]$row.canPromotePackageConsumerRuntime) "$($row.ownerReviewedGolden)/$($row.canPromoteRealModelRuntime)/$($row.canPromotePackageConsumerRuntime)"
  Add-Check "$id-proof-classification" ($row.proofClassification -eq "template-only" -and [string]$row.sourceState -match "owner|candidate") "$($row.proofClassification)/$($row.sourceState)"
  Add-Check "$id-boundary" (-not [string]::IsNullOrWhiteSpace([string]$row.boundary) -and [string]$row.boundary -eq [string]$profile.boundary) $row.boundary
  if ($id -like "yolo-*") {
    Add-Check "$id-yolo-contract-link" (@($row.contractRequiredMetadata).Count -ge 5 -and @($row.contractRequiredMetadata | Where-Object { [string]::IsNullOrWhiteSpace([string]$_) }).Count -eq 0) (@($row.contractRequiredMetadata) -join ",")
  }
}

$candidates = @($matrix.independentReferenceCandidates)
Add-Check "candidate-count" ($candidates.Count -eq 1) "$($candidates.Count)"
if ($candidates.Count -eq 1) {
  $candidate = $candidates[0]
  Add-Check "candidate-identity" ($candidate.id -eq "mnist-onnxruntime-cpu-1.23.2" -and $candidate.task -eq "mnist-classification" -and $candidate.evidenceClassification -eq "independent-framework-reference-candidate-runtime") "$($candidate.id)/$($candidate.task)/$($candidate.evidenceClassification)"
  Add-Check "candidate-provider" ($candidate.framework -eq "ONNX Runtime" -and $candidate.version -eq "1.23.2" -and $candidate.provider -eq "CPUExecutionProvider" -and [bool]$candidate.providerValidated -and [bool]$candidate.deterministicOutput) "$($candidate.framework)/$($candidate.version)/$($candidate.provider)/$($candidate.providerValidated)/$($candidate.deterministicOutput)"
  Add-Check "candidate-hashes" ($candidate.modelSha256 -eq "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf" -and $candidate.inputTensorSha256 -eq "81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564" -and $candidate.referenceSha256 -eq "1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571") "$($candidate.modelSha256)/$($candidate.inputTensorSha256)/$($candidate.referenceSha256)"
  Add-Check "candidate-task-isolation" ((@($candidate.eligibleTaskIds) -join ",") -eq "mnist" -and (@($candidate.ineligibleMatrixTaskIds) -join ",") -eq ($expectedRowIds -join ",") -and -not [bool]$candidate.canReuseForAnyMatrixTask) "$(@($candidate.eligibleTaskIds) -join ',')/$(@($candidate.ineligibleMatrixTaskIds) -join ',')/$($candidate.canReuseForAnyMatrixTask)"
  Add-Check "candidate-owner-boundary" (-not [bool]$candidate.allReuseFingerprintsRecorded -and -not [bool]$candidate.ownerReviewedGolden -and -not [bool]$candidate.repositoryRedistributionApproved) "$($candidate.allReuseFingerprintsRecorded)/$($candidate.ownerReviewedGolden)/$($candidate.repositoryRedistributionApproved)"
  Add-Check "candidate-reason" ($candidate.reason -match "do not match" -and $candidate.reason -match "YoloVision") $candidate.reason
}

$boundary = $matrix.proofBoundary
Add-Check "matrix-proof-boundary" (-not [bool]$boundary.crossTaskReferenceReuseProved -and -not [bool]$boundary.ownerReviewedGoldenAvailable -and -not [bool]$boundary.publicPackageProof -and -not [bool]$boundary.postPublishProof -and -not [bool]$boundary.canPublishPublicly -and -not [bool]$boundary.canCloseReleaseIssue) "$($boundary.crossTaskReferenceReuseProved)/$($boundary.ownerReviewedGoldenAvailable)/$($boundary.publicPackageProof)/$($boundary.postPublishProof)/$($boundary.canPublishPublicly)/$($boundary.canCloseReleaseIssue)"
Add-Check "matrix-proof-statement" ($boundary.statement -match "audits" -and $boundary.statement -match "does not create" -and $boundary.statement -match "Owner golden") $boundary.statement

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "cross-task-reference-provenance-validation.v1"
  strict = [bool]$Strict
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = @($checks)
}
$outputFullPath = Resolve-RepositoryPath $OutputPath
$markdownFullPath = Resolve-RepositoryPath $MarkdownPath
New-Item -ItemType Directory -Path (Split-Path -Parent $outputFullPath) -Force | Out-Null
[IO.File]::WriteAllText($outputFullPath, ($result | ConvertTo-Json -Depth 20) + [Environment]::NewLine, $utf8)
$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# Cross-Task Reference Provenance Validation")
$lines.Add("")
$lines.Add("- strict: ``$([bool]$Strict)``")
$lines.Add("- checks: ``$($checks.Count)``")
$lines.Add("- passed: ``$($checks.Count - $failed.Count)``")
$lines.Add("- failed: ``$($failed.Count)``")
$lines.Add("")
$lines.Add("| Check | Passed | Actual |")
$lines.Add("| --- | --- | --- |")
foreach ($check in $checks) {
  $actual = ([string]$check.actual).Replace("|", "\\|").Replace("`r", " ").Replace("`n", " ")
  $lines.Add("| ``$($check.id)`` | ``$($check.passed)`` | $actual |")
}
$lines.Add("")
$lines.Add("The validator audits contract structure and current readiness only; it does not create or promote runtime, Owner, package, publication, or release proof.")
[IO.File]::WriteAllLines($markdownFullPath, $lines, $utf8)

Write-Output "CrossTaskReferenceProvenanceValidation=$($checks.Count - $failed.Count)/$($checks.Count)"
Write-Output "Validation=$outputFullPath"
if ($Strict -and $failed.Count -gt 0) {
  throw "Cross-task reference provenance validation failed: $($failed.Count) check(s)."
}
