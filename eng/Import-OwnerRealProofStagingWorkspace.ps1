[CmdletBinding()]
param(
  [string]$OwnerStagingRoot = "",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Test-OwnerPathInsideRoot {
  param([string]$CandidatePath, [string]$RootPath)
  if ([string]::IsNullOrWhiteSpace($CandidatePath) -or [string]::IsNullOrWhiteSpace($RootPath)) { return $false }
  $candidate = [System.IO.Path]::GetFullPath($CandidatePath).TrimEnd('\', '/')
  $root = [System.IO.Path]::GetFullPath($RootPath).TrimEnd('\', '/')
  $rootSlash = $root + [System.IO.Path]::DirectorySeparatorChar
  $rootAltSlash = $root + [System.IO.Path]::AltDirectorySeparatorChar
  return $candidate.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
    $candidate.StartsWith($rootSlash, [StringComparison]::OrdinalIgnoreCase) -or
    $candidate.StartsWith($rootAltSlash, [StringComparison]::OrdinalIgnoreCase)
}

function Test-ForbiddenPathFragment {
  param(
    [string]$Path,
    [switch]$AllowNupkgEvidenceFile
  )

  foreach ($fragment in @("ProjectReference", "local feed", "local-feed", "localfeed", "\bin\", "\obj\", "artifacts\final-release", "artifacts/final-release", "NuGet.Config")) {
    if ($Path.IndexOf($fragment, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
  }

  if (-not $AllowNupkgEvidenceFile.IsPresent -and $Path.IndexOf(".nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
    return $true
  }

  return $false
}

$contract = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\owner-real-proof-staging-workspace-contract.json"
if ($null -eq $contract) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealProofStagingWorkspaceContract.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  $contract = Read-JsonOrNull $RepositoryRoot "artifacts\final-release\owner-real-proof-staging-workspace-contract.json"
}

$requiredFiles = @(Convert-ToArray (Get-PropertyOrDefault -Object $contract -Name "requiredFiles" -DefaultValue @()))
$findings = New-Object System.Collections.Generic.List[object]
$mappingResults = New-Object System.Collections.Generic.List[object]

if ([string]::IsNullOrWhiteSpace($OwnerStagingRoot)) {
  $findings.Add((New-OwnerFinding "owner-staging-root-missing" "action-required" "missing-field" "OwnerStagingRoot was not supplied.")) | Out-Null
  $resolvedRoot = ""
  $rootOutsideRepository = $false
}
else {
  $resolvedRoot = Resolve-OwnerPath $RepositoryRoot $OwnerStagingRoot
  $rootOutsideRepository = -not (Test-OwnerPathInsideRoot -CandidatePath $resolvedRoot -RootPath $RepositoryRoot)
  if (-not $rootOutsideRepository) {
    $findings.Add((New-OwnerFinding "owner-staging-root-outside-repository" "action-required" "forbidden-path" "Owner staging root must be outside the repository root.")) | Out-Null
  }
  if (Test-ForbiddenPathFragment -Path $resolvedRoot) {
    $findings.Add((New-OwnerFinding "owner-staging-root-forbidden-fragment" "action-required" "forbidden-substitute" "Owner staging root contains a forbidden local substitute fragment.")) | Out-Null
  }
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
    $findings.Add((New-OwnerFinding "owner-staging-root-exists" "action-required" "missing-file" "Owner staging root does not exist.")) | Out-Null
  }
}

foreach ($file in $requiredFiles) {
  $relativePath = [string](Get-PropertyOrDefault -Object $file -Name "relativePath" -DefaultValue "")
  $evidenceKind = [string](Get-PropertyOrDefault -Object $file -Name "evidenceKind" -DefaultValue "")
  $fullPath = if ([string]::IsNullOrWhiteSpace($resolvedRoot) -or [string]::IsNullOrWhiteSpace($relativePath)) { "" } else { Resolve-OwnerPath $resolvedRoot $relativePath }
  $exists = -not [string]::IsNullOrWhiteSpace($fullPath) -and (Test-Path -LiteralPath $fullPath -PathType Leaf)
  $requiresSha256 = [bool](Get-PropertyOrDefault -Object $file -Name "requiresSha256" -DefaultValue $false)
  $computedSha256 = ""
  $hashValid = $false
  $allowNupkgEvidenceFile = $evidenceKind -eq "public-package-file" -and $relativePath.StartsWith("public-package/", [StringComparison]::OrdinalIgnoreCase)
  $forbiddenPath = -not [string]::IsNullOrWhiteSpace($fullPath) -and (Test-ForbiddenPathFragment -Path $fullPath -AllowNupkgEvidenceFile:$allowNupkgEvidenceFile)

  if ([string]::IsNullOrWhiteSpace($fullPath)) {
    $findings.Add((New-OwnerFinding "$relativePath-missing-root" "action-required" "missing-file" "Cannot resolve $relativePath without OwnerStagingRoot.")) | Out-Null
  }
  elseif ($forbiddenPath) {
    $findings.Add((New-OwnerFinding "$relativePath-forbidden-fragment" "action-required" "forbidden-substitute" "Staging file path contains a forbidden local substitute fragment: $relativePath")) | Out-Null
  }
  elseif ($RequireExistingFiles.IsPresent -and -not $exists) {
    $findings.Add((New-OwnerFinding "$relativePath-exists" "action-required" "missing-file" "Required staging file is missing: $relativePath")) | Out-Null
  }

  if ($exists -and $requiresSha256) {
    $computedSha256 = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashValid = Test-Sha256Text $computedSha256
    if ($RequireHashMatch.IsPresent -and -not $hashValid) {
      $findings.Add((New-OwnerFinding "$relativePath-sha256" "action-required" "missing-sha256" "SHA256 could not be computed for $relativePath.")) | Out-Null
    }
  }

  $mappingResults.Add([pscustomobject]@{
      lane = [string](Get-PropertyOrDefault -Object $file -Name "lane" -DefaultValue "")
      evidenceKind = $evidenceKind
      sourcePath = $relativePath
      resolvedPath = $fullPath
      fileExists = $exists
      requiresSha256 = $requiresSha256
      computedSha256 = $computedSha256
      hashValid = $hashValid
      forbiddenPath = $forbiddenPath
      targetJson = [string](Get-PropertyOrDefault -Object $file -Name "targetJson" -DefaultValue "")
      targetField = [string](Get-PropertyOrDefault -Object $file -Name "targetField" -DefaultValue "")
      targetHashField = [string](Get-PropertyOrDefault -Object $file -Name "targetHashField" -DefaultValue "")
      ownerActionRequired = (-not $exists) -or ($requiresSha256 -and -not $hashValid) -or $forbiddenPath
      passed = $exists -and ((-not $requiresSha256) -or $hashValid) -and (-not $forbiddenPath)
    }) | Out-Null
}

$mappingArray = @($mappingResults.ToArray())
$laneSummaries = foreach ($lane in @($mappingArray | Select-Object -ExpandProperty lane -Unique)) {
  $laneMappings = @($mappingArray | Where-Object { [string]$_.lane -eq [string]$lane })
  [pscustomobject]@{
    lane = [string]$lane
    requiredFileCount = $laneMappings.Count
    existingFileCount = @($laneMappings | Where-Object { [bool]$_.fileExists }).Count
    sha256RequiredFileCount = @($laneMappings | Where-Object { [bool]$_.requiresSha256 }).Count
    sha256ValidFileCount = @($laneMappings | Where-Object { [bool]$_.requiresSha256 -and [bool]$_.hashValid }).Count
    forbiddenPathCount = @($laneMappings | Where-Object { [bool]$_.forbiddenPath }).Count
    laneReadyForStrictImport = ($laneMappings.Count -gt 0 -and @($laneMappings | Where-Object { -not [bool]$_.passed }).Count -eq 0)
  }
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$requiresStrictEvidence = $RequireExistingFiles.IsPresent -and $RequireHashMatch.IsPresent
if (-not $requiresStrictEvidence) {
  $findings.Add((New-OwnerFinding "strict-import-flags-required" "action-required" "missing-strictness" "Strict import readiness requires -RequireExistingFiles and -RequireHashMatch.")) | Out-Null
  $failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
}

$readyForStrictImport = $rootOutsideRepository -and $requiresStrictEvidence -and $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0 -and @($mappingArray | Where-Object { -not [bool]$_.passed }).Count -eq 0
$state = if ($readyForStrictImport) { "owner-real-proof-staging-workspace-import-ready" } else { "blocked-owner-real-proof-staging-workspace-required" }

$candidate = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($readyForStrictImport) { "owner-real-proof-staging-workspace-candidate-ready-for-strict-import" } else { "blocked-owner-real-proof-staging-workspace-candidate" }
  ownerStagingRoot = $OwnerStagingRoot
  resolvedOwnerStagingRoot = $resolvedRoot
  rootOutsideRepository = $rootOutsideRepository
  laneCount = @($laneSummaries).Count
  mappingCount = $mappingArray.Count
  existingFileCount = @($mappingArray | Where-Object { [bool]$_.fileExists }).Count
  sha256RequiredFileCount = @($mappingArray | Where-Object { [bool]$_.requiresSha256 }).Count
  sha256ValidFileCount = @($mappingArray | Where-Object { [bool]$_.requiresSha256 -and [bool]$_.hashValid }).Count
  forbiddenPathCount = @($mappingArray | Where-Object { [bool]$_.forbiddenPath }).Count
  laneSummaries = @($laneSummaries)
  mappingResults = @($mappingArray)
  proofCandidateReady = $false
  readyForStrictImport = $readyForStrictImport
  ownerActionRequired = -not $readyForStrictImport
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
  boundary = "Owner staging workspace candidate only maps files into owner input candidates. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$import = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $state
  ownerStagingRoot = $OwnerStagingRoot
  resolvedOwnerStagingRoot = $resolvedRoot
  rootOutsideRepository = $rootOutsideRepository
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  failOnNotProof = $FailOnNotProof.IsPresent
  laneCount = @($laneSummaries).Count
  mappingCount = $mappingArray.Count
  existingFileCount = @($mappingArray | Where-Object { [bool]$_.fileExists }).Count
  sha256RequiredFileCount = @($mappingArray | Where-Object { [bool]$_.requiresSha256 }).Count
  sha256ValidFileCount = @($mappingArray | Where-Object { [bool]$_.requiresSha256 -and [bool]$_.hashValid }).Count
  forbiddenPathCount = @($mappingArray | Where-Object { [bool]$_.forbiddenPath }).Count
  laneSummaries = @($laneSummaries)
  mappingResults = @($mappingArray)
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  findings = @($findings.ToArray())
  candidatePath = "artifacts/final-release/owner-real-proof-staging-workspace-candidate.json"
  readyForStrictImport = $readyForStrictImport
  proofCandidateReady = $false
  ownerActionRequired = -not $readyForStrictImport
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
  boundary = "Owner staging workspace import validates local owner file layout only. Strict External CleanConsumer, YoloVision, article publication, public package, and release close validators must still accept real evidence; this import is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$importPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-import.json"
$importMdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-import.md"
$candidatePath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-candidate.json"
$candidateMdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-candidate.md"
Write-Utf8File -LiteralPath $importPath -InputObject ($import | ConvertTo-Json -Depth 16)
Write-Utf8File -LiteralPath $candidatePath -InputObject ($candidate | ConvertTo-Json -Depth 16)

$findingRows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.severity)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}
$laneRows = foreach ($lane in $laneSummaries) {
  "| ``$($lane.lane)`` | ``$($lane.existingFileCount)/$($lane.requiredFileCount)`` | ``$($lane.sha256ValidFileCount)/$($lane.sha256RequiredFileCount)`` | ``$($lane.forbiddenPathCount)`` | ``$($lane.laneReadyForStrictImport)`` |"
}
Write-Utf8File -LiteralPath $importMdPath -InputObject @("# Owner Real Proof Staging Workspace Import", "", "- importState: ``$state``", "- readyForStrictImport: ``$readyForStrictImport``", "- rootOutsideRepository: ``$rootOutsideRepository``", "- mappings: ``$($import.existingFileCount)/$($import.mappingCount)``", "- sha256: ``$($import.sha256ValidFileCount)/$($import.sha256RequiredFileCount)``", "- failedActionRequiredCount: ``$($failedActionRequired.Count)``", "", "| Lane | Files | SHA256 | Forbidden Paths | Ready |", "|---|---:|---:|---:|---:|", @($laneRows), "", "| ID | Severity | Category | Message |", "|---|---|---|---|", @($findingRows), "", "## Boundary", "", $import.boundary)
Write-Utf8File -LiteralPath $candidateMdPath -InputObject @("# Owner Real Proof Staging Workspace Candidate", "", "- candidateState: ``$($candidate.candidateState)``", "- readyForStrictImport: ``$($candidate.readyForStrictImport)``", "- proofCandidateReady: ``False``", "- laneCount: ``$($candidate.laneCount)``", "- mappingCount: ``$($candidate.mappingCount)``", "", "## Boundary", "", $candidate.boundary)

Write-Host "OwnerRealProofStagingWorkspaceImportState=$state ReadyForStrictImport=$readyForStrictImport Mappings=$($import.existingFileCount)/$($import.mappingCount) Sha256=$($import.sha256ValidFileCount)/$($import.sha256RequiredFileCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $readyForStrictImport) { throw "Owner real proof staging workspace is not ready for strict import." }
