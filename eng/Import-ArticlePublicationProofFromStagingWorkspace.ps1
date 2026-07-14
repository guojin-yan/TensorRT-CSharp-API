[CmdletBinding()]
param(
  [string]$OwnerStagingRoot = "",
  [string]$StagingImportPath = "artifacts\final-release\owner-real-proof-staging-workspace-import.json",
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

function Test-HttpUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("http://", [StringComparison]::OrdinalIgnoreCase)
}

function Add-ArticleFinding {
  param([string]$Id, [string]$Category, [string]$Message)
  $script:findings.Add((New-OwnerFinding $Id "action-required" $Category $Message)) | Out-Null
}

function Get-StringProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  return [string](Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue "")
}

function Test-TrueProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  return [bool](Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $false)
}

function Test-FalseProperty {
  param([AllowNull()][object]$Object, [string]$Name)
  return -not [bool](Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $true)
}

function Get-LaneMapping {
  param([object[]]$Mappings, [string]$SourcePath)
  $normalized = $SourcePath.Replace("\", "/")
  return @($Mappings | Where-Object { ([string]$_.sourcePath).Replace("\", "/") -eq $normalized } | Select-Object -First 1)[0]
}

if (-not [System.IO.Path]::IsPathRooted($StagingImportPath)) {
  $StagingImportPath = Join-Path $RepositoryRoot $StagingImportPath
}

if (-not [string]::IsNullOrWhiteSpace($OwnerStagingRoot)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerRealProofStagingWorkspace.ps1") `
    -RepositoryRoot $RepositoryRoot `
    -OutputRoot $OutputRoot `
    -OwnerStagingRoot $OwnerStagingRoot `
    -RequireExistingFiles:$RequireExistingFiles.IsPresent `
    -RequireHashMatch:$RequireHashMatch.IsPresent | Out-Null
}
elseif (-not (Test-Path -LiteralPath $StagingImportPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerRealProofStagingWorkspace.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot | Out-Null
}

$stagingImport = Get-Content -LiteralPath $StagingImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$mappings = @(Convert-ToArray (Get-PropertyOrDefault -Object $stagingImport -Name "mappingResults" -DefaultValue @()))
$articleMappings = @($mappings | Where-Object { [string]$_.lane -eq "article-publication" })
$findings = New-Object System.Collections.Generic.List[object]
$script:findings = $findings

$recordsMapping = Get-LaneMapping $articleMappings "article-publication/article-proof-records.json"
$manifestMapping = Get-LaneMapping $articleMappings "article-publication/article-proof-manifest.json"
$screenshotsMapping = Get-LaneMapping $articleMappings "article-publication/screenshots.zip"

foreach ($mapping in @($recordsMapping, $manifestMapping, $screenshotsMapping)) {
  if ($null -eq $mapping -or -not [bool](Get-PropertyOrDefault -Object $mapping -Name "fileExists" -DefaultValue $false)) {
    $source = if ($null -eq $mapping) { "missing-article-publication-mapping" } else { [string]$mapping.sourcePath }
    Add-ArticleFinding ($source.Replace("/", "-") + "-missing") "missing-file" "Article publication staging file is missing: $source"
  }
  elseif (-not [bool](Get-PropertyOrDefault -Object $mapping -Name "hashValid" -DefaultValue $false)) {
    Add-ArticleFinding ([string]$mapping.sourcePath).Replace("/", "-") "missing-sha256" "Article publication staging file has no computable SHA256: $($mapping.sourcePath)"
  }
}

$records = @()
$recordParseReady = $false
$recordFilePath = [string](Get-PropertyOrDefault -Object $recordsMapping -Name "resolvedPath" -DefaultValue "")
if (-not [string]::IsNullOrWhiteSpace($recordFilePath) -and (Test-Path -LiteralPath $recordFilePath -PathType Leaf)) {
  try {
    $rawRecords = Get-Content -LiteralPath $recordFilePath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($rawRecords.PSObject.Properties.Name -contains "articleProofRecords") {
      $records = @(Convert-ToArray $rawRecords.articleProofRecords)
    }
    else {
      $records = @(Convert-ToArray $rawRecords)
    }

    $recordParseReady = $true
  }
  catch {
    Add-ArticleFinding "article-proof-records-json-parse" "invalid-json" "article-proof-records.json must be valid JSON."
  }
}

if ($recordParseReady -and $records.Count -eq 0) {
  Add-ArticleFinding "article-proof-records-empty" "missing-real-owner-input" "article-proof-records.json must contain at least one public article proof record."
}

$readyRecordCount = 0
$blockedRecordCount = 0
for ($index = 0; $index -lt $records.Count; $index++) {
  $record = $records[$index]
  $recordFindingsBefore = $findings.Count
  foreach ($requiredName in @("articleId", "title", "publicUrl", "publishedAtUtc", "platform", "contentSha256", "screenshotPath", "screenshotSha256")) {
    if ([string]::IsNullOrWhiteSpace((Get-StringProperty $record $requiredName))) {
      Add-ArticleFinding "article-record-$index-$requiredName" "missing-field" "Article proof record $index missing required field: $requiredName."
    }
  }

  if (-not (Test-HttpUrl (Get-StringProperty $record "publicUrl"))) {
    Add-ArticleFinding "article-record-$index-public-url" "private-or-invalid-url" "Article proof record $index publicUrl must be http(s)."
  }
  if (-not (Test-Sha256Text (Get-StringProperty $record "contentSha256"))) {
    Add-ArticleFinding "article-record-$index-content-sha256" "missing-sha256" "Article proof record $index contentSha256 must be a SHA256 value."
  }
  if (-not (Test-Sha256Text (Get-StringProperty $record "screenshotSha256"))) {
    Add-ArticleFinding "article-record-$index-screenshot-sha256" "missing-sha256" "Article proof record $index screenshotSha256 must be a SHA256 value."
  }
  if (-not (Test-TrueProperty $record "isPublic")) {
    Add-ArticleFinding "article-record-$index-is-public" "private-url" "Article proof record $index must set isPublic=true."
  }
  if (-not (Test-FalseProperty $record "isDraft")) {
    Add-ArticleFinding "article-record-$index-is-draft" "draft-article" "Article proof record $index must set isDraft=false."
  }
  if (-not (Test-TrueProperty $record "ownerReviewed")) {
    Add-ArticleFinding "article-record-$index-owner-reviewed" "missing-owner-review" "Article proof record $index must set ownerReviewed=true."
  }

  if ($findings.Count -eq $recordFindingsBefore) {
    $readyRecordCount++
  }
  else {
    $blockedRecordCount++
  }
}

$manifestReady = $false
$manifestHashMatches = $false
$screenshotsArchiveHashReady = [bool](Get-PropertyOrDefault -Object $screenshotsMapping -Name "hashValid" -DefaultValue $false)
$manifestFilePath = [string](Get-PropertyOrDefault -Object $manifestMapping -Name "resolvedPath" -DefaultValue "")
if (-not [string]::IsNullOrWhiteSpace($manifestFilePath) -and (Test-Path -LiteralPath $manifestFilePath -PathType Leaf)) {
  try {
    $manifest = Get-Content -LiteralPath $manifestFilePath -Raw -Encoding utf8 | ConvertFrom-Json
    $manifestReady = $true
    $manifestCount = [int](Get-PropertyOrDefault -Object $manifest -Name "articleProofCount" -DefaultValue -1)
    $recordsHash = Get-StringProperty $manifest "articleProofRecordsSha256"
    $screenshotsHash = Get-StringProperty $manifest "screenshotsArchiveSha256"
    $manifestSha = Get-StringProperty $manifest "manifestSha256"
    if ($manifestCount -ne $records.Count) {
      Add-ArticleFinding "article-proof-manifest-count" "manifest-mismatch" "articleProofCount must match article-proof-records count."
    }
    if (-not (Test-HttpUrl (Get-StringProperty $manifest "publicProofManifestUrl"))) {
      Add-ArticleFinding "article-proof-manifest-public-url" "private-or-invalid-url" "publicProofManifestUrl must be http(s)."
    }
    if (-not (Test-Sha256Text $manifestSha)) {
      Add-ArticleFinding "article-proof-manifest-sha256" "missing-sha256" "manifestSha256 must be a SHA256 value."
    }
    if ($recordsHash -ne [string](Get-PropertyOrDefault -Object $recordsMapping -Name "computedSha256" -DefaultValue "")) {
      Add-ArticleFinding "article-proof-manifest-records-hash" "hash-mismatch" "articleProofRecordsSha256 must match article-proof-records.json."
    }
    if ($screenshotsHash -ne [string](Get-PropertyOrDefault -Object $screenshotsMapping -Name "computedSha256" -DefaultValue "")) {
      Add-ArticleFinding "article-proof-manifest-screenshots-hash" "hash-mismatch" "screenshotsArchiveSha256 must match screenshots.zip."
    }

    $manifestHashMatches = $recordsHash -eq [string](Get-PropertyOrDefault -Object $recordsMapping -Name "computedSha256" -DefaultValue "") -and
      $screenshotsHash -eq [string](Get-PropertyOrDefault -Object $screenshotsMapping -Name "computedSha256" -DefaultValue "")
  }
  catch {
    Add-ArticleFinding "article-proof-manifest-json-parse" "invalid-json" "article-proof-manifest.json must be valid JSON."
  }
}

$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$shapeValid = $articleMappings.Count -ge 3 -and
  @($articleMappings | Where-Object { -not [bool]$_.passed }).Count -eq 0 -and
  $recordParseReady -and $records.Count -gt 0 -and $readyRecordCount -eq $records.Count -and
  $manifestReady -and $manifestHashMatches -and $screenshotsArchiveHashReady -and
  $failedActionRequired.Count -eq 0
$state = if ($shapeValid) { "article-publication-staging-shape-valid-non-proof" } else { "blocked-article-publication-staging-owner-proof-required" }

$record = [pscustomobject]@{
  recordKind = "article-publication-proof-from-staging-workspace"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $state
  stagingImportState = [string](Get-PropertyOrDefault -Object $stagingImport -Name "importState" -DefaultValue "")
  stagingReadyForStrictImport = [bool](Get-PropertyOrDefault -Object $stagingImport -Name "readyForStrictImport" -DefaultValue $false)
  laneId = "article-publication"
  laneFileCount = $articleMappings.Count
  existingLaneFileCount = @($articleMappings | Where-Object { [bool]$_.fileExists }).Count
  sha256RequiredFileCount = @($articleMappings | Where-Object { [bool]$_.requiresSha256 }).Count
  sha256ValidFileCount = @($articleMappings | Where-Object { [bool]$_.hashValid }).Count
  articleProofRecordCount = $records.Count
  articleProofReadyRecordCount = $readyRecordCount
  blockedArticleProofRecordCount = $blockedRecordCount
  articleProofRecordsReady = $recordParseReady -and $records.Count -gt 0 -and $readyRecordCount -eq $records.Count
  articleProofManifestReady = $manifestReady
  articleProofManifestHashMatches = $manifestHashMatches
  screenshotsArchiveHashReady = $screenshotsArchiveHashReady
  ownerEvidenceShapeValid = $shapeValid
  findingCount = $findings.Count
  failedActionRequiredCount = $failedActionRequired.Count
  findings = @($findings.ToArray())
  proofCandidateReady = $false
  ownerActionRequired = -not $shapeValid
  passed = $false
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Article publication staging admission validates Owner-provided public article files, hashes, and manifest shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "article-publication-proof-from-staging-workspace.json"
$mdPath = Join-Path $OutputRoot "article-publication-proof-from-staging-workspace.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)
$findingRows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Article Publication Proof From Staging Workspace",
  "",
  "- importState: ``$state``",
  "- articleProofRecords: ``$readyRecordCount/$($records.Count)``",
  "- manifestHashMatches: ``$manifestHashMatches``",
  "- screenshotsArchiveHashReady: ``$screenshotsArchiveHashReady``",
  "- failedActionRequiredCount: ``$($failedActionRequired.Count)``",
  "",
  "| ID | Category | Message |",
  "|---|---|---|",
  @($findingRows),
  "",
  "## Boundary",
  "",
  $record.boundary
)
Write-Host "ArticlePublicationProofFromStagingWorkspaceState=$state Records=$readyRecordCount/$($records.Count) ManifestHashMatches=$manifestHashMatches FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $shapeValid) { throw "Article publication staging workspace evidence is not shape-valid." }
