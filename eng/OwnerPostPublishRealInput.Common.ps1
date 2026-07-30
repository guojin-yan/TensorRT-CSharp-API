. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

function Get-OwnerPostPublishForbiddenSubstitutes {
  @(
    "template",
    "example",
    "draft",
    "misuse",
    "ready fixture",
    "dashboard",
    "audit",
    "bundle",
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "queued workflow",
    "missing runner",
    "dry-run",
    "pre-publish smoke",
    "readiness",
    "tutorial",
    "matrix",
    "local package cache",
    "NuGet.Config local source",
    ".nupkg"
  )
}

function New-OwnerPostPublishFieldSpec {
  param([string]$Name, [string]$Kind, [string]$Description)
  [pscustomobject]@{
    name = $Name
    kind = $Kind
    description = $Description
  }
}

function New-OwnerPostPublishLaneSpec {
  param([string]$Id, [string]$Title, [object[]]$Fields)
  [pscustomobject]@{
    id = $Id
    title = $Title
    fields = @($Fields)
    requiredFieldCount = @($Fields).Count
  }
}

function Get-OwnerPostPublishLaneSpecs {
  @(
    New-OwnerPostPublishLaneSpec -Id "public-package-urls-and-hashes" -Title "Public package URLs and hashes" -Fields @(
      New-OwnerPostPublishFieldSpec "nugetManagedPackageId" "string" "NuGet managed C# API package id."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageVersion" "version" "NuGet managed C# API package version."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageUrl" "url" "Public NuGet package URL."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageSha256" "sha256" "Downloaded NuGet package SHA256."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageId" "string" "Legacy-named field for the GitHub Packages bridge-only package id; vendor runtime package ids are forbidden."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageVersion" "version" "Legacy-named field for the GitHub Packages bridge-only package version."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageUrl" "url" "GitHub Packages bridge-only package URL."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageSha256" "sha256" "Downloaded GitHub bridge-only package SHA256."
      New-OwnerPostPublishFieldSpec "downloadedAtUtc" "utc" "Owner download timestamp in UTC."
      New-OwnerPostPublishFieldSpec "downloadTranscriptSha256" "sha256" "Transcript hash for public package download."
    )
    New-OwnerPostPublishLaneSpec -Id "external-clean-consumer-logs" -Title "External clean consumer logs" -Fields @(
      New-OwnerPostPublishFieldSpec "externalWorkspaceRoot" "external-path" "Repository-external clean consumer workspace."
      New-OwnerPostPublishFieldSpec "consumerProjectFileSha256" "sha256" "External consumer project file SHA256."
      New-OwnerPostPublishFieldSpec "packageRestoreSourceUrl" "url" "Public package source URL used by the external consumer."
      New-OwnerPostPublishFieldSpec "restoreResolvedPackageListSha256" "sha256" "Resolved package list transcript SHA256."
      New-OwnerPostPublishFieldSpec "restoreLogSha256" "sha256" "dotnet restore transcript SHA256."
      New-OwnerPostPublishFieldSpec "buildLogSha256" "sha256" "dotnet build transcript SHA256."
      New-OwnerPostPublishFieldSpec "smokeLogSha256" "sha256" "runtime smoke transcript SHA256."
      New-OwnerPostPublishFieldSpec "hostMetadataSha256" "sha256" "Host metadata JSON SHA256."
      New-OwnerPostPublishFieldSpec "noLocalSubstituteConfirmation" "confirmation" "Owner confirms no local feed/ProjectReference/direct nupkg substitute."
    )
    New-OwnerPostPublishLaneSpec -Id "yolovision-real-model-assets" -Title "YoloVision real model assets" -Fields @(
      New-OwnerPostPublishFieldSpec "taskName" "yolovision-task" "YoloVision task name: det, cls, seg, obb, pose, or sem."
      New-OwnerPostPublishFieldSpec "modelSourceUrl" "url" "Real model source URL."
      New-OwnerPostPublishFieldSpec "modelLicense" "string" "Model license or redistribution permission note."
      New-OwnerPostPublishFieldSpec "modelSha256" "sha256" "Real model file SHA256."
      New-OwnerPostPublishFieldSpec "labelsSha256" "sha256" "Labels file SHA256."
      New-OwnerPostPublishFieldSpec "inputImageSha256" "sha256" "Input image or tensor SHA256."
      New-OwnerPostPublishFieldSpec "assetManifestSha256" "sha256" "YoloVision asset manifest SHA256."
      New-OwnerPostPublishFieldSpec "outputJsonSha256" "sha256" "YoloVision output JSON SHA256."
      New-OwnerPostPublishFieldSpec "stdoutLogSha256" "sha256" "YoloVision stdout log SHA256."
      New-OwnerPostPublishFieldSpec "stderrLogSha256" "sha256" "YoloVision stderr log SHA256."
      New-OwnerPostPublishFieldSpec "hostMetadataSha256" "sha256" "Host metadata JSON SHA256."
      New-OwnerPostPublishFieldSpec "runtimeTranscriptSha256" "sha256" "Full YoloVision runtime command transcript SHA256."
      New-OwnerPostPublishFieldSpec "realModelExecutionConfirmation" "confirmation" "Owner confirms this was a real model execution, not a readiness/tutorial artifact."
      New-OwnerPostPublishFieldSpec "ownerReviewedAtUtc" "utc" "Owner review timestamp in UTC."
    )
    New-OwnerPostPublishLaneSpec -Id "article-publication-urls" -Title "Article publication URLs" -Fields @(
      New-OwnerPostPublishFieldSpec "articleProofCount" "positive-int" "Number of public article proof records included."
      New-OwnerPostPublishFieldSpec "articleProofManifestUrl" "url" "Public article proof manifest URL."
      New-OwnerPostPublishFieldSpec "articleProofManifestSha256" "sha256" "Public article proof manifest SHA256."
      New-OwnerPostPublishFieldSpec "articleId" "string" "Article roadmap id."
      New-OwnerPostPublishFieldSpec "articleTitle" "string" "Published article title."
      New-OwnerPostPublishFieldSpec "articlePublishUrl" "url" "Public article URL."
      New-OwnerPostPublishFieldSpec "articlePublishedAtUtc" "utc" "Article publication timestamp in UTC."
      New-OwnerPostPublishFieldSpec "articleScreenshotSha256" "sha256" "Publication screenshot SHA256."
      New-OwnerPostPublishFieldSpec "linkedPublicPackageUrl" "url" "Public package URL linked by article."
      New-OwnerPostPublishFieldSpec "linkedProofHash" "sha256" "Linked proof/evidence hash."
    )
    New-OwnerPostPublishLaneSpec -Id "release-issue-close-material" -Title "Release Issue close material" -Fields @(
      New-OwnerPostPublishFieldSpec "releaseIssueUrl" "url" "Release issue URL."
      New-OwnerPostPublishFieldSpec "ownerCloseDecision" "enum" "Owner close decision."
      New-OwnerPostPublishFieldSpec "releaseEvidenceBundleSha256" "sha256" "Release evidence bundle SHA256."
      New-OwnerPostPublishFieldSpec "classificationAuditSha256" "sha256" "Classification audit SHA256."
      New-OwnerPostPublishFieldSpec "postPublishProofSha256" "sha256" "Post-publish proof SHA256."
      New-OwnerPostPublishFieldSpec "rollbackDecision" "enum" "Owner rollback decision."
      New-OwnerPostPublishFieldSpec "ownerFinalCloseDecisionImportedAtUtc" "utc" "Owner final close decision import timestamp in UTC."
      New-OwnerPostPublishFieldSpec "manualCloseReviewConfirmation" "confirmation" "Owner confirms manual close review remains required."
      New-OwnerPostPublishFieldSpec "knownLimitationsAcknowledgement" "confirmation" "Owner acknowledges known limitations."
    )
  )
}

function Get-OwnerPostPublishAllFieldSpecs {
  foreach ($lane in Get-OwnerPostPublishLaneSpecs) {
    foreach ($field in @($lane.fields)) {
      [pscustomobject]@{
        laneId = $lane.id
        laneTitle = $lane.title
        name = $field.name
        kind = $field.kind
        description = $field.description
      }
    }
  }
}

function New-OwnerPostPublishTemplateRecord {
  $forbidden = @(Get-OwnerPostPublishForbiddenSubstitutes)
  $lanes = foreach ($lane in Get-OwnerPostPublishLaneSpecs) {
    $fields = foreach ($field in @($lane.fields)) {
      [pscustomobject]@{
        name = $field.name
        kind = $field.kind
        value = "<owner-$($lane.id)-$($field.name)>"
        required = $true
        description = $field.description
      }
    }
    $isArticleProofLane = [string]$lane.id -eq "article-publication-urls"
    $articleProofRecords = if ($isArticleProofLane) {
      @(
        [pscustomobject]@{
          articleId = "<owner-article-publication-urls-articleProofRecords-0-articleId>"
          articleTitle = "<owner-article-publication-urls-articleProofRecords-0-articleTitle>"
          articlePublishUrl = "<owner-article-publication-urls-articleProofRecords-0-articlePublishUrl>"
          articlePublishedAtUtc = "<owner-article-publication-urls-articleProofRecords-0-articlePublishedAtUtc>"
          articleScreenshotSha256 = "<owner-article-publication-urls-articleProofRecords-0-articleScreenshotSha256>"
          linkedPublicPackageUrl = "<owner-article-publication-urls-articleProofRecords-0-linkedPublicPackageUrl>"
          linkedProofHash = "<owner-article-publication-urls-articleProofRecords-0-linkedProofHash>"
        }
      )
    }
    else {
      @()
    }

    [pscustomobject]@{
      id = $lane.id
      title = $lane.title
      laneState = "blocked-owner-real-input-required"
      requiredFieldCount = @($fields).Count
      fields = @($fields)
      supportsMultipleArticleProofRecords = $isArticleProofLane
      minimumArticleProofRecordCount = if ($isArticleProofLane) { 1 } else { 0 }
      articleProofRecordCount = 0
      articleProofRecords = @($articleProofRecords)
      proofReady = $false
      performsPublish = $false
      usesPublishToken = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      isReleaseCloseProof = $false
    }
  }

  [pscustomobject]@{
    recordKind = "owner-post-publish-docs-article-sample-real-input-template"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    templateState = "blocked-owner-post-publish-real-input-template"
    laneCount = @($lanes).Count
    requiredFieldCount = [int](@($lanes | ForEach-Object { $_.requiredFieldCount } | Measure-Object -Sum).Sum)
    placeholderFieldCount = @($lanes | ForEach-Object { $_.fields } | Where-Object { Test-OwnerPlaceholder -Value $_.value }).Count
    lanes = @($lanes)
    forbiddenSubstitutes = @($forbidden)
    forbiddenSubstituteCount = @($forbidden).Count
    ownerActionRequired = $true
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner post-publish docs/article/sample real input template is placeholder input only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

function Get-OwnerPostPublishFieldValue {
  param([AllowNull()][object]$InputRecord, [string]$LaneId, [string]$FieldName)
  if ($null -eq $InputRecord) { return $null }
  foreach ($lane in @(Convert-ToArray (Get-PropertyOrDefault -Object $InputRecord -Name "lanes" -DefaultValue @()))) {
    if ([string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "") -ne $LaneId) { continue }
    foreach ($field in @(Convert-ToArray (Get-PropertyOrDefault -Object $lane -Name "fields" -DefaultValue @()))) {
      if ([string](Get-PropertyOrDefault -Object $field -Name "name" -DefaultValue "") -eq $FieldName) {
        return (Get-PropertyOrDefault -Object $field -Name "value" -DefaultValue $null)
      }
    }
  }

  return $null
}

function Test-OwnerPostPublishUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase) -and
    $text.IndexOf("localhost", [StringComparison]::OrdinalIgnoreCase) -lt 0 -and
    $text.IndexOf("127.0.0.1", [StringComparison]::OrdinalIgnoreCase) -lt 0 -and
    $text.IndexOf("file:", [StringComparison]::OrdinalIgnoreCase) -lt 0
}

function Test-OwnerPostPublishUtc {
  param([AllowNull()][object]$Value)
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse([string]$Value, [ref]$parsed)
}

function Test-OwnerPostPublishConfirmation {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text.Equals("true", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("owner-confirmed", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("acknowledged", [StringComparison]::OrdinalIgnoreCase)
}

function Test-OwnerPostPublishEnum {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return -not [string]::IsNullOrWhiteSpace($text) -and -not (Test-OwnerPlaceholder -Value $text)
}

function Test-OwnerPostPublishPositiveInt {
  param([AllowNull()][object]$Value)
  $parsed = 0
  return [int]::TryParse([string]$Value, [ref]$parsed) -and $parsed -gt 0
}

function Test-OwnerPostPublishYoloVisionTask {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text -in @("det", "cls", "seg", "obb", "pose", "sem")
}

function Test-OwnerPostPublishOwnerCloseDecision {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text -in @("close-release-issue-after-manual-review", "keep-release-issue-open", "defer-release-issue-close")
}

function Test-OwnerPostPublishRollbackDecision {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text -in @("rollback-not-required", "rollback-required", "defer-rollback-decision")
}

function Get-OwnerPostPublishUrlHost {
  param([AllowNull()][object]$Value)
  $uri = $null
  if ([System.Uri]::TryCreate([string]$Value, [System.UriKind]::Absolute, [ref]$uri)) {
    return $uri.Host.ToLowerInvariant()
  }

  return ""
}

function Get-OwnerPostPublishExternalPathValidation {
  param([AllowNull()][object]$Value, [string]$RepositoryRoot)
  $text = [string]$Value
  $reasons = New-Object System.Collections.Generic.List[string]
  $normalizedPath = ""

  if ([string]::IsNullOrWhiteSpace($text) -or (Test-OwnerPlaceholder -Value $text)) {
    $reasons.Add("external-path-missing-or-placeholder") | Out-Null
  }
  elseif (-not [System.IO.Path]::IsPathRooted($text)) {
    $reasons.Add("external-path-not-rooted") | Out-Null
  }
  else {
    try {
      $normalizedPath = [System.IO.Path]::GetFullPath($text).TrimEnd('\', '/')
      $root = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
      $rootWithSlash = $root + [System.IO.Path]::DirectorySeparatorChar
      $rootWithAltSlash = $root + [System.IO.Path]::AltDirectorySeparatorChar
      if ($normalizedPath.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
          $normalizedPath.StartsWith($rootWithSlash, [StringComparison]::OrdinalIgnoreCase) -or
          $normalizedPath.StartsWith($rootWithAltSlash, [StringComparison]::OrdinalIgnoreCase)) {
        $reasons.Add("external-path-inside-repository-root") | Out-Null
      }
    }
    catch {
      $reasons.Add("external-path-invalid") | Out-Null
    }
  }

  foreach ($fragment in @("ProjectReference", "local feed", "local-feed", "localfeed", ".nupkg", "artifacts\final-release", "artifacts/final-release", "\bin\", "\obj\", "\packages\")) {
    if ($text.IndexOf($fragment, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $reasons.Add("forbidden-clean-consumer-path-fragment=$fragment") | Out-Null
    }
  }

  [pscustomobject]@{
    formatValid = $reasons.Count -eq 0
    reasons = @($reasons.ToArray())
    normalizedPath = $normalizedPath
  }
}

function Test-OwnerPostPublishInputPathForbidden {
  param([string]$InputPath)
  $leaf = [System.IO.Path]::GetFileName($InputPath)
  foreach ($suffix in @(".template.json", ".example.json", ".draft.json", ".misuse.json", ".ready.json", "-validation.json")) {
    if ($leaf.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
  }

  return $false
}

function Get-OwnerPostPublishLaneObject {
  param([AllowNull()][object]$InputRecord, [string]$LaneId)
  foreach ($lane in @(Convert-ToArray (Get-PropertyOrDefault -Object $InputRecord -Name "lanes" -DefaultValue @()))) {
    if ([string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "") -eq $LaneId) {
      return $lane
    }
  }

  return $null
}

function Get-OwnerPostPublishArticleProofRecords {
  param([AllowNull()][object]$InputRecord)
  $lane = Get-OwnerPostPublishLaneObject -InputRecord $InputRecord -LaneId "article-publication-urls"
  $laneRecords = Convert-ToArray (Get-PropertyOrDefault -Object $lane -Name "articleProofRecords" -DefaultValue @())
  if ($laneRecords.Count -gt 0) { return @($laneRecords) }
  return @(Convert-ToArray (Get-PropertyOrDefault -Object $InputRecord -Name "articleProofRecords" -DefaultValue @()))
}

function Test-OwnerPostPublishArticleProofRecordValue {
  param([string]$Name, [AllowNull()][object]$Value)
  switch ($Name) {
    "articlePublishUrl" { return Test-OwnerPostPublishUrl -Value $Value }
    "articlePublishedAtUtc" { return Test-OwnerPostPublishUtc -Value $Value }
    "articleScreenshotSha256" { return Test-Sha256Text -Value $Value }
    "linkedPublicPackageUrl" { return Test-OwnerPostPublishUrl -Value $Value }
    "linkedProofHash" { return Test-Sha256Text -Value $Value }
    default { return -not (Test-OwnerPlaceholder -Value $Value) -and -not [string]::IsNullOrWhiteSpace([string]$Value) }
  }
}

function Get-OwnerPostPublishArticleProofRecordResults {
  param([AllowNull()][object]$InputRecord)
  $records = @(Get-OwnerPostPublishArticleProofRecords -InputRecord $InputRecord)
  $requiredFields = @("articleId", "articleTitle", "articlePublishUrl", "articlePublishedAtUtc", "articleScreenshotSha256", "linkedPublicPackageUrl", "linkedProofHash")
  $forbidden = @(Get-OwnerPostPublishForbiddenSubstitutes)
  $index = 0
  foreach ($record in $records) {
    $fieldResults = New-Object System.Collections.Generic.List[object]
    foreach ($fieldName in $requiredFields) {
      $value = Get-PropertyOrDefault -Object $record -Name $fieldName -DefaultValue $null
      $valueText = [string]$value
      $placeholder = Test-OwnerPlaceholder -Value $value
      $forbiddenHits = @($forbidden | Where-Object { $valueText.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
      $formatValid = Test-OwnerPostPublishArticleProofRecordValue -Name $fieldName -Value $value
      $fieldResults.Add([pscustomobject]@{
          fieldName = $fieldName
          suppliedValue = $valueText
          placeholder = $placeholder
          formatValid = $formatValid
          forbiddenSubstituteHits = @($forbiddenHits)
          fieldReady = (-not $placeholder) -and $formatValid -and $forbiddenHits.Count -eq 0
        }) | Out-Null
    }

    $fields = @($fieldResults.ToArray())
    [pscustomobject]@{
      recordIndex = $index
      articleId = [string](Get-PropertyOrDefault -Object $record -Name "articleId" -DefaultValue "")
      requiredFieldCount = $fields.Count
      readyFieldCount = @($fields | Where-Object { [bool]$_.fieldReady }).Count
      blockedFieldCount = @($fields | Where-Object { -not [bool]$_.fieldReady }).Count
      recordReady = ($fields.Count -gt 0 -and @($fields | Where-Object { -not [bool]$_.fieldReady }).Count -eq 0)
      fieldResults = @($fields)
    }
    $index++
  }
}

function Get-OwnerPostPublishImportAnalysis {
  param([AllowNull()][object]$InputRecord, [string]$InputPath, [string]$RepositoryRoot)

  $forbidden = @(Get-OwnerPostPublishForbiddenSubstitutes)
  $inputPathForbidden = Test-OwnerPostPublishInputPathForbidden -InputPath $InputPath
  $fieldResults = New-Object System.Collections.Generic.List[object]

  foreach ($spec in Get-OwnerPostPublishAllFieldSpecs) {
    $value = Get-OwnerPostPublishFieldValue -InputRecord $InputRecord -LaneId $spec.laneId -FieldName $spec.name
    $valueText = [string]$value
    $placeholder = Test-OwnerPlaceholder -Value $value
    $forbiddenHits = @($forbidden | Where-Object { $valueText.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
    $formatValid = $false
    $formatValidationReasons = @()
    $normalizedValue = ""
    $urlHost = ""
    switch ([string]$spec.kind) {
      "sha256" { $formatValid = Test-Sha256Text -Value $valueText }
      "url" {
        $formatValid = Test-OwnerPostPublishUrl -Value $valueText
        $urlHost = Get-OwnerPostPublishUrlHost -Value $valueText
        if (-not $formatValid) { $formatValidationReasons = @("url-must-be-public-https") }
      }
      "utc" { $formatValid = Test-OwnerPostPublishUtc -Value $valueText }
      "confirmation" { $formatValid = Test-OwnerPostPublishConfirmation -Value $valueText }
      "enum" {
        if ([string]$spec.name -eq "ownerCloseDecision") {
          $formatValid = Test-OwnerPostPublishOwnerCloseDecision -Value $valueText
          if (-not $formatValid) { $formatValidationReasons = @("owner-close-decision-invalid-or-not-manual-review-safe") }
        }
        elseif ([string]$spec.name -eq "rollbackDecision") {
          $formatValid = Test-OwnerPostPublishRollbackDecision -Value $valueText
          if (-not $formatValid) { $formatValidationReasons = @("rollback-decision-invalid") }
        }
        else {
          $formatValid = Test-OwnerPostPublishEnum -Value $valueText
        }
      }
      "positive-int" { $formatValid = Test-OwnerPostPublishPositiveInt -Value $valueText }
      "yolovision-task" { $formatValid = Test-OwnerPostPublishYoloVisionTask -Value $valueText }
      "external-path" {
        $pathValidation = Get-OwnerPostPublishExternalPathValidation -Value $valueText -RepositoryRoot $RepositoryRoot
        $formatValid = [bool]$pathValidation.formatValid
        $formatValidationReasons = @($pathValidation.reasons)
        $normalizedValue = [string]$pathValidation.normalizedPath
      }
      default { $formatValid = -not $placeholder -and -not [string]::IsNullOrWhiteSpace($valueText) }
    }

    $ready = (-not $placeholder) -and $formatValid -and $forbiddenHits.Count -eq 0 -and (-not $inputPathForbidden)
    $fieldResults.Add([pscustomobject]@{
        laneId = $spec.laneId
        fieldName = $spec.name
        kind = $spec.kind
        suppliedValue = $valueText
        placeholder = $placeholder
        formatValid = $formatValid
        formatValidationReasons = @($formatValidationReasons)
        normalizedValue = $normalizedValue
        urlHost = $urlHost
        forbiddenSubstituteHits = @($forbiddenHits)
        fieldReady = $ready
      }) | Out-Null
  }

  $fields = @($fieldResults.ToArray())
  $articleProofRecordResults = @(Get-OwnerPostPublishArticleProofRecordResults -InputRecord $InputRecord)
  $articleProofRecordCount = $articleProofRecordResults.Count
  $articleProofReadyRecordCount = @($articleProofRecordResults | Where-Object { [bool]$_.recordReady }).Count
  $articleProofRecordFailedCount = @($articleProofRecordResults | Where-Object { -not [bool]$_.recordReady }).Count
  $articleProofCountValue = Get-OwnerPostPublishFieldValue -InputRecord $InputRecord -LaneId "article-publication-urls" -FieldName "articleProofCount"
  $articleProofCountParsed = 0
  $articleProofCountFormatValid = [int]::TryParse([string]$articleProofCountValue, [ref]$articleProofCountParsed)
  $articleProofCountMatchesRecordCount = $articleProofCountFormatValid -and $articleProofCountParsed -eq $articleProofRecordCount
  $articleProofRecordsReady = $articleProofRecordCount -gt 0 -and $articleProofRecordFailedCount -eq 0 -and $articleProofCountMatchesRecordCount
  [pscustomobject]@{
    inputPath = $InputPath
    inputPathForbidden = $inputPathForbidden
    fieldResults = @($fields)
    fieldResultCount = $fields.Count
    readyFieldCount = @($fields | Where-Object { [bool]$_.fieldReady }).Count
    blockedFieldCount = @($fields | Where-Object { -not [bool]$_.fieldReady }).Count
    placeholderFieldCount = @($fields | Where-Object { [bool]$_.placeholder }).Count
    invalidFormatFieldCount = @($fields | Where-Object { -not [bool]$_.formatValid }).Count
    forbiddenSubstituteFieldCount = @($fields | Where-Object { @($_.forbiddenSubstituteHits).Count -gt 0 }).Count
    articleProofRecordCount = $articleProofRecordCount
    articleProofReadyRecordCount = $articleProofReadyRecordCount
    articleProofRecordFailedCount = $articleProofRecordFailedCount
    articleProofCountMatchesRecordCount = $articleProofCountMatchesRecordCount
    articleProofRecordsReady = $articleProofRecordsReady
    articleProofRecordResults = @($articleProofRecordResults)
    allFieldsReady = ($fields.Count -gt 0 -and @($fields | Where-Object { -not [bool]$_.fieldReady }).Count -eq 0 -and $articleProofRecordsReady)
  }
}

function New-OwnerPostPublishLaneCandidate {
  param(
    [object]$ImportRecord,
    [string]$LaneId,
    [string]$RecordKind,
    [string]$CandidateState,
    [string]$Title,
    [string]$Boundary
  )

  $analysis = Get-PropertyOrDefault -Object $ImportRecord -Name "analysis" -DefaultValue $null
  $fieldResults = @(Convert-ToArray (Get-PropertyOrDefault -Object $analysis -Name "fieldResults" -DefaultValue @()) | Where-Object { [string]$_.laneId -eq $LaneId })
  $readyFieldCount = @($fieldResults | Where-Object { [bool]$_.fieldReady }).Count
  $blockedFieldCount = @($fieldResults | Where-Object { -not [bool]$_.fieldReady }).Count
  $laneSpecificBlockedReasons = New-Object System.Collections.Generic.List[string]
  $articleProofRecordResults = @()
  $articleProofRecordCount = 0
  $articleProofReadyRecordCount = 0
  $articleProofRecordsReady = $true
  $supportsMultipleArticleProofRecords = [string]$LaneId -eq "article-publication-urls"
  if ($supportsMultipleArticleProofRecords) {
    $articleProofRecordResults = @(Convert-ToArray (Get-PropertyOrDefault -Object $analysis -Name "articleProofRecordResults" -DefaultValue @()))
    $articleProofRecordCount = [int](Get-PropertyOrDefault -Object $analysis -Name "articleProofRecordCount" -DefaultValue 0)
    $articleProofReadyRecordCount = [int](Get-PropertyOrDefault -Object $analysis -Name "articleProofReadyRecordCount" -DefaultValue 0)
    $articleProofRecordsReady = [bool](Get-PropertyOrDefault -Object $analysis -Name "articleProofRecordsReady" -DefaultValue $false)
    if (-not $articleProofRecordsReady) {
      $laneSpecificBlockedReasons.Add("article-proof-records-missing-invalid-or-count-mismatch") | Out-Null
    }
  }

  $externalWorkspaceField = $fieldResults | Where-Object { [string]$_.fieldName -eq "externalWorkspaceRoot" } | Select-Object -First 1
  $packageRestoreSourceField = $fieldResults | Where-Object { [string]$_.fieldName -eq "packageRestoreSourceUrl" } | Select-Object -First 1
  $taskNameField = $fieldResults | Where-Object { [string]$_.fieldName -eq "taskName" } | Select-Object -First 1
  $realModelExecutionConfirmationField = $fieldResults | Where-Object { [string]$_.fieldName -eq "realModelExecutionConfirmation" } | Select-Object -First 1
  $ownerCloseDecisionField = $fieldResults | Where-Object { [string]$_.fieldName -eq "ownerCloseDecision" } | Select-Object -First 1
  $manualCloseReviewConfirmationField = $fieldResults | Where-Object { [string]$_.fieldName -eq "manualCloseReviewConfirmation" } | Select-Object -First 1
  $laneSpecificReady = $laneSpecificBlockedReasons.Count -eq 0
  $candidateReady = ($fieldResults.Count -gt 0 -and $blockedFieldCount -eq 0 -and -not [bool](Get-PropertyOrDefault -Object $analysis -Name "inputPathForbidden" -DefaultValue $true) -and $laneSpecificReady)

  [pscustomobject]@{
    recordKind = $RecordKind
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    title = $Title
    laneId = $LaneId
    candidateState = $CandidateState
    candidateReady = $candidateReady
    proofReady = $false
    requiredFieldCount = $fieldResults.Count
    readyFieldCount = $readyFieldCount
    blockedFieldCount = $blockedFieldCount
    fieldResults = @($fieldResults)
    laneSpecificReady = $laneSpecificReady
    laneSpecificBlockedReasonCount = $laneSpecificBlockedReasons.Count
    laneSpecificBlockedReasons = @($laneSpecificBlockedReasons.ToArray())
    externalWorkspacePathReady = [bool](Get-PropertyOrDefault -Object $externalWorkspaceField -Name "fieldReady" -DefaultValue $false)
    externalWorkspacePathValidationReasons = @(Get-PropertyOrDefault -Object $externalWorkspaceField -Name "formatValidationReasons" -DefaultValue @())
    packageRestoreSourceUrlHost = [string](Get-PropertyOrDefault -Object $packageRestoreSourceField -Name "urlHost" -DefaultValue "")
    rejectsLocalFeedProjectReferenceAndDirectNupkg = $true
    yoloVisionTaskReady = [bool](Get-PropertyOrDefault -Object $taskNameField -Name "fieldReady" -DefaultValue $false)
    yoloVisionHashFieldCount = @($fieldResults | Where-Object { [string]$_.kind -eq "sha256" }).Count
    realModelExecutionConfirmationReady = [bool](Get-PropertyOrDefault -Object $realModelExecutionConfirmationField -Name "fieldReady" -DefaultValue $false)
    rejectsReadinessTutorialMatrixArtifacts = $true
    supportsMultipleArticleProofRecords = $supportsMultipleArticleProofRecords
    minimumArticleProofRecordCount = if ($supportsMultipleArticleProofRecords) { 1 } else { 0 }
    articleProofRecordCount = $articleProofRecordCount
    articleProofReadyRecordCount = $articleProofReadyRecordCount
    articleProofRecordsReady = $articleProofRecordsReady
    articleProofRecordResults = @($articleProofRecordResults)
    ownerCloseDecisionReady = [bool](Get-PropertyOrDefault -Object $ownerCloseDecisionField -Name "fieldReady" -DefaultValue $false)
    ownerCloseDecisionValue = [string](Get-PropertyOrDefault -Object $ownerCloseDecisionField -Name "suppliedValue" -DefaultValue "")
    manualCloseReviewConfirmationReady = [bool](Get-PropertyOrDefault -Object $manualCloseReviewConfirmationField -Name "fieldReady" -DefaultValue $false)
    manualCloseReviewOnly = $true
    ownerActionRequired = $blockedFieldCount -gt 0 -or (-not $laneSpecificReady)
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = $Boundary
  }
}

function Assert-OwnerPostPublishFalseFlags {
  param([object]$Record)
  return ((-not [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "usesPublishToken" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)) -and
    (-not [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)))
}

function Export-OwnerPostPublishLaneCandidateArtifact {
  param(
    [string]$ImportPath,
    [string]$OutputRoot,
    [string]$RepositoryRoot,
    [string]$LaneId,
    [string]$RecordKind,
    [string]$FileStem,
    [string]$CandidateState,
    [string]$Title,
    [string]$Boundary,
    [AllowNull()][object]$ExtraProperties
  )

  $resolvedImportPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $ImportPath
  if (-not (Test-Path -LiteralPath $resolvedImportPath -PathType Leaf)) {
    & (Join-Path $RepositoryRoot "eng\Import-OwnerPostPublishDocsArticleSampleRealInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  }

  $import = Get-Content -LiteralPath $resolvedImportPath -Raw -Encoding utf8 | ConvertFrom-Json
  $candidate = New-OwnerPostPublishLaneCandidate -ImportRecord $import -LaneId $LaneId -RecordKind $RecordKind -CandidateState $CandidateState -Title $Title -Boundary $Boundary
  if ($null -ne $ExtraProperties) {
    foreach ($property in $ExtraProperties.PSObject.Properties) {
      Add-Member -InputObject $candidate -NotePropertyName $property.Name -NotePropertyValue $property.Value -Force
    }
  }

  $jsonPath = Join-Path $OutputRoot "$FileStem.json"
  $mdPath = Join-Path $OutputRoot "$FileStem.md"
  Write-Utf8File -LiteralPath $jsonPath -InputObject ($candidate | ConvertTo-Json -Depth 16)
  Write-Utf8File -LiteralPath $mdPath -InputObject @(
    "# $Title",
    "",
    "- candidateState: ``$($candidate.candidateState)``",
    "- candidateReady: ``$($candidate.candidateReady)``",
    "- proofReady: ``False``",
    "- requiredFieldCount: ``$($candidate.requiredFieldCount)``",
    "- readyFieldCount: ``$($candidate.readyFieldCount)``",
    "- blockedFieldCount: ``$($candidate.blockedFieldCount)``",
    "- canCloseReleaseIssue: ``False``",
    "",
    $candidate.boundary
  )

  return $candidate
}

function Test-OwnerPostPublishLaneCandidateArtifact {
  param(
    [object]$Record,
    [string]$RecordKind,
    [string]$LaneId,
    [string]$ExpectedBlockedState,
    [string]$RequiredBoundaryText
  )

  $items = New-Object System.Collections.Generic.List[object]
  $items.Add((New-OwnerValidationItem "record-kind" ([string]$Record.recordKind -eq $RecordKind) "blocker" "Candidate recordKind must match.")) | Out-Null
  $items.Add((New-OwnerValidationItem "lane-id" ([string]$Record.laneId -eq $LaneId) "blocker" "Candidate lane id must match.")) | Out-Null
  $items.Add((New-OwnerValidationItem "blocked-state" ([string]$Record.candidateState -eq $ExpectedBlockedState -and -not [bool]$Record.proofReady) "blocker" "Candidate must remain blocked/non-proof by default.")) | Out-Null
  $items.Add((New-OwnerValidationItem "field-coverage" ([int]$Record.requiredFieldCount -gt 0 -and ([int]$Record.blockedFieldCount -gt 0 -or [bool](Get-PropertyOrDefault -Object $Record -Name "candidateReady" -DefaultValue $false))) "blocker" "Candidate must carry lane field results and either blocked fields or an explicit candidateReady=true state.")) | Out-Null
  $items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $Record) "blocker" "Candidate must not publish, close, or promote proof.")) | Out-Null
  $items.Add((New-OwnerValidationItem "boundary" ([string]$Record.boundary -like "*$RequiredBoundaryText*" -and [string]$Record.boundary -like "*not package push*") "blocker" "Candidate boundary must explicitly reject proof substitution.")) | Out-Null
  return @($items.ToArray())
}

function New-OwnerPostPublishProofValidatorSpec {
  param(
    [string]$LaneId,
    [string]$RecordKind,
    [string]$FileStem,
    [string]$CandidateStem,
    [string]$ExportScript,
    [string]$TestScript,
    [string]$Title,
    [string]$BlockedState,
    [string]$AcceptedState,
    [string]$ProofKind,
    [bool]$IsFinalBridge
  )

  [pscustomobject]@{
    laneId = $LaneId
    recordKind = $RecordKind
    fileStem = $FileStem
    candidateStem = $CandidateStem
    exportScript = $ExportScript
    testScript = $TestScript
    title = $Title
    blockedState = $BlockedState
    acceptedState = $AcceptedState
    proofKind = $ProofKind
    isFinalBridge = $IsFinalBridge
  }
}

function Get-OwnerPostPublishProofValidatorSpecs {
  @(
    New-OwnerPostPublishProofValidatorSpec `
      -LaneId "public-package-urls-and-hashes" `
      -RecordKind "public-package-url-hash-proof-validator" `
      -FileStem "public-package-url-hash-proof-validator" `
      -CandidateStem "public-package-url-hash-verification-candidate" `
      -ExportScript "Export-PublicPackageUrlHashVerificationCandidate.ps1" `
      -TestScript "Test-PublicPackageUrlHashVerificationCandidate.ps1" `
      -Title "Public Package URL/Hash Proof Validator" `
      -BlockedState "blocked-public-package-url-hash-real-owner-proof-required" `
      -AcceptedState "public-package-url-hash-real-owner-evidence-accepted" `
      -ProofKind "public-package-url-hash" `
      -IsFinalBridge $false
    New-OwnerPostPublishProofValidatorSpec `
      -LaneId "external-clean-consumer-logs" `
      -RecordKind "external-clean-consumer-post-publish-proof-validator" `
      -FileStem "external-clean-consumer-post-publish-proof-validator" `
      -CandidateStem "external-clean-consumer-post-publish-candidate" `
      -ExportScript "Export-ExternalCleanConsumerPostPublishCandidate.ps1" `
      -TestScript "Test-ExternalCleanConsumerPostPublishCandidate.ps1" `
      -Title "External Clean Consumer Post-Publish Proof Validator" `
      -BlockedState "blocked-external-clean-consumer-real-owner-proof-required" `
      -AcceptedState "external-clean-consumer-real-owner-evidence-accepted" `
      -ProofKind "external-clean-consumer" `
      -IsFinalBridge $false
    New-OwnerPostPublishProofValidatorSpec `
      -LaneId "yolovision-real-model-assets" `
      -RecordKind "yolovision-real-model-post-publish-proof-validator" `
      -FileStem "yolovision-real-model-post-publish-proof-validator" `
      -CandidateStem "yolovision-real-model-post-publish-candidate" `
      -ExportScript "Export-YoloVisionRealModelPostPublishCandidate.ps1" `
      -TestScript "Test-YoloVisionRealModelPostPublishCandidate.ps1" `
      -Title "YoloVision Real Model Post-Publish Proof Validator" `
      -BlockedState "blocked-yolovision-real-model-real-owner-proof-required" `
      -AcceptedState "yolovision-real-model-real-owner-evidence-accepted" `
      -ProofKind "yolovision-real-model" `
      -IsFinalBridge $false
    New-OwnerPostPublishProofValidatorSpec `
      -LaneId "article-publication-urls" `
      -RecordKind "article-publication-proof-validator" `
      -FileStem "article-publication-proof-validator" `
      -CandidateStem "article-publication-proof-candidate" `
      -ExportScript "Export-ArticlePublicationProofCandidate.ps1" `
      -TestScript "Test-ArticlePublicationProofCandidate.ps1" `
      -Title "Article Publication Proof Validator" `
      -BlockedState "blocked-article-publication-real-owner-proof-required" `
      -AcceptedState "article-publication-real-owner-evidence-accepted" `
      -ProofKind "article-publication" `
      -IsFinalBridge $false
    New-OwnerPostPublishProofValidatorSpec `
      -LaneId "release-issue-close-material" `
      -RecordKind "release-close-final-bridge-proof-validator" `
      -FileStem "release-close-final-bridge-proof-validator" `
      -CandidateStem "release-issue-close-material-candidate" `
      -ExportScript "Export-ReleaseIssueCloseMaterialCandidate.ps1" `
      -TestScript "Test-ReleaseIssueCloseMaterialCandidate.ps1" `
      -Title "Release Close Final Bridge Proof Validator" `
      -BlockedState "blocked-release-close-final-bridge-real-owner-proof-required" `
      -AcceptedState "release-close-final-bridge-real-owner-evidence-accepted" `
      -ProofKind "release-close-final-bridge" `
      -IsFinalBridge $true
  )
}

function Get-OwnerPostPublishProofValidatorSpec {
  param([string]$RecordKind)
  foreach ($spec in Get-OwnerPostPublishProofValidatorSpecs) {
    if ([string]$spec.recordKind -eq $RecordKind) { return $spec }
  }

  throw "Unknown Owner post-publish proof validator record kind: $RecordKind"
}

function Read-OwnerPostPublishJsonOrNull {
  param([string]$RepositoryRoot, [string]$Path)
  $resolved = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-OwnerPostPublishValidatorDependencyResults {
  param([string]$OutputRoot)

  foreach ($spec in @(Get-OwnerPostPublishProofValidatorSpecs | Where-Object { -not [bool]$_.isFinalBridge })) {
    $path = Join-Path $OutputRoot "$($spec.fileStem)-validation.json"
    $validation = $null
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      $validation = Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
    }

    [pscustomobject]@{
      recordKind = [string]$spec.recordKind
      validationPath = $path
      validationPresent = $null -ne $validation
      ownerEvidenceAccepted = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerEvidenceAccepted" -DefaultValue $false)
      failedBlockerCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedBlockerCount" -DefaultValue 999)
    }
  }
}

function Export-OwnerPostPublishProofValidatorArtifact {
  param(
    [string]$ImportPath,
    [string]$OutputRoot,
    [string]$RepositoryRoot,
    [string]$RecordKind
  )

  $spec = Get-OwnerPostPublishProofValidatorSpec -RecordKind $RecordKind
  $resolvedImportPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $ImportPath
  if (-not (Test-Path -LiteralPath $resolvedImportPath -PathType Leaf)) {
    & (Join-Path $RepositoryRoot "eng\Import-OwnerPostPublishDocsArticleSampleRealInput.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  }

  $candidatePath = Join-Path $OutputRoot "$($spec.candidateStem).json"
  if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) {
    & (Join-Path $RepositoryRoot "eng\$($spec.exportScript)") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  }

  $candidateValidationPath = Join-Path $OutputRoot "$($spec.candidateStem)-validation.json"
  if (-not (Test-Path -LiteralPath $candidateValidationPath -PathType Leaf)) {
    & (Join-Path $RepositoryRoot "eng\$($spec.testScript)") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
  }

  $import = Get-Content -LiteralPath $resolvedImportPath -Raw -Encoding utf8 | ConvertFrom-Json
  $candidate = Get-Content -LiteralPath $candidatePath -Raw -Encoding utf8 | ConvertFrom-Json
  $candidateValidation = Get-Content -LiteralPath $candidateValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
  $analysis = Get-PropertyOrDefault -Object $import -Name "analysis" -DefaultValue $null
  $fieldResults = @(Convert-ToArray (Get-PropertyOrDefault -Object $analysis -Name "fieldResults" -DefaultValue @()) | Where-Object { [string]$_.laneId -eq [string]$spec.laneId })
  $readyFieldCount = @($fieldResults | Where-Object { [bool]$_.fieldReady }).Count
  $blockedFieldCount = @($fieldResults | Where-Object { -not [bool]$_.fieldReady }).Count
  $realOwnerInputPresent = [bool](Get-PropertyOrDefault -Object $import -Name "realOwnerInputPresent" -DefaultValue $false)
  $inputPathForbidden = [bool](Get-PropertyOrDefault -Object $import -Name "inputPathForbidden" -DefaultValue $true)
  $importCandidateReady = [bool](Get-PropertyOrDefault -Object $import -Name "candidateReady" -DefaultValue $false)
  $candidateReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "candidateReady" -DefaultValue $false)
  $candidateValidationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $candidateValidation -Name "failedBlockerCount" -DefaultValue 999)
  $laneSpecificReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "laneSpecificReady" -DefaultValue $true)
  $laneSpecificBlockedReasonCount = [int](Get-PropertyOrDefault -Object $candidate -Name "laneSpecificBlockedReasonCount" -DefaultValue 0)
  $dependencyResults = @()
  $dependencyAcceptedCount = 0
  $dependencyRequiredCount = 0
  if ([bool]$spec.isFinalBridge) {
    $dependencyResults = @(Get-OwnerPostPublishValidatorDependencyResults -OutputRoot $OutputRoot)
    $dependencyRequiredCount = $dependencyResults.Count
    $dependencyAcceptedCount = @($dependencyResults | Where-Object { [bool]$_.ownerEvidenceAccepted -and [int]$_.failedBlockerCount -eq 0 }).Count
  }

  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if (-not $realOwnerInputPresent) { $blockedReasons.Add("real-owner-input-file-missing") | Out-Null }
  if ($inputPathForbidden) { $blockedReasons.Add("owner-input-path-forbidden-template-draft-or-validation") | Out-Null }
  if (-not $importCandidateReady) { $blockedReasons.Add("owner-real-input-import-not-candidate-ready") | Out-Null }
  if (-not $candidateReady) { $blockedReasons.Add("lane-candidate-not-ready") | Out-Null }
  if (-not $laneSpecificReady -or $laneSpecificBlockedReasonCount -gt 0) { $blockedReasons.Add("lane-specific-proof-surface-blocked=$laneSpecificBlockedReasonCount") | Out-Null }
  if ($blockedFieldCount -gt 0) { $blockedReasons.Add("lane-fields-blocked=$blockedFieldCount") | Out-Null }
  if ($candidateValidationFailedBlockerCount -gt 0) { $blockedReasons.Add("candidate-validation-failed-blockers=$candidateValidationFailedBlockerCount") | Out-Null }
  if ([bool]$spec.isFinalBridge -and $dependencyAcceptedCount -lt $dependencyRequiredCount) { $blockedReasons.Add("dependency-proof-validators-not-accepted=$dependencyAcceptedCount/$dependencyRequiredCount") | Out-Null }

  $ownerEvidenceAccepted = $realOwnerInputPresent -and
    (-not $inputPathForbidden) -and
    $importCandidateReady -and
    $candidateReady -and
    ($blockedFieldCount -eq 0) -and
    ($candidateValidationFailedBlockerCount -eq 0) -and
    ((-not [bool]$spec.isFinalBridge) -or ($dependencyRequiredCount -gt 0 -and $dependencyAcceptedCount -eq $dependencyRequiredCount))
  $validatorState = if ($ownerEvidenceAccepted) { [string]$spec.acceptedState } else { [string]$spec.blockedState }

  $record = [pscustomobject]@{
    recordKind = [string]$spec.recordKind
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    title = [string]$spec.title
    validatorState = $validatorState
    proofKind = [string]$spec.proofKind
    laneId = [string]$spec.laneId
    sourceImportPath = $resolvedImportPath
    sourceCandidatePath = $candidatePath
    sourceCandidateValidationPath = $candidateValidationPath
    realOwnerInputPresent = $realOwnerInputPresent
    inputPathForbidden = $inputPathForbidden
    importCandidateReady = $importCandidateReady
    laneCandidateReady = $candidateReady
    laneSpecificReady = $laneSpecificReady
    laneSpecificBlockedReasonCount = $laneSpecificBlockedReasonCount
    laneSpecificBlockedReasons = @(Get-PropertyOrDefault -Object $candidate -Name "laneSpecificBlockedReasons" -DefaultValue @())
    ownerEvidenceAccepted = $ownerEvidenceAccepted
    proofReady = $false
    proofPromotionReady = $false
    requiredFieldCount = $fieldResults.Count
    readyFieldCount = $readyFieldCount
    blockedFieldCount = $blockedFieldCount
    fieldResults = @($fieldResults)
    externalWorkspacePathReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "externalWorkspacePathReady" -DefaultValue $false)
    externalWorkspacePathValidationReasons = @(Get-PropertyOrDefault -Object $candidate -Name "externalWorkspacePathValidationReasons" -DefaultValue @())
    packageRestoreSourceUrlHost = [string](Get-PropertyOrDefault -Object $candidate -Name "packageRestoreSourceUrlHost" -DefaultValue "")
    rejectsLocalFeedProjectReferenceAndDirectNupkg = [bool](Get-PropertyOrDefault -Object $candidate -Name "rejectsLocalFeedProjectReferenceAndDirectNupkg" -DefaultValue $false)
    yoloVisionTaskReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "yoloVisionTaskReady" -DefaultValue $false)
    yoloVisionHashFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "yoloVisionHashFieldCount" -DefaultValue 0)
    realModelExecutionConfirmationReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "realModelExecutionConfirmationReady" -DefaultValue $false)
    rejectsReadinessTutorialMatrixArtifacts = [bool](Get-PropertyOrDefault -Object $candidate -Name "rejectsReadinessTutorialMatrixArtifacts" -DefaultValue $false)
    supportsMultipleArticleProofRecords = [bool](Get-PropertyOrDefault -Object $candidate -Name "supportsMultipleArticleProofRecords" -DefaultValue $false)
    minimumArticleProofRecordCount = [int](Get-PropertyOrDefault -Object $candidate -Name "minimumArticleProofRecordCount" -DefaultValue 0)
    articleProofRecordCount = [int](Get-PropertyOrDefault -Object $candidate -Name "articleProofRecordCount" -DefaultValue 0)
    articleProofReadyRecordCount = [int](Get-PropertyOrDefault -Object $candidate -Name "articleProofReadyRecordCount" -DefaultValue 0)
    articleProofRecordsReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "articleProofRecordsReady" -DefaultValue $true)
    articleProofRecordResults = @(Get-PropertyOrDefault -Object $candidate -Name "articleProofRecordResults" -DefaultValue @())
    ownerCloseDecisionReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "ownerCloseDecisionReady" -DefaultValue $false)
    ownerCloseDecisionValue = [string](Get-PropertyOrDefault -Object $candidate -Name "ownerCloseDecisionValue" -DefaultValue "")
    manualCloseReviewConfirmationReady = [bool](Get-PropertyOrDefault -Object $candidate -Name "manualCloseReviewConfirmationReady" -DefaultValue $false)
    manualCloseReviewOnly = [bool](Get-PropertyOrDefault -Object $candidate -Name "manualCloseReviewOnly" -DefaultValue $true)
    candidateValidationFailedBlockerCount = $candidateValidationFailedBlockerCount
    dependencyRequiredCount = $dependencyRequiredCount
    dependencyAcceptedCount = $dependencyAcceptedCount
    dependencyResults = @($dependencyResults)
    blockedReasonCount = $blockedReasons.Count
    blockedReasons = @($blockedReasons.ToArray())
    ownerActionRequired = -not $ownerEvidenceAccepted
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "$($spec.title) is a strict admission validator over Owner-supplied evidence fields only. It does not download packages, run a clean consumer, run YoloVision, publish articles, execute dotnet nuget push, approve public release, close the release issue, or become proof by itself; it is not runtime proof, not post-publish proof, not release close approval, and not package push."
  }

  $jsonPath = Join-Path $OutputRoot "$($spec.fileStem).json"
  $mdPath = Join-Path $OutputRoot "$($spec.fileStem).md"
  Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)
  Write-Utf8File -LiteralPath $mdPath -InputObject @(
    "# $($spec.title)",
    "",
    "- validatorState: ``$($record.validatorState)``",
    "- realOwnerInputPresent: ``$($record.realOwnerInputPresent)``",
    "- ownerEvidenceAccepted: ``$($record.ownerEvidenceAccepted)``",
    "- proofReady: ``False``",
    "- proofPromotionReady: ``False``",
    "- readyFieldCount: ``$readyFieldCount/$($fieldResults.Count)``",
    "- blockedReasonCount: ``$($record.blockedReasonCount)``",
    "- canCloseReleaseIssue: ``False``",
    "",
    $record.boundary
  )

  return $record
}

function Test-OwnerPostPublishProofValidatorArtifact {
  param(
    [object]$Record,
    [string]$RecordKind,
    [string]$ExpectedBlockedState,
    [string]$RequiredBoundaryText
  )

  $items = New-Object System.Collections.Generic.List[object]
  $items.Add((New-OwnerValidationItem "record-kind" ([string]$Record.recordKind -eq $RecordKind) "blocker" "Validator recordKind must match.")) | Out-Null
  $items.Add((New-OwnerValidationItem "validator-state" ([string]$Record.validatorState -in @($ExpectedBlockedState, [string](Get-OwnerPostPublishProofValidatorSpec -RecordKind $RecordKind).acceptedState)) "blocker" "Validator state must be explicit blocked or accepted state.")) | Out-Null
  $items.Add((New-OwnerValidationItem "field-coverage" ([int]$Record.requiredFieldCount -gt 0) "blocker" "Validator must carry lane field results.")) | Out-Null
  $items.Add((New-OwnerValidationItem "blocked-without-owner-input" (([bool]$Record.realOwnerInputPresent) -or ((-not [bool]$Record.ownerEvidenceAccepted) -and [int]$Record.blockedReasonCount -gt 0)) "blocker" "Validator must remain blocked when real Owner input is absent.")) | Out-Null
  $items.Add((New-OwnerValidationItem "no-proof-promotion-flags" (Assert-OwnerPostPublishFalseFlags -Record $Record) "blocker" "Validator must not publish, close, or promote proof by itself.")) | Out-Null
  $items.Add((New-OwnerValidationItem "proof-ready-false" ((-not [bool](Get-PropertyOrDefault -Object $Record -Name "proofReady" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $Record -Name "proofPromotionReady" -DefaultValue $true))) "blocker" "Validator must not mark itself proofReady or proofPromotionReady.")) | Out-Null
  $items.Add((New-OwnerValidationItem "boundary" ([string]$Record.boundary -like "*$RequiredBoundaryText*" -and [string]$Record.boundary -like "*not package push*") "blocker" "Validator boundary must explicitly reject proof substitution.")) | Out-Null
  return @($items.ToArray())
}

function Get-OwnerPostPublishProofValidatorValidationResults {
  param([string]$OutputRoot)

  foreach ($spec in Get-OwnerPostPublishProofValidatorSpecs) {
    $recordPath = Join-Path $OutputRoot "$($spec.fileStem).json"
    $validationPath = Join-Path $OutputRoot "$($spec.fileStem)-validation.json"
    $record = $null
    $validation = $null
    if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
      $record = Get-Content -LiteralPath $recordPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    if (Test-Path -LiteralPath $validationPath -PathType Leaf) {
      $validation = Get-Content -LiteralPath $validationPath -Raw -Encoding utf8 | ConvertFrom-Json
    }

    [pscustomobject]@{
      recordKind = [string]$spec.recordKind
      proofKind = [string]$spec.proofKind
      laneId = [string]$spec.laneId
      isFinalBridge = [bool]$spec.isFinalBridge
      recordPath = $recordPath
      validationPath = $validationPath
      recordPresent = $null -ne $record
      validationPresent = $null -ne $validation
      validatorState = [string](Get-PropertyOrDefault -Object $record -Name "validatorState" -DefaultValue "missing-validator-record")
      validationState = [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "missing-validator-validation")
      ownerEvidenceAccepted = [bool](Get-PropertyOrDefault -Object $validation -Name "ownerEvidenceAccepted" -DefaultValue $false)
      requiredFieldCount = [int](Get-PropertyOrDefault -Object $validation -Name "requiredFieldCount" -DefaultValue 0)
      readyFieldCount = [int](Get-PropertyOrDefault -Object $validation -Name "readyFieldCount" -DefaultValue 0)
      blockedFieldCount = [int](Get-PropertyOrDefault -Object $validation -Name "blockedFieldCount" -DefaultValue 0)
      blockedReasonCount = [int](Get-PropertyOrDefault -Object $validation -Name "blockedReasonCount" -DefaultValue 0)
      failedBlockerCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedBlockerCount" -DefaultValue 999)
    }
  }
}

function Export-OwnerPostPublishProofAcceptanceManifestArtifact {
  param(
    [string]$OutputRoot,
    [string]$RepositoryRoot
  )

  $results = @(Get-OwnerPostPublishProofValidatorValidationResults -OutputRoot $OutputRoot)
  $validatorCount = $results.Count
  $acceptedValidatorCount = @($results | Where-Object { [bool]$_.ownerEvidenceAccepted -and [int]$_.failedBlockerCount -eq 0 }).Count
  $blockedValidatorCount = $validatorCount - $acceptedValidatorCount
  $missingValidationCount = @($results | Where-Object { -not [bool]$_.validationPresent }).Count
  $failedBlockerCount = @($results | Where-Object { [int]$_.failedBlockerCount -gt 0 }).Count
  $allValidatorsAccepted = $validatorCount -gt 0 -and $acceptedValidatorCount -eq $validatorCount -and $failedBlockerCount -eq 0

  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if ($missingValidationCount -gt 0) { $blockedReasons.Add("validator-validation-missing=$missingValidationCount") | Out-Null }
  if ($blockedValidatorCount -gt 0) { $blockedReasons.Add("validator-owner-evidence-not-accepted=$blockedValidatorCount/$validatorCount") | Out-Null }
  if ($failedBlockerCount -gt 0) { $blockedReasons.Add("validator-failed-blockers=$failedBlockerCount") | Out-Null }
  if (-not $allValidatorsAccepted) { $blockedReasons.Add("real-owner-post-publish-evidence-incomplete") | Out-Null }

  $manifestState = if ($allValidatorsAccepted) { "owner-post-publish-proof-acceptance-ready-for-manual-close-review" } else { "blocked-owner-post-publish-proof-acceptance-real-evidence-required" }
  $record = [pscustomobject]@{
    recordKind = "owner-post-publish-proof-acceptance-manifest"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    manifestState = $manifestState
    validatorCount = $validatorCount
    acceptedValidatorCount = $acceptedValidatorCount
    blockedValidatorCount = $blockedValidatorCount
    missingValidationCount = $missingValidationCount
    validatorFailedBlockerCount = $failedBlockerCount
    allValidatorsAccepted = $allValidatorsAccepted
    readyForManualReleaseCloseReview = $allValidatorsAccepted
    releaseCloseReady = $false
    closeIssueCommandReady = $false
    validatorResults = @($results)
    blockedReasonCount = $blockedReasons.Count
    blockedReasons = @($blockedReasons.ToArray())
    ownerActionRequired = -not $allValidatorsAccepted
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    sourceArtifacts = @($results | ForEach-Object { "artifacts/final-release/$([System.IO.Path]::GetFileName($_.validationPath))" })
    boundary = "Owner post-publish proof acceptance manifest aggregates strict validators only. It does not download packages, run external consumers, run YoloVision, publish articles, execute dotnet nuget push, approve public release, close the release issue, or become proof by itself; it is not runtime proof, not post-publish proof, not release close approval, and not package push."
  }

  $jsonPath = Join-Path $OutputRoot "owner-post-publish-proof-acceptance-manifest.json"
  $mdPath = Join-Path $OutputRoot "owner-post-publish-proof-acceptance-manifest.md"
  Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)
  Write-Utf8File -LiteralPath $mdPath -InputObject @(
    "# Owner Post-Publish Proof Acceptance Manifest",
    "",
    "- manifestState: ``$($record.manifestState)``",
    "- validators: ``$acceptedValidatorCount/$validatorCount``",
    "- blockedValidatorCount: ``$blockedValidatorCount``",
    "- readyForManualReleaseCloseReview: ``$($record.readyForManualReleaseCloseReview)``",
    "- releaseCloseReady: ``False``",
    "- canCloseReleaseIssue: ``False``",
    "",
    $record.boundary
  )

  return $record
}

function Test-OwnerPostPublishProofAcceptanceManifestArtifact {
  param([object]$Record)

  $items = New-Object System.Collections.Generic.List[object]
  $items.Add((New-OwnerValidationItem "record-kind" ([string]$Record.recordKind -eq "owner-post-publish-proof-acceptance-manifest") "blocker" "Acceptance manifest recordKind must match.")) | Out-Null
  $items.Add((New-OwnerValidationItem "validator-count" ([int]$Record.validatorCount -ge 5) "blocker" "Acceptance manifest must inspect all Owner post-publish proof validators.")) | Out-Null
  $items.Add((New-OwnerValidationItem "blocked-or-accepted-state" ([string]$Record.manifestState -in @("blocked-owner-post-publish-proof-acceptance-real-evidence-required", "owner-post-publish-proof-acceptance-ready-for-manual-close-review")) "blocker" "Acceptance manifest state must be explicit.")) | Out-Null
  $items.Add((New-OwnerValidationItem "no-auto-close" ((-not [bool](Get-PropertyOrDefault -Object $Record -Name "releaseCloseReady" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $Record -Name "closeIssueCommandReady" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $true))) "blocker" "Acceptance manifest must not close the release issue automatically.")) | Out-Null
  $items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $Record) "blocker" "Acceptance manifest must not publish, close, or promote proof by itself.")) | Out-Null
  $items.Add((New-OwnerValidationItem "boundary" ([string]$Record.boundary -like "*not post-publish proof*" -and [string]$Record.boundary -like "*not package push*") "blocker" "Acceptance manifest boundary must explicitly reject proof substitution.")) | Out-Null
  if (-not [bool](Get-PropertyOrDefault -Object $Record -Name "allValidatorsAccepted" -DefaultValue $false)) {
    $items.Add((New-OwnerValidationItem "blocked-reasons" ([int](Get-PropertyOrDefault -Object $Record -Name "blockedReasonCount" -DefaultValue 0) -gt 0) "blocker" "Blocked acceptance manifest must include blocked reasons.")) | Out-Null
  }

  return @($items.ToArray())
}

function Get-OwnerPostPublishImportFieldResult {
  param([object]$ImportRecord, [string]$LaneId, [string]$FieldName)
  $analysis = Get-PropertyOrDefault -Object $ImportRecord -Name "analysis" -DefaultValue $null
  foreach ($field in @(Convert-ToArray (Get-PropertyOrDefault -Object $analysis -Name "fieldResults" -DefaultValue @()))) {
    if ([string]$field.laneId -eq $LaneId -and [string]$field.fieldName -eq $FieldName) {
      return $field
    }
  }

  return $null
}

function Get-OwnerPostPublishImportFieldValue {
  param([object]$ImportRecord, [string]$LaneId, [string]$FieldName)
  $result = Get-OwnerPostPublishImportFieldResult -ImportRecord $ImportRecord -LaneId $LaneId -FieldName $FieldName
  return [string](Get-PropertyOrDefault -Object $result -Name "suppliedValue" -DefaultValue "")
}

function Get-FileSha256Text {
  param([string]$Path)
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
      $hash = $sha.ComputeHash($stream)
      return -join ($hash | ForEach-Object { $_.ToString("x2") })
    }
    finally {
      $stream.Dispose()
    }
  }
  finally {
    $sha.Dispose()
  }
}

