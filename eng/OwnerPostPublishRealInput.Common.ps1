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
    "pre-publish smoke"
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
      New-OwnerPostPublishFieldSpec "nugetManagedPackageId" "string" "NuGet small bridge/core package id."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageVersion" "version" "NuGet small bridge/core package version."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageUrl" "url" "Public NuGet package URL."
      New-OwnerPostPublishFieldSpec "nugetManagedPackageSha256" "sha256" "Downloaded NuGet package SHA256."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageId" "string" "GitHub Packages full runtime package id."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageVersion" "version" "GitHub Packages full runtime package version."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageUrl" "url" "GitHub Packages runtime package URL."
      New-OwnerPostPublishFieldSpec "githubRuntimePackageSha256" "sha256" "Downloaded GitHub runtime package SHA256."
      New-OwnerPostPublishFieldSpec "downloadedAtUtc" "utc" "Owner download timestamp in UTC."
      New-OwnerPostPublishFieldSpec "downloadTranscriptSha256" "sha256" "Transcript hash for public package download."
    )
    New-OwnerPostPublishLaneSpec -Id "external-clean-consumer-logs" -Title "External clean consumer logs" -Fields @(
      New-OwnerPostPublishFieldSpec "externalWorkspaceRoot" "external-path" "Repository-external clean consumer workspace."
      New-OwnerPostPublishFieldSpec "restoreLogSha256" "sha256" "dotnet restore transcript SHA256."
      New-OwnerPostPublishFieldSpec "buildLogSha256" "sha256" "dotnet build transcript SHA256."
      New-OwnerPostPublishFieldSpec "smokeLogSha256" "sha256" "runtime smoke transcript SHA256."
      New-OwnerPostPublishFieldSpec "hostMetadataSha256" "sha256" "Host metadata JSON SHA256."
      New-OwnerPostPublishFieldSpec "noLocalSubstituteConfirmation" "confirmation" "Owner confirms no local feed/ProjectReference/direct nupkg substitute."
    )
    New-OwnerPostPublishLaneSpec -Id "yolovision-real-model-assets" -Title "YoloVision real model assets" -Fields @(
      New-OwnerPostPublishFieldSpec "modelSourceUrl" "url" "Real model source URL."
      New-OwnerPostPublishFieldSpec "modelSha256" "sha256" "Real model file SHA256."
      New-OwnerPostPublishFieldSpec "labelsSha256" "sha256" "Labels file SHA256."
      New-OwnerPostPublishFieldSpec "inputImageSha256" "sha256" "Input image or tensor SHA256."
      New-OwnerPostPublishFieldSpec "outputJsonSha256" "sha256" "YoloVision output JSON SHA256."
      New-OwnerPostPublishFieldSpec "stdoutLogSha256" "sha256" "YoloVision stdout log SHA256."
      New-OwnerPostPublishFieldSpec "stderrLogSha256" "sha256" "YoloVision stderr log SHA256."
      New-OwnerPostPublishFieldSpec "hostMetadataSha256" "sha256" "Host metadata JSON SHA256."
      New-OwnerPostPublishFieldSpec "ownerReviewedAtUtc" "utc" "Owner review timestamp in UTC."
    )
    New-OwnerPostPublishLaneSpec -Id "article-publication-urls" -Title "Article publication URLs" -Fields @(
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

    [pscustomobject]@{
      id = $lane.id
      title = $lane.title
      laneState = "blocked-owner-real-input-required"
      requiredFieldCount = @($fields).Count
      fields = @($fields)
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

function Test-OwnerPostPublishInputPathForbidden {
  param([string]$InputPath)
  $leaf = [System.IO.Path]::GetFileName($InputPath)
  foreach ($suffix in @(".template.json", ".example.json", ".draft.json", ".misuse.json", ".ready.json", "-validation.json")) {
    if ($leaf.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
  }

  return $false
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
    switch ([string]$spec.kind) {
      "sha256" { $formatValid = Test-Sha256Text -Value $valueText }
      "url" { $formatValid = Test-OwnerPostPublishUrl -Value $valueText }
      "utc" { $formatValid = Test-OwnerPostPublishUtc -Value $valueText }
      "confirmation" { $formatValid = Test-OwnerPostPublishConfirmation -Value $valueText }
      "enum" { $formatValid = Test-OwnerPostPublishEnum -Value $valueText }
      "external-path" {
        $formatValid = -not $placeholder -and -not [string]::IsNullOrWhiteSpace($valueText)
        if ($formatValid -and -not [System.IO.Path]::IsPathRooted($valueText)) {
          $formatValid = $false
        }
        if ($formatValid) {
          $root = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
          $candidate = [System.IO.Path]::GetFullPath($valueText).TrimEnd('\', '/')
          if ($candidate.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $formatValid = $false }
        }
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
        forbiddenSubstituteHits = @($forbiddenHits)
        fieldReady = $ready
      }) | Out-Null
  }

  $fields = @($fieldResults.ToArray())
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
    allFieldsReady = ($fields.Count -gt 0 -and @($fields | Where-Object { -not [bool]$_.fieldReady }).Count -eq 0)
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
  $candidateReady = ($fieldResults.Count -gt 0 -and $blockedFieldCount -eq 0 -and -not [bool](Get-PropertyOrDefault -Object $analysis -Name "inputPathForbidden" -DefaultValue $true))

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
    ownerActionRequired = $blockedFieldCount -gt 0
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
  $items.Add((New-OwnerValidationItem "field-coverage" ([int]$Record.requiredFieldCount -gt 0 -and [int]$Record.blockedFieldCount -gt 0) "blocker" "Candidate must carry lane field results and blocked fields.")) | Out-Null
  $items.Add((New-OwnerValidationItem "non-proof-flags" (Assert-OwnerPostPublishFalseFlags -Record $Record) "blocker" "Candidate must not publish, close, or promote proof.")) | Out-Null
  $items.Add((New-OwnerValidationItem "boundary" ([string]$Record.boundary -like "*$RequiredBoundaryText*" -and [string]$Record.boundary -like "*not package push*") "blocker" "Candidate boundary must explicitly reject proof substitution.")) | Out-Null
  return @($items.ToArray())
}

