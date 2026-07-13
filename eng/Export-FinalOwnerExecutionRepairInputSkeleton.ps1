[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-InputSlot {
  param([string]$Name, [string]$Kind, [string]$Description)

  [pscustomobject]@{
    name = $Name
    kind = $Kind
    required = $true
    placeholder = "<owner-real-$Kind-required>"
    description = $Description
  }
}

function New-SkeletonItem {
  param([object]$RepairItem)

  $sourceRepairItemId = [string](Get-PropertyOrDefault -Object $RepairItem -Name "id" -DefaultValue "")
  $sourceExecutionStepId = [string](Get-PropertyOrDefault -Object $RepairItem -Name "sourceExecutionStepId" -DefaultValue "")
  $fileFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "requiredFileFields" -DefaultValue @())
  $shaFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "requiredSha256Fields" -DefaultValue @())
  $identityFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "requiredIdentityFields" -DefaultValue @())
  $nonSubstituteConfirmations = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "requiredNonSubstituteConfirmations" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "validatorCommands" -DefaultValue @())
  $expectedResultArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "expectedResultArtifacts" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $RepairItem -Name "cannotUseMarkers" -DefaultValue @())

  $fileInputs = @($fileFields | ForEach-Object { New-InputSlot -Name $_ -Kind "file-path" -Description "Owner must provide an existing evidence file under an allowed evidence root." })
  $sha256Inputs = @($shaFields | ForEach-Object { New-InputSlot -Name $_ -Kind "sha256" -Description "Owner must provide the exact 64-character SHA256 for the matching file input." })
  $identityInputs = @($identityFields | ForEach-Object { New-InputSlot -Name $_ -Kind "identity" -Description "Owner must provide real host, package, reviewer, timestamp, source, or exit-code metadata." })

  [pscustomobject]@{
    sourceRepairItemId = $sourceRepairItemId
    sourceExecutionStepId = $sourceExecutionStepId
    laneId = [string](Get-PropertyOrDefault -Object $RepairItem -Name "laneId" -DefaultValue "")
    actionRequiredId = [string](Get-PropertyOrDefault -Object $RepairItem -Name "actionRequiredId" -DefaultValue "")
    inputState = "owner-real-evidence-required"
    ownerMustFill = @(
      "Replace every placeholder with real external evidence values.",
      "Provide existing stdout/stderr/transcript/validator output files where requested.",
      "Provide matching SHA256 values for each supplied file.",
      "Provide exitCode=0 only when the external command actually succeeded.",
      "Provide real host identity, package identity, owner reviewer, and review timestamp.",
      "Confirm the evidence is not local feed, ProjectReference, direct .nupkg, template, draft, dry-run, dashboard, candidate, build-only, dependency-probe-only, or blocked-by-cuda-driver."
    )
    fileInputs = @($fileInputs)
    sha256Inputs = @($sha256Inputs)
    identityInputs = @($identityInputs)
    nonSubstituteConfirmations = @($nonSubstituteConfirmations)
    expectedValidatorCommands = @($validatorCommands)
    expectedResultArtifacts = @($expectedResultArtifacts)
    evidenceRootPolicy = [pscustomobject]@{
      mustExist = $true
      mustStayUnderAllowedEvidenceRoot = $true
      defaultAllowedEvidenceRoots = @(
        "artifacts/final-release/owner-real-inputs",
        "artifacts/final-release/owner-external-results",
        "artifacts/user-acceptance",
        "artifacts/package-consumer"
      )
      pathTraversalAllowed = $false
      absolutePathsAllowedOnlyWhenExplicitlyApproved = $true
    }
    cannotUseMarkers = @($cannotUseMarkers)
    importPreflight = [pscustomobject]@{
      importer = "eng/Import-OwnerExternalProofExecutionResult.ps1"
      validator = "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict"
      requiresExistingFiles = $true
      requiresSha256Match = $true
      requiresExitCodeZero = $true
      requiresOwnerReview = $true
      requiresNonSubstituteConfirmations = $true
    }
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner input skeleton only. It is a fillable contract for real evidence import; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$repairChecklist = Read-JsonOrNull "artifacts\final-release\final-owner-execution-repair-checklist.json"
$repairChecklistValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-repair-checklist-validation.json"
$executionPackage = Read-JsonOrNull "artifacts\final-release\final-owner-execution-package.json"

$repairItems = @()
if ($null -ne $repairChecklist) {
  $repairItems = @((Get-PropertyOrDefault -Object $repairChecklist -Name "repairItems" -DefaultValue @()))
}

$skeletonItems = @($repairItems | ForEach-Object { New-SkeletonItem -RepairItem $_ } | Sort-Object sourceExecutionStepId)
$fileInputCount = @($skeletonItems | ForEach-Object { $_.fileInputs } | Where-Object { $null -ne $_ }).Count
$sha256InputCount = @($skeletonItems | ForEach-Object { $_.sha256Inputs } | Where-Object { $null -ne $_ }).Count
$identityInputCount = @($skeletonItems | ForEach-Object { $_.identityInputs } | Where-Object { $null -ne $_ }).Count
$nonSubstituteConfirmationCount = @($skeletonItems | ForEach-Object { $_.nonSubstituteConfirmations } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count

$record = [ordered]@{
  recordKind = "final-owner-execution-repair-input-skeleton"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  skeletonState = "blocked-owner-real-evidence-input-required"
  sourceRepairChecklistState = [string](Get-PropertyOrDefault -Object $repairChecklist -Name "checklistState" -DefaultValue "missing-final-owner-execution-repair-checklist")
  sourceRepairChecklistValidationState = [string](Get-PropertyOrDefault -Object $repairChecklistValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-repair-checklist-validation")
  sourceExecutionPackageState = [string](Get-PropertyOrDefault -Object $executionPackage -Name "packageState" -DefaultValue "missing-final-owner-execution-package")
  skeletonItemCount = $skeletonItems.Count
  blockedSkeletonItemCount = $skeletonItems.Count
  readySkeletonItemCount = 0
  fileInputCount = $fileInputCount
  sha256InputCount = $sha256InputCount
  identityInputCount = $identityInputCount
  nonSubstituteConfirmationCount = $nonSubstituteConfirmationCount
  skeletonItems = @($skeletonItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-repair-checklist.json",
    "artifacts/final-release/final-owner-execution-repair-checklist.md",
    "artifacts/final-release/final-owner-execution-repair-checklist-validation.json",
    "artifacts/final-release/final-owner-execution-repair-checklist-validation.md",
    "artifacts/final-release/final-owner-execution-package.json",
    "artifacts/final-release/final-owner-execution-package-validation.json"
  )
  boundary = "Final owner execution repair input skeleton is input skeleton only. It prepares owner-filled real evidence for strict import; it does not publish, does not close the release, is not runtime proof, is not post-publish proof, is not publish approval, is not release close approval, and is not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-repair-input-skeleton.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-repair-input-skeleton.md"
$record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $skeletonItems) {
  "| ``$(ConvertTo-MarkdownCell $item.sourceRepairItemId)`` | ``$(ConvertTo-MarkdownCell $item.sourceExecutionStepId)`` | ``$(ConvertTo-MarkdownCell $item.laneId)`` | ``$(@($item.fileInputs).Count)`` | ``$(@($item.sha256Inputs).Count)`` | ``$(@($item.identityInputs).Count)`` | ``$(@($item.nonSubstituteConfirmations).Count)`` |"
}

$markdown = @(
  "# Final Owner Execution Repair Input Skeleton",
  "",
  "- skeletonState: ``$($record.skeletonState)``",
  "- skeletonItemCount: ``$($record.skeletonItemCount)``",
  "- blockedSkeletonItemCount: ``$($record.blockedSkeletonItemCount)``",
  "- readySkeletonItemCount: ``0``",
  "- fileInputCount: ``$fileInputCount``",
  "- sha256InputCount: ``$sha256InputCount``",
  "- identityInputCount: ``$identityInputCount``",
  "- nonSubstituteConfirmationCount: ``$nonSubstituteConfirmationCount``",
  "- boundary: $($record.boundary)",
  "",
  "> This is input skeleton only. It cannot substitute real Owner evidence, external command logs, matching SHA256 values, package identity, host identity, or post-publish verification.",
  "",
  "| Repair Item | Execution Step | Lane | File Inputs | SHA256 Inputs | Identity Inputs | Non-Substitute Confirmations |",
  "|---|---|---|---:|---:|---:|---:|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
