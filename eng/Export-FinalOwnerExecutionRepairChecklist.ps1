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

function Get-RequiredFileFields {
  param([string]$StepId, [string[]]$RequiredInputs)

  $fields = @("stdoutPath", "stderrPath", "mergedTranscriptPath", "validatorOutputPath")
  foreach ($input in $RequiredInputs) {
    if ($input -match "path|log|output|transcript|nupkg|package|artifact|file") {
      $normalized = ([string]$input).Trim()
      if (-not [string]::IsNullOrWhiteSpace($normalized)) {
        $fields += $normalized
      }
    }
  }

  switch ($StepId) {
    { $_ -match "real-model|runtime|consumer|post-publish|import|bridge" } {
      $fields += @("runtimeLogPath", "proofRecordPath")
    }
  }

  return @($fields | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
}

function Get-RequiredSha256Fields {
  param([string]$StepId, [string[]]$RequiredInputs)

  $fields = @("stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "validatorOutputSha256")
  foreach ($input in $RequiredInputs) {
    if ($input -match "sha256|hash|digest") {
      $fields += ([string]$input).Trim()
    }
  }

  switch ($StepId) {
    { $_ -match "real-model|runtime|consumer|post-publish|import|bridge" } {
      $fields += @("packageSha256", "runtimeLogSha256", "proofRecordSha256")
    }
  }

  return @($fields | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
}

function Get-RequiredIdentityFields {
  param([string]$StepId, [string[]]$RequiredInputs)

  $fields = @("exitCode", "hostIdentity", "packageIdentity", "ownerReviewer", "ownerReviewTimestampUtc")
  foreach ($input in $RequiredInputs) {
    if ($input -match "identity|host|package|owner|review|exitCode|version|source") {
      $fields += ([string]$input).Trim()
    }
  }

  switch ($StepId) {
    "03-post-publish-verification-public-channel" {
      $fields += @("publicPackageSourceUrl", "publishedPackageUrl", "downloadedNupkgSha256", "cleanExternalConsumerRoot")
    }
    "02-package-consumer-runtime-clean-external-proof" {
      $fields += @("publicOrOwnerApprovedPackageSource", "cleanExternalConsumerRoot", "installedPackageIdentity")
    }
  }

  return @($fields | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
}

function New-RepairItem {
  param([object]$Step)

  $id = [string](Get-PropertyOrDefault -Object $Step -Name "id" -DefaultValue "")
  $requiredInputs = Convert-ToStringArray (Get-PropertyOrDefault -Object $Step -Name "requiredInputs" -DefaultValue @())
  $ownerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Step -Name "ownerCommands" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $Step -Name "validatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $Step -Name "expectedResultArtifacts" -DefaultValue @())
  $fileFields = Get-RequiredFileFields -StepId $id -RequiredInputs $requiredInputs
  $shaFields = Get-RequiredSha256Fields -StepId $id -RequiredInputs $requiredInputs
  $identityFields = Get-RequiredIdentityFields -StepId $id -RequiredInputs $requiredInputs

  [pscustomobject]@{
    id = "repair-$id"
    order = [int](Get-PropertyOrDefault -Object $Step -Name "order" -DefaultValue 0)
    laneId = [string](Get-PropertyOrDefault -Object $Step -Name "laneId" -DefaultValue "")
    actionRequiredId = [string](Get-PropertyOrDefault -Object $Step -Name "actionRequiredId" -DefaultValue "")
    repairState = "blocked-real-owner-evidence-required"
    sourceExecutionStepId = $id
    requiredInputs = @($requiredInputs)
    requiredFileFields = @($fileFields)
    requiredSha256Fields = @($shaFields)
    requiredIdentityFields = @($identityFields)
    requiredNonSubstituteConfirmations = @(
      "not local feed",
      "not ProjectReference",
      "not direct .nupkg",
      "not template",
      "not draft",
      "not dry-run",
      "not dashboard",
      "not candidate",
      "not build-only",
      "not dependency-probe-only",
      "not blocked-by-cuda-driver",
      "not runbook-only"
    )
    ownerCommands = @($ownerCommands)
    validatorCommands = @($validatorCommands)
    expectedResultArtifacts = @($expectedArtifacts)
    firstOwnerAction = if (@($ownerCommands).Count -gt 0) { [string]$ownerCommands[0] } else { "Collect real owner evidence for $id, then run the strict validator." }
    cannotUseMarkers = @(
      "local feed",
      "ProjectReference",
      "direct .nupkg",
      "template",
      "draft",
      "dry-run",
      "dashboard",
      "candidate",
      "build-only",
      "dependency-probe-only",
      "blocked-by-cuda-driver"
    )
    boundary = "Repair guidance only. This item points the Owner to missing real files, SHA256 hashes, exitCode=0, host identity, package identity, owner reviewer, and non-substitute confirmations; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$executionPackage = Read-JsonOrNull "artifacts\final-release\final-owner-execution-package.json"
$executionPackageValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-package-validation.json"
$worklist = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist.json"
$finalGate = Read-JsonOrNull "artifacts\final-release\final-publish-proof-gate-report.json"

$executionSteps = @()
if ($null -ne $executionPackage) {
  $executionSteps = @((Get-PropertyOrDefault -Object $executionPackage -Name "executionSteps" -DefaultValue @()))
}

$repairItems = @($executionSteps | ForEach-Object { New-RepairItem -Step $_ } | Sort-Object order)
$requiredFileFieldCount = @($repairItems | ForEach-Object { $_.requiredFileFields } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count
$requiredSha256FieldCount = @($repairItems | ForEach-Object { $_.requiredSha256Fields } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count
$requiredIdentityFieldCount = @($repairItems | ForEach-Object { $_.requiredIdentityFields } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count
$requiredNonSubstituteConfirmationCount = @($repairItems | ForEach-Object { $_.requiredNonSubstituteConfirmations } | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count

$record = [ordered]@{
  recordKind = "final-owner-execution-repair-checklist"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  checklistState = "blocked-final-owner-execution-repair-real-owner-evidence-required"
  sourceExecutionPackageState = [string](Get-PropertyOrDefault -Object $executionPackage -Name "packageState" -DefaultValue "missing-final-owner-execution-package")
  sourceExecutionPackageValidationState = [string](Get-PropertyOrDefault -Object $executionPackageValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-package-validation")
  sourceWorklistState = [string](Get-PropertyOrDefault -Object $worklist -Name "worklistState" -DefaultValue "missing-final-owner-proof-action-worklist")
  finalPublishProofGateState = [string](Get-PropertyOrDefault -Object $finalGate -Name "validationState" -DefaultValue "missing-final-publish-proof-gate-report")
  executionStepCount = $executionSteps.Count
  repairItemCount = $repairItems.Count
  blockedRepairItemCount = $repairItems.Count
  readyRepairItemCount = 0
  requiredFileFieldCount = $requiredFileFieldCount
  requiredSha256FieldCount = $requiredSha256FieldCount
  requiredIdentityFieldCount = $requiredIdentityFieldCount
  requiredNonSubstituteConfirmationCount = $requiredNonSubstituteConfirmationCount
  repairItems = @($repairItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isPackagePush = $false
  sourceArtifacts = @(
    "artifacts/final-release/final-owner-execution-package.json",
    "artifacts/final-release/final-owner-execution-package.md",
    "artifacts/final-release/final-owner-execution-package-validation.json",
    "artifacts/final-release/final-owner-execution-package-validation.md",
    "artifacts/final-release/final-owner-proof-action-worklist.json",
    "artifacts/final-release/final-owner-proof-action-worklist.md",
    "artifacts/final-release/final-publish-proof-gate-report.json"
  )
  boundary = "Final owner execution repair checklist is owner guidance only. It enumerates missing real evidence fields for each execution step; it does not publish, does not close the release, is not runtime proof, is not post-publish proof, is not publish approval, is not release close approval, and is not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-repair-checklist.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-repair-checklist.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $repairItems) {
  "| ``$($item.order)`` | ``$(ConvertTo-MarkdownCell $item.sourceExecutionStepId)`` | ``$(ConvertTo-MarkdownCell $item.actionRequiredId)`` | ``$(@($item.requiredFileFields).Count)`` | ``$(@($item.requiredSha256Fields).Count)`` | ``$(@($item.requiredIdentityFields).Count)`` | ``$(ConvertTo-MarkdownCell $item.firstOwnerAction)`` |"
}

$markdown = @(
  "# Final Owner Execution Repair Checklist",
  "",
  "- checklistState: ``$($record.checklistState)``",
  "- executionStepCount: ``$($record.executionStepCount)``",
  "- repairItemCount: ``$($record.repairItemCount)``",
  "- blockedRepairItemCount: ``$($record.blockedRepairItemCount)``",
  "- readyRepairItemCount: ``0``",
  "- requiredFileFieldCount: ``$requiredFileFieldCount``",
  "- requiredSha256FieldCount: ``$requiredSha256FieldCount``",
  "- requiredIdentityFieldCount: ``$requiredIdentityFieldCount``",
  "- requiredNonSubstituteConfirmationCount: ``$requiredNonSubstituteConfirmationCount``",
  "- boundary: $($record.boundary)",
  "",
  "> 本清单只用于 Owner 回填修复，不执行公开发布，不把模板、候选、dashboard、local feed、ProjectReference 或 direct .nupkg 当 proof。",
  "",
  "| Order | Source Step | Action Required | File Fields | SHA256 Fields | Identity Fields | First Owner Action |",
  "|---:|---|---|---:|---:|---:|---|",
  @($rows),
  "",
  "## Cannot Use",
  "",
  "- local feed",
  "- ProjectReference",
  "- direct .nupkg",
  "- template / draft / dry-run / dashboard / candidate",
  "- build-only / dependency-probe-only / blocked-by-cuda-driver"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
