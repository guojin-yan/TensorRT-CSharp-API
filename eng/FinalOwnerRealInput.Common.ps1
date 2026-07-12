[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')

$script:FinalOwnerRealInputBoundary = 'not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push'
$script:FinalOwnerRealInputDefaultRoot = 'artifacts\final-release\owner-real-inputs'
$script:FinalOwnerRealInputGlobalForbiddenSubstitutes = @(
  'local feed',
  'local .nupkg',
  'ProjectReference',
  'direct nupkg',
  'direct .nupkg',
  'template',
  'draft',
  'dry-run',
  'dashboard',
  'runbook',
  'audit pack',
  'candidate',
  'local-only scan',
  'build-only',
  'dependency probe only',
  'blocked-by-driver',
  'Owner execution package',
  'real proof readiness gate'
)

function Get-FinalOwnerRealInputLaneSpecs {
  [CmdletBinding()]
  param()

  $lanes = Get-StrictCloseRealInputFinalBlockerLanes
  foreach ($lane in $lanes) {
    $baseName = [IO.Path]::GetFileNameWithoutExtension([string]$lane.ownerInputFile)
    [pscustomobject]@{
      laneId = [string]$lane.blockerId
      title = [string]$lane.title
      baseName = $baseName
      templateFileName = "$baseName.template.json"
      templateMarkdownFileName = "$baseName.template.md"
      validationFileName = "$baseName-validation.json"
      validationMarkdownFileName = "$baseName-validation.md"
      ownerInputFile = [string]$lane.ownerInputFile
      validator = [string]$lane.validator
      releaseCloseTarget = [string]$lane.releaseCloseTarget
      requiredEvidenceFields = @($lane.requiredEvidenceFields | ForEach-Object { [string]$_ } | Select-Object -Unique)
      forbiddenSubstitutes = @($script:FinalOwnerRealInputGlobalForbiddenSubstitutes + @($lane.forbiddenSubstitutes | ForEach-Object { [string]$_ }) | Select-Object -Unique)
      acceptanceRule = [string]$lane.acceptanceRule
    }
  }
}

function Resolve-FinalOwnerRealInputPath {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory)]
    [string]$Path
  )

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function New-FinalOwnerRealInputTemplateObject {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)]
    [object]$Lane
  )

  $placeholderFields = [ordered]@{}
  foreach ($field in $Lane.requiredEvidenceFields) {
    $placeholderFields[$field] = "<owner-fill-$field>"
  }

  [pscustomobject][ordered]@{
    recordKind = 'final-owner-real-input-template'
    templateKind = "$($Lane.laneId)-owner-input-template"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    laneId = $Lane.laneId
    title = $Lane.title
    ownerInputState = 'template-owner-real-input-required'
    realInputState = 'blocked-final-owner-real-input-required'
    isTemplate = $true
    realInputRequired = $true
    ownerInputFile = $Lane.ownerInputFile
    validator = $Lane.validator
    releaseCloseTarget = $Lane.releaseCloseTarget
    requiredEvidenceFields = @($Lane.requiredEvidenceFields)
    ownerInput = [pscustomobject]$placeholderFields
    forbiddenSubstitutes = @($Lane.forbiddenSubstitutes)
    nonSubstituteConfirmations = @($Lane.forbiddenSubstitutes | ForEach-Object {
      [pscustomobject]@{
        marker = $_
        confirmedAbsent = '<owner-fill-true-after-real-check>'
      }
    })
    acceptanceRule = $Lane.acceptanceRule
    performsPublish = $false
    approvesPublicRelease = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    isReleaseCloseRecordProof = $false
    boundary = $script:FinalOwnerRealInputBoundary
  }
}

function New-FinalOwnerRealInputTemplate {
  [CmdletBinding()]
  param(
    [string]$LaneId,
    [string]$OutputRoot = $script:FinalOwnerRealInputDefaultRoot,
    [string]$RepositoryRoot
  )

  if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  }

  $templateRoot = Resolve-FinalOwnerRealInputPath -RepositoryRoot $RepositoryRoot -Path $OutputRoot
  $finalReleaseRoot = Join-Path $RepositoryRoot 'artifacts\final-release'
  New-Item -ItemType Directory -Force -Path $templateRoot | Out-Null
  New-Item -ItemType Directory -Force -Path $finalReleaseRoot | Out-Null

  $lanes = @(Get-FinalOwnerRealInputLaneSpecs)
  if (-not [string]::IsNullOrWhiteSpace($LaneId)) {
    $lanes = @($lanes | Where-Object { $_.laneId -eq $LaneId })
    if ($lanes.Count -eq 0) {
      throw "Unknown final owner input lane: $LaneId"
    }
  }

  $writtenTemplates = New-Object System.Collections.Generic.List[object]
  foreach ($lane in $lanes) {
    $template = New-FinalOwnerRealInputTemplateObject -Lane $lane
    $jsonPath = Join-Path $templateRoot $lane.templateFileName
    $markdownPath = Join-Path $templateRoot $lane.templateMarkdownFileName
    $template | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

    $lines = @(
      "# $($lane.title) Owner Input Template",
      '',
      "- Lane: ``$($lane.laneId)``",
      "- State: ``template-owner-real-input-required``",
      "- Validator: ``$($lane.validator)``",
      "- ReleaseClose target: ``$($lane.releaseCloseTarget)``",
      "- Boundary: $script:FinalOwnerRealInputBoundary",
      '',
      '## Required Evidence Fields',
      ''
    )
    $lines += @($lane.requiredEvidenceFields | ForEach-Object { "- ``$_``" })
    $lines += @('', '## Forbidden Substitutes', '')
    $lines += @($lane.forbiddenSubstitutes | ForEach-Object { "- ``$_``" })
    $lines += @('', '## Acceptance Rule', '', $lane.acceptanceRule)
    $lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

    $writtenTemplates.Add([pscustomobject]@{
      laneId = $lane.laneId
      templatePath = $jsonPath
      templateMarkdownPath = $markdownPath
      validator = $lane.validator
      releaseCloseTarget = $lane.releaseCloseTarget
    }) | Out-Null
  }

  $allLaneIds = @((Get-FinalOwnerRealInputLaneSpecs) | ForEach-Object { $_.laneId })
  $pack = [pscustomobject][ordered]@{
    recordKind = 'final-owner-real-input-template-pack'
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    packState = 'blocked-final-owner-real-input-required'
    landingRoot = 'artifacts/final-release/owner-real-inputs'
    templateCount = $allLaneIds.Count
    writtenTemplateCount = $writtenTemplates.Count
    finalBlockerIds = @($allLaneIds)
    templates = @($writtenTemplates.ToArray())
    performsPublish = $false
    approvesPublicRelease = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = $script:FinalOwnerRealInputBoundary
  }

  $packJsonPath = Join-Path $finalReleaseRoot 'final-owner-real-input-template-pack.json'
  $packMarkdownPath = Join-Path $finalReleaseRoot 'final-owner-real-input-template-pack.md'
  $pack | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $packJsonPath -Encoding utf8

  $packLines = @(
    '# Final Owner Real Input Template Pack',
    '',
    "- State: ``$($pack.packState)``",
    "- Landing root: ``$($pack.landingRoot)``",
    "- Template count: ``$($pack.templateCount)``",
    "- Boundary: $script:FinalOwnerRealInputBoundary",
    '',
    '| Lane | Validator | ReleaseClose Target |',
    '| --- | --- | --- |'
  )
  foreach ($lane in Get-FinalOwnerRealInputLaneSpecs) {
    $packLines += "| ``$($lane.laneId)`` | ``$($lane.validator)`` | ``$($lane.releaseCloseTarget)`` |"
  }
  $packLines | Set-Content -LiteralPath $packMarkdownPath -Encoding utf8

  Write-Host "Final owner real input template pack written:"
  Write-Host "  Json=$packJsonPath"
  Write-Host "  Markdown=$packMarkdownPath"
}

function Test-FinalOwnerRealInputTemplate {
  [CmdletBinding()]
  param(
    [string]$LaneId,
    [string]$InputRoot = $script:FinalOwnerRealInputDefaultRoot,
    [string]$OutputRoot = 'artifacts\final-release',
    [string]$RepositoryRoot,
    [switch]$Strict
  )

  if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
  }

  $templateRoot = Resolve-FinalOwnerRealInputPath -RepositoryRoot $RepositoryRoot -Path $InputRoot
  $finalReleaseRoot = Resolve-FinalOwnerRealInputPath -RepositoryRoot $RepositoryRoot -Path $OutputRoot
  New-Item -ItemType Directory -Force -Path $templateRoot | Out-Null
  New-Item -ItemType Directory -Force -Path $finalReleaseRoot | Out-Null

  $isFullPackValidation = [string]::IsNullOrWhiteSpace($LaneId)
  $lanes = @(Get-FinalOwnerRealInputLaneSpecs)
  if (-not [string]::IsNullOrWhiteSpace($LaneId)) {
    $lanes = @($lanes | Where-Object { $_.laneId -eq $LaneId })
    if ($lanes.Count -eq 0) {
      throw "Unknown final owner input lane: $LaneId"
    }
  }

  if (-not (Test-Path -LiteralPath (Join-Path $finalReleaseRoot 'final-owner-real-input-template-pack.json') -PathType Leaf)) {
    New-FinalOwnerRealInputTemplate -OutputRoot $InputRoot -RepositoryRoot $RepositoryRoot
  }

  $requiredCommonFields = @('stdoutPath', 'stderrPath', 'transcriptPath', 'logPath', 'logSha256', 'exitCode', 'hostIdentity', 'ownerReviewer', 'nonSubstituteConfirmations')
  $laneValidations = New-Object System.Collections.Generic.List[object]
  $blockerFindings = New-Object System.Collections.Generic.List[object]
  $actionRequiredFindings = New-Object System.Collections.Generic.List[object]

  foreach ($lane in $lanes) {
    $templatePath = Join-Path $templateRoot $lane.templateFileName
    if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
      New-FinalOwnerRealInputTemplate -LaneId $lane.laneId -OutputRoot $InputRoot -RepositoryRoot $RepositoryRoot
    }

    $template = Get-Content -LiteralPath $templatePath -Raw -Encoding utf8 | ConvertFrom-Json
    $laneFindings = New-Object System.Collections.Generic.List[object]
    function Add-LaneFinding([string]$Id, [string]$Severity, [string]$Message) {
      $finding = [pscustomobject]@{ id = $Id; severity = $Severity; message = $Message; laneId = $lane.laneId }
      $laneFindings.Add($finding) | Out-Null
      if ($Severity -eq 'blocker') { $blockerFindings.Add($finding) | Out-Null }
      if ($Severity -eq 'action-required') { $actionRequiredFindings.Add($finding) | Out-Null }
    }

    if ([string]$template.recordKind -ne 'final-owner-real-input-template') { Add-LaneFinding 'record-kind' 'blocker' 'Template recordKind must be final-owner-real-input-template.' }
    if ([string]$template.laneId -ne [string]$lane.laneId) { Add-LaneFinding 'lane-id' 'blocker' 'Template laneId does not match wrapper lane.' }
    if (-not [bool]$template.isTemplate) { Add-LaneFinding 'is-template' 'blocker' 'Generated owner input must remain a template until Owner fills real evidence.' }
    if ([bool]$template.performsPublish -or [bool]$template.canPublishPublicly -or [bool]$template.canCloseReleaseIssue -or [bool]$template.canPromoteRuntimeProof -or [bool]$template.isRuntimeExecutionProof -or [bool]$template.isPostPublishProof -or [bool]$template.isReleaseCloseProof) {
      Add-LaneFinding 'non-proof-flags' 'blocker' 'Template must not publish, promote proof, or close release.'
    }

    foreach ($token in @('not runtime proof', 'not post-publish proof', 'not publish approval', 'not release close approval', 'not package push')) {
      if ([string]$template.boundary -notmatch [regex]::Escape($token)) {
        Add-LaneFinding "boundary-$token" 'blocker' "Missing boundary token $token."
      }
    }

    $fieldList = @($template.requiredEvidenceFields | ForEach-Object { [string]$_ })
    foreach ($field in $requiredCommonFields) {
      if ($fieldList -notcontains $field) {
        Add-LaneFinding "required-field-$field" 'blocker' "Missing required evidence field $field."
      }
    }

    $substitutes = @($template.forbiddenSubstitutes | ForEach-Object { [string]$_ })
    foreach ($substitute in @('local feed', 'ProjectReference', 'direct nupkg', 'template', 'draft', 'dry-run', 'dashboard', 'candidate')) {
      if ($substitutes -notcontains $substitute) {
        Add-LaneFinding "forbidden-$substitute" 'blocker' "Missing forbidden substitute marker $substitute."
      }
    }

    Add-LaneFinding 'real-owner-input-required' 'action-required' 'Owner must replace the template with real stdout/stderr/log/SHA256/exitCode/host/package evidence before this lane can pass.'

    $laneValidation = [pscustomobject][ordered]@{
      laneId = $lane.laneId
      templatePath = $templatePath
      validator = $lane.validator
      releaseCloseTarget = $lane.releaseCloseTarget
      validationState = if (@($laneFindings | Where-Object { $_.severity -eq 'blocker' }).Count -eq 0) { 'blocked-final-owner-real-input-required' } else { 'invalid-final-owner-real-input-template' }
      failedBlockerCount = @($laneFindings | Where-Object { $_.severity -eq 'blocker' }).Count
      failedActionRequiredCount = @($laneFindings | Where-Object { $_.severity -eq 'action-required' }).Count
      findings = @($laneFindings.ToArray())
      performsPublish = $false
      approvesPublicRelease = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      canPromoteRuntimeProof = $false
      isRuntimeExecutionProof = $false
      isPostPublishProof = $false
      isReleaseCloseProof = $false
      boundary = $script:FinalOwnerRealInputBoundary
    }

    $laneValidations.Add($laneValidation) | Out-Null
    $laneValidationJsonPath = Join-Path $finalReleaseRoot $lane.validationFileName
    $laneValidationMarkdownPath = Join-Path $finalReleaseRoot $lane.validationMarkdownFileName
    $laneValidation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $laneValidationJsonPath -Encoding utf8
    @(
      "# $($lane.title) Owner Input Validation",
      '',
      "- validationState: ``$($laneValidation.validationState)``",
      "- failedBlockerCount: ``$($laneValidation.failedBlockerCount)``",
      "- failedActionRequiredCount: ``$($laneValidation.failedActionRequiredCount)``",
      "- canPublishPublicly: ``False``",
      "- canCloseReleaseIssue: ``False``",
      "- Boundary: $script:FinalOwnerRealInputBoundary"
    ) | Set-Content -LiteralPath $laneValidationMarkdownPath -Encoding utf8
  }

  $packValidation = [pscustomobject][ordered]@{
    recordKind = 'final-owner-real-input-template-pack-validation'
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    validationState = if ($blockerFindings.Count -eq 0) { 'blocked-final-owner-real-input-required' } else { 'invalid-final-owner-real-input-template-pack' }
    laneCount = $lanes.Count
    failedBlockerCount = $blockerFindings.Count
    failedActionRequiredCount = $actionRequiredFindings.Count
    laneValidations = @($laneValidations.ToArray())
    findings = @($blockerFindings.ToArray() + $actionRequiredFindings.ToArray())
    performsPublish = $false
    approvesPublicRelease = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = $script:FinalOwnerRealInputBoundary
  }

  if ($isFullPackValidation) {
    $packValidationJsonPath = Join-Path $finalReleaseRoot 'final-owner-real-input-template-pack-validation.json'
    $packValidationMarkdownPath = Join-Path $finalReleaseRoot 'final-owner-real-input-template-pack-validation.md'
    $packValidation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $packValidationJsonPath -Encoding utf8
    @(
      '# Final Owner Real Input Template Pack Validation',
      '',
      "- validationState: ``$($packValidation.validationState)``",
      "- laneCount: ``$($packValidation.laneCount)``",
      "- failedBlockerCount: ``$($packValidation.failedBlockerCount)``",
      "- failedActionRequiredCount: ``$($packValidation.failedActionRequiredCount)``",
      "- canPublishPublicly: ``False``",
      "- canCloseReleaseIssue: ``False``",
      "- Boundary: $script:FinalOwnerRealInputBoundary"
    ) | Set-Content -LiteralPath $packValidationMarkdownPath -Encoding utf8

    Write-Host "Final owner real input template pack validation written:"
    Write-Host "  Json=$packValidationJsonPath"
    Write-Host "  Markdown=$packValidationMarkdownPath"
  }
  else {
    Write-Host "Final owner real input lane validation written; full template pack validation was left unchanged."
  }
  Write-Host "ValidationState=$($packValidation.validationState) Lanes=$($packValidation.laneCount) FailedBlockers=$($packValidation.failedBlockerCount) FailedActionRequired=$($packValidation.failedActionRequiredCount)"

  if ($Strict -and $blockerFindings.Count -gt 0) {
    throw "Final owner real input template validation failed with $($blockerFindings.Count) blocker finding(s)."
  }
}
