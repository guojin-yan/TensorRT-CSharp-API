[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$LedgerPath = "docs/articles/zh-cn/publishing/technical-article-closure-ledger.json",
  [string]$OutputRoot = "docs/articles/zh-cn/publishing"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-RepositoryPath {
  param([string]$RelativePath)

  $normalized = $RelativePath.Replace('/', '\')
  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $normalized))
}

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Resolve-RepositoryPath $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
    return $DefaultValue
  }

  return $Object.PSObject.Properties[$Name].Value
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return @()
  }

  return @($Value)
}

function Convert-ToRepositoryPath {
  param([string]$Path)

  return $Path.Replace('\', '/')
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
}

function Get-HandoffLine {
  param(
    [object[]]$HandoffLines,
    [string]$LaneId
  )

  return @($HandoffLines | Where-Object { [string]$_.id -eq $LaneId } | Select-Object -First 1)[0]
}

function New-FallbackLane {
  param([string]$LaneId)

  $common = @(
    'template',
    'draft',
    'runbook',
    'collection package',
    'handoff',
    'local feed',
    'ProjectReference',
    'direct .nupkg reference',
    'build-only',
    'parse-only',
    'precheck-only',
    'dry-run-only',
    'schema-only',
    'readiness snapshot',
    'dependency-probe-only',
    'blocked-by-cuda-driver',
    'Windows handoff for Linux proof',
    'synthetic runtime'
  )

  switch ($LaneId) {
    'callback-runtime' {
      return [pscustomobject][ordered]@{
        id = 'callback-runtime'
        proofClass = 'real-callback-runtime'
        currentState = 'callbackRuntimeProofState=blocked-real-callback-runtime-proof-required; isRealCallbackRuntimeProof=False'
        blockerReason = 'Real callback invocation evidence with InvocationCount>0 and owner-safe attach/detach validation is still required.'
        ownerNextAction = 'Run the callback proof execution pack on a compatible host, capture invocation and failure-injection evidence, then pass the strict callback validator.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CallbackRuntimeProofExecutionPack.ps1'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CallbackRuntimeProofExecutionPack.ps1'
        requiredRealInputs = @('compatible TensorRT/CUDA host', 'callback-specific package consumer', 'InvocationCount>0', 'attach/detach and in-flight lifecycle evidence', 'native failure and managed exception evidence')
        expectedArtifacts = @('artifacts/final-release/callback-runtime-proof-execution-pack.json', 'artifacts/final-release/callback-runtime-proof-execution-pack-validation.json')
        missingRealInputCount = 5
        cannotUse = $common
      }
    }
    'linux-runner' {
      return [pscustomobject][ordered]@{
        id = 'linux-runner-proof'
        proofClass = 'linux-runner-proof'
        currentState = 'linuxRunnerValidationState=template-only; isRealLinuxRunnerProof=False'
        blockerReason = 'A real Linux x64 runner record and matching smoke log are still required; Windows handoff is not Linux proof.'
        ownerNextAction = 'Run the Linux x64 runtime proof on a real runner and validate linux-runner-evidence-record.json.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerEvidenceRecordTemplate.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof'
        requiredRealInputs = @('real Linux x64 runner identity', 'Linux runtime package key', 'Linux restore/build/smoke commands', 'Linux host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata', 'existing Linux runner smoke log with matching SHA256')
        expectedArtifacts = @('artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record.json', 'artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.json')
        missingRealInputCount = 5
        cannotUse = $common
      }
    }
    'owner-authorization' {
      return [pscustomobject][ordered]@{
        id = 'owner-authorization'
        proofClass = 'owner-authorization'
        currentState = 'ownerProofInputValidationState=blocked-template-only; ownerProofInputCanPromote=False'
        blockerReason = 'Owner authorization and command approval are still template guidance until real owner fields validate.'
        ownerNextAction = 'Fill release-owner-proof-input-record.json with owner decision, selected channel, package hashes, runtime log hash, host metadata, and non-placeholder command approval.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1 -InputPath artifacts/final-release/release-owner-proof-input-record.json -RequireExistingLogs -FailOnNotProof'
        requiredRealInputs = @('owner name and decision id', 'selected channel source URI', 'managed/runtime nupkg SHA256', 'runtime smoke log SHA256', 'TensorRT runtime and host metadata', 'reviewed acknowledgement hashes', 'non-placeholder command approval')
        expectedArtifacts = @('artifacts/final-release/release-owner-proof-input-record.json', 'artifacts/final-release/release-owner-proof-input-record-validation.json', 'artifacts/final-release/owner-authorized-publish-command-plan-validation.json')
        missingRealInputCount = 7
        cannotUse = $common
      }
    }
    'package-consumer-runtime' {
      return [pscustomobject][ordered]@{
        id = 'package-consumer-runtime'
        proofClass = 'package-consumer-runtime'
        currentState = 'externalRuntimeValidationState=template-only; externalRuntimeCanPromoteRuntimeProof=False'
        blockerReason = 'Package-consumer runtime remains blocked until a real clean external consumer smoke and hash-matching validation pass.'
        ownerNextAction = 'Collect compatible-host package consumer runtime smoke from a clean external consumer, then validate external-runtime-proof-record.json with -FailOnNotProof.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof'
        requiredRealInputs = @('clean consumer root outside repository', 'managed/runtime package identities from selected channel', 'managed/runtime nupkg SHA256', 'runtime package key', 'compatible host and driver metadata', 'restore/build/runtime smoke commands', 'stdout and stderr summaries', 'existing smoke log with matching SHA256', 'forbidden-substitute-free consumer')
        expectedArtifacts = @('artifacts/final-release/external-runtime-proof-record.json', 'artifacts/final-release/external-runtime-proof-validation.json')
        missingRealInputCount = 9
        cannotUse = $common
      }
    }
    'post-publish-verification' {
      return [pscustomobject][ordered]@{
        id = 'post-publish-verification'
        proofClass = 'post-publish-verification'
        currentState = 'postPublishValidationState=template-only; isPostPublishVerificationProof=False; postPublishCanCloseReleaseIssue=False'
        blockerReason = 'Post-publish verification can only happen after real channel publication and a clean consumer validation.'
        ownerNextAction = 'After owner performs real selected-channel publication outside automation, download and validate a clean consumer post-publish record.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof'
        requiredRealInputs = @('selected channel and source URI', 'published managed/runtime package URLs', 'downloaded package SHA256', 'clean consumer outside repository', 'restore/build/runtime smoke commands', 'runtime package key in smoke command', 'logs and summaries with matching hashes', 'host metadata and reviewer identity')
        expectedArtifacts = @('artifacts/final-release/post-publish-verification-record.json', 'artifacts/final-release/post-publish-verification-validation.json')
        missingRealInputCount = 8
        cannotUse = $common
      }
    }
    'real-model-runtime' {
      return [pscustomobject][ordered]@{
        id = 'real-model-runtime'
        proofClass = 'real-model-runtime'
        currentState = 'sampleRunValidationState=owner-action-required; isRealSampleRunProof=False'
        blockerReason = 'Real-model runtime requires owner-approved assets, hashes, license/input metadata and a real runner log.'
        ownerNextAction = 'Backfill real Classification/YoloVision assets, model/input/label hashes, runner log and sample-run evidence validation.'
        firstCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleAssetManifestTemplate.ps1'
        validatorCommand = 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof'
        requiredRealInputs = @('model source and license', 'model SHA256', 'input image/tensor SHA256', 'labels/class metadata SHA256', 'TensorRtExec build sidecar', 'sample runner command and log SHA256', 'Classification or YoloVision task metadata', 'YOLO family coverage confirmed by owner', 'task coverage det/cls/seg/obb/pose/sem')
        expectedArtifacts = @('artifacts/user-acceptance/sample-asset-manifest.json', 'artifacts/user-acceptance/sample-run-evidence-record.json', 'artifacts/user-acceptance/sample-run-evidence-record-validation.json')
        missingRealInputCount = 9
        cannotUse = $common
      }
    }
    default { throw "Unknown proof lane: $LaneId" }
  }
}

$ledger = Read-JsonOrNull $LedgerPath
if ($null -eq $ledger) {
  throw "Technical article closure ledger not found: $(Resolve-RepositoryPath $LedgerPath)"
}

$articles = @($ledger.articles | Where-Object { [bool]$_.proofRequired -and -not [bool]$_.proofComplete } | Sort-Object articleId)
if ($articles.Count -ne [int]$ledger.ownerOrRuntimeProofRequiredCount) {
  throw "Ledger proof count mismatch: filtered=$($articles.Count) reported=$($ledger.ownerOrRuntimeProofRequiredCount)"
}

$handoff = Read-JsonOrNull "artifacts/final-release/owner-proof-execution-handoff.json"
$handoffLines = @(Get-PropertyOrDefault -Object $handoff -Name "handoffLines" -DefaultValue @())
$laneIds = @('callback-runtime', 'linux-runner', 'owner-authorization', 'package-consumer-runtime', 'post-publish-verification', 'real-model-runtime')
$dependencyToLane = @{
  'real-callback-runtime' = 'callback-runtime'
  'linux-runner-proof' = 'linux-runner'
  'owner-authorization' = 'owner-authorization'
  'package-consumer-runtime' = 'package-consumer-runtime'
  'post-publish-verification' = 'post-publish-verification'
  'real-model-runtime' = 'real-model-runtime'
}

$lanes = @()
foreach ($laneId in $laneIds) {
  $fallback = New-FallbackLane $laneId
  $handoffId = if ($laneId -eq 'linux-runner') { 'linux-runner-proof' } else { $laneId }
  $source = Get-HandoffLine -HandoffLines $handoffLines -LaneId $handoffId
  $lane = [pscustomobject][ordered]@{
    id = $laneId
    proofClass = [string](Get-PropertyOrDefault -Object $source -Name 'proofClass' -DefaultValue $fallback.proofClass)
    currentState = [string](Get-PropertyOrDefault -Object $source -Name 'currentState' -DefaultValue $fallback.currentState)
    blockerReason = [string](Get-PropertyOrDefault -Object $source -Name 'blockerReason' -DefaultValue $fallback.blockerReason)
    ownerNextAction = [string](Get-PropertyOrDefault -Object $source -Name 'ownerNextAction' -DefaultValue $fallback.ownerNextAction)
    firstCommand = [string](Get-PropertyOrDefault -Object $source -Name 'firstCommand' -DefaultValue $fallback.firstCommand)
    validatorCommand = [string](Get-PropertyOrDefault -Object $source -Name 'validatorCommand' -DefaultValue $fallback.validatorCommand)
    requiredRealInputs = @(Get-PropertyOrDefault -Object $source -Name 'requiredRealInputs' -DefaultValue $fallback.requiredRealInputs)
    expectedArtifacts = @(Get-PropertyOrDefault -Object $source -Name 'expectedArtifacts' -DefaultValue $fallback.expectedArtifacts)
    missingRealInputCount = [int](Get-PropertyOrDefault -Object $source -Name 'missingRealInputCount' -DefaultValue $fallback.missingRealInputCount)
    cannotUse = @(Get-PropertyOrDefault -Object $source -Name 'cannotUse' -DefaultValue $fallback.cannotUse)
    canPromoteProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
  $lanes += $lane
}

$laneById = @{}
foreach ($lane in $lanes) { $laneById[$lane.id] = $lane }
$articleRows = @()
foreach ($article in $articles) {
  $articleLanes = @(
    foreach ($dependency in @(Convert-ToArray $article.proofDependencies)) {
      if ($dependencyToLane.ContainsKey([string]$dependency)) { $dependencyToLane[[string]$dependency] }
    }
  ) | Sort-Object -Unique
  if ($articleLanes.Count -eq 0) { throw "Article $($article.articleId) has proofRequired=true but no supported proof dependency." }

  $articleRows += [pscustomobject][ordered]@{
    articleId = [int]$article.articleId
    title = [string]$article.title
    canonicalArticlePath = [string]$article.canonicalArticlePath
    contentComplete = [bool]$article.contentComplete
    proofState = [string]$article.proofState
    proofComplete = [bool]$article.proofComplete
    proofDependencies = @(Convert-ToArray $article.proofDependencies)
    proofLanes = @($articleLanes)
    currentBlocker = @($articleLanes | ForEach-Object { $laneById[$_].blockerReason } | Select-Object -Unique)
    ownerNextActions = @($articleLanes | ForEach-Object { $laneById[$_].ownerNextAction } | Select-Object -Unique)
    validatorCommands = @($articleLanes | ForEach-Object { $laneById[$_].validatorCommand } | Select-Object -Unique)
    firstCommands = @($articleLanes | ForEach-Object { $laneById[$_].firstCommand } | Select-Object -Unique)
    expectedArtifacts = @($articleLanes | ForEach-Object { $laneById[$_].expectedArtifacts } | Select-Object -Unique)
    promotionFlags = [pscustomobject][ordered]@{ canPromoteRuntimeProof = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; performsPublish = $false }
  }
}

$sourceArtifacts = @(
  [pscustomobject][ordered]@{ path = 'docs/articles/zh-cn/publishing/technical-article-closure-ledger.json'; role = '42 article proof dependencies and canonical paths'; requiredFields = @('ownerOrRuntimeProofRequiredCount', 'articles[].proofDependencies', 'articles[].proofState') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/owner-proof-execution-handoff.json'; role = 'six owner execution lanes and first commands'; requiredFields = @('handoffLines[].ownerNextAction', 'handoffLines[].validatorCommand', 'handoffLines[].requiredRealInputs') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/real-owner-proof-convergence-dashboard-validation.json'; role = 'real owner input convergence state'; requiredFields = @('convergenceState', 'blockedLaneCount', 'canPromoteRuntimeProof') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/owner-external-proof-input-preflight.json'; role = 'external proof input preflight'; requiredFields = @('preflightState', 'proofLineCount', 'blockedProofLineCount') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/release-evidence-bundle.json'; role = 'release evidence and publish freeze'; requiredFields = @('bundleState', 'canPublishPublicly', 'canCloseReleaseIssue') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/release-close-preflight.json'; role = 'release close blocker state'; requiredFields = @('preflightState', 'failedItemCount', 'canCloseReleaseIssue') }
  [pscustomobject][ordered]@{ path = 'artifacts/final-release/callback-runtime-proof-execution-pack.json'; role = 'callback-specific invocation proof contract'; requiredFields = @('InvocationCount', 'isRealCallbackRuntimeProof', 'canPromoteRuntimeProof') }
)

$observed = [pscustomobject][ordered]@{
  convergenceState = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull 'artifacts/final-release/real-owner-proof-convergence-dashboard-validation.json') -Name 'convergenceState' -DefaultValue 'blocked-real-owner-proof-convergence-real-owner-input-required')
  ownerProofPreflightState = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull 'artifacts/final-release/owner-external-proof-input-preflight.json') -Name 'preflightState' -DefaultValue 'blocked-real-proof-required')
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull 'artifacts/final-release/release-evidence-bundle.json') -Name 'bundleState' -DefaultValue 'blocked-evidence-incomplete')
  releaseClosePreflightState = [string](Get-PropertyOrDefault -Object (Read-JsonOrNull 'artifacts/final-release/release-close-preflight.json') -Name 'preflightState' -DefaultValue 'blocked-real-proof-required')
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
}

$laneSummaries = @($lanes | ForEach-Object {
  $laneId = $_.id
  $rows = @($articleRows | Where-Object { $_.proofLanes -contains $laneId })
  [pscustomobject][ordered]@{
    id = $laneId
    proofClass = $_.proofClass
    articleCount = $rows.Count
    articleIds = @($rows | ForEach-Object { $_.articleId })
    currentState = $_.currentState
    blockerReason = $_.blockerReason
    ownerNextAction = $_.ownerNextAction
    firstCommand = $_.firstCommand
    validatorCommand = $_.validatorCommand
    requiredRealInputs = @($_.requiredRealInputs)
    expectedArtifacts = @($_.expectedArtifacts)
    missingRealInputCount = $_.missingRealInputCount
    canPromoteProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
})

$record = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = 'technical-article-proof-backlog'
  backlogState = 'blocked-real-proof-owner-action-required'
  sourceLedger = (Convert-ToRepositoryPath $LedgerPath)
  articleProofCount = $articleRows.Count
  articleProofIds = @($articleRows | ForEach-Object { $_.articleId })
  laneCount = $laneSummaries.Count
  proofRelationCount = @($articleRows | ForEach-Object { $_.proofLanes }).Count
  blockedArticleCount = @($articleRows | Where-Object { -not $_.promotionFlags.canPromoteRuntimeProof }).Count
  readyArticleCount = 0
  ownerOrRuntimeProofRequiredCount = [int]$ledger.ownerOrRuntimeProofRequiredCount
  contentCompleteCount = [int]$ledger.contentCompleteCount
  observedReleaseState = $observed
  laneSummaries = $laneSummaries
  articles = $articleRows
  sourceArtifacts = $sourceArtifacts
  nonSubstituteProofKinds = @($lanes | ForEach-Object { $_.cannotUse } | Select-Object -Unique)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  proofBoundary = 'This backlog is a deterministic projection of content proof dependencies and owner handoff contracts. It is not runtime proof, package-consumer proof, callback invocation proof, Linux runner proof, real-model proof, post-publish proof, publish approval, or release issue close approval.'
}

$outputPath = Resolve-RepositoryPath $OutputRoot
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$jsonPath = Join-Path $outputPath 'technical-article-proof-backlog.json'
$markdownPath = Join-Path $outputPath 'technical-article-proof-backlog.md'
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($jsonPath, ($record | ConvertTo-Json -Depth 20) + [Environment]::NewLine, $utf8)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# Technical Article Proof Backlog')
$lines.Add('')
$lines.Add('- backlogState: ``' + $record.backlogState + '``')
$lines.Add('- article proof rows: ``' + $record.articleProofCount + '``')
$lines.Add('- proof relations: ``' + $record.proofRelationCount + '``')
$lines.Add('- lanes: ``' + $record.laneCount + '``')
$lines.Add('- blocked articles: ``' + $record.blockedArticleCount + '``')
$lines.Add('- content complete count: ``' + $record.contentCompleteCount + '``')
$lines.Add('- performsPublish=false; canPublishPublicly=false; canCloseReleaseIssue=false; canPromoteRuntimeProof=false')
$lines.Add('')
$lines.Add($record.proofBoundary)
$lines.Add('')
$lines.Add('## Observed Release State')
$lines.Add('')
$lines.Add('| Field | Value |')
$lines.Add('| --- | --- |')
foreach ($property in $observed.PSObject.Properties) { $lines.Add('| ' + $property.Name + ' | ``' + (ConvertTo-MarkdownCell $property.Value) + '`` |') }
$lines.Add('')
$lines.Add('## Lane Summaries')
$lines.Add('')
$lines.Add('| Lane | Articles | State | Blocker | Validator |')
$lines.Add('| --- | ---: | --- | --- | --- |')
foreach ($lane in $laneSummaries) { $lines.Add('| ``' + $lane.id + '`` | ' + $lane.articleCount + ' | ``' + (ConvertTo-MarkdownCell $lane.currentState) + '`` | ' + (ConvertTo-MarkdownCell $lane.blockerReason) + ' | ``' + (ConvertTo-MarkdownCell $lane.validatorCommand) + '`` |') }
$lines.Add('')
$lines.Add('## Article Rows')
$lines.Add('')
$lines.Add('| ID | Title | Proof lane(s) | Current blocker | Owner action |')
$lines.Add('| ---: | --- | --- | --- | --- |')
foreach ($article in $articleRows) { $lines.Add('| ' + $article.articleId + ' | ' + (ConvertTo-MarkdownCell $article.title) + ' | ``' + (($article.proofLanes -join ', ') + '``') + ' | ' + (ConvertTo-MarkdownCell ($article.currentBlocker -join ' / ')) + ' | ' + (ConvertTo-MarkdownCell ($article.ownerNextActions -join ' / ')) + ' |') }
$lines.Add('')
$lines.Add('## Source Contracts')
$lines.Add('')
foreach ($source in $sourceArtifacts) { $lines.Add('- ``' + $source.path + '``: ' + $source.role + '; required fields: ``' + (($source.requiredFields -join ', ') + '``') ) }
$lines.Add('')
$lines.Add('## Non-Substitutes')
$lines.Add('')
foreach ($item in $record.nonSubstituteProofKinds) { $lines.Add('- ``' + $item + '``') }
[System.IO.File]::WriteAllText($markdownPath, ($lines -join [Environment]::NewLine) + [Environment]::NewLine, $utf8)

Write-Host "Technical article proof backlog written: ArticleProofRows=$($record.articleProofCount) Relations=$($record.proofRelationCount) Lanes=$($record.laneCount) Blocked=$($record.blockedArticleCount)"
Write-Host "PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False CanPromoteRuntimeProof=False"
