[CmdletBinding()]
param(
  [string]$ContractPath = "samples/assets/cross-task-reference-provenance-contract.json",
  [string]$ClassificationManifestPath = "samples/assets/classification-assets.template.json",
  [string]$YoloTaskContractPath = "applications/YoloVision/yolovision-task-output-contract.json",
  [string]$YoloOwnerInputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$IndependentReferenceEvidencePath = "artifacts/interface-coverage/tensorrtexec-mnist-onnxruntime-reference-evidence.json",
  [string]$OutputDirectory = "artifacts/interface-coverage",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else { $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot) }

$utf8 = [Text.UTF8Encoding]::new($false)

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-RelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)
  return [IO.Path]::GetRelativePath($RepositoryRoot, [IO.Path]::GetFullPath($Path)).Replace('\', '/')
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Json {
  param([Parameter(Mandatory = $true)][string]$Path)
  $fullPath = Resolve-RepositoryPath $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { throw "Required JSON file is missing: $fullPath" }
  return Get-Content -LiteralPath $fullPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
}

function Get-PropertyValue {
  param(
    [AllowNull()][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [AllowNull()][object]$DefaultValue = $null
  )
  if ($null -eq $Object) { return $DefaultValue }
  $property = $Object.PSObject.Properties[$Name]
  if ($null -eq $property) { return $DefaultValue }
  return $property.Value
}

function Test-ReadyValue {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return $false }
  if ($Value -is [bool]) { return [bool]$Value }
  if ($Value -is [ValueType] -and $Value -isnot [char]) { return $true }
  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }
  return $text -notmatch '(?i)(owner|required|placeholder|not-provided|not-captured|not-run|candidate-not-downloaded|unresolved)'
}

function Test-ReviewedLicense {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  return (Test-ReadyValue $text) -and $text -notmatch '(?i)(review|before redistribution|terms)'
}

function Test-Sha256 {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match '^[0-9a-f]{64}$'
}

function New-FieldCheck {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [AllowNull()][object]$Value,
    [Parameter(Mandatory = $true)][string]$Source,
    [ValidateSet("value", "sha256", "license")][string]$Rule = "value"
  )
  $ready = switch ($Rule) {
    "sha256" { Test-Sha256 $Value }
    "license" { Test-ReviewedLicense $Value }
    default { Test-ReadyValue $Value }
  }
  return [pscustomobject][ordered]@{
    id = $Id
    ready = [bool]$ready
    source = $Source
    valueState = if ($ready) { "recorded" } else { "owner-action-required" }
  }
}

function Get-SemanticValue {
  param(
    [Parameter(Mandatory = $true)][object]$Case,
    [Parameter(Mandatory = $true)][string]$Field
  )
  switch ($Field) {
    "imagePreprocessContract" { return Get-PropertyValue $Case.input "preprocessContract" }
    "labelMappingSha256" { return Get-PropertyValue $Case.labels "sha256" }
    "sourceImageInversePolicy" { return Get-PropertyValue $Case.outputMetadata "sourceImageInversePolicy" }
    default { return Get-PropertyValue $Case.outputMetadata $Field }
  }
}

$contractFullPath = Resolve-RepositoryPath $ContractPath
$classificationFullPath = Resolve-RepositoryPath $ClassificationManifestPath
$yoloContractFullPath = Resolve-RepositoryPath $YoloTaskContractPath
$yoloOwnerFullPath = Resolve-RepositoryPath $YoloOwnerInputPath
$independentFullPath = Resolve-RepositoryPath $IndependentReferenceEvidencePath
$outputFullPath = Resolve-RepositoryPath $OutputDirectory

$contract = Read-Json $ContractPath
$classification = Read-Json $ClassificationManifestPath
$yoloContract = Read-Json $YoloTaskContractPath
$yoloOwner = Read-Json $YoloOwnerInputPath
$independent = Read-Json $IndependentReferenceEvidencePath

$rows = [Collections.Generic.List[object]]::new()
$classificationProfile = @($contract.taskProfiles | Where-Object id -eq "classification")[0]
$classificationCommon = @(
  New-FieldCheck "modelSource" $classification.model.sourceUrl "classification.model.sourceUrl"
  New-FieldCheck "modelLicense" $classification.model.license "classification.model.license" -Rule license
  New-FieldCheck "modelSha256" $classification.model.sha256 "classification.model.sha256" -Rule sha256
  New-FieldCheck "labelsSource" $classification.labels.sourceUrl "classification.labels.sourceUrl"
  New-FieldCheck "labelsLicense" $classification.labels.license "classification.labels.license" -Rule license
  New-FieldCheck "labelsSha256" $classification.labels.sha256 "classification.labels.sha256" -Rule sha256
  New-FieldCheck "inputSource" $classification.input.sourceUrl "classification.input.sourceUrl"
  New-FieldCheck "inputLicense" $classification.input.license "classification.input.license" -Rule license
  New-FieldCheck "inputSha256" $classification.input.sha256 "classification.input.sha256" -Rule sha256
  New-FieldCheck "inputTensorName" $classification.tensor.inputName "classification.tensor.inputName"
  New-FieldCheck "inputShape" $classification.tensor.inputShape "classification.tensor.inputShape"
  New-FieldCheck "outputTensorName" $classification.tensor.outputName "classification.tensor.outputName"
  New-FieldCheck "outputShape" $classification.tensor.outputShape "classification.tensor.outputShape"
  New-FieldCheck "independentReference" $null "classification.evidence.independentReference"
  New-FieldCheck "ownerGoldenDecision" $null "classification.ownerReview.goldenAcceptanceDecision"
)
$classificationSemanticValues = [ordered]@{
  imageResizePolicy = $classification.preprocess.resize
  imageCropPolicy = $classification.preprocess.crop
  colorOrder = $classification.preprocess.colorOrder
  scale = $classification.preprocess.scale
  mean = $classification.preprocess.mean
  std = $classification.preprocess.std
  outputValueKind = $null
  scoreTransform = $null
  labelMappingSha256 = $classification.labels.sha256
  topK = $classification.postprocess.topK
  argmaxRule = $null
}
$classificationSemantic = foreach ($field in @($classificationProfile.requiredSemanticFields)) {
  $rule = if ($field -eq "labelMappingSha256") { "sha256" } else { "value" }
  New-FieldCheck $field $classificationSemanticValues[$field] "classification.taskSemantics.$field" -Rule $rule
}
$classificationReady = @($classificationCommon | Where-Object ready).Count + @($classificationSemantic | Where-Object ready).Count
$classificationTotal = $classificationCommon.Count + @($classificationSemantic).Count
$rows.Add([pscustomobject][ordered]@{
  id = "classification"
  sample = "Classification"
  task = "classification"
  sourceState = [string]$classification.status
  proofClassification = [string]$classification.proofClassification
  commonFieldChecks = @($classificationCommon)
  taskSemanticChecks = @($classificationSemantic)
  readyFieldCount = $classificationReady
  requiredFieldCount = $classificationTotal
  missingFieldCount = $classificationTotal - $classificationReady
  runtimeContractState = "implemented-managed-contract-owner-assets-required"
  runtimeContractArtifacts = @(
    "samples/ComputerVision/01.Classification/ClassificationImagePreprocessor.cs",
    "samples/ComputerVision/01.Classification/ClassificationOutputArtifacts.cs",
    "samples/ComputerVision/01.Classification/classification-reference.schema.json",
    "samples/ComputerVision/01.Classification/classification-output.schema.json"
  )
  runtimeContractIsRuntimeProof = $false
  independentReferenceState = "not-captured-for-classification"
  referenceReuseEligible = $false
  ownerReviewedGolden = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  boundary = [string]$classificationProfile.boundary
}) | Out-Null

$taskToProfile = @{
  det = "yolo-det"
  cls = "yolo-cls"
  seg = "yolo-seg"
  obb = "yolo-obb"
  pose = "yolo-pose"
  sem = "yolo-sem"
}
foreach ($task in @("det", "cls", "seg", "obb", "pose", "sem")) {
  $case = @($yoloOwner.cases | Where-Object task -eq $task)[0]
  $profile = @($contract.taskProfiles | Where-Object id -eq $taskToProfile[$task])[0]
  $taskContract = @($yoloContract.tasks | Where-Object task -eq $task)[0]
  if ($null -eq $case -or $null -eq $profile -or $null -eq $taskContract) { throw "YoloVision task contract is incomplete for '$task'." }

  $common = @(
    New-FieldCheck "modelSource" $case.model.sourceUrl "$($case.caseId).model.sourceUrl"
    New-FieldCheck "modelLicense" $case.model.license "$($case.caseId).model.license" -Rule license
    New-FieldCheck "modelSha256" $case.model.sha256 "$($case.caseId).model.sha256" -Rule sha256
    New-FieldCheck "labelsPath" $case.labels.path "$($case.caseId).labels.path"
    New-FieldCheck "labelsLicense" $case.labels.license "$($case.caseId).labels.license" -Rule license
    New-FieldCheck "labelsSha256" $case.labels.sha256 "$($case.caseId).labels.sha256" -Rule sha256
    New-FieldCheck "inputImagePath" $case.input.imagePath "$($case.caseId).input.imagePath"
    New-FieldCheck "inputImageLicense" $case.input.imageLicense "$($case.caseId).input.imageLicense" -Rule license
    New-FieldCheck "inputImageSha256" $case.input.imageSha256 "$($case.caseId).input.imageSha256" -Rule sha256
    New-FieldCheck "preprocessedTensorPath" $case.input.preprocessedTensorPath "$($case.caseId).input.preprocessedTensorPath"
    New-FieldCheck "preprocessedTensorSha256" $case.input.preprocessedTensorSha256 "$($case.caseId).input.preprocessedTensorSha256" -Rule sha256
    New-FieldCheck "inputShape" $case.input.inputShape "$($case.caseId).input.inputShape"
    New-FieldCheck "outputJsonSha256" $case.yoloVision.outputJsonSha256 "$($case.caseId).yoloVision.outputJsonSha256" -Rule sha256
    New-FieldCheck "runLogSha256" $case.yoloVision.runLogSha256 "$($case.caseId).yoloVision.runLogSha256" -Rule sha256
    New-FieldCheck "ownerGoldenDecision" $case.ownerReview.acceptanceDecision "$($case.caseId).ownerReview.acceptanceDecision"
  )
  $semantic = foreach ($field in @($profile.requiredSemanticFields)) {
    $value = Get-SemanticValue -Case $case -Field $field
    $rule = if ($field -in @("labelMappingSha256", "paletteSha256")) { "sha256" } else { "value" }
    New-FieldCheck $field $value "$($case.caseId).taskSemantics.$field" -Rule $rule
  }
  $ready = @($common | Where-Object ready).Count + @($semantic | Where-Object ready).Count
  $total = $common.Count + @($semantic).Count
  $rows.Add([pscustomobject][ordered]@{
    id = [string]$profile.id
    sample = "YoloVision"
    task = $task
    caseId = [string]$case.caseId
    sourceState = [string]$case.ownerInputState
    proofClassification = [string]$case.proofClassification
    contractRequiredMetadata = @($taskContract.requiredMetadata)
    commonFieldChecks = @($common)
    taskSemanticChecks = @($semantic)
    readyFieldCount = $ready
    requiredFieldCount = $total
    missingFieldCount = $total - $ready
    independentReferenceState = "not-captured-for-$task"
    referenceReuseEligible = $false
    ownerReviewedGolden = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    boundary = [string]$profile.boundary
  }) | Out-Null
}

$candidate = [pscustomobject][ordered]@{
  id = "mnist-onnxruntime-cpu-1.23.2"
  task = "mnist-classification"
  state = [string]$independent.state
  evidenceClassification = [string]$independent.evidenceClassification
  framework = [string]$independent.runtime.name
  version = [string]$independent.runtime.version
  provider = [string]$independent.runtime.requestedProvider
  providerValidated = [bool]$independent.runtime.providerValidated
  deterministicOutput = [bool]$independent.reference.deterministicOutput
  modelSha256 = [string]$independent.model.sha256
  inputTensorSha256 = [string]$independent.input.sha256
  referenceSha256 = [string]$independent.reference.sha256
  eligibleTaskIds = @("mnist")
  ineligibleMatrixTaskIds = @($rows | ForEach-Object id)
  allReuseFingerprintsRecorded = $false
  ownerReviewedGolden = [bool]$independent.proofBoundary.ownerReviewedGolden
  repositoryRedistributionApproved = [bool]$independent.proofBoundary.repositoryRedistributionApproved
  canReuseForAnyMatrixTask = $false
  reason = "The MNIST model/input/preprocess/output/labels/task semantics do not match the generic Classification sample or any YoloVision task profile."
}

$readyRows = @($rows | Where-Object { $_.missingFieldCount -eq 0 -and $_.ownerReviewedGolden })
$matrix = [ordered]@{
  schemaVersion = "cross-task-reference-provenance-matrix.v1"
  generatedDate = [DateTime]::Now.ToString("yyyy-MM-dd")
  state = "owner-action-required"
  contract = [ordered]@{
    path = Get-RelativePath $contractFullPath
    sha256 = Get-Sha256 $contractFullPath
    schemaVersion = [string]$contract.schemaVersion
  }
  sources = @(
    [ordered]@{ role = "classification-manifest"; path = Get-RelativePath $classificationFullPath; sha256 = Get-Sha256 $classificationFullPath },
    [ordered]@{ role = "yolovision-task-contract"; path = Get-RelativePath $yoloContractFullPath; sha256 = Get-Sha256 $yoloContractFullPath },
    [ordered]@{ role = "yolovision-owner-input-template"; path = Get-RelativePath $yoloOwnerFullPath; sha256 = Get-Sha256 $yoloOwnerFullPath },
    [ordered]@{ role = "independent-reference-candidate"; path = Get-RelativePath $independentFullPath; sha256 = Get-Sha256 $independentFullPath }
  )
  rowCount = $rows.Count
  readyRowCount = $readyRows.Count
  ownerActionRequiredRowCount = $rows.Count - $readyRows.Count
  rows = @($rows)
  independentReferenceCandidates = @($candidate)
  proofBoundary = [ordered]@{
    crossTaskReferenceReuseProved = $false
    ownerReviewedGoldenAvailable = $false
    publicPackageProof = $false
    postPublishProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    statement = "This matrix audits provenance readiness and task semantics. It does not create reference output, approve licenses, accept an Owner golden, prove public-package consumption, or provide post-publish/release proof."
  }
}

New-Item -ItemType Directory -Path $outputFullPath -Force | Out-Null
$jsonPath = Join-Path $outputFullPath "cross-task-reference-provenance-matrix.json"
$markdownPath = Join-Path $outputFullPath "cross-task-reference-provenance-matrix.md"
[IO.File]::WriteAllText($jsonPath, ($matrix | ConvertTo-Json -Depth 30) + [Environment]::NewLine, $utf8)
$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# Cross-Task Reference Provenance Matrix")
$lines.Add("")
$lines.Add("- state: ``$($matrix.state)``")
$lines.Add("- rows: ``$($matrix.rowCount)``")
$lines.Add("- ready rows: ``$($matrix.readyRowCount)``")
$lines.Add("- owner action required rows: ``$($matrix.ownerActionRequiredRowCount)``")
$lines.Add("")
$lines.Add("| Row | Sample | Task | Runtime contract | Ready / Required | Missing | Independent reference | Owner golden |")
$lines.Add("| --- | --- | --- | --- | ---: | ---: | --- | --- |")
foreach ($row in $rows) {
  $runtimeContract = if ($row.PSObject.Properties["runtimeContractState"]) { [string]$row.runtimeContractState } else { "not-audited-in-this-batch" }
  $lines.Add("| ``$($row.id)`` | $($row.sample) | ``$($row.task)`` | ``$runtimeContract`` | $($row.readyFieldCount) / $($row.requiredFieldCount) | $($row.missingFieldCount) | ``$($row.independentReferenceState)`` | ``$($row.ownerReviewedGolden)`` |")
}
$lines.Add("")
$lines.Add("## Independent Candidates")
$lines.Add("")
$lines.Add("- ``$($candidate.id)``: provider ``$($candidate.provider)`` / deterministic ``$($candidate.deterministicOutput)`` / eligible only for ``mnist`` / reusable for matrix rows ``False``.")
$lines.Add("")
$lines.Add("$($matrix.proofBoundary.statement)")
[IO.File]::WriteAllLines($markdownPath, $lines, $utf8)

Write-Output "Cross-task reference provenance matrix written."
Write-Output "JSON=$jsonPath"
Write-Output "Markdown=$markdownPath"
Write-Output "Rows=$($matrix.rowCount) Ready=$($matrix.readyRowCount) OwnerActionRequired=$($matrix.ownerActionRequiredRowCount)"
Write-Output "MnistReferenceReusableForMatrixTasks=$($candidate.canReuseForAnyMatrixTask)"
