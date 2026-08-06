[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string[]] $InputPath = @(
        "samples/assets/yolovision-yolov8-det-candidate.template.json",
        "samples/assets/yolovision-yolov8-seg-candidate.template.json",
        "samples/assets/yolovision-yolov8-pose-candidate.template.json",
        "samples/assets/yolovision-yolov8-obb-candidate.template.json",
        "samples/assets/yolovision-yolov8-cls-candidate.template.json",
        "samples/assets/yolovision-yolov8-sem-candidate.template.json"
    ),

    [Parameter(Mandatory = $false)]
    [switch] $Strict,

    [Parameter(Mandatory = $false)]
    [string] $ContractPath = "applications/YoloVision/yolovision-task-output-contract.json",

    [Parameter(Mandatory = $false)]
    [string] $OutputPath = "artifacts/yolovision/yolovision-real-asset-candidate-validation.json"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Test-Placeholder {
    param([AllowNull()][string] $Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $true
    }

    return $Value -match "owner-required|owner-confirmed|\.\.\.|template-only|notProofUntil"
}

function Test-Sha256 {
    param([AllowNull()][string] $Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    return $Value -cmatch "^[0-9a-fA-F]{64}$"
}

function Get-JsonString {
    param(
        [Parameter(Mandatory = $true)] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    if ($null -eq $Object.PSObject.Properties[$Name]) {
        return $null
    }

    $value = $Object.$Name
    if ($null -eq $value) {
        return $null
    }

    return [string]$value
}

function Add-ValidationItem {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]] $Items,
        [Parameter(Mandatory = $true)] [string] $Id,
        [Parameter(Mandatory = $true)] [bool] $Passed,
        [Parameter(Mandatory = $true)] [string] $Severity,
        [Parameter(Mandatory = $true)] [string] $Message
    )

    $Items.Add([ordered]@{
        id = $Id
        passed = $Passed
        severity = $Severity
        message = $Message
    }) | Out-Null
}

function Test-JsonProperty {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name] -and $null -ne $Object.$Name
}

function Test-ContractMetadataCoverage {
    param(
        [AllowNull()] $Candidate,
        [AllowNull()] $Metadata,
        [Parameter(Mandatory = $true)] [string] $Task,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    $commands = $Candidate.commands
    $runCommand = Get-JsonString $commands "yoloVisionRunCommand"
    $buildCommand = Get-JsonString $commands "tensorRtExecBuildCommand"

    switch ($Name) {
        "inputShape" { return Test-JsonProperty $Candidate.input "inputShape" }
        "classCount" { return (Test-JsonProperty $Candidate.labels "classCount") -or (Test-JsonProperty $Metadata "classCount") }
        "labelsPath" { return Test-JsonProperty $Candidate.labels "localPath" }
        "classificationOutput" { return (Test-JsonProperty $Metadata "classificationOutput") -or (Test-JsonProperty $Metadata "classScoreField") -or ($runCommand -like "*--class-score-field*") }
        "outputLayout" { return (Test-JsonProperty $Metadata "outputLayout") -or (Test-JsonProperty $Metadata "layout") }
        "outputRoleMap" { return (Test-JsonProperty $Metadata "outputRoleMap") -or ($runCommand -like "*--output-role-map*") }
        "boxFormat" { return (Test-JsonProperty $Metadata "boxFormat") -or (Test-JsonProperty $Metadata "rotatedBoxLayout") -or ($runCommand -like "*--rotated-box-layout*") }
        "scoreRule" { return (Test-JsonProperty $Metadata "scoreRule") -or (Test-JsonProperty $Metadata "classScoreField") -or (Test-JsonProperty $Metadata.postprocessMetadata "confidence") }
        "nmsMode" { return (Test-JsonProperty $Metadata "nmsMode") -or (Test-JsonProperty $Metadata.postprocessMetadata "nmsMode") -or ($runCommand -like "*--nms-mode*") }
        "maskCoefficientCount" { return (Test-JsonProperty $Metadata "maskCoefficientCount") -or ($runCommand -like "*--mask-coefficient-count*") }
        "prototypeShape" { return (Test-JsonProperty $Metadata "prototypeShape") -or (Test-JsonProperty $Metadata "prototypeOutputShape") }
        "maskResizePolicy" { return (Test-JsonProperty $Metadata "maskResizePolicy") -or (Test-JsonProperty $Metadata.postprocessMetadata "maskCropAndScaleRule") }
        "angleOutput" { return (Test-JsonProperty $Metadata "angleOutput") -or (Test-JsonProperty $Metadata "rotatedBoxLayout") -or ($runCommand -like "*--rotated-box-layout*") }
        "angleUnit" { return (Test-JsonProperty $Metadata "angleUnit") -or ($runCommand -like "*--angle-unit*") }
        "keypointCount" { return (Test-JsonProperty $Metadata "keypointCount") -or ($runCommand -like "*--keypoint-count*") }
        "keypointStride" { return (Test-JsonProperty $Metadata "keypointStride") -or (Test-JsonProperty $Metadata "keypointLayout") }
        "coordinateLayout" { return (Test-JsonProperty $Metadata "coordinateLayout") -or (Test-JsonProperty $Metadata "keypointLayout") -or (Test-JsonProperty $Metadata "coordinateSpace") }
        "semanticOutput" { return (Test-JsonProperty $Metadata "semanticOutput") -or (Test-JsonProperty $Metadata "semanticOutputRole") -or ($runCommand -like "*--semantic-output*") }
        "mapWidth" { return (Test-JsonProperty $Metadata "mapWidth") -or (Test-JsonProperty $Metadata "semanticMapShape") }
        "mapHeight" { return (Test-JsonProperty $Metadata "mapHeight") -or (Test-JsonProperty $Metadata "semanticMapShape") }
        "argmaxRule" { return (Test-JsonProperty $Metadata "argmaxRule") -or (Test-JsonProperty $Metadata.postprocessMetadata "activation") }
        default { return Test-JsonProperty $Metadata $Name }
    }
}

$resolvedContract = Resolve-RepoPath $ContractPath
if (-not (Test-Path -LiteralPath $resolvedContract)) {
    throw "Missing YoloVision task/output contract: $ContractPath"
}

$contract = Get-Content -LiteralPath $resolvedContract -Raw | ConvertFrom-Json -Depth 64
$contractTasks = @($contract.tasks)
$contractTaskMap = @{}
foreach ($contractTask in $contractTasks) {
    $contractTaskMap[[string]$contractTask.task] = $contractTask
}

$records = New-Object System.Collections.Generic.List[object]
$allItems = New-Object System.Collections.Generic.List[object]

foreach ($path in $InputPath) {
    $resolvedPath = Resolve-RepoPath $path
    $items = New-Object System.Collections.Generic.List[object]

    if (-not (Test-Path -LiteralPath $resolvedPath)) {
        Add-ValidationItem $items "file-exists" $false "blocker" "Candidate file does not exist: $path"
        $recordValidationItems = [object[]]@($items.ToArray())
        $records.Add([ordered]@{
            inputPath = $path
            resolvedPath = $resolvedPath
            validationState = "invalid"
            canPromoteRealModelRuntime = $false
            canPromotePackageConsumerRuntime = $false
            validationItems = $recordValidationItems
        }) | Out-Null
        foreach ($item in $items) { $allItems.Add($item) | Out-Null }
        continue
    }

    $candidate = Get-Content -LiteralPath $resolvedPath -Raw | ConvertFrom-Json -Depth 32
    $task = Get-JsonString $candidate "task"
    $runtimeProofState = Get-JsonString $candidate "runtimeProofState"
    $proofClassification = Get-JsonString $candidate "proofClassification"

    Add-ValidationItem $items "file-exists" $true "required" "Candidate file exists."
    Add-ValidationItem $items "sample-name-yolovision" ((Get-JsonString $candidate "sampleName") -eq "YoloVision") "blocker" "sampleName must be YoloVision."
    Add-ValidationItem $items "family-yolov8" ((Get-JsonString $candidate "family") -eq "YOLOv8") "blocker" "family must be YOLOv8 for this validator batch."
    Add-ValidationItem $items "task-supported" ($task -in @("det", "seg", "pose", "obb", "cls", "sem")) "blocker" "task must be det, seg, pose, obb, cls, or sem."
    Add-ValidationItem $items "task-output-contract-linked" ((Get-JsonString $candidate "taskOutputContract") -eq $ContractPath) "blocker" "Candidate template must link yolovision-task-output-contract.json."
    Add-ValidationItem $items "case-task-in-contract" ($contractTaskMap.ContainsKey($task)) "blocker" "Candidate task must exist in yolovision-task-output-contract.json."
    Add-ValidationItem $items "runtime-proof-state-known" ($runtimeProofState -in @("owner-action-required", "owner-backfilled", "validated-real-model-runtime")) "blocker" "runtimeProofState must be owner-action-required, owner-backfilled, or validated-real-model-runtime."
    Add-ValidationItem $items "package-consumer-never-promoted" (-not [bool]$candidate.canPromotePackageConsumerRuntime) "blocker" "YoloVision real asset candidates cannot promote package-consumer-runtime proof."

    $metadata = $candidate.outputMetadata
    $contractEntry = $null
    $contractRequiredMetadata = @()
    $contractProfileHint = $null
    $contractArticleEntrypoints = @()
    if ($contractTaskMap.ContainsKey($task)) {
        $contractEntry = $contractTaskMap[$task]
        $contractRequiredMetadata = @($contractEntry.requiredMetadata | ForEach-Object { [string]$_ })
        $contractProfileHint = Get-JsonString $contractEntry "tensorRtExecProfileHint"
        $contractArticleEntrypoints = @($contractEntry.articleEntrypoints | ForEach-Object { [string]$_ })
    }

    Add-ValidationItem $items "contract-required-metadata-present" ($contractRequiredMetadata.Count -ge 4) "blocker" "Contract requiredMetadata must be present for task $task."
    Add-ValidationItem $items "contract-profile-hint-present" (-not [string]::IsNullOrWhiteSpace($contractProfileHint)) "blocker" "Contract TensorRtExec profile hint must be present for task $task."
    Add-ValidationItem $items "contract-article-entrypoint-present" ($contractArticleEntrypoints.Count -ge 1) "blocker" "Contract article entrypoint must be present for task $task."
    Add-ValidationItem $items "tensorrtexec-profile-hint-aligned" ((Get-JsonString $candidate.commands "tensorRtExecBuildCommand").Contains($contractProfileHint)) "blocker" "TensorRtExec build command must include the contract profile hint for task $task."

    foreach ($metadataName in $contractRequiredMetadata) {
        $metadataCovered = Test-ContractMetadataCoverage $candidate $metadata $task $metadataName
        Add-ValidationItem $items ("contract-required-metadata-" + $metadataName) $metadataCovered "blocker" "Task $task must map contract requiredMetadata.$metadataName in the candidate template."
    }

    switch ($task) {
        "pose" {
            Add-ValidationItem $items "pose-keypoint-count-present" (Test-JsonProperty $metadata "keypointCount") "blocker" "Pose candidate must include outputMetadata.keypointCount."
            Add-ValidationItem $items "pose-keypoint-layout-present" (Test-JsonProperty $metadata "keypointLayout") "blocker" "Pose candidate must include outputMetadata.keypointLayout."
            Add-ValidationItem $items "pose-score-field-present" (Test-JsonProperty $metadata "keypointScoreField") "blocker" "Pose candidate must include outputMetadata.keypointScoreField."
        }
        "obb" {
            Add-ValidationItem $items "obb-angle-unit-present" (Test-JsonProperty $metadata "angleUnit") "blocker" "OBB candidate must include outputMetadata.angleUnit."
            Add-ValidationItem $items "obb-rotated-box-layout-present" (Test-JsonProperty $metadata "rotatedBoxLayout") "blocker" "OBB candidate must include outputMetadata.rotatedBoxLayout."
            Add-ValidationItem $items "obb-coordinate-space-present" (Test-JsonProperty $metadata "coordinateSpace") "blocker" "OBB candidate must include outputMetadata.coordinateSpace."
        }
        "cls" {
            Add-ValidationItem $items "cls-topk-present" (Test-JsonProperty $metadata "topK") "blocker" "Classification candidate must include outputMetadata.topK."
            Add-ValidationItem $items "cls-score-field-present" (Test-JsonProperty $metadata "classScoreField") "blocker" "Classification candidate must include outputMetadata.classScoreField."
            Add-ValidationItem $items "cls-labels-required-present" (Test-JsonProperty $metadata "labelsRequired") "blocker" "Classification candidate must include outputMetadata.labelsRequired."
        }
        "sem" {
            Add-ValidationItem $items "sem-map-shape-present" (Test-JsonProperty $metadata "semanticMapShape") "blocker" "Semantic segmentation candidate must include outputMetadata.semanticMapShape."
            Add-ValidationItem $items "sem-class-map-layout-present" (Test-JsonProperty $metadata "classMapLayout") "blocker" "Semantic segmentation candidate must include outputMetadata.classMapLayout."
            Add-ValidationItem $items "sem-palette-required-present" (Test-JsonProperty $metadata "paletteRequired") "blocker" "Semantic segmentation candidate must include outputMetadata.paletteRequired."
        }
    }

    $modelSha = Get-JsonString $candidate.model "sha256"
    $labelsSha = Get-JsonString $candidate.labels "sha256"
    $imageSha = Get-JsonString $candidate.input "imageSha256"
    $preprocessedSha = Get-JsonString $candidate.input "preprocessedTensorSha256"
    $runLogSha = Get-JsonString $candidate.proofChecklist "runLogSha256"

    if ([string]::IsNullOrWhiteSpace($runLogSha) -and $null -ne $candidate.proofChecklist.requiredHashes) {
        $runLogSha = Get-JsonString $candidate.proofChecklist "actualRunLogSha256"
    }

    $hashes = [ordered]@{
        modelSha256 = $modelSha
        labelsSha256 = $labelsSha
        imageSha256 = $imageSha
        preprocessedTensorSha256 = $preprocessedSha
        runLogSha256 = $runLogSha
    }

    foreach ($entry in $hashes.GetEnumerator()) {
        $isTemplatePlaceholder = Test-Placeholder $entry.Value
        $isRealHash = Test-Sha256 $entry.Value
        $ok = $isTemplatePlaceholder -or $isRealHash
        Add-ValidationItem $items ("hash-format-or-placeholder-" + $entry.Key) $ok "blocker" "$($entry.Key) must be a 64-character SHA256 when backfilled, or an owner-required placeholder while template-only."
    }

    $requiredHashes = @()
    if ($null -ne $candidate.proofChecklist.requiredHashes) {
        $requiredHashes = @($candidate.proofChecklist.requiredHashes | ForEach-Object { [string]$_ })
    }

    foreach ($requiredHash in @("modelSha256", "labelsSha256", "imageSha256", "preprocessedTensorSha256", "runLogSha256")) {
        Add-ValidationItem $items ("required-hash-listed-" + $requiredHash) ($requiredHashes -contains $requiredHash) "required" "proofChecklist.requiredHashes must list $requiredHash."
    }

    $evidenceLines = @()
    if ($null -ne $candidate.proofChecklist.requiredEvidenceLines) {
        $evidenceLines = @($candidate.proofChecklist.requiredEvidenceLines | ForEach-Object { [string]$_ })
    }

    $hasPassedLine = $false
    foreach ($line in $evidenceLines) {
        if ($line.Contains("YoloVision Passed=True")) {
            $hasPassedLine = $true
            break
        }
    }

    Add-ValidationItem $items "expected-evidence-passed-line" $hasPassedLine "blocker" "Expected/required evidence lines must include YoloVision Passed=True."

    $stdoutSummary = Get-JsonString $candidate.proofChecklist "stdoutSummary"
    $stderrSummary = Get-JsonString $candidate.proofChecklist "stderrSummary"
    $hasStdout = -not (Test-Placeholder $stdoutSummary)
    $hasStderr = -not (Test-Placeholder $stderrSummary)
    Add-ValidationItem $items "stdout-summary-real-or-placeholder" ((Test-Placeholder $stdoutSummary) -or $hasStdout) "required" "stdoutSummary must remain owner-required or contain real stdout summary."
    Add-ValidationItem $items "stderr-summary-real-or-placeholder" ((Test-Placeholder $stderrSummary) -or $hasStderr) "required" "stderrSummary must remain owner-required/no-stderr placeholder or contain real stderr summary."

    $allRealHashes = (Test-Sha256 $modelSha) -and (Test-Sha256 $labelsSha) -and (Test-Sha256 $imageSha) -and (Test-Sha256 $preprocessedSha) -and (Test-Sha256 $runLogSha)
    $hasRealEvidenceText = $hasStdout -and $hasStderr -and $hasPassedLine
    $isTemplateOnly = $runtimeProofState -eq "owner-action-required" -or $proofClassification -eq "template-only"
    $canPromoteReal = $allRealHashes -and $hasRealEvidenceText -and -not $isTemplateOnly -and [bool]$candidate.canPromoteRealModelRuntime

    Add-ValidationItem $items "template-cannot-promote-real-model-runtime" (-not ($isTemplateOnly -and [bool]$candidate.canPromoteRealModelRuntime)) "blocker" "Template-only / owner-action-required candidates cannot promote real-model-runtime."
    Add-ValidationItem $items "real-runtime-promotion-requires-all-hashes" ((-not [bool]$candidate.canPromoteRealModelRuntime) -or $allRealHashes) "blocker" "canPromoteRealModelRuntime requires real model, labels, image, preprocessed tensor, and run log SHA256 hashes."
    Add-ValidationItem $items "real-runtime-promotion-requires-stdout-stderr" ((-not [bool]$candidate.canPromoteRealModelRuntime) -or $hasRealEvidenceText) "blocker" "canPromoteRealModelRuntime requires stdout/stderr summaries and YoloVision Passed=True evidence."

    $failedBlockerCount = @($items | Where-Object { $_.severity -eq "blocker" -and -not $_.passed }).Count
    $failedActionRequiredCount = @($items | Where-Object { -not $_.passed }).Count
    $validationState = if ($failedBlockerCount -gt 0) {
        "invalid"
    }
    elseif ($canPromoteReal) {
        "validated-real-model-runtime"
    }
    elseif ($isTemplateOnly) {
        "owner-action-required"
    }
    else {
        "owner-backfill-incomplete"
    }

    $recordValidationItems = [object[]]@($items.ToArray())
    $records.Add([ordered]@{
        inputPath = $path
        resolvedPath = $resolvedPath
        contractPath = $ContractPath
        contractRequiredMetadata = [object[]]$contractRequiredMetadata
        tensorRtExecProfileHint = $contractProfileHint
        sampleName = (Get-JsonString $candidate "sampleName")
        family = (Get-JsonString $candidate "family")
        task = $task
        runtimeProofState = $runtimeProofState
        proofClassification = $proofClassification
        validationState = $validationState
        failedBlockerCount = $failedBlockerCount
        failedActionRequiredCount = $failedActionRequiredCount
        canPromoteRealModelRuntime = $canPromoteReal
        canPromotePackageConsumerRuntime = $false
        validationItems = $recordValidationItems
    }) | Out-Null

    foreach ($item in $items) { $allItems.Add($item) | Out-Null }
}

$totalFailedBlockers = @($allItems | Where-Object { $_.severity -eq "blocker" -and -not $_.passed }).Count
$totalFailedActionRequired = @($allItems | Where-Object { -not $_.passed }).Count
$allRealPromotable = $records.Count -gt 0 -and @($records | Where-Object { -not $_.canPromoteRealModelRuntime }).Count -eq 0

$result = [ordered]@{
    recordKind = "yolovision-real-asset-candidate-validation"
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    contractPath = $ContractPath
    contractTaskCount = $contractTasks.Count
    performsPublish = $false
    canPublishPublicly = $false
    canPromotePackageConsumerRuntime = $false
    canPromoteRealModelRuntime = $allRealPromotable
    validationState = if ($totalFailedBlockers -gt 0) { "invalid" } elseif ($allRealPromotable) { "validated-real-model-runtime" } else { "owner-action-required" }
    failedBlockerCount = $totalFailedBlockers
    failedActionRequiredCount = $totalFailedActionRequired
    records = [object[]]@($records.ToArray())
}

$resolvedOutput = Resolve-RepoPath $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutput
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$json = $result | ConvertTo-Json -Depth 32
Set-Content -LiteralPath $resolvedOutput -Value $json -Encoding UTF8
Write-Host "Wrote $resolvedOutput"

if ($Strict -and $totalFailedBlockers -gt 0) {
    throw "YoloVision real asset candidate validation failed with $totalFailedBlockers blocker(s)."
}
