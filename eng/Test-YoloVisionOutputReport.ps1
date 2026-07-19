[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string[]] $InputPath = @(
        "samples/YoloVision/examples/yolovision-output-det.example.json",
        "samples/YoloVision/examples/yolovision-output-cls.example.json",
        "samples/YoloVision/examples/yolovision-output-seg.example.json",
        "samples/YoloVision/examples/yolovision-output-obb.example.json",
        "samples/YoloVision/examples/yolovision-output-pose.example.json",
        "samples/YoloVision/examples/yolovision-output-sem.example.json"
    ),

    [Parameter(Mandatory = $false)]
    [string] $OutputPath = "artifacts/yolovision/yolovision-output-report-validation.json",

    [Parameter(Mandatory = $false)]
    [switch] $Strict
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

$supportedTasks = @("det", "cls", "seg", "obb", "pose", "sem")
$supportedFamilies = @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "yolox", "custom")
$supportedInputSourceKinds = @("external-tensor", "synthetic-pattern", "preprocessed-image-tensor")
$sha256Pattern = "^[a-fA-F0-9]{64}$"
$forbiddenSubstitutes = @(
    "build-only",
    "dry-run",
    "template",
    "local feed",
    "ProjectReference",
    'direct `.nupkg`',
    "TensorRtExec report",
    "YoloVision matrix",
    "OnnxToEngine report",
    "readonly diagnostics"
)

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Test-JsonProperty {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name] -and $null -ne $Object.$Name
}

function Get-JsonString {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    if (-not (Test-JsonProperty $Object $Name)) {
        return ""
    }

    return [string]$Object.$Name
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

function Test-PositiveNumber {
    param([AllowNull()] $Value)

    if ($null -eq $Value) {
        return $false
    }

    [double] $number = 0
    return [double]::TryParse([string]$Value, [ref]$number) -and $number -gt 0
}

function Test-NonNegativeNumber {
    param([AllowNull()] $Value)

    if ($null -eq $Value) {
        return $false
    }

    [double] $number = 0
    return [double]::TryParse([string]$Value, [ref]$number) -and $number -ge 0
}

function Test-OptionalSha256 {
    param([AllowNull()] $Value)

    if ($null -eq $Value) {
        return $true
    }

    $text = [string]$Value
    return [string]::IsNullOrWhiteSpace($text) -or ($text -match $sha256Pattern)
}

function Test-TaskPrediction {
    param(
        [AllowNull()] $Prediction,
        [Parameter(Mandatory = $true)] [string] $Task
    )

    if ($null -eq $Prediction) {
        return $false
    }

    if ((Get-JsonString $Prediction "task") -ne $Task) {
        return $false
    }

    switch ($Task) {
        "det" {
            return (Test-JsonProperty $Prediction "box") -and
                (Test-JsonProperty $Prediction "classId") -and
                (Test-JsonProperty $Prediction "className") -and
                (Test-NonNegativeNumber $Prediction.score)
        }
        "cls" {
            return (Test-JsonProperty $Prediction "classId") -and
                (Test-JsonProperty $Prediction "className") -and
                (Test-NonNegativeNumber $Prediction.score)
        }
        "seg" {
            return (Test-JsonProperty $Prediction "box") -and
                (Test-JsonProperty $Prediction "maskShape") -and
                (Test-JsonProperty $Prediction "maskPixelCount") -and
                (Test-JsonProperty $Prediction "maskThreshold")
        }
        "obb" {
            return (Test-JsonProperty $Prediction "center") -and
                (Test-JsonProperty $Prediction "size") -and
                (Test-JsonProperty $Prediction "angle") -and
                (Test-JsonProperty $Prediction "angleUnit") -and
                (Test-JsonProperty $Prediction "angleRange")
        }
        "pose" {
            return (Test-JsonProperty $Prediction "box") -and
                (Test-JsonProperty $Prediction "keypoints") -and
                @($Prediction.keypoints).Count -gt 0
        }
        "sem" {
            return (Test-JsonProperty $Prediction "classCount") -and
                (Test-JsonProperty $Prediction "width") -and
                (Test-JsonProperty $Prediction "height") -and
                (Test-JsonProperty $Prediction "valueCount")
        }
        default {
            return $false
        }
    }
}

$records = New-Object System.Collections.Generic.List[object]
$allItems = New-Object System.Collections.Generic.List[object]

foreach ($path in $InputPath) {
    $resolvedPath = Resolve-RepoPath $path
    $items = New-Object System.Collections.Generic.List[object]

    if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
        Add-ValidationItem $items "file-exists" $false "blocker" "YoloVision output report does not exist: $path"
        $records.Add([ordered]@{
            inputPath = $path
            resolvedPath = $resolvedPath
            validationState = "invalid"
            canPromoteRealModelRuntime = $false
            canPromotePackageConsumerRuntime = $false
            validationItems = [object[]]@($items.ToArray())
        }) | Out-Null
        foreach ($item in $items) { $allItems.Add($item) | Out-Null }
        continue
    }

    $text = Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8
    $report = $text | ConvertFrom-Json -Depth 64
    $task = Get-JsonString $report "task"
    $family = Get-JsonString $report "modelFamily"
    $predictions = @($report.predictions)
    $outputs = @($report.outputs)
    $labels = $null
    if (Test-JsonProperty $report "labels") {
        $labels = $report.labels
    }
    $forbidden = @()
    if (Test-JsonProperty $report "boundary") {
        $forbidden = @($report.boundary.forbiddenSubstitutes | ForEach-Object { [string]$_ })
    }

    Add-ValidationItem $items "file-exists" $true "required" "YoloVision output report exists."
    Add-ValidationItem $items "schema-version" ((Get-JsonString $report "schemaVersion") -eq "yolovision-output.v1") "blocker" "schemaVersion must be yolovision-output.v1."
    Add-ValidationItem $items "task-supported" ($supportedTasks -contains $task) "blocker" "task must be det, cls, seg, obb, pose, or sem."
    Add-ValidationItem $items "family-supported" ($supportedFamilies -contains $family) "blocker" "modelFamily must be a supported YOLO family or custom."
    Add-ValidationItem $items "input-present" (Test-JsonProperty $report "input") "blocker" "input object must be present."
    Add-ValidationItem $items "engine-present" (Test-JsonProperty $report "engine") "blocker" "engine object must be present."
    Add-ValidationItem $items "runtime-present" (Test-JsonProperty $report "runtime") "blocker" "runtime object must be present."
    Add-ValidationItem $items "outputs-present" ($outputs.Count -gt 0) "blocker" "outputs array must contain copied tensor summaries."
    Add-ValidationItem $items "predictions-present" ($predictions.Count -gt 0) "blocker" "predictions array must contain at least one task prediction."
    Add-ValidationItem $items "labels-sha256-well-formed" ((-not (Test-JsonProperty $report "labels")) -or (Test-OptionalSha256 (Get-JsonString $labels "sha256"))) "blocker" "labels.sha256 must be empty or a 64-character SHA256 when labels metadata is present."
    Add-ValidationItem $items "input-source-kind-supported" ($supportedInputSourceKinds -contains (Get-JsonString $report.input "sourceKind")) "blocker" "input.sourceKind must be synthetic-pattern, external-tensor, or preprocessed-image-tensor."
    Add-ValidationItem $items "input-sha256-well-formed" (Test-OptionalSha256 (Get-JsonString $report.input "sha256")) "blocker" "input.sha256 must be empty or a 64-character SHA256."
    Add-ValidationItem $items "input-image-sha256-well-formed" ((-not (Test-JsonProperty $report.input "image")) -or (Test-OptionalSha256 (Get-JsonString $report.input.image "sha256"))) "blocker" "input.image.sha256 must be empty or a 64-character SHA256 when image metadata is present."
    Add-ValidationItem $items "input-preprocessed-tensor-sha256-well-formed" ((-not (Test-JsonProperty $report.input "preprocessedTensor")) -or (Test-OptionalSha256 (Get-JsonString $report.input.preprocessedTensor "sha256"))) "blocker" "input.preprocessedTensor.sha256 must be empty or a 64-character SHA256 when preprocessing metadata is present."
    Add-ValidationItem $items "output-value-sha256-well-formed" ((@($outputs | Where-Object { (Test-JsonProperty $_ "valueSha256") -and -not (Test-OptionalSha256 (Get-JsonString $_ "valueSha256")) }).Count) -eq 0) "blocker" "outputs[].valueSha256 must be empty or a 64-character SHA256 when present."
    Add-ValidationItem $items "input-size-positive" ((Test-PositiveNumber $report.input.width) -and (Test-PositiveNumber $report.input.height)) "blocker" "input width and height must be positive."
    Add-ValidationItem $items "engine-shape-present" (@($report.engine.inputShape).Count -ge 4) "blocker" "engine.inputShape must contain the runtime tensor shape."
    Add-ValidationItem $items "boundary-present" (Test-JsonProperty $report "boundary") "blocker" "boundary object must be present."
    Add-ValidationItem $items "boundary-not-runtime-proof" ((Test-JsonProperty $report "boundary") -and -not [bool]$report.boundary.isRuntimeProof) "blocker" "boundary.isRuntimeProof must be false."
    Add-ValidationItem $items "boundary-evidence-kind" ((Get-JsonString $report.boundary "evidenceKind") -eq "YoloVision output schema; readonly diagnostics; not runtime proof") "blocker" "boundary.evidenceKind must keep the non-proof classification."
    Add-ValidationItem $items "forbidden-substitutes-complete" (($forbiddenSubstitutes | Where-Object { $forbidden -notcontains $_ }).Count -eq 0) "blocker" "forbiddenSubstitutes must list every non-proof substitute."
    Add-ValidationItem $items "no-yolodet" (-not $text.Contains("YoloDet", [System.StringComparison]::OrdinalIgnoreCase)) "blocker" "YoloVision output records must not refer to the retired YoloDet sample name."
    Add-ValidationItem $items "task-predictions-match" ((@($predictions | Where-Object { -not (Test-TaskPrediction $_ $task) }).Count) -eq 0) "blocker" "Every prediction must match the declared task and required metadata."

    $failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
    $state = if ($failedBlockers.Count -eq 0) { "yolovision-output-report-ready" } else { "invalid" }

    $records.Add([ordered]@{
        inputPath = $path
        resolvedPath = $resolvedPath
        task = $task
        modelFamily = $family
        validationState = $state
        canPromoteRealModelRuntime = $false
        canPromotePackageConsumerRuntime = $false
        validationItems = [object[]]@($items.ToArray())
    }) | Out-Null

    foreach ($item in $items) { $allItems.Add($item) | Out-Null }
}

$failedBlockerCount = @($allItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "yolovision-output-report-ready" } else { "invalid" }

$result = [ordered]@{
    recordKind = "yolovision-output-report-validation"
    validationState = $validationState
    failedBlockers = $failedBlockerCount
    inputCount = $InputPath.Count
    tasks = [object[]]@($records | ForEach-Object { $_.task } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    proofBoundary = "YoloVision output reports are owner-review/golden-output artifacts. They are not runtime proof or package-consumer proof."
    records = [object[]]@($records.ToArray())
}

$resolvedOutput = Resolve-RepoPath $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutput
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$result | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $resolvedOutput -Encoding utf8

Write-Host "ValidationState=$validationState FailedBlockers=$failedBlockerCount Records=$($InputPath.Count)"
Write-Host "YoloVision output report validation written:"
Write-Host "  Json=$resolvedOutput"

if ($Strict.IsPresent -and $failedBlockerCount -gt 0) {
    exit 1
}
