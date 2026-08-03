[CmdletBinding()]
param(
  [string]$WorkspaceRoot = "",
  [string]$TensorRtMnistModelPath = "",
  [string]$ReportPath = "artifacts\demo-models\inventory-validation.json",
  [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resolvedWorkspaceRoot = if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
  Split-Path -Parent $repositoryRoot
} else {
  [IO.Path]::GetFullPath($WorkspaceRoot)
}
$modelRoot = [IO.Path]::GetFullPath((Join-Path $resolvedWorkspaceRoot "models"))
$repositoryPrefix = [IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if ($modelRoot.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "The demo model root must stay outside the Git repository: $modelRoot"
}

$inventoryPath = Join-Path $repositoryRoot "samples\assets\demo-model-inventory.json"
$inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
$sourceById = @{
  "classification-resnet18-imagenet1k-v1" = Join-Path $modelRoot "Classification\resnet18-torchvision-v0.25.0\resnet18-imagenet1k-v1.onnx"
  "onnxtoengine-nvidia-mnist-opset8" = if ([string]::IsNullOrWhiteSpace($TensorRtMnistModelPath)) {
    $tensorRtRoot = if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_TENSORRT_ROOT)) { $env:JYPPX_TENSORRT_ROOT } else { $env:TENSORRT_PATH }
    if ([string]::IsNullOrWhiteSpace($tensorRtRoot)) {
      Join-Path $modelRoot "OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx"
    }
    else {
      Join-Path $tensorRtRoot "data\mnist\mnist.onnx"
    }
  } else { [IO.Path]::GetFullPath($TensorRtMnistModelPath) }
  "yolovision-yolov8n-detection-v8.3.0" = Join-Path $resolvedWorkspaceRoot "downloads\yolov8n-det-ultralytics-v8.3.0\source\yolov8n.onnx"
  "yolovision-yolov10n-detection-v1.1" = Join-Path $resolvedWorkspaceRoot "downloads\yolov10-agpl\source\yolov10n.onnx"
  "yolovision-yolox-s-detection-0.1.1rc0" = Join-Path $resolvedWorkspaceRoot "downloads\yolox-apache\source\yolox_s.onnx"
  "yolovision-yolov8n-classification-v8.3.0" = Join-Path $resolvedWorkspaceRoot "downloads\yolov8n-cls-ultralytics-v8.3.0\source\yolov8n-cls.onnx"
  "yolovision-yolov8n-instance-segmentation-v8.3.0" = Join-Path $resolvedWorkspaceRoot "downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.onnx"
  "yolovision-yolov8n-pose-v8.3.0" = Join-Path $resolvedWorkspaceRoot "downloads\yolov8n-pose-ultralytics-v8.3.0\source\yolov8n-pose.onnx"
  "yolovision-yolov8n-obb-v8.3.0" = Join-Path $resolvedWorkspaceRoot "downloads\yolov8n-obb-ultralytics-v8.3.0\source\yolov8n-obb.onnx"
  "yolovision-lraspp-mobilenet-v3-large-v0.25.0" = Join-Path $resolvedWorkspaceRoot "downloads\lraspp-mobilenet-v3-large-torchvision-v0.25.0\derived\lraspp-mobilenet-v3-large-320.onnx"
}

function Get-FileIdentity {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int64]$ExpectedLength,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return [pscustomobject][ordered]@{ exists = $false; passed = $false; length = 0L; sha256 = "" }
  }
  $item = Get-Item -LiteralPath $Path
  $sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  return [pscustomobject][ordered]@{
    exists = $true
    passed = $item.Length -eq $ExpectedLength -and $sha256 -eq $ExpectedSha256
    length = $item.Length
    sha256 = $sha256
  }
}

$results = [Collections.Generic.List[object]]::new()
foreach ($model in $inventory.models) {
  $relativePath = [string]$model.onnx.workspaceRelativePath
  $targetPath = [IO.Path]::GetFullPath((Join-Path $resolvedWorkspaceRoot ($relativePath -replace '/', '\')))
  $modelPrefix = $modelRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  if (-not $targetPath.StartsWith($modelPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Inventory target escapes the workspace model root: $relativePath"
  }

  $expectedLength = [int64]$model.onnx.expectedLength
  $expectedSha256 = ([string]$model.onnx.sha256).ToLowerInvariant()
  $targetIdentity = Get-FileIdentity -Path $targetPath -ExpectedLength $expectedLength -ExpectedSha256 $expectedSha256
  $copied = $false
  if (-not $targetIdentity.passed -and -not $VerifyOnly) {
    $sourcePath = [string]$sourceById[[string]$model.id]
    $sourceIdentity = Get-FileIdentity -Path $sourcePath -ExpectedLength $expectedLength -ExpectedSha256 $expectedSha256
    if (-not $sourceIdentity.passed) {
      throw "No hash-matching source is available for '$($model.id)': $sourcePath"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null
    if (-not [string]::Equals($sourcePath, $targetPath, [StringComparison]::OrdinalIgnoreCase)) {
      Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
      $copied = $true
    }
    $targetIdentity = Get-FileIdentity -Path $targetPath -ExpectedLength $expectedLength -ExpectedSha256 $expectedSha256
  }

  $results.Add([pscustomobject][ordered]@{
    id = [string]$model.id
    path = $targetPath
    expectedLength = $expectedLength
    expectedSha256 = $expectedSha256
    actualLength = $targetIdentity.length
    actualSha256 = $targetIdentity.sha256
    copied = $copied
    passed = $targetIdentity.passed
  }) | Out-Null
}

$failed = @($results | Where-Object { -not $_.passed })
$resolvedReportPath = if ([IO.Path]::IsPathRooted($ReportPath)) {
  [IO.Path]::GetFullPath($ReportPath)
} else {
  [IO.Path]::GetFullPath((Join-Path $repositoryRoot $ReportPath))
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedReportPath) | Out-Null
[pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "demo-model-inventory-local-validation"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  workspaceRoot = $resolvedWorkspaceRoot
  modelRoot = $modelRoot
  verifyOnly = [bool]$VerifyOnly
  modelCount = $results.Count
  failedCount = $failed.Count
  passed = $failed.Count -eq 0
  models = @($results)
  policy = [pscustomobject][ordered]@{
    modelRootOutsideGitRepository = $true
    uploadsAssets = $false
    performsPublish = $false
  }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resolvedReportPath -Encoding utf8

Write-Host "DemoModelInventory Models=$($results.Count) Failed=$($failed.Count) Passed=$($failed.Count -eq 0)"
Write-Host "ModelRoot=$modelRoot"
Write-Host "Report=$resolvedReportPath"
if ($failed.Count -ne 0) {
  foreach ($item in $failed) {
    Write-Error "Model validation failed: $($item.id) path=$($item.path)"
  }
  exit 1
}
