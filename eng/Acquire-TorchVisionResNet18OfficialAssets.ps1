[CmdletBinding()]
param(
  [string]$AssetDirectory = "",
  [string]$ModelDirectory = "",
  [string]$PythonPath = "python",
  [switch]$AllowDownload,
  [switch]$ExportOnnx
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repositoryRoot
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Confirm-Asset {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int64]$ExpectedLength,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256,
    [Parameter(Mandatory = $true)][string]$Name
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Name was not found: $Path"
  }
  $item = Get-Item -LiteralPath $Path
  if ($item.Length -ne $ExpectedLength) {
    throw "$Name length mismatch. Expected $ExpectedLength, actual $($item.Length): $Path"
  }
  $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualSha256 -ne $ExpectedSha256) {
    throw "$Name SHA256 mismatch. Expected $ExpectedSha256, actual ${actualSha256}: $Path"
  }
  return $actualSha256
}

$resolvedAssetDirectory = if ([string]::IsNullOrWhiteSpace($AssetDirectory)) {
  Join-Path $workspaceRoot "downloads\resnet18-torchvision-v0.25.0\source"
} else { [IO.Path]::GetFullPath($AssetDirectory) }
$resolvedModelDirectory = if ([string]::IsNullOrWhiteSpace($ModelDirectory)) {
  Join-Path $workspaceRoot "models\Classification\resnet18-torchvision-v0.25.0"
} else { [IO.Path]::GetFullPath($ModelDirectory) }
New-Item -ItemType Directory -Force -Path $resolvedAssetDirectory, $resolvedModelDirectory | Out-Null

$weightsPath = Join-Path $resolvedAssetDirectory "resnet18-f37072fd.pth"
if (-not (Test-Path -LiteralPath $weightsPath -PathType Leaf)) {
  if (-not $AllowDownload) {
    throw "ResNet18 weights are missing. Re-run with -AllowDownload: $weightsPath"
  }
  Invoke-WebRequest `
    -Uri "https://download.pytorch.org/models/resnet18-f37072fd.pth" `
    -OutFile $weightsPath
}
Confirm-Asset `
  -Path $weightsPath `
  -ExpectedLength 46830571L `
  -ExpectedSha256 "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec" `
  -Name "torchvision ResNet18 weights" | Out-Null

$onnxPath = Join-Path $resolvedModelDirectory "resnet18-imagenet1k-v1.onnx"
$labelsPath = Join-Path $resolvedModelDirectory "imagenet1k.names"
$reportPath = Join-Path $resolvedModelDirectory "resnet18-onnx-export.json"
if ($ExportOnnx) {
  & $PythonPath (Join-Path $PSScriptRoot "Export-ClassificationResNet18Onnx.py") `
    --weights $weightsPath `
    --onnx $onnxPath `
    --labels $labelsPath `
    --report $reportPath
  if ($LASTEXITCODE -ne 0) {
    throw "ResNet18 ONNX export failed with exit code $LASTEXITCODE."
  }
}

$onnxReady = Test-Path -LiteralPath $onnxPath -PathType Leaf
$onnxSha256 = if ($onnxReady) {
  Confirm-Asset `
    -Path $onnxPath `
    -ExpectedLength 46748553L `
    -ExpectedSha256 "ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903" `
    -Name "exported ResNet18 ONNX"
} else { "" }

$record = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "torchvision-resnet18-official-asset-acquisition"
  model = "ResNet18 IMAGENET1K_V1"
  torchvisionTag = "v0.25.0"
  torchvisionCommit = "8ac84ee75afb1c327902156b5336f56ad63b7e2f"
  weights = [pscustomobject][ordered]@{
    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    path = [IO.Path]::GetFullPath($weightsPath)
    length = 46830571L
    sha256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
  }
  export = [pscustomobject][ordered]@{
    script = "eng/Export-ClassificationResNet18Onnx.py"
    opset = 17
    contract = "images:[1,3,224,224] -> logits:[1,1000]"
    onnxPath = [IO.Path]::GetFullPath($onnxPath)
    onnxReady = $onnxReady
    onnxLength = 46748553L
    onnxSha256 = $onnxSha256
    labelsPath = [IO.Path]::GetFullPath($labelsPath)
  }
  policy = [pscustomobject][ordered]@{
    modelFilesStoredOutsideGitRepository = $true
    uploadsAssets = $false
    performsPublish = $false
    redistributionApprovedForRepository = $false
  }
}

$acquisitionReport = Join-Path $resolvedModelDirectory "asset-acquisition.json"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $acquisitionReport -Encoding utf8
Write-Host "ResNet18 official weights are hash-verified."
Write-Host "ModelDirectory=$resolvedModelDirectory"
Write-Host "OnnxPath=$onnxPath OnnxReady=$onnxReady OnnxSha256=$onnxSha256"
Write-Host "Report=$acquisitionReport"
