[CmdletBinding()]
param(
  [string]$AssetDirectory = "",
  [string]$ModelDirectory = "",
  [string]$PythonPath = "python",
  [string]$ReferenceOutputDirectory = "artifacts\yolovision\semantic-lraspp-reference",
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

function Resolve-PathFromBase {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Base
  )

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }
  return [IO.Path]::GetFullPath((Join-Path $Base $Path))
}

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

if ([string]::IsNullOrWhiteSpace($AssetDirectory)) {
  $resolvedAssetDirectory = Join-Path $workspaceRoot "downloads\lraspp-mobilenet-v3-large-torchvision-v0.25.0\source"
}
else {
  $resolvedAssetDirectory = Resolve-PathFromBase -Path $AssetDirectory -Base $workspaceRoot
}

if ([string]::IsNullOrWhiteSpace($ModelDirectory)) {
  $resolvedModelDirectory = Join-Path $workspaceRoot "models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0"
}
else {
  $resolvedModelDirectory = Resolve-PathFromBase -Path $ModelDirectory -Base $workspaceRoot
}

$resolvedReferenceOutputDirectory = Resolve-PathFromBase -Path $ReferenceOutputDirectory -Base $repositoryRoot
New-Item -ItemType Directory -Force -Path $resolvedAssetDirectory, $resolvedModelDirectory, $resolvedReferenceOutputDirectory | Out-Null

$assets = @(
  [pscustomobject][ordered]@{
    id = "torchvision-lraspp-mobilenet-v3-large-weights"
    fileName = "lraspp_mobilenet_v3_large-d234d4ea.pth"
    destination = "model"
    url = "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth"
    length = 13097061L
    sha256 = "d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280"
  },
  [pscustomobject][ordered]@{
    id = "torchvision-voc-category-metadata"
    fileName = "_meta.py"
    destination = "asset"
    url = "https://raw.githubusercontent.com/pytorch/vision/8ac84ee75afb1c327902156b5336f56ad63b7e2f/torchvision/models/_meta.py"
    length = 28875L
    sha256 = "7eaa5e401b1ff441186e602943c4342cac8cf7f505c238ab7882399b00dbc096"
  },
  [pscustomobject][ordered]@{
    id = "torchvision-license"
    fileName = "LICENSE.torchvision.txt"
    destination = "asset"
    url = "https://raw.githubusercontent.com/pytorch/vision/8ac84ee75afb1c327902156b5336f56ad63b7e2f/LICENSE"
    length = 1517L
    sha256 = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"
  },
  [pscustomobject][ordered]@{
    id = "pytorch-hub-dog-input"
    fileName = "dog.jpg"
    destination = "asset"
    url = "https://raw.githubusercontent.com/pytorch/hub/c7895df70c7767403e36f82786d6b611b7984557/images/dog.jpg"
    length = 661378L
    sha256 = "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"
  }
)

$results = [Collections.Generic.List[object]]::new()
foreach ($asset in $assets) {
  $directory = if ($asset.destination -eq "model") { $resolvedModelDirectory } else { $resolvedAssetDirectory }
  $path = Join-Path $directory $asset.fileName
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    if (-not $AllowDownload) {
      throw "Asset '$($asset.id)' is missing. Re-run with -AllowDownload: $path"
    }
    Invoke-WebRequest -Uri $asset.url -OutFile $path
  }
  $actualSha256 = Confirm-Asset -Path $path -ExpectedLength $asset.length -ExpectedSha256 $asset.sha256 -Name $asset.id
  $results.Add([pscustomobject][ordered]@{
    id = $asset.id
    path = [IO.Path]::GetFullPath($path)
    sourceUrl = $asset.url
    length = $asset.length
    sha256 = $actualSha256
    redistributionApprovedForRepository = $false
  }) | Out-Null
}

$weightsPath = Join-Path $resolvedModelDirectory "lraspp_mobilenet_v3_large-d234d4ea.pth"
$imagePath = Join-Path $resolvedAssetDirectory "dog.jpg"
$onnxPath = Join-Path $resolvedModelDirectory "lraspp-mobilenet-v3-large-320.onnx"
if ($ExportOnnx) {
  $referenceScript = Join-Path $PSScriptRoot "Invoke-YoloVisionSemanticReference.py"
  & $PythonPath $referenceScript `
    --weights $weightsPath `
    --image $imagePath `
    --onnx $onnxPath `
    --output-directory $resolvedReferenceOutputDirectory `
    --export-onnx
  if ($LASTEXITCODE -ne 0) {
    throw "LRASPP ONNX export/reference command failed with exit code $LASTEXITCODE."
  }
}

$onnxReady = Test-Path -LiteralPath $onnxPath -PathType Leaf
$onnxSha256 = if ($onnxReady) {
  Confirm-Asset `
    -Path $onnxPath `
    -ExpectedLength 12879801L `
    -ExpectedSha256 "3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8" `
    -Name "exported LRASPP ONNX"
}
else { "" }
$record = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "torchvision-lraspp-official-asset-acquisition"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  model = "LRASPP MobileNetV3 Large"
  torchvisionTag = "v0.25.0"
  torchvisionCommit = "8ac84ee75afb1c327902156b5336f56ad63b7e2f"
  hubInputCommit = "c7895df70c7767403e36f82786d6b611b7984557"
  modelDirectory = [IO.Path]::GetFullPath($resolvedModelDirectory)
  assetDirectory = [IO.Path]::GetFullPath($resolvedAssetDirectory)
  assets = @($results)
  onnx = [pscustomobject][ordered]@{
    path = [IO.Path]::GetFullPath($onnxPath)
    exists = $onnxReady
    sha256 = $onnxSha256
    expectedLength = 12879801L
    expectedSha256 = "3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8"
    expectedContract = "images:[1,3,320,320] -> semantic:[1,21,320,320]"
  }
  policy = [pscustomobject][ordered]@{
    modelFilesStoredOutsideGitRepository = $true
    uploadsAssets = $false
    performsPublish = $false
    redistributionApprovedForRepository = $false
  }
}

$reportPath = Join-Path $resolvedReferenceOutputDirectory "asset-acquisition.json"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Host "LRASPP official assets are hash-verified."
Write-Host "ModelDirectory=$resolvedModelDirectory"
Write-Host "OnnxPath=$onnxPath OnnxReady=$onnxReady OnnxSha256=$onnxSha256"
Write-Host "Report=$reportPath"
