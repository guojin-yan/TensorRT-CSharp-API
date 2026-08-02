[CmdletBinding()]
param(
    [string]$AssetRoot = "E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-det-ultralytics-v8.3.0",
    [string]$PythonPath = $env:JYPPX_YOLO_PYTHON
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Test-DriveIsNotC {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($fullPath)
    if ([string]::Equals($root, "C:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Official model, image, tensor, and reference assets must not be written to C:. Path=$fullPath"
    }
    return $fullPath
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-FileContract {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$ExpectedLength,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $ExpectedLength) {
        throw "Asset length mismatch. Path=$Path Expected=$ExpectedLength Actual=$($item.Length)"
    }
    $actualSha256 = Get-Sha256 -Path $Path
    if (-not [string]::Equals($actualSha256, $ExpectedSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset SHA256 mismatch. Path=$Path Expected=$ExpectedSha256 Actual=$actualSha256"
    }
}

function Invoke-PinnedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [string]$FallbackUrl,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$ExpectedLength,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (Test-Path -LiteralPath $Destination) {
        try {
            Assert-FileContract -Path $Destination -ExpectedLength $ExpectedLength -ExpectedSha256 $ExpectedSha256
            return
        }
        catch {
            Remove-Item -LiteralPath $Destination -Force
        }
    }

    $urls = @($Url)
    if (-not [string]::IsNullOrWhiteSpace($FallbackUrl)) {
        $urls += $FallbackUrl
    }
    $lastError = $null
    foreach ($candidate in $urls) {
        try {
            Invoke-WebRequest -Uri $candidate -OutFile $Destination -MaximumRedirection 10
            Assert-FileContract -Path $Destination -ExpectedLength $ExpectedLength -ExpectedSha256 $ExpectedSha256
            return
        }
        catch {
            $lastError = $_
            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
        }
    }
    throw "Unable to acquire pinned asset $Destination. LastError=$lastError"
}

$resolvedRoot = Test-DriveIsNotC -Path $AssetRoot
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = "python"
}

$sourceDirectory = Join-Path $resolvedRoot "source"
$derivedDirectory = Join-Path $resolvedRoot "derived"
$reportDirectory = Join-Path $resolvedRoot "reports"
New-Item -ItemType Directory -Force -Path $sourceDirectory, $derivedDirectory, $reportDirectory | Out-Null

$assets = @(
    [ordered]@{
        id = "yolov8n-det-pt"
        fileName = "yolov8n.pt"
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        fallbackUrl = "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt"
        githubReleaseAssetId = 195719301
        expectedLength = 6549796
        expectedSha256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
        hashProvenance = "repository-pinned-after-first-download-from-the-exact-official-release-asset-id; upstream Release did not publish a digest"
    },
    [ordered]@{
        id = "ultralytics-coco-yaml"
        fileName = "coco.yaml"
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/cfg/datasets/coco.yaml"
        fallbackUrl = "https://cdn.jsdelivr.net/gh/ultralytics/ultralytics@6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/cfg/datasets/coco.yaml"
        expectedLength = 2586
        expectedSha256 = "bd6f98a2e18775c39a4d5214080c87fcb163d367c18a2fcf2609371bab00c0b8"
        hashProvenance = "source-commit-content-hash-pinned"
    },
    [ordered]@{
        id = "ultralytics-license-v8.3.0"
        fileName = "LICENSE.ultralytics-6e43d1e1.txt"
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/6e43d1e1e5db72afbf686dee6745669bcb124b0a/LICENSE"
        fallbackUrl = "https://cdn.jsdelivr.net/gh/ultralytics/ultralytics@6e43d1e1e5db72afbf686dee6745669bcb124b0a/LICENSE"
        expectedLength = 34523
        expectedSha256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
        hashProvenance = "source-commit-content-hash-pinned"
    },
    [ordered]@{
        id = "ultralytics-bus-jpg"
        fileName = "bus.jpg"
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/assets/bus.jpg"
        fallbackUrl = "https://cdn.jsdelivr.net/gh/ultralytics/ultralytics@6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/assets/bus.jpg"
        expectedLength = 137419
        expectedSha256 = "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63"
        hashProvenance = "source-commit-content-hash-pinned"
    }
)

foreach ($asset in $assets) {
    $destination = Join-Path $sourceDirectory $asset.fileName
    Invoke-PinnedDownload `
        -Url $asset.url `
        -FallbackUrl $asset.fallbackUrl `
        -Destination $destination `
        -ExpectedLength $asset.expectedLength `
        -ExpectedSha256 $asset.expectedSha256
    $asset.localPath = $destination
}

$deriveCode = @'
import sys
from pathlib import Path
import yaml
from PIL import Image

image_path, yaml_path, ppm_path, labels_path = map(Path, sys.argv[1:])
Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
names = document["names"]
labels = [str(names[index]) for index in range(len(names))] if isinstance(names, dict) else list(names)
if len(labels) != 80:
    raise ValueError(f"Expected 80 COCO class names, got {len(labels)}")
labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")
'@

$ppmPath = Join-Path $derivedDirectory "bus.ppm"
$labelsPath = Join-Path $derivedDirectory "coco.names"
& $PythonPath -c $deriveCode `
    (Join-Path $sourceDirectory "bus.jpg") `
    (Join-Path $sourceDirectory "coco.yaml") `
    $ppmPath `
    $labelsPath
if ($LASTEXITCODE -ne 0) {
    throw "Python derivation failed with exit code $LASTEXITCODE."
}
Assert-FileContract -Path $ppmPath -ExpectedLength 2624416 -ExpectedSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688"
Assert-FileContract -Path $labelsPath -ExpectedLength 621 -ExpectedSha256 "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a"

$report = [ordered]@{
    schemaVersion = 1
    recordKind = "yolovision-yolov8n-det-official-asset-acquisition-report"
    generatedUtc = [DateTimeOffset]::UtcNow.ToString("O")
    assetRoot = $resolvedRoot
    assets = $assets
    derived = @(
        [ordered]@{ id = "bus-ppm"; localPath = $ppmPath; expectedLength = 2624416; expectedSha256 = "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" },
        [ordered]@{ id = "coco-labels"; localPath = $labelsPath; classCount = 80; expectedLength = 621; expectedSha256 = "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a" }
    )
    boundaries = [ordered]@{
        performsExport = $false
        performsRuntime = $false
        performsPublish = $false
        uploadsAssets = $false
        publicRedistributionOwnerApproval = $false
    }
}
$reportPath = Join-Path $reportDirectory "detection-acquisition-report.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding utf8

$markdownPath = Join-Path $reportDirectory "detection-acquisition-report.md"
@(
    "# YOLOv8n Detection Official Asset Acquisition",
    "",
    "- Asset root: ``$resolvedRoot``",
    "- Assets: $($assets.Count)",
    "- Derived labels: 80 exact COCO class names",
    "- Export/runtime/publish: false/false/false",
    "- Report: ``$reportPath``"
) | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "YoloV8DetectionAssets=$resolvedRoot"
Write-Output "AcquisitionReport=$reportPath"
Write-Output "PerformsExport=False PerformsRuntime=False PerformsPublish=False UploadsAssets=False"
