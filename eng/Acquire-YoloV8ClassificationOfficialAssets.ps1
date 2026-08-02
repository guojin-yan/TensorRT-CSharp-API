[CmdletBinding()]
param(
    [string]$AssetRoot = "E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0",
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
        id = "yolov8n-cls-pt"
        fileName = "yolov8n-cls.pt"
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt"
        fallbackUrl = ""
        githubReleaseAssetId = 195719213
        expectedLength = 5563076
        expectedSha256 = "11fa19f2aea79bc960d680a13f82f22105982b325eb9e17a4a5e1a9f8245980a"
        hashProvenance = "repository-pinned-after-first-download-from-the-exact-official-release-asset-id; upstream Release did not publish a digest"
    },
    [ordered]@{
        id = "ultralytics-imagenet-yaml"
        fileName = "ImageNet.yaml"
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/cfg/datasets/ImageNet.yaml"
        fallbackUrl = "https://cdn.jsdelivr.net/gh/ultralytics/ultralytics@6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/cfg/datasets/ImageNet.yaml"
        expectedLength = 42507
        expectedSha256 = "3f9b74af030d657da2bbba28064779c528bbe66eba71c0733b45e20400461a15"
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
labels = list(document["map"].values())
if len(labels) != 1000:
    raise ValueError(f"Expected 1000 ImageNet map entries, got {len(labels)}")
labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")
'@

$ppmPath = Join-Path $derivedDirectory "bus.ppm"
$labelsPath = Join-Path $derivedDirectory "imagenet-yolov8n-cls.names"
& $PythonPath -c $deriveCode `
    (Join-Path $sourceDirectory "bus.jpg") `
    (Join-Path $sourceDirectory "ImageNet.yaml") `
    $ppmPath `
    $labelsPath
if ($LASTEXITCODE -ne 0) {
    throw "Python derivation failed with exit code $LASTEXITCODE."
}
Assert-FileContract -Path $ppmPath -ExpectedLength 2624416 -ExpectedSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688"
Assert-FileContract -Path $labelsPath -ExpectedLength 10511 -ExpectedSha256 "dcc60e7297d33ea2b0efeab10074e4ac07d3fdd702fb1fb7ace169ee684240dd"

$report = [ordered]@{
    schemaVersion = 1
    recordKind = "yolovision-yolov8n-cls-official-asset-acquisition-report"
    generatedUtc = [DateTimeOffset]::UtcNow.ToString("O")
    assetRoot = $resolvedRoot
    assets = $assets
    derived = @(
        [ordered]@{ id = "bus-ppm"; localPath = $ppmPath; expectedLength = 2624416; expectedSha256 = "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" },
        [ordered]@{ id = "imagenet-model-map-labels"; localPath = $labelsPath; classCount = 1000; expectedLength = 10511; expectedSha256 = "dcc60e7297d33ea2b0efeab10074e4ac07d3fdd702fb1fb7ace169ee684240dd" }
    )
    boundaries = [ordered]@{
        performsExport = $false
        performsRuntime = $false
        performsPublish = $false
        uploadsAssets = $false
        publicRedistributionOwnerApproval = $false
    }
}
$reportPath = Join-Path $reportDirectory "classification-acquisition-report.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding utf8

$markdownPath = Join-Path $reportDirectory "classification-acquisition-report.md"
@(
    "# YOLOv8n Classification Official Asset Acquisition",
    "",
    "- Asset root: ``$resolvedRoot``",
    "- Assets: $($assets.Count)",
    "- Derived labels: 1000 exact ImageNet map entries",
    "- Export/runtime/publish: false/false/false",
    "- Report: ``$reportPath``"
) | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "YoloV8ClassificationAssets=$resolvedRoot"
Write-Output "AcquisitionReport=$reportPath"
Write-Output "PerformsExport=False PerformsRuntime=False PerformsPublish=False UploadsAssets=False"
