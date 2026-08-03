[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$EvidencePath,
  [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if ([string]::IsNullOrWhiteSpace($EvidencePath)) {
  $EvidencePath = Join-Path $RepositoryRoot "samples\assets\yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "docs\images\yolovision-lraspp-semantic-local-package-result.png"
}
$EvidencePath = [IO.Path]::GetFullPath($EvidencePath)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

if (-not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) {
  throw "Semantic runtime evidence does not exist: $EvidencePath"
}
$evidence = Get-Content -LiteralPath $EvidencePath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$evidence.recordName -ne "yolovision-lraspp-semantic-local-package-consumer-trt10.11" -or
    [string]$evidence.proofClassification -ne "local-package-consumer-runtime") {
  throw "Unsupported semantic runtime evidence contract."
}
if ([int]$evidence.packageConsumer.packageCount -ne 3 -or
    [int]$evidence.packageConsumer.runtimeExitCode -ne 0 -or
    [int]$evidence.rawTensorReferenceValidation.mismatchCount -ne 0 -or
    [int]$evidence.classIndexArtifactValidation.mismatchCount -ne 0 -or
    -not [bool]$evidence.controlledNegativeValidation.failClosed -or
    -not [bool]$evidence.controlledArtifactIntegrityValidation.failClosed) {
  throw "Semantic runtime evidence is not a passed, fail-closed three-package result."
}
if ([bool]$evidence.proofBoundary.publicPackageProof -or
    [bool]$evidence.proofBoundary.postPublishProof -or
    [bool]$evidence.proofBoundary.publicRedistributionOwnerApproval -or
    [bool]$evidence.proofBoundary.performsPublish -or
    [bool]$evidence.proofBoundary.uploadsAssets) {
  throw "Semantic runtime evidence crosses the approved publication or asset boundary."
}
$invariantCulture = [Globalization.CultureInfo]::InvariantCulture
$maximumAbsoluteError = [double]$evidence.rawTensorReferenceValidation.tensor.maximumAbsoluteError
$maximumAbsoluteErrorText = $maximumAbsoluteError.ToString("0.0000000e+00", $invariantCulture)
$generatedUtc = ([DateTime]$evidence.generatedAtUtc).ToUniversalTime().ToString(
  "yyyy-MM-dd HH:mm:ss 'UTC'",
  $invariantCulture)

Add-Type -AssemblyName System.Drawing

$width = 1600
$height = 1000
$bitmap = [Drawing.Bitmap]::new($width, $height, [Drawing.Imaging.PixelFormat]::Format24bppRgb)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([Drawing.Color]::FromArgb(15, 20, 28))

$fontTitle = [Drawing.Font]::new("Segoe UI Semibold", 34, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$fontSubtitle = [Drawing.Font]::new("Segoe UI", 18, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$fontHeading = [Drawing.Font]::new("Segoe UI Semibold", 21, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$fontMetric = [Drawing.Font]::new("Consolas", 31, [Drawing.FontStyle]::Bold, [Drawing.GraphicsUnit]::Pixel)
$fontBody = [Drawing.Font]::new("Consolas", 17, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$fontSmall = [Drawing.Font]::new("Consolas", 14, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)

$brushPrimary = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(238, 243, 249))
$brushSecondary = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(155, 167, 183))
$brushGreen = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(86, 211, 139))
$brushCyan = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(91, 192, 222))
$brushYellow = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(246, 196, 83))
$brushPanel = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(23, 30, 41))
$brushPanelStrong = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(28, 37, 50))
$brushBackgroundBar = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(78, 91, 109))
$brushDogBar = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(86, 211, 139))
$penBorder = [Drawing.Pen]::new([Drawing.Color]::FromArgb(48, 61, 79), 1)
$penDivider = [Drawing.Pen]::new([Drawing.Color]::FromArgb(57, 70, 88), 1)

function Draw-Text {
  param(
    [string]$Text,
    [Drawing.Font]$Font,
    [Drawing.Brush]$Brush,
    [float]$X,
    [float]$Y
  )
  $graphics.DrawString($Text, $Font, $Brush, $X, $Y)
}

function Draw-Panel {
  param([int]$X, [int]$Y, [int]$Width, [int]$Height, [Drawing.Brush]$Brush = $brushPanel)
  $graphics.FillRectangle($Brush, $X, $Y, $Width, $Height)
  $graphics.DrawRectangle($penBorder, $X, $Y, $Width, $Height)
}

try {
  Draw-Text "YoloVision LRASPP Semantic Segmentation" $fontTitle $brushPrimary 72 48
  Draw-Text "clean local PackageReference consumer / TensorRT 10.11 / CUDA 12.9" $fontSubtitle $brushSecondary 74 98
  Draw-Text "PASS" $fontHeading $brushGreen 1420 62

  Draw-Panel 72 150 700 435
  Draw-Text "EXECUTION CONTRACT" $fontHeading $brushCyan 104 180
  Draw-Text "Packages" $fontBody $brushSecondary 104 232
  Draw-Text "3 isolated local feeds" $fontBody $brushPrimary 330 232
  Draw-Text "References" $fontBody $brushSecondary 104 273
  Draw-Text "0 project / 0 assembly" $fontBody $brushPrimary 330 273
  Draw-Text "Input" $fontBody $brushSecondary 104 314
  Draw-Text "images [1,3,320,320] FP32" $fontBody $brushPrimary 330 314
  Draw-Text "Preprocess" $fontBody $brushSecondary 104 355
  Draw-Text "RGB NCHW / ImageNet mean+std" $fontBody $brushPrimary 330 355
  Draw-Text "Output" $fontBody $brushSecondary 104 396
  Draw-Text "semantic [1,21,320,320]" $fontBody $brushPrimary 330 396
  Draw-Text "GPU" $fontBody $brushSecondary 104 437
  Draw-Text ([string]$evidence.runtimeEnvironment.gpu) $fontBody $brushPrimary 330 437
  Draw-Text "Native deps" $fontBody $brushSecondary 104 478
  Draw-Text "user-installed / bridge-only package" $fontBody $brushPrimary 330 478
  Draw-Text "Runtime exit" $fontBody $brushSecondary 104 519
  Draw-Text "0" $fontMetric $brushGreen 330 507

  Draw-Panel 804 150 724 435
  Draw-Text "VERIFIED RESULT" $fontHeading $brushCyan 836 180
  Draw-Text "2,150,400" $fontMetric $brushPrimary 836 226
  Draw-Text "logits compared" $fontBody $brushSecondary 1065 238
  Draw-Text "0" $fontMetric $brushGreen 836 284
  Draw-Text "raw mismatches" $fontBody $brushSecondary 900 296
  Draw-Text "102,400" $fontMetric $brushPrimary 836 342
  Draw-Text "argmax pixels compared" $fontBody $brushSecondary 1020 354
  Draw-Text "0" $fontMetric $brushGreen 836 400
  Draw-Text "class-index mismatches" $fontBody $brushSecondary 900 412
  Draw-Text ("max abs error  " + $maximumAbsoluteErrorText) $fontBody $brushYellow 836 471

  $pixelCount = [double]$evidence.classIndexArtifactValidation.pixelCount
  $backgroundCount = [double](@($evidence.classIndexArtifactValidation.histogram | Where-Object { [int]$_.classId -eq 0 })[0].pixelCount)
  $dogCount = [double](@($evidence.classIndexArtifactValidation.histogram | Where-Object { [int]$_.classId -eq 12 })[0].pixelCount)
  $barX = 836
  $barY = 521
  $barWidth = 620
  $backgroundWidth = [int][Math]::Round($barWidth * $backgroundCount / $pixelCount)
  $graphics.FillRectangle($brushBackgroundBar, $barX, $barY, $backgroundWidth, 18)
  $graphics.FillRectangle($brushDogBar, $barX + $backgroundWidth, $barY, $barWidth - $backgroundWidth, 18)
  $backgroundText = "background {0:N0} ({1:0.0}%)" -f $backgroundCount, (100.0 * $backgroundCount / $pixelCount)
  $dogText = "dog {0:N0} ({1:0.0}%)" -f $dogCount, (100.0 * $dogCount / $pixelCount)
  Draw-Text $backgroundText $fontSmall $brushSecondary 836 546
  Draw-Text $dogText $fontSmall $brushGreen 1240 546

  Draw-Panel 72 617 1456 220 $brushPanelStrong
  Draw-Text "FAIL-CLOSED NEGATIVE CHECKS" $fontHeading $brushCyan 104 647
  Draw-Text "Raw reference mutation" $fontBody $brushSecondary 104 704
  Draw-Text "exit 1 / mismatch 1 / first index 0" $fontBody $brushGreen 410 704
  Draw-Text "Class-index byte mutation" $fontBody $brushSecondary 104 750
  Draw-Text "exit 1 / artifact-sha256 rejected" $fontBody $brushGreen 410 750
  Draw-Text "Public package / post-publish / asset redistribution" $fontBody $brushSecondary 820 704
  Draw-Text "NOT CLAIMED" $fontHeading $brushYellow 1335 698
  Draw-Text "CUDA, cuDNN and TensorRT remain external host dependencies" $fontSmall $brushSecondary 820 756

  $graphics.DrawLine($penDivider, 72, 877, 1528, 877)
  Draw-Text ("class-index sha256  " + [string]$evidence.classIndexArtifactValidation.classIndexSha256) $fontSmall $brushSecondary 74 899
  Draw-Text ("evidence generated  " + $generatedUtc) $fontSmall $brushSecondary 74 930
  Draw-Text "Image generated only from the committed runtime evidence; no source image or model is embedded." $fontSmall $brushSecondary 74 960

  $outputDirectory = Split-Path -Parent $OutputPath
  New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
  $bitmap.Save($OutputPath, [Drawing.Imaging.ImageFormat]::Png)
}
finally {
  foreach ($item in @(
    $fontTitle, $fontSubtitle, $fontHeading, $fontMetric, $fontBody, $fontSmall,
    $brushPrimary, $brushSecondary, $brushGreen, $brushCyan, $brushYellow,
    $brushPanel, $brushPanelStrong, $brushBackgroundBar, $brushDogBar,
    $penBorder, $penDivider, $graphics, $bitmap
  )) {
    if ($null -ne $item) {
      $item.Dispose()
    }
  }
}

$hash = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
$file = Get-Item -LiteralPath $OutputPath
Write-Host "SemanticArticleResultImage=$OutputPath"
Write-Host "Dimensions=${width}x${height} Length=$($file.Length) SHA256=$hash"
Write-Host "PublicPublicationAuthorized=False PerformsPublish=False UploadsAssets=False"
