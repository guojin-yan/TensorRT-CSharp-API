[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$CatalogPath,
  [string]$OutputPath,
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$outerRoot = [IO.Path]::GetFullPath((Split-Path -Parent $RepositoryRoot))
if ([string]::IsNullOrWhiteSpace($CatalogPath)) {
  $CatalogPath = Join-Path $RepositoryRoot "docs\articles\zh-cn\publication-catalog.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "artifacts\article-publication\technical-article-completeness.json"
}
$CatalogPath = [IO.Path]::GetFullPath($CatalogPath)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$utf8 = [Text.UTF8Encoding]::new($false)

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-RelativePathCompat {
  param(
    [Parameter(Mandatory = $true)][string]$BasePath,
    [Parameter(Mandatory = $true)][string]$TargetPath
  )

  $nativeMethod = [IO.Path].GetMethod("GetRelativePath", [type[]]@([string], [string]))
  if ($null -ne $nativeMethod) {
    return [IO.Path]::GetRelativePath($BasePath, $TargetPath)
  }

  $baseWithSeparator = $BasePath.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
  $baseUri = [Uri]::new($baseWithSeparator)
  $targetUri = [Uri]::new($TargetPath)
  return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', [IO.Path]::DirectorySeparatorChar)
}

function Normalize-PathText {
  param([Parameter(Mandatory = $true)][string]$Path)
  return $Path.Replace('\', '/').TrimStart('./')
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not (Test-Path -LiteralPath $CatalogPath -PathType Leaf)) {
  throw "Technical article publication catalog does not exist: $CatalogPath"
}
$catalog = Get-Content -LiteralPath $CatalogPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([int]$catalog.schemaVersion -ne 2 -or [string]$catalog.recordKind -ne "technical-article-publication-catalog") {
  throw "Unsupported technical article publication catalog contract."
}
if ([bool]$catalog.publicPublicationAuthorized -or [bool]$catalog.performsPublish) {
  throw "The content completeness catalog must not authorize or perform publication."
}

$articleRoot = Join-Path $RepositoryRoot "docs\articles\zh-cn"
$markdownFiles = @(Get-ChildItem -LiteralPath $articleRoot -Recurse -File -Filter *.md)
$markdownWithImages = @(
  foreach ($file in $markdownFiles) {
    $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8
    if ($content -match '!\[[^\]]*\]\([^\)]+\)|<img\s') {
      $file.FullName
    }
  }
)

$results = @(
  foreach ($article in @($catalog.articles)) {
    $failures = [Collections.Generic.List[string]]::new()
    $articlePath = Resolve-RepositoryPath -Path ([string]$article.path)
    if (-not (Test-Path -LiteralPath $articlePath -PathType Leaf)) {
      $failures.Add("article-missing")
    }

    $content = if (Test-Path -LiteralPath $articlePath -PathType Leaf) {
      Get-Content -LiteralPath $articlePath -Raw -Encoding utf8
    }
    else {
      ""
    }
    $requiredHeadingGroups = @(
      @("## 本文使用的项目与库"),
      @("## 模型获取与许可证"),
      @("## ONNX 转换与暂存"),
      @("## 使用公开包准备应用", "## 使用公开包准备案例", "## 创建本地包消费项目"),
      @("## 编写程序入口"),
      @("## 编译并运行"),
      @("## 已验证结果"),
      @("## 复查与边界")
    )
    foreach ($headingGroup in $requiredHeadingGroups) {
      $matchedHeading = @($headingGroup | Where-Object {
        $content.IndexOf($_, [StringComparison]::Ordinal) -ge 0
      }).Count -gt 0
      if (-not $matchedHeading) {
        $failures.Add("missing-heading:$($headingGroup -join '|')")
      }
    }

    $hasTerminalRuntimeStatement =
      $content.IndexOf("终端截图来自", [StringComparison]::Ordinal) -ge 0 -and
      $content.IndexOf("真实运行", [StringComparison]::Ordinal) -ge 0 -and
      $content.IndexOf("stdout", [StringComparison]::OrdinalIgnoreCase) -ge 0
    $hasTensorRtRuntimeStatement =
      $content.IndexOf("真实 TensorRT", [StringComparison]::OrdinalIgnoreCase) -ge 0
    $hasSharedResultSourceStatement =
      $content.IndexOf("两张图都来自同一次", [StringComparison]::Ordinal) -ge 0 -or
      $content.IndexOf("同次运行输出", [StringComparison]::Ordinal) -ge 0 -or
      $content.IndexOf("结果图使用同一个", [StringComparison]::Ordinal) -ge 0
    if (-not $hasTerminalRuntimeStatement -or
        -not $hasTensorRtRuntimeStatement -or
        -not $hasSharedResultSourceStatement) {
      $failures.Add("missing-runtime-and-annotated-image-source-statement")
    }

    $machineSpecificAbsolutePaths = @([regex]::Matches($content, '(?im)(?:[A-Z]:\\|/Users/[^/\s]+/|/home/[^/\s]+/)'))
    if ($machineSpecificAbsolutePaths.Count -gt [int]$catalog.completenessRule.maximumMachineSpecificAbsolutePathCount) {
      $failures.Add("too-many-machine-specific-absolute-paths:$($machineSpecificAbsolutePaths.Count)")
    }

    $evidencePath = Resolve-RepositoryPath -Path ([string]$article.realExecutionEvidence)
    if (-not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
      $failures.Add("real-execution-evidence-missing")
    }
    else {
      $evidence = Get-Content -LiteralPath $evidencePath -Raw -Encoding utf8 | ConvertFrom-Json
      if ([string]$evidence.proofClassification -notin @("real-model-runtime", "local-package-consumer-runtime", "package-consumer-runtime", "post-publish-runtime")) {
        $failures.Add("unsupported-real-execution-classification")
      }
    }

    $markdownImagePaths = @(
      foreach ($match in [regex]::Matches($content, '!\[[^\]]*\]\((?<path>[^\)\s]+)(?:\s+"[^"]*")?\)')) {
        $rawPath = [Uri]::UnescapeDataString($match.Groups["path"].Value)
        if (-not $rawPath.StartsWith("http", [StringComparison]::OrdinalIgnoreCase)) {
          [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $articlePath) $rawPath))
        }
      }
    )
    $resultImages = @($article.resultImages)
    if ($resultImages.Count -lt [int]$catalog.completenessRule.minimumResultImageCount) {
      $failures.Add("insufficient-result-images")
    }
    $resultImageRoles = @($resultImages | ForEach-Object { [string]$_.role })
    foreach ($requiredRole in @($catalog.completenessRule.requiredResultImageRoles)) {
      if ($resultImageRoles -notcontains [string]$requiredRole) {
        $failures.Add("required-result-image-role-missing:$requiredRole")
      }
    }
    if (@($resultImageRoles | Group-Object | Where-Object Count -ne 1).Count -gt 0) {
      $failures.Add("result-image-role-not-unique")
    }
    foreach ($image in $resultImages) {
      $imagePath = Resolve-RepositoryPath -Path ([string]$image.path)
      $extension = [IO.Path]::GetExtension($imagePath).ToLowerInvariant()
      if ($extension -notin @($catalog.completenessRule.allowedImageExtensions)) {
        $failures.Add("unsupported-result-image-extension:$extension")
      }
      if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
        $failures.Add("result-image-missing:$($image.path)")
        continue
      }
      if ((Get-Sha256 -Path $imagePath) -ne [string]$image.sha256) {
        $failures.Add("result-image-sha256-mismatch:$($image.path)")
      }
      if ($markdownImagePaths -notcontains $imagePath) {
        $failures.Add("result-image-not-referenced-by-article:$($image.path)")
      }
      if ([string]::IsNullOrWhiteSpace([string]$image.source)) {
        $failures.Add("result-image-source-missing:$($image.path)")
      }
    }

    $visualAssetEvidencePath = Resolve-RepositoryPath -Path ([string]$article.visualAssetEvidence)
    if (-not (Test-Path -LiteralPath $visualAssetEvidencePath -PathType Leaf)) {
      $failures.Add("visual-asset-evidence-missing")
    }
    else {
      $visualAssetEvidence = Get-Content -LiteralPath $visualAssetEvidencePath -Raw -Encoding utf8 | ConvertFrom-Json
      if ([string]$visualAssetEvidence.recordKind -ne "technical-article-visual-assets") {
        $failures.Add("visual-asset-evidence-kind-invalid")
      }
      if ([string]$visualAssetEvidence.articleId -ne [string]$article.id) {
        $failures.Add("visual-asset-evidence-article-id-mismatch")
      }
      foreach ($propertyName in @("descriptionUrl", "downloadUrl", "downloadedSha256", "license", "licenseUrl")) {
        if ([string]::IsNullOrWhiteSpace([string]$visualAssetEvidence.sourceImage.$propertyName)) {
          $failures.Add("visual-source-field-missing:$propertyName")
        }
      }
      if (-not [bool]$visualAssetEvidence.sourceImage.publicRedistributionPermittedByLicense) {
        $failures.Add("visual-source-public-redistribution-not-permitted")
      }
      if ([bool]$visualAssetEvidence.modelFilesTrackedByGit -or
          [bool]$visualAssetEvidence.uploadsModelFiles -or
          [bool]$visualAssetEvidence.performsPublish) {
        $failures.Add("visual-asset-evidence-boundary-invalid")
      }
      foreach ($image in $resultImages) {
        $evidenceImage = @($visualAssetEvidence.resultImages | Where-Object { [string]$_.role -eq [string]$image.role })
        if ($evidenceImage.Count -ne 1 -or
            [string]$evidenceImage[0].path -ne [string]$image.path -or
            [string]$evidenceImage[0].sha256 -ne [string]$image.sha256) {
          $failures.Add("visual-asset-evidence-image-mismatch:$($image.role)")
        }
      }
    }

    $model = $article.model
    foreach ($propertyName in @("sourceUrl", "pinnedRevision", "license", "conversionCommand", "onnxWorkspacePath", "onnxSha256")) {
      if ([string]::IsNullOrWhiteSpace([string]$model.$propertyName)) {
        $failures.Add("model-field-missing:$propertyName")
      }
    }
    $normalizedContent = $content.Replace('\', '/')
    foreach ($requiredText in @([string]$model.sourceUrl, [string]$model.conversionCommand, [string]$model.onnxWorkspacePath, [string]$model.onnxSha256)) {
      if ($normalizedContent.IndexOf((Normalize-PathText -Path $requiredText), [StringComparison]::Ordinal) -lt 0) {
        $failures.Add("model-contract-not-documented:$requiredText")
      }
    }
    $onnxPath = [IO.Path]::GetFullPath((Join-Path $outerRoot ([string]$model.onnxWorkspacePath)))
    if (-not (Test-Path -LiteralPath $onnxPath -PathType Leaf)) {
      $failures.Add("outer-model-missing")
    }
    elseif ((Get-Sha256 -Path $onnxPath) -ne [string]$model.onnxSha256) {
      $failures.Add("outer-model-sha256-mismatch")
    }
    if ([bool]$model.trackedByGit -or [bool]$model.uploadsModelFiles) {
      $failures.Add("model-git-or-upload-boundary-invalid")
    }
    if ([bool]$article.publicationAuthorized) {
      $failures.Add("article-must-not-self-authorize-publication")
    }
    if ([string]::IsNullOrWhiteSpace([string]$article.proofBoundary)) {
      $failures.Add("proof-boundary-missing")
    }

    [pscustomobject][ordered]@{
      id = [string]$article.id
      path = [string]$article.path
      classification = [string]$article.classification
      status = [string]$article.status
      resultImageCount = $resultImages.Count
      resultImageRoles = $resultImageRoles
      machineSpecificAbsolutePathCount = $machineSpecificAbsolutePaths.Count
      failureCount = $failures.Count
      failures = @($failures)
      passed = ($failures.Count -eq 0)
      publicPublicationAuthorized = $false
    }
  }
)

$failed = @($results | Where-Object { -not $_.passed })
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "technical-article-completeness-validation"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  catalogPath = (Get-RelativePathCompat -BasePath $RepositoryRoot -TargetPath $CatalogPath).Replace('\', '/')
  repositoryMarkdownCount = $markdownFiles.Count
  repositoryMarkdownWithImageCount = $markdownWithImages.Count
  catalogArticleCount = $results.Count
  catalogPassedCount = @($results | Where-Object { $_.passed }).Count
  catalogFailedCount = $failed.Count
  defaultUnlistedClassification = [string]$catalog.defaultClassification
  articles = $results
  passed = ($failed.Count -eq 0 -and (-not $Strict -or $results.Count -gt 0))
  publicPublicationAuthorized = $false
  performsPublish = $false
  boundary = "Content completeness validation only; not article publication, public-package proof, post-publish proof, Owner acceptance, or release proof."
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
[IO.File]::WriteAllText($OutputPath, ($report | ConvertTo-Json -Depth 12) + "`n", $utf8)
Write-Host "TechnicalArticleCompleteness Markdown=$($report.repositoryMarkdownCount) WithImages=$($report.repositoryMarkdownWithImageCount) Catalog=$($report.catalogArticleCount) Passed=$($report.catalogPassedCount) Failed=$($report.catalogFailedCount)"
Write-Host "PublicPublicationAuthorized=False PerformsPublish=False"
Write-Host "Report=$OutputPath"
if (-not $report.passed) {
  throw "Technical article completeness validation failed for $($report.catalogFailedCount) catalog article(s)."
}
