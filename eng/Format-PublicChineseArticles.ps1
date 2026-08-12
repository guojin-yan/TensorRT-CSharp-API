[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
  $RepositoryRoot = (Resolve-Path $RepositoryRoot).Path
}

function Resolve-RepoPath {
  param([string]$RelativePath)
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot ($RelativePath -replace '/', [IO.Path]::DirectorySeparatorChar)))
}

function Get-CompatibleRelativePath {
  param(
    [string]$BasePath,
    [string]$TargetPath
  )

  $normalizedBase = [IO.Path]::GetFullPath($BasePath)
  if (-not $normalizedBase.EndsWith([IO.Path]::DirectorySeparatorChar.ToString(), [StringComparison]::Ordinal)) {
    $normalizedBase += [IO.Path]::DirectorySeparatorChar
  }

  $baseUri = [Uri]::new($normalizedBase)
  $targetUri = [Uri]::new([IO.Path]::GetFullPath($TargetPath))
  return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}

function Get-ProjectPreface {
  param([string]$Module)

  $programUrl = switch ($Module) {
    "02-samples" { "https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples" }
    "03-applications" { "https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications" }
    default { "https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0" }
  }

  return @(
    "<!-- public-article-project-preface:start -->",
    "TensorRT CSharp API v4.0 是一个面向 C#/.NET 开发者的 TensorRT 与 CUDA 工程化接口项目。它把 NVIDIA 原生运行时、生成式绑定、C++ Bridge、托管对象模型和可验证的示例程序组织成一条完整链路，使使用者可以在熟悉的 .NET 项目中完成 Engine 构建、反序列化、ExecutionContext 管理、CUDA 内存操作、异步流同步和结果校验。项目的目标不是隐藏 TensorRT 的概念，而是把这些概念转换为有明确生命周期、所有权和错误边界的 C# API。",
    '',
    "4.0.0 是一次完整重构后的正式版本。核心接口、Bridge 边界、Runtime 包命名、样例目录和验证方式都以 4.x 设计为准，不能把 3.x 的类型名、旧包名或旧 DLL 目录直接复制到新项目。托管包只提供项目接口和自有 Bridge；TensorRT、CUDA、cuDNN、显卡驱动以及对应许可证仍由使用者按目标平台安装和管理。",
    '',
    "单篇文章也应能够独立阅读：读者可以先从项目入口确认源码和包，再根据本文的程序路径准备依赖，最后用输出中的状态、计数、Shape、哈希或结果图片判断流程是否真的完成。对于尚未具备兼容 GPU 的环境，本文会把静态检查、期望输出和真实运行结果分开标记，不把帮助命令或 build-only 结果包装成推理成功。",
    '',
    "项目、包和源码入口（以下地址保留明文，便于复制到不完整支持 Markdown 链接的平台）：",
    '',
    "项目主页：",
    '',
    '```text',
    "https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0",
    '```',
    '',
    "核心 NuGet：",
    '',
    '```text',
    "https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0",
    '```',
    '',
    "Runtime Bridge 包列表：",
    '',
    '```text',
    "https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance",
    '```',
    '',
    "运行库清单：",
    '',
    '```text',
    "https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json",
    '```',
    '',
    "### 程序出处与输出说明",
    '',
    "本文涉及的程序、脚本或命令均以仓库中的实现为准；对应源码入口：",
    '',
    '```text',
    $programUrl,
    '```',
    '',
    '运行示例必须同时给出程序输出和判定标准。终端中的 `status`、`Ready`、`Bound`、`enqueueCount`、`OutputMatch`、进程退出码、报告文件或结果图片分别说明不同层次的事实；只有明确写出这些结果，读者才能区分程序启动、Engine 构建、GPU enqueue 和业务结果语义。',
    "<!-- public-article-project-preface:end -->"
  )
}

function Convert-ArticleContent {
  param(
    [string[]]$Lines,
    [string]$Module,
    [string]$SourcePath,
    [string]$ArticleTitle
  )

  $withoutPreface = [System.Collections.Generic.List[string]]::new()
  $insidePreface = $false
  $insideLayout = $false
  $skipBlankAfterLayout = $false
  foreach ($sourceLine in $Lines) {
    if ($sourceLine -eq '<!-- public-article-project-preface:start -->') { $insidePreface = $true; continue }
    if ($sourceLine -eq '<!-- public-article-project-preface:end -->') { $insidePreface = $false; continue }
    if ($sourceLine -eq '<!-- public-article-layout:start -->') {
      while ($withoutPreface.Count -gt 0 -and [string]::IsNullOrWhiteSpace($withoutPreface[$withoutPreface.Count - 1])) {
        $withoutPreface.RemoveAt($withoutPreface.Count - 1)
      }
      $insideLayout = $true
      continue
    }
    if ($sourceLine -eq '<!-- public-article-layout:end -->') {
      $insideLayout = $false
      $skipBlankAfterLayout = $true
      continue
    }
    if ($skipBlankAfterLayout -and [string]::IsNullOrWhiteSpace($sourceLine)) { continue }
    $skipBlankAfterLayout = $false
    if (-not $insidePreface -and -not $insideLayout) { $withoutPreface.Add($sourceLine) }
  }
  $Lines = $withoutPreface.ToArray()

  $result = [System.Collections.Generic.List[string]]::new()
  $articleDirectory = Split-Path (Resolve-RepoPath $SourcePath) -Parent
  $inFence = $false
  $prefaceAdded = $false
  $skipSection = $false
  $preface = Get-ProjectPreface -Module $Module

  foreach ($originalLine in $Lines) {
    $line = $originalLine

    if ($line -match '^\s*(```+|~~~+)') {
      $inFence = -not $inFence
      $result.Add($line)
      continue
    }

    if (-not $inFence -and $line -match '^##\s+(?:\d+(?:\.\d+)*[\.、]?\s*)?(下一步|下一篇|后续文章)') {
      $skipSection = $true
      continue
    }

    if ($skipSection) {
      if ($line -match '^##\s+') {
        $skipSection = $false
      }
      else {
        continue
      }
    }

    if (-not $inFence -and $line -match '^##\s+(?:\d+\.\s*)?前言\s*$' -and -not $prefaceAdded) {
      $result.Add($line)
      foreach ($prefaceLine in $preface) { $result.Add($prefaceLine) }
      $prefaceAdded = $true
      continue
    }

    if ($line -match '^\s*(?:flowchart|graph)\s+LR\s*$') {
      $line = $line -replace '\bLR\b', 'TD'
    }

    if (-not $inFence) {
      $line = [regex]::Replace(
        $line,
        '!\[([^\]]*)\]\((?!https?://)([^)\s]+)(?:\s+"[^"]*")?\)',
        {
          param($match)
          $alt = $match.Groups[1].Value.Replace('"', '&quot;')
          $target = $match.Groups[2].Value
          return '<img src="{0}" alt="{1}" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />' -f $target, $alt
        }
      )

      $line = [regex]::Replace(
        $line,
        '(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)',
        {
          param($match)
          return '{0}：<{1}>' -f $match.Groups[1].Value, $match.Groups[2].Value
        }
      )

      $line = [regex]::Replace(
        $line,
        '(?<!!)\[([^\]]+)\]\(([^)\s]+)\)',
        {
          param($match)
          $label = $match.Groups[1].Value
          $target = $match.Groups[2].Value
          if ($target -match '^(https?://|mailto:|#)') {
            return $match.Value
          }

          $parts = $target.Split('#', 2)
          $resolvedTarget = [IO.Path]::GetFullPath((Join-Path $articleDirectory $parts[0]))
          $repoRelative = (Get-CompatibleRelativePath -BasePath $RepositoryRoot -TargetPath $resolvedTarget).Replace('\', '/')
          if ($repoRelative.StartsWith('../', [StringComparison]::Ordinal)) {
            throw "Link '$target' in '$SourcePath' points outside the repository."
          }

          $encodedPath = (($repoRelative.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/')
          $url = "https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/$encodedPath"
          if ($parts.Count -eq 2) { $url += "#$($parts[1])" }
          return '{0}：<{1}>' -f $label, $url
        }
      )
    }

    if ($line -match '^#\s+') {
      $line = "# $ArticleTitle"
      $result.Add($line)
      $result.Add('')
      $result.Add('<!-- public-article-layout:start -->')
      $result.Add('<style>')
      $result.Add('.content article { min-width: 0; overflow-wrap: anywhere; }')
      $result.Add('.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }')
      $result.Add('.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }')
      $result.Add('.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }')
      $result.Add('.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }')
      $result.Add('</style>')
      $result.Add('<!-- public-article-layout:end -->')
      $result.Add('')
      continue
    }
    elseif ($line -match '^#{2,6}\s+') {
      $line = $line -replace 'TensorRtSharp4\.0', 'TensorRT CSharp API v4.0'
    }

    $result.Add($line)
  }

  if (-not $prefaceAdded) {
    throw "Canonical article '$SourcePath' has no '## 前言' heading."
  }

  return $result.ToArray()
}

function Number-Headings {
  param([string[]]$Lines)

  $result = [System.Collections.Generic.List[string]]::new()
  $inFence = $false
  $h2 = 0
  $h3 = 0
  $h4 = 0

  foreach ($line in $Lines) {
    if ($line -match '^\s*(```+|~~~+)') {
      $inFence = -not $inFence
      $result.Add($line)
      continue
    }

    if (-not $inFence -and $line -match '^(#{2,4})\s+(.+?)\s*$') {
      $level = $Matches[1].Length
      $title = $Matches[2] -replace '^\d+(?:\.\d+)*[\.、]?\s*', ''
      $title = $title -replace '^[一二三四五六七八九十]+、\s*', ''
      if ($level -eq 2) {
        $h2++
        $h3 = 0
        $h4 = 0
        $line = "## $h2. $title"
      }
      elseif ($level -eq 3) {
        if ($h2 -eq 0) { throw "Found H3 before H2: '$line'" }
        $h3++
        $h4 = 0
        $line = "### $h2.$h3 $title"
      }
      else {
        if ($h2 -eq 0 -or $h3 -eq 0) { throw "Found H4 before H3: '$line'" }
        $h4++
        $line = "#### $h2.$h3.$h4 $title"
      }
    }

    $result.Add($line)
  }

  return $result.ToArray()
}

function Add-ArticleDeclaration {
  param(
    [string[]]$Lines,
    [string]$SourcePath
  )

  $clean = [System.Collections.Generic.List[string]]::new()
  $insideDeclaration = $false
  foreach ($line in $Lines) {
    if ($line -eq '<!-- public-article-declaration:start -->') { $insideDeclaration = $true; continue }
    if ($line -eq '<!-- public-article-declaration:end -->') { $insideDeclaration = $false; continue }
    if (-not $insideDeclaration) { $clean.Add($line) }
  }

  while ($clean.Count -gt 0 -and [string]::IsNullOrWhiteSpace($clean[$clean.Count - 1])) {
    $clean.RemoveAt($clean.Count - 1)
  }

  $articleDirectory = Split-Path (Resolve-RepoPath $SourcePath) -Parent
  $imagePath = Resolve-RepoPath 'docs/images/personal-contact-banner-v6-zh.png'
  $relativeImage = (Get-CompatibleRelativePath -BasePath $articleDirectory -TargetPath $imagePath).Replace('\', '/')

  $declaration = @(
    '',
    '<!-- public-article-declaration:start -->',
    '## 文章声明',
    '',
    '### 1. 开源协议声明',
    '作者所有开源项目代码均遵循 Apache License 2.0 开源协议。',
    '特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。',
    '',
    '### 2. 代码开发与质量说明',
    'AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。',
    '安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。',
    '技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。',
    '测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。',
    '',
    '### 3. 免责声明（重要）',
    '请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。',
    '',
    '### 4. 代码开源范围',
    ('本项目承诺核心逻辑代码完全开源，但上述提到的{0}第三方库{1}的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。' -f [char]0x201C, [char]0x201D),
    '',
    '### 5. 社区与反馈',
    '尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。',
    '',
    ('<img src="{0}" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />' -f $relativeImage),
    '<!-- public-article-declaration:end -->'
  )
  foreach ($line in $declaration) { $clean.Add($line) }
  return $clean.ToArray()
}

$indexPath = Resolve-RepoPath 'docs/articles/zh-cn/article-index.json'
$index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json
$imagePath = Resolve-RepoPath 'docs/images/personal-contact-banner-v6-zh.png'
if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
  throw "Missing contact image '$imagePath'."
}

foreach ($article in $index.articles) {
  $sourcePath = [string]$article.sourcePath
  $fullPath = Resolve-RepoPath $sourcePath
  $lines = Get-Content -LiteralPath $fullPath -Encoding UTF8
  $lines = Convert-ArticleContent -Lines $lines -Module ([string]$article.module) -SourcePath $sourcePath -ArticleTitle ([string]$article.title)
  $lines = Add-ArticleDeclaration -Lines $lines -SourcePath $sourcePath
  $lines = Number-Headings -Lines $lines
  [IO.File]::WriteAllLines($fullPath, [string[]]$lines, [Text.UTF8Encoding]::new($false))
}

Write-Host "Formatted $($index.articles.Count) canonical Chinese public articles."
