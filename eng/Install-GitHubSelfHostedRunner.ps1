[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Repository,
  [string]$RunnerName,
  [string]$RunnerRoot,
  [string]$RunnerVersion = "latest",
  [string[]]$RunnerLabels = @("ubuntu-20.04", "tensorrt-csharp"),
  [AllowEmptyString()]
  [string]$RegistrationToken = $env:GITHUB_RUNNER_REGISTRATION_TOKEN,
  [switch]$UseGhRegistrationToken,
  [switch]$Replace,
  [switch]$InstallService,
  [switch]$InstallRunnerDependencies,
  [switch]$NoDownload,
  [switch]$DryRun,
  [string]$RunnerGroup,
  [string]$WorkDirectory = "_work"
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Test-IsLinuxHost {
  return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)
}

function Invoke-CheckedNativeCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$ArgumentList
  )

  Write-Host "> $FilePath $($ArgumentList -join ' ')"
  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
  }
}

function Expand-LabelList {
  param(
    [string[]]$Values
  )

  $labels = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $labels.Add($trimmed)
      }
    }
  }

  @($labels | Select-Object -Unique)
}

function Resolve-RunnerVersion {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Version
  )

  if ($Version -ne "latest") {
    return $Version.TrimStart("v")
  }

  if ($DryRun.IsPresent) {
    return "<latest>"
  }

  $release = Invoke-RestMethod -Uri "https://api.github.com/repos/actions/runner/releases/latest" -Headers @{
    "Accept" = "application/vnd.github+json"
    "User-Agent" = "jyppx-tensorrt-runner-setup"
  }

  return ([string]$release.tag_name).TrimStart("v")
}

function Get-RegistrationTokenFromGh {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryName
  )

  if ($DryRun.IsPresent) {
    Write-Host "Dry-run: would request a short-lived registration token with:"
    Write-Host "  gh api -X POST /repos/$RepositoryName/actions/runners/registration-token --jq .token"
    return "<registration-token>"
  }

  $token = (& gh api -X POST "/repos/$RepositoryName/actions/runners/registration-token" --jq ".token" | Out-String).Trim()
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($token)) {
    throw "Unable to request a GitHub Actions runner registration token for '$RepositoryName'."
  }

  return $token
}

$repositoryParts = $Repository.Split("/", 2)
if ($repositoryParts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($repositoryParts[0]) -or [string]::IsNullOrWhiteSpace($repositoryParts[1])) {
  throw "Repository must use the 'owner/name' format. Value: $Repository"
}

if (-not (Test-IsLinuxHost) -and -not $DryRun.IsPresent) {
  throw "This runner installer is intended to run on Linux. Use -DryRun from other operating systems to inspect the commands."
}

$architecture = "x64"
if (Test-IsLinuxHost) {
  $osArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
  if ($osArchitecture -notin @("x64", "x86_64")) {
    throw "The current runtime package line requires Linux x64. Detected architecture: $osArchitecture"
  }
}

$repoName = $repositoryParts[1]
if ([string]::IsNullOrWhiteSpace($RunnerName)) {
  $hostName = if ([string]::IsNullOrWhiteSpace($env:HOSTNAME)) { "linux" } else { $env:HOSTNAME }
  $RunnerName = "$hostName-$repoName-ubuntu20"
}

if ([string]::IsNullOrWhiteSpace($RunnerRoot)) {
  if (Test-IsLinuxHost) {
    $homeDirectory = if (-not [string]::IsNullOrWhiteSpace($env:HOME)) { $env:HOME } else { (Get-Location).Path }
    $RunnerRoot = "$homeDirectory/actions-runner-tensorrt-csharp"
  }
  else {
    $RunnerRoot = "/opt/actions-runner-tensorrt-csharp"
  }
}

$runnerLabelsExpanded = @(Expand-LabelList -Values $RunnerLabels)
if ($runnerLabelsExpanded.Count -eq 0) {
  throw "At least one custom runner label is required. Use ubuntu-20.04 for the Ubuntu 20.04 runtime lane."
}

if (@($runnerLabelsExpanded | ForEach-Object { $_.ToLowerInvariant() }) -notcontains "ubuntu-20.04") {
  Write-Warning "The runner labels do not include 'ubuntu-20.04'. The release preflight expects self-hosted,linux,x64,ubuntu-20.04."
}

$automaticLabels = @("self-hosted", "linux", "x64")
$customLabels = @($runnerLabelsExpanded | Where-Object { $automaticLabels -notcontains $_.ToLowerInvariant() } | Select-Object -Unique)
$expectedLabels = @($automaticLabels + $customLabels | Select-Object -Unique)
$resolvedVersion = Resolve-RunnerVersion -Version $RunnerVersion
$archiveName = "actions-runner-linux-$architecture-$resolvedVersion.tar.gz"
$downloadUrl = "https://github.com/actions/runner/releases/download/v$resolvedVersion/$archiveName"
$githubUrl = "https://github.com/$Repository"

if ($UseGhRegistrationToken.IsPresent -and [string]::IsNullOrWhiteSpace($RegistrationToken)) {
  $RegistrationToken = Get-RegistrationTokenFromGh -RepositoryName $Repository
}

if (-not $DryRun.IsPresent -and [string]::IsNullOrWhiteSpace($RegistrationToken)) {
  throw "A registration token is required. Pass -RegistrationToken, set GITHUB_RUNNER_REGISTRATION_TOKEN, or pass -UseGhRegistrationToken with a gh identity that can request runner registration tokens."
}

$configArgs = @("--url", $githubUrl, "--token", $RegistrationToken, "--name", $RunnerName, "--work", $WorkDirectory, "--unattended")
if ($customLabels.Count -gt 0) {
  $configArgs += @("--labels", ($customLabels -join ","))
}
if (-not [string]::IsNullOrWhiteSpace($RunnerGroup)) {
  $configArgs += @("--runnergroup", $RunnerGroup)
}
if ($Replace.IsPresent) {
  $configArgs += "--replace"
}

$safeConfigArgs = @(
  for ($i = 0; $i -lt $configArgs.Count; $i++) {
    if ($i -gt 0 -and $configArgs[$i - 1] -eq "--token") {
      "<redacted>"
    }
    else {
      $configArgs[$i]
    }
  }
)

Write-Host "GitHub Actions self-hosted runner setup"
Write-Host "Repository      : $Repository"
Write-Host "Runner name     : $RunnerName"
Write-Host "Runner root     : $RunnerRoot"
Write-Host "Runner version  : $resolvedVersion"
Write-Host "Expected labels : $($expectedLabels -join ',')"
Write-Host "Config labels   : $($customLabels -join ',')"

if ($DryRun.IsPresent) {
  Write-Host ""
  Write-Host "Dry-run commands:"
  Write-Host "  mkdir -p '$RunnerRoot'"
  if (-not $NoDownload.IsPresent) {
    Write-Host "  curl -fsSL -o '$archiveName' '$downloadUrl'"
    Write-Host "  tar xzf '$archiveName'"
  }
  if ($InstallRunnerDependencies.IsPresent) {
    Write-Host "  sudo ./bin/installdependencies.sh"
  }
  Write-Host "  ./config.sh $($safeConfigArgs -join ' ')"
  if ($InstallService.IsPresent) {
    Write-Host "  sudo ./svc.sh install"
    Write-Host "  sudo ./svc.sh start"
  }
  else {
    Write-Host "  ./run.sh"
  }
  Write-Host ""
  Write-Host "The registration token is never written to disk by this script."
  exit 0
}

New-Item -ItemType Directory -Path $RunnerRoot -Force | Out-Null
Push-Location $RunnerRoot
try {
  if ((Test-Path -LiteralPath ".runner" -PathType Leaf) -and -not $Replace.IsPresent) {
    throw "A runner is already configured at '$RunnerRoot'. Re-run with -Replace only when you deliberately want to replace it."
  }

  if (-not $NoDownload.IsPresent -and -not (Test-Path -LiteralPath "config.sh" -PathType Leaf)) {
    Invoke-CheckedNativeCommand -FilePath "curl" -ArgumentList @("-fsSL", "-o", $archiveName, $downloadUrl)
    Invoke-CheckedNativeCommand -FilePath "tar" -ArgumentList @("xzf", $archiveName)
  }

  if (-not (Test-Path -LiteralPath "config.sh" -PathType Leaf)) {
    throw "config.sh was not found in '$RunnerRoot'. Remove -NoDownload or unpack the GitHub Actions runner archive first."
  }

  if ($InstallRunnerDependencies.IsPresent) {
    Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList @("./bin/installdependencies.sh")
  }

  Write-Host "> ./config.sh $($safeConfigArgs -join ' ')"
  & ./config.sh @configArgs
  if ($LASTEXITCODE -ne 0) {
    throw "GitHub Actions runner configuration failed with exit code $LASTEXITCODE."
  }

  if ($InstallService.IsPresent) {
    Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList @("./svc.sh", "install")
    Invoke-CheckedNativeCommand -FilePath "sudo" -ArgumentList @("./svc.sh", "start")
  }
  else {
    Write-Host "Runner configured. Start it with:"
    Write-Host "  cd '$RunnerRoot' && ./run.sh"
  }
}
finally {
  Pop-Location
}
