[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$Ref = "TensorRtSharp4.0",
  [string]$Version = "4.0.0",
  [string]$RuntimeVersion,
  [string]$Configuration = "Release",
  [object]$RunDocsRelease = $true,
  [switch]$RunWindowsRuntimePackaging,
  [string[]]$WindowsRuntimeKeys = @(),
  [string[]]$WindowsSplitPackageRoles = @(),
  [ValidateSet("full", "split")]
  [string]$WindowsRuntimeDeliveryMode = "split",
  [string]$WindowsMetaPackageVersion,
  [string]$WindowsBridgePackageVersion,
  [string]$WindowsCudaCudnnPackageVersion,
  [string]$WindowsCudaCudnnPackageVersionMap,
  [string]$WindowsCudaCudnnPackageReleaseTag,
  [string]$WindowsCudaCudnnPackageReleaseTagMap,
  [string]$WindowsTensorRtPackageVersion,
  [string]$WindowsTensorRtPackageVersionMap,
  [string]$WindowsTensorRtPackageReleaseTag,
  [string]$WindowsTensorRtPackageReleaseTagMap,
  [switch]$WindowsIncludeMetaPackage,
  [switch]$SkipWindowsConsumerValidation,
  [switch]$RunLinuxRuntimePackaging,
  [string[]]$LinuxRuntimeKeys = @(),
  [string]$LinuxRuntimeKeySet = "hosted-all",
  [ValidateSet("hosted", "hosted-container", "self-hosted")]
  [string]$LinuxRunnerMode = "hosted",
  [string[]]$LinuxSplitPackageRoles = @(),
  [ValidateSet("full", "split")]
  [string]$LinuxRuntimeDeliveryMode = "split",
  [string]$LinuxMetaPackageVersion,
  [string]$LinuxBridgePackageVersion,
  [string]$LinuxCudaCudnnPackageVersion,
  [string]$LinuxCudaCudnnPackageVersionMap,
  [string]$LinuxCudaCudnnPackageReleaseTag,
  [string]$LinuxCudaCudnnPackageReleaseTagMap,
  [string]$LinuxTensorRtPackageVersion,
  [string]$LinuxTensorRtPackageVersionMap,
  [string]$LinuxTensorRtPackageReleaseTag,
  [string]$LinuxTensorRtPackageReleaseTagMap,
  [switch]$LinuxIncludeMetaPackage,
  [switch]$SkipLinuxConsumerValidation,
  [Alias("RunLinuxSelfHostedUbuntu20RuntimePackaging")]
  [switch]$RunLinuxUbuntu20RuntimePackaging,
  [Alias("LinuxSelfHostedUbuntu20RuntimeKeys")]
  [string[]]$LinuxUbuntu20RuntimeKeys = @(),
  [Alias("LinuxSelfHostedUbuntu20RuntimeKeySet")]
  [string]$LinuxUbuntu20RuntimeKeySet = "hosted-container-ubuntu20",
  [object]$RunWindowsSmoke = $true,
  [object]$RunLinuxSmoke = $false,
  [object]$PublishManagedToNuGet = $false,
  [object]$PublishManagedToGitHubPackages = $true,
  [object]$PublishRuntimeToGitHubPackages = $false,
  [object]$AttachRuntimeToGitHubRelease = $true,
  [switch]$DryRun
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Expand-TokenList {
  param(
    [string[]]$Values
  )

  $tokens = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $tokens.Add($trimmed)
      }
    }
  }

  @($tokens | Select-Object -Unique)
}

function ConvertTo-WorkflowBoolean {
  param(
    [bool]$Value
  )

  if ($Value) { "true" } else { "false" }
}

function ConvertFrom-BooleanInput {
  param(
    [AllowNull()]
    [object]$Value,
    [bool]$DefaultValue
  )

  if ($null -eq $Value) {
    return $DefaultValue
  }

  if ($Value -is [bool]) {
    return [bool]$Value
  }

  if ($Value -is [switch]) {
    return [bool]$Value.IsPresent
  }

  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $DefaultValue
  }

  switch -Regex ($text) {
    '^(1|true|yes|y|on)$' { return $true }
    '^(0|false|no|n|off)$' { return $false }
    default {
      throw "Invalid boolean value '$text'. Use true/false, 1/0, yes/no, or on/off."
    }
  }
}

function Add-WorkflowInput {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.Generic.List[string]]$ArgumentList,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [AllowEmptyString()]
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return
  }

  $ArgumentList.Add("-f") | Out-Null
  $ArgumentList.Add("$Name=$Value") | Out-Null
}

function Add-ReleaseConfigValue {
  param(
    [Parameter(Mandatory = $true)]
    [hashtable]$Config,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [AllowEmptyString()]
    [string]$Value
  )

  if (-not [string]::IsNullOrWhiteSpace($Value)) {
    $Config[$Name] = $Value
  }
}

function Add-ReleaseConfigBoolean {
  param(
    [Parameter(Mandatory = $true)]
    [hashtable]$Config,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [bool]$Value
  )

  if ($Value) {
    $Config[$Name] = $true
  }
}

function Format-CommandLine {
  param(
    [string[]]$ArgumentList
  )

  @(
    "gh"
    foreach ($argument in $ArgumentList) {
      if ($argument -match "[\s`"']") {
        "'" + ($argument -replace "'", "''") + "'"
      }
      else {
        $argument
      }
    }
  ) -join " "
}

$windowsKeys = @(Expand-TokenList -Values $WindowsRuntimeKeys)
$windowsRoles = @(Expand-TokenList -Values $WindowsSplitPackageRoles)
$linuxKeys = @(Expand-TokenList -Values $LinuxRuntimeKeys)
$linuxRoles = @(Expand-TokenList -Values $LinuxSplitPackageRoles)
$linuxUbuntu20Keys = @(Expand-TokenList -Values $LinuxUbuntu20RuntimeKeys)
$runDocsReleaseValue = ConvertFrom-BooleanInput -Value $RunDocsRelease -DefaultValue $true
$runWindowsSmokeValue = ConvertFrom-BooleanInput -Value $RunWindowsSmoke -DefaultValue $true
$runLinuxSmokeValue = ConvertFrom-BooleanInput -Value $RunLinuxSmoke -DefaultValue $false
$publishManagedToNuGetValue = ConvertFrom-BooleanInput -Value $PublishManagedToNuGet -DefaultValue $false
$publishManagedToGitHubPackagesValue = ConvertFrom-BooleanInput -Value $PublishManagedToGitHubPackages -DefaultValue $true
$publishRuntimeToGitHubPackagesValue = ConvertFrom-BooleanInput -Value $PublishRuntimeToGitHubPackages -DefaultValue $false
$attachRuntimeToGitHubReleaseValue = ConvertFrom-BooleanInput -Value $AttachRuntimeToGitHubRelease -DefaultValue $true

$releaseConfig = @{}
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_runtime_delivery_mode" -Value $WindowsRuntimeDeliveryMode
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_meta_package_version" -Value $WindowsMetaPackageVersion
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_bridge_package_version" -Value $WindowsBridgePackageVersion
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_cuda_cudnn_package_version_map" -Value $WindowsCudaCudnnPackageVersionMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_cuda_cudnn_package_release_tag" -Value $WindowsCudaCudnnPackageReleaseTag
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_cuda_cudnn_package_release_tag_map" -Value $WindowsCudaCudnnPackageReleaseTagMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_tensorrt_package_version_map" -Value $WindowsTensorRtPackageVersionMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_tensorrt_package_release_tag" -Value $WindowsTensorRtPackageReleaseTag
Add-ReleaseConfigValue -Config $releaseConfig -Name "windows_tensorrt_package_release_tag_map" -Value $WindowsTensorRtPackageReleaseTagMap
Add-ReleaseConfigBoolean -Config $releaseConfig -Name "windows_include_meta_package" -Value $WindowsIncludeMetaPackage.IsPresent
Add-ReleaseConfigBoolean -Config $releaseConfig -Name "windows_skip_consumer_validation" -Value $SkipWindowsConsumerValidation.IsPresent
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_runtime_delivery_mode" -Value $LinuxRuntimeDeliveryMode
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_meta_package_version" -Value $LinuxMetaPackageVersion
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_bridge_package_version" -Value $LinuxBridgePackageVersion
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_cuda_cudnn_package_version_map" -Value $LinuxCudaCudnnPackageVersionMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_cuda_cudnn_package_release_tag" -Value $LinuxCudaCudnnPackageReleaseTag
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_cuda_cudnn_package_release_tag_map" -Value $LinuxCudaCudnnPackageReleaseTagMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_tensorrt_package_version_map" -Value $LinuxTensorRtPackageVersionMap
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_tensorrt_package_release_tag" -Value $LinuxTensorRtPackageReleaseTag
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_tensorrt_package_release_tag_map" -Value $LinuxTensorRtPackageReleaseTagMap
Add-ReleaseConfigBoolean -Config $releaseConfig -Name "linux_include_meta_package" -Value $LinuxIncludeMetaPackage.IsPresent
Add-ReleaseConfigBoolean -Config $releaseConfig -Name "linux_skip_consumer_validation" -Value $SkipLinuxConsumerValidation.IsPresent
Add-ReleaseConfigValue -Config $releaseConfig -Name "linux_ubuntu20_runtime_key_set" -Value $LinuxUbuntu20RuntimeKeySet

$releaseConfigJson = if ($releaseConfig.Count -gt 0) {
  $releaseConfig | ConvertTo-Json -Compress -Depth 5
}
else {
  "{}"
}

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.AddRange([string[]]@("workflow", "run", "release-bundle.yml", "--repo", $Repository, "--ref", $Ref))

Add-WorkflowInput -ArgumentList $arguments -Name "version" -Value $Version
Add-WorkflowInput -ArgumentList $arguments -Name "configuration" -Value $Configuration
Add-WorkflowInput -ArgumentList $arguments -Name "run_docs_release" -Value (ConvertTo-WorkflowBoolean -Value $runDocsReleaseValue)
Add-WorkflowInput -ArgumentList $arguments -Name "runtime_version" -Value $RuntimeVersion
Add-WorkflowInput -ArgumentList $arguments -Name "run_windows_runtime_packaging" -Value (ConvertTo-WorkflowBoolean -Value $RunWindowsRuntimePackaging.IsPresent)
Add-WorkflowInput -ArgumentList $arguments -Name "windows_runtime_keys" -Value ($windowsKeys -join ",")
Add-WorkflowInput -ArgumentList $arguments -Name "windows_split_package_roles" -Value ($windowsRoles -join ",")
Add-WorkflowInput -ArgumentList $arguments -Name "windows_cuda_cudnn_package_version" -Value $WindowsCudaCudnnPackageVersion
Add-WorkflowInput -ArgumentList $arguments -Name "windows_tensorrt_package_version" -Value $WindowsTensorRtPackageVersion
Add-WorkflowInput -ArgumentList $arguments -Name "run_linux_runtime_packaging" -Value (ConvertTo-WorkflowBoolean -Value $RunLinuxRuntimePackaging.IsPresent)
Add-WorkflowInput -ArgumentList $arguments -Name "linux_runtime_keys" -Value ($linuxKeys -join ",")
Add-WorkflowInput -ArgumentList $arguments -Name "linux_runtime_key_set" -Value $LinuxRuntimeKeySet
Add-WorkflowInput -ArgumentList $arguments -Name "linux_runner_mode" -Value $LinuxRunnerMode
Add-WorkflowInput -ArgumentList $arguments -Name "linux_split_package_roles" -Value ($linuxRoles -join ",")
Add-WorkflowInput -ArgumentList $arguments -Name "linux_cuda_cudnn_package_version" -Value $LinuxCudaCudnnPackageVersion
Add-WorkflowInput -ArgumentList $arguments -Name "linux_tensorrt_package_version" -Value $LinuxTensorRtPackageVersion
Add-WorkflowInput -ArgumentList $arguments -Name "run_linux_ubuntu20_runtime_packaging" -Value (ConvertTo-WorkflowBoolean -Value $RunLinuxUbuntu20RuntimePackaging.IsPresent)
Add-WorkflowInput -ArgumentList $arguments -Name "linux_ubuntu20_runtime_keys" -Value ($linuxUbuntu20Keys -join ",")
Add-WorkflowInput -ArgumentList $arguments -Name "run_windows_smoke" -Value (ConvertTo-WorkflowBoolean -Value $runWindowsSmokeValue)
Add-WorkflowInput -ArgumentList $arguments -Name "run_linux_smoke" -Value (ConvertTo-WorkflowBoolean -Value $runLinuxSmokeValue)
Add-WorkflowInput -ArgumentList $arguments -Name "publish_managed_to_nuget" -Value (ConvertTo-WorkflowBoolean -Value $publishManagedToNuGetValue)
Add-WorkflowInput -ArgumentList $arguments -Name "publish_managed_to_github_packages" -Value (ConvertTo-WorkflowBoolean -Value $publishManagedToGitHubPackagesValue)
Add-WorkflowInput -ArgumentList $arguments -Name "publish_runtime_to_github_packages" -Value (ConvertTo-WorkflowBoolean -Value $publishRuntimeToGitHubPackagesValue)
Add-WorkflowInput -ArgumentList $arguments -Name "attach_runtime_to_github_release" -Value (ConvertTo-WorkflowBoolean -Value $attachRuntimeToGitHubReleaseValue)
Add-WorkflowInput -ArgumentList $arguments -Name "release_config_json" -Value $releaseConfigJson

$commandText = Format-CommandLine -ArgumentList @($arguments)
Write-Host $commandText

if ($DryRun.IsPresent) {
  Write-Host ""
  Write-Host "Dry run only. release_config_json:"
  Write-Host $releaseConfigJson
  return
}

& gh @arguments
if ($LASTEXITCODE -ne 0) {
  throw "gh workflow run release-bundle.yml failed with exit code $LASTEXITCODE."
}
