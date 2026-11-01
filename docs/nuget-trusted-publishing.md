# NuGet Trusted Publishing 配置

项目向 NuGet.org 发布时使用 GitHub Actions OIDC，通过 `NuGet/login@v1` 换取有效期 1 小时的临时 API key，再交给现有上传脚本。无需保存或轮换长期 `NUGET_API_KEY` secret。GitHub Packages 的 `GITHUB_TOKEN`、正式仓库限制、Owner 授权、`production-release` 环境门禁和包内容校验继续生效。

## 1. 在 NuGet.org 创建信任策略

登录 [NuGet.org Trusted Publishing](https://www.nuget.org/account/trustedpublishing)，选择 Create。每个实际执行 NuGet 登录的 workflow 文件创建一条策略。

所有策略的共同字段：

| NuGet.org 字段 | 填写值 |
| --- | --- |
| Publisher / Provider | GitHub Actions（如页面显示为 GitHub，则选 GitHub） |
| Package Owner | `GuojinYan` |
| Repository Owner | `guojin-yan` |
| Repository | `TensorRT-CSharp-API` |
| Environment | `production-release` |

`GuojinYan` 是当前[核心包页面](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0.1)列出的 NuGet 所有者。它与 GitHub owner `guojin-yan` 不同。使用该 NuGet 账户创建策略；若将来包转交组织，应选择实际拥有包的组织，并使用有权限的 NuGet 用户登录工作流。

按使用方式填写策略名、工作流与包范围：

| 用途 | Policy Name（建议） | Workflow File | Glob Patterns and Packages |
| --- | --- | --- | --- |
| C# 核心包单独迭代，优先创建 | `tensorrt-managed` | `package-managed.yml` | `JYPPX.TensorRT.CSharp.API` |
| 正式完整版本，建议创建 | `tensorrt-release-bundle` | `release-bundle.yml` | `JYPPX.TensorRT.CSharp.API` 与 `JYPPX.TensorRT.CSharp.API.Runtime.*.Bridge`，分两行 |
| 独立发布 Windows Bridge，按需创建 | `tensorrt-runtime-windows` | `runtime-windows.yml` | `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.*.Bridge` |
| 独立发布 Linux Bridge，按需创建 | `tensorrt-runtime-linux` | `runtime-linux.yml` | `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.*.Bridge` |

Workflow File **只填文件名**，例如 `package-managed.yml`，不要填写 `.github/workflows/` 路径、显示名称或分支。Environment 虽是 NuGet 页面上的可选项，本项目应填写 `production-release`，与发布 job 一致。策略名可自行调整，不影响工作流匹配。

如果页面提供 Select Scopes，选 Push 下的“发布已有包的新版本”（Push only new package versions），满足当前已经存在的核心包和 Bridge ID 的后续迭代。将来新增 Bridge 包 ID 时，再明确启用“发布新包及新版本”权限。无需授予 Unlist 或其他操作权限。

完整版本策略的包范围可直接粘贴：

```text
JYPPX.TensorRT.CSharp.API
JYPPX.TensorRT.CSharp.API.Runtime.*.Bridge
```

不要使用 `*` 或 `JYPPX.TensorRT.CSharp.API*` 这样的过宽范围；`YoloVision`、`Classification` 和 NVIDIA 厂商运行库不属于公开发布集合。需要限定到当前 18 个 Bridge ID 时，也可从 `pack/runtime-split/split-runtime-packages.manifest.json` 中取出 Bridge 包 ID，逐行填写。

`release-bundle.yml` 会调度子工作流构建，并在自身集中执行 NuGet 登录和上传。因此只使用完整发布入口时，配置 `release-bundle.yml` 策略即可；只有直接从 Windows/Linux 工作流发包时才需要对应策略。`release-quality-gate.yml` 的 reusable 调用固定关闭发布，不需要为它创建 NuGet 策略。现有 reusable 发布入口若被其他工作流调用，调用者也必须允许 `id-token: write`；NuGet 按执行登录的工作流身份匹配策略。

## 2. 设置 GitHub 仓库变量

打开正式仓库的 [Settings → Secrets and variables → Actions → Variables](https://github.com/guojin-yan/TensorRT-CSharp-API/settings/variables/actions)，新增 Repository variable：

| Name | Value |
| --- | --- |
| `NUGET_USER` | `GuojinYan` |

此处填 NuGet **账户用户名**，不要填写邮箱、GitHub 用户名或 API key。该值不是密钥，工作流使用 `vars.NUGET_USER` 读取。也可由仓库管理员执行：

```powershell
gh variable set NUGET_USER --repo guojin-yan/TensorRT-CSharp-API --body GuojinYan
```

保留 GitHub Environment `production-release` 及已有审批/分支限制、`PRODUCTION_RELEASE_READY` 标记。当前开发分支为 `TensorRtSharp4.0`；发布所选 ref 必须符合该环境的部署规则。新的认证流程不再读取旧 `NUGET_API_KEY` secret，可以在确认没有其他消费者后移除旧 secret。

## 3. 后续发布

先将工作流修改提交并推送到用于发布的 ref，再创建/确认上述策略和变量。准备好新版本后，按已有授权流程手工触发发布；本次配置迁移本身不上传包。

- C# 核心小版本使用 `package-managed.yml`，填写待发布的新 `version`，设置 `owner_publish_approved=true`、`publish_to_nuget=true`。此入口只发布核心包，Bridge 无需跟随升级。已存在的 `4.0.0.1` 不能覆盖上传。
- 完整版本使用 `release-bundle.yml`，同时开启 `publish_managed_to_nuget` 和 `publish_runtime_to_nuget`，按现有要求提供完整 18 个 Bridge 构建结果。不要用此入口代替核心包的单独迭代。
- 构建验证、GitHub Packages 或 Release 附件上传未启用 NuGet 发布时，不执行 NuGet 登录。

NuGet 登录放在包下载和内容校验之后、NuGet 上传之前，以减少 1 小时有效期被构建占用。临时凭据为空时，上传脚本会立即失败，不回退到机器上的旧 NuGet.Config key。日志中的 `NUGET_API_KEY` 是登录 action 的临时输出变量，不再指仓库 secret。

`release-publication-audit.yml` 及本地 readiness/prerequisite 脚本只读取用户名配置，不申请 OIDC token，也不能证明 NuGet 策略已创建或匹配成功。审计入参已从 `require_nuget_api_key` 改为 `require_nuget_trusted_publishing_configuration`，脚本使用 `-NuGetUser`；旧的 API key 可用性入参已移除。首次真实登录仍须在策略创建后的 GitHub Actions 发布 job 中验证。

如 NuGet 提示策略仅临时激活 7 天，应在窗口内完成首次有效发布，或在过期后重新启用策略；按 NuGet 页面的实际状态操作。

## 官方参考

- [NuGet Trusted Publishing](https://learn.microsoft.com/nuget/nuget-org/trusted-publishing)
- [NuGet/login action](https://github.com/NuGet/login)
