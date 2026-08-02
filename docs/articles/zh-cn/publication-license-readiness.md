# 公开发布许可证门禁

TensorRtSharp4.0 的本地编译、测试和包 dry run 可以在许可证尚待 Owner 决策时继续，但任何 NuGet、GitHub Packages 或 GitHub Release 发布都必须先声明许可证。自动化不会替项目选择 MIT、Apache-2.0 或商业许可证，也不会把 `NOASSERTION`、`NONE`、`TBD` 等占位值当成有效授权。

## 为什么独立成门禁

包能够编译和被消费，不代表具备公开分发条件。许可证是发布物本身的合同：

- NuGet 包需要在 nuspec 中使用 `<license type="expression">...</license>`，或者用 `<license type="file">...</license>` 指向包内非空许可证文件。
- Git 跟踪源码归档需要在归档根目录包含非空 `LICENSE` 或 `COPYING` 文件。
- `licenseUrl`、README 中的口头说明、SBOM 的 `NOASSERTION` 和 Owner 尚未执行的计划都不能替代许可证元数据。

仓库用 `pack/publication-license-policy.json` 保存机器可读策略。当前 `ownerDecisionState` 为 `required`，`selectedPackageLicense` 和 `selectedSourceArchiveLicenseFileName` 均为空，表示 Owner 尚未选择许可证；即使某个临时包自行写入了 MIT 或 Apache-2.0，也不能把它视为默认授权。

## 验证命令

只检查工作流和 push 入口是否接入 fail-closed 门禁：

```powershell
pwsh -NoProfile -File .\eng\Test-PublicationLicenseReadiness.ps1 -StaticOnly
```

检查准备发布的 NuGet 包：

```powershell
pwsh -NoProfile -File .\eng\Test-PublicationLicenseReadiness.ps1 `
  -ArtifactPath .\artifacts\managed, .\artifacts\runtime-split-nupkg
```

检查 Git tracked source archive：

```powershell
pwsh -NoProfile -File .\eng\Test-PublicationLicenseReadiness.ps1 `
  -ArtifactPath .\artifacts\source\TensorRtSharp4.0-source-4.0.0.zip
```

当前仓库尚未声明许可证时，后两条命令应返回非零并明确提示 Owner 选择许可证。这是正确的发布阻断结果，不是构建失败。

## 工作流顺序

所有公开发布路径必须保持以下顺序：

1. `owner_publish_approved=true`，并且仓库 Owner 为 `guojin-yan`。
2. 构建 managed、YoloVision、bridge 或 tracked source archive。
3. 执行厂商运行库排除策略和许可证门禁。
4. 门禁通过后，才允许创建 GitHub Release、上传资产或执行 NuGet push。

`grape-yan` 继续只做日常 Actions 编译检查。dry run 可以上传 GitHub Actions artifact 供审计，但不能创建 Release 或发布 package。

## Owner 选择许可证后

Owner 需要一次性完成以下变更，避免仓库、nuspec、源码归档和 SBOM 相互矛盾：

1. 在仓库根目录加入最终许可证文件，并确认 NuGet 使用合法的 SPDX expression 还是包内 license file。
2. 为 managed、YoloVision 和所有 bridge 包配置相同的许可证元数据。
3. 将 `ownerDecisionState` 更新为 `approved`，在 `selectedPackageLicense.type/value` 和 `selectedSourceArchiveLicenseFileName` 中记录 Owner 的准确选择。
4. 重新提交、重打全部候选包和源码归档，重新生成 SHA256、SBOM 与 Owner 审批指纹。
5. 运行许可证门禁、external vendor runtime policy、clean consumer 和发布质量门禁。

旧候选包即使功能验证通过，只要许可证元数据仍为空，就不能直接提升为公开发布物。

## 证据边界

许可证门禁只证明发布物声明了允许的许可证形态，并且引用的文件真实存在。它不判断自定义条款的法律充分性，不替代 Owner/法律审核，不执行发布，也不是 runtime、package-consumer-runtime 或 post-publish proof。
