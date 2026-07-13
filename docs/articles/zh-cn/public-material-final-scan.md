# Public Material Final Scan

`public-material-final-scan` 是公开材料最终扫描合同。它面向 README、docs index、toc 和中文文章，检查旧样例名回流、proof substitute 晋级和 post-publish 过度声明。

机器可读文件：

`artifacts/final-release/public-material-final-scan.json`

## 适用读者

- 公开文档审核者。
- release owner。
- 准备发布文章和 README 的维护者。

## 解决问题

项目已经迁移到 `samples/YoloVision`，并建立了多层 proof boundary。如果公开材料混入旧 live path、把 report/matrix 写成 runtime proof，或把 clean consumer 证据写成“发布后已验证”，就会误导使用者。

本扫描要求检查：

- README.md。
- README.zh-CN.md。
- docs/index.md。
- docs/toc.yml。
- docs/articles/zh-cn。

## 边界说明

旧检测样例路径不能作为 live sample path 回流。历史说明可以提旧名，但必须明确 live sample 是 `samples/YoloVision`。

以下内容不能晋级为 runtime proof、post-publish proof、publish approval 或 release close approval：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- blocked-by-cuda-driver

`sample-run-evidence` 不能替代 `package-consumer-runtime`；`package-consumer-runtime` 也不能替代 `post-publish verification`。真实 post-publish verification 必须等 selected-channel 真实发布后，在 clean consumer 中验证。

## 可复制验证命令

```powershell
rg -n "旧样例名|旧 live path|legacy sample" README.md README.zh-CN.md docs/index.md docs/toc.yml docs/articles/zh-cn

rg -n "runtime proof|post-publish proof|release-ready|发布后已验证|package push command" README.md README.zh-CN.md docs/index.md docs/toc.yml docs/articles/zh-cn
```

这些命令只做扫描，不执行真实 package push。

## 下一步

1. 修复 live old-name path。
2. 修复 proof overclaim。
3. 保留 report、matrix、dashboard 的 non-proof 边界。
4. 将扫描结果纳入 final owner proof blocker dashboard。
