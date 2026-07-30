# NVIDIA 运行库外置与发布策略

自 2026-07-30 起，TensorRtSharp4.0 不再打包或发布 NVIDIA 原厂运行库。CUDA、cuDNN、TensorRT、NVRTC、parser、plugin 与 builder-resource 文件由用户根据目标版本自行安装。

正式发布物只包括：

- `JYPPX.TensorRT.CSharp.API`：C# 托管接口程序集与 XML 文档；
- 以 `.Bridge` 结尾的版本化桥接包：仅包含项目自行编译的 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`；
- 从 Git 跟踪文件生成的源码归档。

`pack/runtime/runtime-packages.manifest.json` 继续保存 TensorRT/CUDA/cuDNN 兼容和编译输入矩阵。其中的 vendor 文件名只用于检查用户本机依赖，不是发布资产。

所有发布工作流在上传前运行 `eng/Test-ExternalVendorRuntimePackagePolicy.ps1`。该门禁会拒绝 full-runtime、CUDA/cuDNN、TensorRT、CUDA RTC、collection/meta 和 builder-resource 包，也会扫描实际 `.nupkg` 内容，阻止 NVIDIA DLL 或 `.so` 进入包。

旧 GitHub Packages 版本和 GitHub Release `.nupkg` 资产不会自动删除。先运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Export-RetiredVendorPackageCleanupPlan.ps1
```

该命令只生成候选清单，不访问远程、不执行删除。正式账号完成授权后，必须核对实际版本、资产名与哈希，再由 Owner 确认删除。托管包、`.Bridge` 包以及 GitHub 自动生成的源码归档必须保留。

`grape-yan` 账号仅用于日常 Actions 编译检查，不允许执行 publish、package push、Release create/upload 等远程发布动作。正式账号继续承担源码同步和正式发布。
