# 本地 GitHub Actions 检查

GitHub Actions 远端 workflow 仍然是正式发布证据来源，但部分 workflow 调度图可以用 [`act`](https://github.com/nektos/act) 在本机做轻量检查。

## 本地 `act` 适合做什么

推荐把本地 `act` 用在不发布、不上传的快速检查上：

- 解析 workflow 文件，确认 job 图没有明显错误。
- dry-run `workflow_dispatch` 输入和表达式。
- 检查 `release-bundle.yml` 这类 Linux hosted 编排 job，以及 `runtime-linux.yml` 的 `prepare` job。
- 在不创建额外 GitHub Actions run 记录的情况下复现小脚本问题。

示例 dry-run：

```powershell
act workflow_dispatch -W .github/workflows/runtime-linux.yml -j prepare -n -P ubuntu-latest=catthehacker/ubuntu:act-latest --pull=false

act workflow_dispatch -W .github/workflows/release-bundle.yml -j orchestrate -n -P ubuntu-latest=catthehacker/ubuntu:act-latest --pull=false
```

## 本地 `act` 不适合证明什么

不要把本地 `act` 当作本仓库的正式发布证据：

- Windows hosted job 不能被 Linux 容器镜像可靠复刻。
- self-hosted Windows runtime job 仍依赖本机 CUDA、cuDNN、TensorRT、Visual Studio、CMake、签名和包源配置。
- Linux runtime 发布仍需要真实 self-hosted Linux x64 runner，并且该 runner 上要有匹配的 NVIDIA roots。
- `act` dry-run 不会上传 artifacts、发布 NuGet、部署 Pages，也不能证明从 GitHub Releases/GitHub Packages 还原 package 的完整链路。

正式发布证据仍然应通过 `gh workflow run` 远程触发，并检查 GitHub hosted 或 self-hosted runner 日志。

## 当前本机设置提示

Windows 上可以用 `winget install --id nektos.act -e` 安装 `act.exe`，但当前 PowerShell 会话可能不会立刻刷新 `PATH`。如果命令暂时不可见，可以直接调用 WinGet 包目录里的 `act.exe`，或打开一个新终端。

非 dry-run 的 `act` job 需要 Docker Desktop 已经启动并可连接 Docker engine。

