# 案例共享源码

[English](README.md) | 简体中文

该目录保存通过 `build/JYPPX.SampleSupport.props` 编译到多个案例和应用中的共享实现。它不是可运行项目，也不会生成 NuGet 包。

可复用的命令行解析、Tensor 输入、参考结果验证和图像解码支持统一放在这里。面向用户的案例应位于 `samples` 下带编号的模块目录；仅用于验证的模板应位于 `tests/fixtures`。
