# Owner Real Input Landing Pack

`owner-real-input-landing-pack` 是真实 Owner 输入落地包，把 `owner-authorization`、`package-consumer-runtime`、`linux-runner-proof`、`real-model-runtime` 和 `post-publish-verification` 映射到必须回填的真实文件、字段、strict validator 和 forbidden substitutes。

- 主要产物：`artifacts/final-release/owner-real-input-landing-pack.json`
- 校验产物：`artifacts/final-release/owner-real-input-landing-pack-validation.json`
- 默认状态：`blocked-owner-real-input-required`
- 边界：它不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

该包只能作为 Owner 回填入口。local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate、draft、build-only、parse-only、sidecar-only 和 template 都不能替代真实 proof。
