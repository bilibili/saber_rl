# SABER 论文宣传网页

以 LAIT (https://lait.cs.sfu.ca/) 的暖纸色、衬线字体、学术长页和交互式叙事为视觉参考，原创绘制 SABER 的四条思考路径。

## 打开

直接双击 `index.html` 即可使用。全部样式、脚本、图表数据与论文 PDF 都保存在本目录，没有构建步骤或外部字体依赖。

也可以通过本地服务预览：

```sh
python3 -m http.server 8767 --bind 127.0.0.1 --directory .
```

打开 http://127.0.0.1:8767/ 。若已有服务运行，无需再次启动。

## 内容与交互

- 中英文切换，自动保存本地语言偏好。
- 四种模式的原创 SVG 思考路径示意。
- 1.5B / 7B、数学 / 代码 / 逻辑测试集的交互结果图和完整数值表。
- 四种思考模式的案例摘要。
- 原论文框架图、论文 PDF、JSON 数据下载与 BibTeX 复制。
- 手机、平板和桌面响应式布局；键盘操作和减少动画偏好支持。

## 资料依据

主要依据两个指定文件夹中的最终版 `SABER_AAAI2026_CR.pdf`，以论文表 1–3、图 1 和图 3 为准。

- 作者与题目：最终版第 1 页。
- 方法：第 2–4 页。直接 GRPO，基座是蒸馏模型，SABER 无额外 SFT warmup。
- 实验图：第 5 页表 1、表 2；`assets/benchmarks.json` 保留原始数值。
- 首屏结果：1.5B / MATH / FastThink，7,866 → 2,719 思考 token，80.6% → 83.5% 准确率；即 token 减少 65.4%，准确率提升 2.9 个百分点（相对提高 3.6%）。
- 消融结果：第 6 页表 3。移除 NoThink 数据后的 62.3% 指 1.5B / MATH。
- 例子：第 7 页图 3 的 196 约数问题；网页为摘要示意，不是实时推理。NoThink 直接回答为补充示意。
- 框架图：最终版论文源文件 `Figure/structure_cmyk.pdf`，对应图 1。
- arXiv：https://arxiv.org/abs/2508.10026 。引用区明确使用 2025 年的 arXiv 版本。

NoThink 的零长度是零显式思考 token，不是零答案 token。预算是提示与训练目标，不是保证截断的硬上限。未将 IJP 的业务结果或 SFT 流程作为 SABER 论文结果展示。公开训练代码：https://github.com/bilibili/saber_rl 。

## 发布

公开网站：https://bilibili.github.io/saber_rl/

本目录对应仓库中的 `website/`。GitHub Pages 的 Source 使用 **GitHub Actions**；推送到 `main` 的 `website/` 或 `.github/workflows/saber-pages.yml` 变更会自动触发 `Deploy SABER website` 发布。也可在 Actions 中手动运行。

`index.html` 为入口，无构建步骤。图片、论文、JSON、样式和脚本均使用相对路径；canonical 与 Open Graph 元信息使用上方正式网址。
