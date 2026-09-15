# vendored: FireRedTeam/FireRedASR(Apache-2.0)

- 來源:https://github.com/FireRedTeam/FireRedASR
- commit:834635e4cf277ed8ca92049fc375b17c3dc20748(2026-02-25)
- 對上游的**唯二**改動:①補 `pyproject.toml`(上游無 packaging;讓它成為
  uv path dependency 裝進 site-packages → Nuitka 打包時隨依賴編進 binary,
  **部署不需外帶本目錄**)②各層補空 `__init__.py`(namespace → regular
  package,Nuitka 收錄最可靠)。推論程式碼本身一字未改。
- 原因:官方 repo 無 PyPI 發佈/packaging,PyPI 上的 `fireredasr` 為第三方
  fork(額外拖 modelscope 重依賴)。vendor 官方 `fireredasr/` 套件目錄 +
  LICENSE,一字未改。
- 使用:主專案 pyproject 以 `[tool.uv.sources] fireredasr = { path = ... }`
  指到本目錄,`uv sync` 即裝;fireredasr_provider 正常 `import fireredasr`。
  權重另放 `assets/fireredasr/FireRedASR-AED-L/`(不進 repo,部署要外帶)。
- 升級:重抓上游 tarball 覆蓋 `fireredasr/`(保留 pyproject 與 __init__.py)、
  更新本檔 commit 記錄、`uv sync --reinstall-package fireredasr`、跑 ASR evals。
