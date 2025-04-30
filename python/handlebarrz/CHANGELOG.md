# Changelog

## [0.1.1](https://github.com/hendrixmar/dotprompt/compare/handlebarrz-v0.1.0...handlebarrz-0.1.1) (2025-04-21)


### Features

* **handlebarrz:** CI to publish rust python package ([#152](https://github.com/hendrixmar/dotprompt/issues/152)) ([20765d8](https://github.com/hendrixmar/dotprompt/commit/20765d83c50537cf935fe461a24f5c86d970d787))
* **handlebarrz:** native ifEquals, unlessEquals, and json helpers ([#121](https://github.com/hendrixmar/dotprompt/issues/121)) ([e3619e9](https://github.com/hendrixmar/dotprompt/commit/e3619e906cd0b69d854ca50d798e36cf44c130bd))
* partial test runner implementation now loads tests ([#139](https://github.com/hendrixmar/dotprompt/issues/139)) ([b09dd2f](https://github.com/hendrixmar/dotprompt/commit/b09dd2f9b8029317ce484d6f32d5a3fb89f5f7e1))
* **py/dotpromptz:** add initial Dotprompt._resolve_partials implementation ([#215](https://github.com/hendrixmar/dotprompt/issues/215)) ([03a161c](https://github.com/hendrixmar/dotprompt/commit/03a161c3440a680bc0df472f35efa155fe0d5151))
* **py/dotpromptz:** configure handlebars to not escape by default ([#163](https://github.com/hendrixmar/dotprompt/issues/163)) ([f7c33e1](https://github.com/hendrixmar/dotprompt/commit/f7c33e1303476fd473e803f930ac1e1f9e1d87c9))
* **py/dotpromptz:** implement helpers in terms of the rust implementation of handlebars-rust and fix go flakiness ([#115](https://github.com/hendrixmar/dotprompt/issues/115)) ([314c0b5](https://github.com/hendrixmar/dotprompt/commit/314c0b5182aaad25bf4cfccb8207faa60f63256f))
* **py/dotpromptz:** initial bits of Dotprompt class ([#148](https://github.com/hendrixmar/dotprompt/issues/148)) ([90f7838](https://github.com/hendrixmar/dotprompt/commit/90f78384a958d41d78dee48497a78dfde11f4476))
* **py/handlebarrz:** add a Template.compile method to the Python wrapper to make it easier to port the JS implementation ([#201](https://github.com/hendrixmar/dotprompt/issues/201)) ([9295972](https://github.com/hendrixmar/dotprompt/commit/92959720fbf2e8ee410d5b8c0c174c6ef464e667))
* **py/handlebarrz:** Python bindings for handlebars-rust ([#113](https://github.com/hendrixmar/dotprompt/issues/113)) ([6b6a97e](https://github.com/hendrixmar/dotprompt/commit/6b6a97e01acc49f53586eb5b8b2b410ae82ce6ce))
* python implementations of helpers ([#129](https://github.com/hendrixmar/dotprompt/issues/129)) ([79c6ef3](https://github.com/hendrixmar/dotprompt/commit/79c6ef3e9cc472fed3a832c00a1515ceef0981da))
* **python:** support lower versions of python (&gt;=3.10) ([#187](https://github.com/hendrixmar/dotprompt/issues/187)) ([4240f9d](https://github.com/hendrixmar/dotprompt/commit/4240f9d720891e350f9116aa4401ce6ea7fac5a3))
* use the more maintained YAML parsing lib at https://github.com/goccy/go-yaml ([#151](https://github.com/hendrixmar/dotprompt/issues/151)) ([910b3a7](https://github.com/hendrixmar/dotprompt/commit/910b3a72f3756296c3b01b96936a5bc4c9fa88ef))


### Bug Fixes

* change project name for pypi publish ([#200](https://github.com/hendrixmar/dotprompt/issues/200)) ([2c07132](https://github.com/hendrixmar/dotprompt/commit/2c0713264fb2c30bdc43f1bd9e51d416f96d1b7e))
* **docs:** update docs for handlebarrz ([#116](https://github.com/hendrixmar/dotprompt/issues/116)) ([ce643b1](https://github.com/hendrixmar/dotprompt/commit/ce643b1f5299ba2a6b214fb57965980d412c1a7b))
* **handlebarrz:** Compatibility tests for python 3.10 ([#190](https://github.com/hendrixmar/dotprompt/issues/190)) ([e459e9c](https://github.com/hendrixmar/dotprompt/commit/e459e9ce94f76d42615593987f99221b0f55a0d3))
* **license:** add license header to stub types file ([#144](https://github.com/hendrixmar/dotprompt/issues/144)) ([0abd498](https://github.com/hendrixmar/dotprompt/commit/0abd49848548f2148a37ec686d703126d8fe8504))
* **license:** use the full license header in rust source, script files and yaml spec files ([#143](https://github.com/hendrixmar/dotprompt/issues/143)) ([77ccec9](https://github.com/hendrixmar/dotprompt/commit/77ccec93a4bf5ccd65932a701676554866e68c6f))
* **license:** use the full license header in source code ([#142](https://github.com/hendrixmar/dotprompt/issues/142)) ([64894ef](https://github.com/hendrixmar/dotprompt/commit/64894ef898876b861c6c244d522f634cd8fcc842))
* **py/dotpromptz:** address compatibility with python 3.10 and add tox configuration for parallelized tests ([#188](https://github.com/hendrixmar/dotprompt/issues/188)) ([d2ba21f](https://github.com/hendrixmar/dotprompt/commit/d2ba21ff3e54f4ca4328b7e574bb6492699095bc))
