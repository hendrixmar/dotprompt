# Changelog

## [0.1.1](https://github.com/hendrixmar/dotprompt/compare/dotpromptz-v0.1.0...dotpromptz-0.1.1) (2025-04-21)


### Features

* add implementation of helpers and util modules; move interfaces into dotpromptz project ([#73](https://github.com/hendrixmar/dotprompt/issues/73)) ([8c7aea1](https://github.com/hendrixmar/dotprompt/commit/8c7aea1faffaf823d01b132e55cb175a4fca5ccb))
* add stub spec_test.py and script to monitor tests. ([#138](https://github.com/hendrixmar/dotprompt/issues/138)) ([65966e9](https://github.com/hendrixmar/dotprompt/commit/65966e9bfc077e85d0b83d04d0384150470dbfd3))
* **go/parse:** parse.go implementation [#62](https://github.com/hendrixmar/dotprompt/issues/62) ([#87](https://github.com/hendrixmar/dotprompt/issues/87)) ([d5dc13c](https://github.com/hendrixmar/dotprompt/commit/d5dc13c0bf0437875a3b133511ffed474a8b3bf9))
* parseDocument python ([#80](https://github.com/hendrixmar/dotprompt/issues/80)) ([82ebc36](https://github.com/hendrixmar/dotprompt/commit/82ebc3672e8de051dfbdd92968ed3f84c79a247f))
* partial test runner implementation now loads tests ([#139](https://github.com/hendrixmar/dotprompt/issues/139)) ([b09dd2f](https://github.com/hendrixmar/dotprompt/commit/b09dd2f9b8029317ce484d6f32d5a3fb89f5f7e1))
* Port JS types to Python  ([#65](https://github.com/hendrixmar/dotprompt/issues/65)) ([edcb037](https://github.com/hendrixmar/dotprompt/commit/edcb03765f3cb6e5743d107a35cf255a60ab0369))
* **py/dotpromptz:** _resolve_metadata for dotprompt ([#226](https://github.com/hendrixmar/dotprompt/issues/226)) ([cfcc87b](https://github.com/hendrixmar/dotprompt/commit/cfcc87b57e49785c2356b03fbc5b7bf773472683))
* **py/dotpromptz:** add initial Dotprompt._resolve_json_schema implementation ([#217](https://github.com/hendrixmar/dotprompt/issues/217)) ([0b62136](https://github.com/hendrixmar/dotprompt/commit/0b621363a394c6b5c0fac6a957098eccff6891ca))
* **py/dotpromptz:** add initial Dotprompt._resolve_partials implementation ([#215](https://github.com/hendrixmar/dotprompt/issues/215)) ([03a161c](https://github.com/hendrixmar/dotprompt/commit/03a161c3440a680bc0df472f35efa155fe0d5151))
* **py/dotpromptz:** add initial Dotprompt._resolve_tools implementation and raise ValueError when resolver is None ([#214](https://github.com/hendrixmar/dotprompt/issues/214)) ([57caf5d](https://github.com/hendrixmar/dotprompt/commit/57caf5d9a9f4fe720c67f99fd10439d5ebe434dc))
* **py/dotpromptz:** add resolve_json_schema and test case for ResolverCallable that returns an asyncio.Future ([#211](https://github.com/hendrixmar/dotprompt/issues/211)) ([7bbe80d](https://github.com/hendrixmar/dotprompt/commit/7bbe80d6a1d9dc18c4d1baacfccf2f33fc8b7e26))
* **py/dotpromptz:** add resolvers module ([#207](https://github.com/hendrixmar/dotprompt/issues/207)) ([826f257](https://github.com/hendrixmar/dotprompt/commit/826f2572e710cebd0138bd757d2bef2e4898d730))
* **py/dotpromptz:** because the Picoschema parser can resolve schemas at runtime, we make it an async implementation ([#220](https://github.com/hendrixmar/dotprompt/issues/220)) ([ae285a8](https://github.com/hendrixmar/dotprompt/commit/ae285a88f3502e0a02c85d04c49c2e2e6ef88766))
* **py/dotpromptz:** configure handlebars to not escape by default ([#163](https://github.com/hendrixmar/dotprompt/issues/163)) ([f7c33e1](https://github.com/hendrixmar/dotprompt/commit/f7c33e1303476fd473e803f930ac1e1f9e1d87c9))
* **py/dotpromptz:** directory-based async and sync implementations of PromptStore and PromptStoreWritable ([#164](https://github.com/hendrixmar/dotprompt/issues/164)) ([ac92fbf](https://github.com/hendrixmar/dotprompt/commit/ac92fbf3af7ac3207102c94c20d294d8c54b9dd4))
* **py/dotpromptz:** implement helpers in terms of the rust implementation of handlebars-rust and fix go flakiness ([#115](https://github.com/hendrixmar/dotprompt/issues/115)) ([314c0b5](https://github.com/hendrixmar/dotprompt/commit/314c0b5182aaad25bf4cfccb8207faa60f63256f))
* **py/dotpromptz:** initial bits of Dotprompt class ([#148](https://github.com/hendrixmar/dotprompt/issues/148)) ([90f7838](https://github.com/hendrixmar/dotprompt/commit/90f78384a958d41d78dee48497a78dfde11f4476))
* **py/dotpromptz:** render_metadata implementation and associated tests ([#229](https://github.com/hendrixmar/dotprompt/issues/229)) ([e66dc45](https://github.com/hendrixmar/dotprompt/commit/e66dc453c718222e1633be951923a35335296dd5))
* **py/dotpromptz:** translate render_metadata for dotprompt.py from ts ([#227](https://github.com/hendrixmar/dotprompt/issues/227)) ([ae1919b](https://github.com/hendrixmar/dotprompt/commit/ae1919b3457824241c734fdf8328f61279fb6710))
* **py/dotpromptz:** use async picoschema_to_json_schema in dotprompt._render_picoschema ([#221](https://github.com/hendrixmar/dotprompt/issues/221)) ([072d95d](https://github.com/hendrixmar/dotprompt/commit/072d95deb01d9b09a7f60d9f6b3fb53a8067e497))
* **py:** add SafeString implementation that works with js2py ([#104](https://github.com/hendrixmar/dotprompt/issues/104)) ([1ebeca3](https://github.com/hendrixmar/dotprompt/commit/1ebeca3976faf2dc91d8d7f4a74c218824aac353))
* **py:** implement identify_partials in terms of regexps since we do not have an AST to walk [#90](https://github.com/hendrixmar/dotprompt/issues/90) ([#150](https://github.com/hendrixmar/dotprompt/issues/150)) ([f802275](https://github.com/hendrixmar/dotprompt/commit/f8022755d7eef716bbb54dd08a2c3a061250d393))
* **py:** implementation of parse.py; refactor parse.ts and update tests. ([#79](https://github.com/hendrixmar/dotprompt/issues/79)) ([47e7245](https://github.com/hendrixmar/dotprompt/commit/47e7245c0aae710b102178019d1f3449c2f1af66))
* python implementations of helpers ([#129](https://github.com/hendrixmar/dotprompt/issues/129)) ([79c6ef3](https://github.com/hendrixmar/dotprompt/commit/79c6ef3e9cc472fed3a832c00a1515ceef0981da))
* python: picoschema support  ISSUE: [#36](https://github.com/hendrixmar/dotprompt/issues/36)  CHANGELOG: - [x] Port relevant functionality - [x] Add tests ([#95](https://github.com/hendrixmar/dotprompt/issues/95)) ([0da188c](https://github.com/hendrixmar/dotprompt/commit/0da188c52540f041309e39fa6bc798eaf7fd7a81))
* **python:** add OpenAI adapter implementation for dotprompt [#38](https://github.com/hendrixmar/dotprompt/issues/38) ([#97](https://github.com/hendrixmar/dotprompt/issues/97)) ([d171f87](https://github.com/hendrixmar/dotprompt/commit/d171f8792ecf08f446e18ea3bbd5309cafa1d8a3))
* **python:** support lower versions of python (&gt;=3.10) ([#187](https://github.com/hendrixmar/dotprompt/issues/187)) ([4240f9d](https://github.com/hendrixmar/dotprompt/commit/4240f9d720891e350f9116aa4401ce6ea7fac5a3))
* **py:** utility function to unquote a string literal coming from js2py handlebars helpers ([#107](https://github.com/hendrixmar/dotprompt/issues/107)) ([b3672ca](https://github.com/hendrixmar/dotprompt/commit/b3672ca6192de4895585b28b8bbd301f8294090f))
* **py:** utility to remove undefined fields from dicts/lists recursively ([#105](https://github.com/hendrixmar/dotprompt/issues/105)) ([d25c911](https://github.com/hendrixmar/dotprompt/commit/d25c911bc1e84e5691b961a4c38a8bcd73c80aa0))


### Bug Fixes

* change project name for pypi publish ([#200](https://github.com/hendrixmar/dotprompt/issues/200)) ([2c07132](https://github.com/hendrixmar/dotprompt/commit/2c0713264fb2c30bdc43f1bd9e51d416f96d1b7e))
* **docs:** update docs for helpers.py functions ([#118](https://github.com/hendrixmar/dotprompt/issues/118)) ([40f74d4](https://github.com/hendrixmar/dotprompt/commit/40f74d4cf75a47d8b7f9f85801a1bb5969bae082))
* **docs:** update helper docs ([#132](https://github.com/hendrixmar/dotprompt/issues/132)) ([9b84245](https://github.com/hendrixmar/dotprompt/commit/9b842459e8faa5f4afe7d389deb6c351ab1271be))
* **go,py:** type fixes and ensure we build/lint the go code in hooks and ci ([#83](https://github.com/hendrixmar/dotprompt/issues/83)) ([19a8257](https://github.com/hendrixmar/dotprompt/commit/19a8257f4f73b776229d5324a0366fd9a79c20aa))
* **helpers:** use ctx instead of hash to get the fn and inverse ([#131](https://github.com/hendrixmar/dotprompt/issues/131)) ([8749d1f](https://github.com/hendrixmar/dotprompt/commit/8749d1f78ee754742ae7fcc9247854021178bdbc))
* **license:** use the full license header in source code ([#142](https://github.com/hendrixmar/dotprompt/issues/142)) ([64894ef](https://github.com/hendrixmar/dotprompt/commit/64894ef898876b861c6c244d522f634cd8fcc842))
* **py/dotpromptz:** add todo about caching resolved schema for later ([#223](https://github.com/hendrixmar/dotprompt/issues/223)) ([728dbaa](https://github.com/hendrixmar/dotprompt/commit/728dbaaef1d0569148426a97242edd1f8064cdbe))
* **py/dotpromptz:** add unit tests for resolver and fix sync resolver error handling ([#208](https://github.com/hendrixmar/dotprompt/issues/208)) ([5e04e28](https://github.com/hendrixmar/dotprompt/commit/5e04e28c99eff0f83c9c8a15df5ef56ff3ebd85f))
* **py/dotpromptz:** address compatibility with python 3.10 and add tox configuration for parallelized tests ([#188](https://github.com/hendrixmar/dotprompt/issues/188)) ([d2ba21f](https://github.com/hendrixmar/dotprompt/commit/d2ba21ff3e54f4ca4328b7e574bb6492699095bc))
* **py/dotpromptz:** dotprompt.define_tool takes just a tool def ([#224](https://github.com/hendrixmar/dotprompt/issues/224)) ([72039fc](https://github.com/hendrixmar/dotprompt/commit/72039fc689d82344bfc1300345059340603404bc))
* **py/dotpromptz:** fix broken picoschema parser tests ([#232](https://github.com/hendrixmar/dotprompt/issues/232)) ([4d154aa](https://github.com/hendrixmar/dotprompt/commit/4d154aaae99f2f31500d86ed6191a05298700b91))
* **py/dotpromptz:** fix pydantic warnings and tabulate the various types in typing.py ([#196](https://github.com/hendrixmar/dotprompt/issues/196)) ([24dcfdd](https://github.com/hendrixmar/dotprompt/commit/24dcfdd320884452ee48fca859619b400fe61327))
* **py/dotpromptz:** make resolver concrete in resolve, resolve_tool, and resolve_partial ([#210](https://github.com/hendrixmar/dotprompt/issues/210)) ([3730190](https://github.com/hendrixmar/dotprompt/commit/37301903c8f26f1b8363ad1f2d515a6df19303e2))
* **py/dotpromptz:** remove stray reference to inspect.isasyncgenfunction since we dont deal with those as resolvers ([#216](https://github.com/hendrixmar/dotprompt/issues/216)) ([c300515](https://github.com/hendrixmar/dotprompt/commit/c3005159aea0af906e3f704d03fd5859a9540d4f))
* **py/dotpromptz:** remove unused Options type from dotprompt.py and fix typo ([#213](https://github.com/hendrixmar/dotprompt/issues/213)) ([ae59431](https://github.com/hendrixmar/dotprompt/commit/ae5943179e77aa6fa775d092254d8adf21b06eb2))
* **py/dotpromptz:** some lint and add docstrings ([#235](https://github.com/hendrixmar/dotprompt/issues/235)) ([2e1c893](https://github.com/hendrixmar/dotprompt/commit/2e1c893bbbe7c29480d67e2d693db90e9bef9b1b))
* **py/spec_test:** hex-encoded SHA-256 digest rather than base64-encoded SHA-256 digest for module IDs ([#140](https://github.com/hendrixmar/dotprompt/issues/140)) ([796c644](https://github.com/hendrixmar/dotprompt/commit/796c6442a3c1836de2170c466966382a0577a940))
* **py:** lint reporting missing docstrings in test files ([#199](https://github.com/hendrixmar/dotprompt/issues/199)) ([4d91514](https://github.com/hendrixmar/dotprompt/commit/4d9151468ddd13f334454701daae42d5717d4dcf))
* remove spurious role type `assistant` ([#169](https://github.com/hendrixmar/dotprompt/issues/169)) ([1b5142c](https://github.com/hendrixmar/dotprompt/commit/1b5142c4a7ad20ef722d438cefa0b93a82d7adbb))


### Documentation

* add initial mkdocs documentation for eng [#43](https://github.com/hendrixmar/dotprompt/issues/43) ([#44](https://github.com/hendrixmar/dotprompt/issues/44)) ([31be336](https://github.com/hendrixmar/dotprompt/commit/31be336d14899acf7ea1cefb4b782f5b2d1c31d1))
* **py/dotpromptz:** document awaitable type hierarchy as ASCII diagram in comments to make code readable ([#212](https://github.com/hendrixmar/dotprompt/issues/212)) ([0926749](https://github.com/hendrixmar/dotprompt/commit/0926749e72d17264b8078151b10f616a906cfd66))
* **py:** add docstring to parse.py and dotprompt.py modules. ([#246](https://github.com/hendrixmar/dotprompt/issues/246)) ([c9d53bb](https://github.com/hendrixmar/dotprompt/commit/c9d53bb3cad96f9d8ef5778ada93ee65276afa09))
* **py:** add documentation for the resolvers module ([#209](https://github.com/hendrixmar/dotprompt/issues/209)) ([04b6691](https://github.com/hendrixmar/dotprompt/commit/04b6691227ef5b01ebf5261d4846a3ba7e723ab2))
* **py:** ascii diagrams to explain the relationships between the types for easy visualization ([#197](https://github.com/hendrixmar/dotprompt/issues/197)) ([6d775c4](https://github.com/hendrixmar/dotprompt/commit/6d775c4bf1301c63d111b0d45db53dba61117555))
