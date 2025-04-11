# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Dotpromptz is a library for generating prompts using Handlebars templates."""

from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Any, TypedDict, Callable
from inspect import iscoroutinefunction

from dotpromptz.helpers import register_all_helpers
from dotpromptz.parse import parse_document
from dotpromptz.typing import (
    JsonSchema,
    ParsedPrompt,
    PartialResolver,
    PromptStore,
    SchemaResolver,
    ToolDefinition,
    ToolResolver,
    PromptMetadata,
    DataArgument,
    RenderedPrompt,
)
from dotpromptz.picoschema import picoschema, PicoschemaOptions
from handlebarrz import EscapeFunction, Handlebars, HelperFn

# Pre-compiled regex for finding partial references in handlebars templates

# Since the handlebars-rust implementation doesn't expose a visitor
# to walk the AST to find partial nodes, we're using a crude regular expression
# to find partials.
_PARTIAL_PATTERN = re.compile(r'{{\s*>\s*([a-zA-Z0-9_.-]+)\s*}}')


class Options(TypedDict, total=False):
    """Options for dotprompt."""

    # The default model to use for the prompt when not specified in the
    # template.
    default_model: str | None
    # Assign a set of default configuration options to be used with a particular
    # model.
    model_configs: dict[str, Any] | None
    # Helpers to pre-register.
    helpers: dict[str, HelperFn] | None
    # Partials to pre-register.
    partials: dict[str, str] | None
    # Provide a static mapping of tool definitions that should be used when
    # resolving tool names.
    tools: dict[str, ToolDefinition] | None
    # Provide a lookup implementation to resolve tool names to definitions.
    tool_resolver: ToolResolver | None
    # Provide a static mapping of schema names to their JSON schema definitions.
    schemas: dict[str, JsonSchema] | None
    # Provide a lookup implementation to resolve schema names to JSON schema
    # definitions.
    schema_resolver: SchemaResolver | None
    # Provide a lookup implementation to resolve partial names to their content.
    partial_resolver: PartialResolver | None


class Dotprompt:
    """Dotprompt extends a Handlebars template for use with Gen AI prompts."""

    def __init__(
        self,
        options: Options | None = None,
        escape_fn: EscapeFunction = EscapeFunction.NO_ESCAPE,
    ) -> None:
        """Initialize Dotprompt with a Handlebars template.

        Args:
            options: Options for Dotprompt.
        """
        self._options: Options = options or {}
        self._handlebars: Handlebars = Handlebars(escape_fn=escape_fn)

        self._known_helpers: dict[str, bool] = {}
        self._default_model: str | None = self._options.get('default_model')
        self._model_configs: dict[str, Any] | None = self._options.get('model_configs', {})
        self._tools: dict[str, ToolDefinition] = self._options.get('tools', {}) or {}
        self._tool_resolver: ToolResolver | None = self._options.get('tool_resolver')
        self._schemas: dict[str, JsonSchema] | None = self._options.get('schemas', {})
        self._schema_resolver: SchemaResolver | None = self._options.get('schema_resolver')
        self._partial_resolver: PartialResolver | None = self._options.get('partial_resolver')
        self._store: PromptStore | None = None

        self._register_initial_helpers()
        self._register_initial_partials()

    def _register_initial_helpers(self) -> None:
        """Register the initial helpers."""
        register_all_helpers(self._handlebars)
        for name, fn in (self._options.get('helpers') or {}).items():
            self._handlebars.register_helper(name, fn)

    def _register_initial_partials(self) -> None:
        """Register the initial partials."""
        for name, source in (self._options.get('partials') or {}).items():
            self._handlebars.register_partial(name, source)

    def define_helper(self, name: str, fn: HelperFn) -> Dotprompt:
        """Define a helper function for the template.

        Args:
            name: The name of the helper function.
            fn: The function to be called when the helper is used in the
                template.

        Returns:
            The Dotprompt instance.
        """
        self._handlebars.register_helper(name, fn)
        self._known_helpers[name] = True
        return self

    def define_partial(self, name: str, source: str) -> Dotprompt:
        """Define a partial template for the template.

        Args:
            name: The name of the partial template.
            source: The source code for the partial.

        Returns:
            The Dotprompt instance.
        """
        self._handlebars.register_partial(name, source)
        return self

    def define_tool(self, name: str, definition: ToolDefinition) -> Dotprompt:
        """Define a tool for the template.

        Args:
            name: The name of the tool.
            definition: The definition of the tool.

        Returns:
            The Dotprompt instance.
        """
        self._tools[name] = definition
        return self

    def parse(self, source: str) -> ParsedPrompt[Any] | None:
        """Parse a prompt from a string.

        Args:
            source: The source code for the prompt.

        Returns:
            The parsed prompt.
        """
        return parse_document(source)

    def identify_partials(self, template: str) -> set[str]:
        """Identify all partial references in a template.

        Args:
            template: The template to scan for partial references.

        Returns:
            A set of partial names referenced in the template.
        """
        partials = set(_PARTIAL_PATTERN.findall(template))
        return partials

    async def render_metadata(self, source: str | ParsedPrompt, additional_metadata: PromptMetadata | None = None) -> PromptMetadata:
        """Processes and resolves all metadata for a prompt template.

        Args:
            source: The template source or parsed prompt.
            additional_metadata: Additional metadata to include.

        Returns:
            A future or awaitable resolving to the fully processed metadata.
        """
        match source:
            case str():
                parsed_source = self.parse(source)
                if parsed_source is None:
                    return PromptMetadata()
            case ParsedPrompt():
                parsed_source = source
            case _:
                return PromptMetadata()

        if additional_metadata is None:
            additional_metadata = ParsedPrompt(
                ext={},
                config=None,
                metadata={},
                toolDefs=None,
                template=source
            )

        selected_model = (
            additional_metadata.model or
            parsed_source.model or
            self._default_model
        )


        model_config = self._model_configs.get(selected_model, {})

        metadata = [
            parsed_source.prompt_metadata,
            additional_metadata
        ]
        return await self._resolve_metadata(PromptMetadata(config=model_config), metadata)

    async def _resolve_metadata(self, base: PromptMetadata, merges: list[PromptMetadata]) -> PromptMetadata:
        """Merges multiple metadata objects together, resolving tools and schemas.

        Args:
            base: The base metadata object.
            merges: Additional metadata objects to merge into the base.

        Returns:
            The merged and processed metadata.
        """

        out = base
        for merge in merges:
            out = PromptMetadata(**{**out.model_dump(), **merge.model_dump(exclude_none=True)})

            if out.config is not None:
                out.config.update(merge.config or {})

        out = await self._resolve_tools(out)
        return self._render_pico_schema(out)


    def _render_pico_schema(self, meta: PromptMetadata) -> PromptMetadata:
        """Processes schema definitions in picoschema format into standard JSON Schema.

        Args:
            meta: The prompt metadata containing schema definitions.

        Returns:
            A processed metadata with expanded schemas.
        """

        if meta.output.schema_ is None and meta.input.schema_ is None:
            return meta

        new_meta = meta

        if meta.input.schema_:
            schema = picoschema(
                schema=meta.input.schema_,
                options=PicoschemaOptions(
                    schema_resolver=self._wrapped_schema_resolver()
                )
            )
            new_meta.input.schema_ = schema

        if meta.output.schema_:
            schema = picoschema(
                schema=meta.output.schema_,
                options=PicoschemaOptions(
                    schema_resolver=self._wrapped_schema_resolver()
                )
            )
            new_meta.output.schema_ = schema

        return new_meta

    def _wrapped_schema_resolver(self) -> Callable[[str], JsonSchema | None | Awaitable[JsonSchema | None]]:
        """Wraps a schema resolver to return a callable that resolves schema name to its definition, using registered schemas or a schema resolver.

            Returns:
                Callable[[str], JsonSchema | None | Awaitable[JsonSchema | None]]:
                    A function that takes a schema name and returns either the resolved schema,
                    an awaitable resolving to it, or `None` if not found.
        """
        async def wrapper(name: str) -> JsonSchema | None | Awaitable[JsonSchema | None]:

            if schema := self._schemas.get(name):
                return schema

            if self._schema_resolver and iscoroutinefunction(self._schema_resolver):
                return await self._schema_resolver(name)

            if self._schema_resolver:
                return self._schema_resolver(name)

            return None

        return wrapper


    async def _resolve_tools(self, base: PromptMetadata) -> PromptMetadata | None:
        """Resolves tool names to their definitions using registered tools or a tool resolver.

        Args:
            base: The metadata containing tool references to resolve.

        Returns:
            Metadata with the tool definitions fully resolved.

        """
        out = base
        if not base.tools:
            return out
        if out.tool_defs is None:
            out.tool_defs = []

        out_tools = []
        for tool_name in out.tools:
            if tool := self._tools.get(tool_name) and out.tool_defs:
                out.tool_defs.append(tool)
            elif self._tool_resolver is not None:
                if iscoroutinefunction(self._tool_resolver):
                    resolved_tool = await self._tool_resolver(tool_name)
                else:
                    resolved_tool = self._tool_resolver(tool_name)

                if resolved_tool is None:
                    return None
                if out.tool_defs:
                    out.tool_defs.append(resolved_tool)
            else:
                out_tools.append(tool_name)

        out.tools = out_tools

        return out

