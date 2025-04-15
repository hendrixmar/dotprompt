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


import anyio
from collections.abc import Awaitable
from typing import Any, Callable, TypedDict
from inspect import iscoroutinefunction

from dotpromptz.helpers import register_all_helpers
from dotpromptz.parse import parse_document
from dotpromptz.resolvers import resolve_tool
from dotpromptz.typing import (
    JsonSchema,
    ModelConfigT,
    ParsedPrompt,
    PartialResolver,
    PromptStore,
    SchemaResolver,
    ToolDefinition,
    ToolResolver,
    PromptMetadata,
)
from dotpromptz.picoschema import picoschema
from handlebarrz import EscapeFunction, Handlebars, HelperFn

# Pre-compiled regex for finding partial references in handlebars templates

# Since the handlebars-rust implementation doesn't expose a visitor
# to walk the AST to find partial nodes, we're using a crude regular expression
# to find partials.
_PARTIAL_PATTERN = re.compile(r'{{\s*>\s*([a-zA-Z0-9_.-]+)\s*}}')


class Options(TypedDict, total=False):
    """Options for dotprompt."""


class Dotprompt:
    """Dotprompt extends a Handlebars template for use with Gen AI prompts."""

    def __init__(
        self,
        default_model: str | None = None,
        model_configs: dict[str, Any] | None = None,
        helpers: dict[str, HelperFn] | None = None,
        partials: dict[str, str] | None = None,
        tools: dict[str, ToolDefinition] | None = None,
        tool_resolver: ToolResolver | None = None,
        schemas: dict[str, JsonSchema] | None = None,
        schema_resolver: SchemaResolver | None = None,
        partial_resolver: PartialResolver | None = None,
        escape_fn: EscapeFunction = EscapeFunction.NO_ESCAPE,
    ) -> None:
        """Initialize Dotprompt with a Handlebars template.

        Args:
            default_model: The default model to use for the prompt when not specified in the template.
            model_configs: Assign a set of default configuration options to be used with a particular model.
            helpers: Helpers to pre-register.
            partials: Partials to pre-register.
            tools: Provide a static mapping of tool definitions that should be used when resolving tool names.
            tool_resolver: Provide a lookup implementation to resolve tool names to definitions.
            schemas: Provide a static mapping of schema names to their JSON schema definitions.
            schema_resolver: resolver for schema names to JSON schema definitions.
            partial_resolver: resolver for partial names to their content.
            escape_fn: ecape function
        """
        self._handlebars: Handlebars = Handlebars(escape_fn=escape_fn)

        self._known_helpers: dict[str, bool] = {}
        self._default_model: str | None = default_model
        self._model_configs: dict[str, Any] = model_configs or {}
        self._helpers: dict[str, HelperFn] = helpers or {}
        self._partials: dict[str, str] = partials or {}
        self._tools: dict[str, ToolDefinition] = tools or {}
        self._tool_resolver: ToolResolver | None = tool_resolver
        self._schemas: dict[str, JsonSchema] = schemas or {}
        self._schema_resolver: SchemaResolver | None = schema_resolver
        self._partial_resolver: PartialResolver | None = partial_resolver
        self._store: PromptStore | None = None

        self._register_initial_helpers()
        self._register_initial_partials()

    def _register_initial_helpers(self) -> None:
        """Register the initial helpers."""
        register_all_helpers(self._handlebars)
        for name, fn in self._helpers.items():
            self._handlebars.register_helper(name, fn)

    def _register_initial_partials(self) -> None:
        """Register the initial partials."""
        for name, source in self._partials.items():
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

    def define_tool(self, definition: ToolDefinition) -> Dotprompt:
        """Define a tool for the template.

        Args:
            name: The name of the tool.
            definition: The definition of the tool.

        Returns:
            The Dotprompt instance.
        """
        self._tools[definition.name] = definition
        return self

    def parse(self, source: str) -> ParsedPrompt[Any]:
        """Parse a prompt from a string.

        Args:
            source: The source code for the prompt.

        Returns:
            The parsed prompt.
        """
        return parse_document(source)

    def _identify_partials(self, template: str) -> set[str]:
        """Identify all partial references in a template.

        Args:
            template: The template to scan for partial references.

        Returns:
            A set of partial names referenced in the template.
        """
        return set(_PARTIAL_PATTERN.findall(template))

    async def _resolve_tools(self, metadata: PromptMetadata[ModelConfigT]) -> PromptMetadata[ModelConfigT]:
        """Resolve all tools in a prompt.

        Args:
            metadata: The prompt metadata.

        Returns:
            A copy of the prompt metadata with the tools resolved.

        Raises:
            ToolNotFoundError: If a tool is not found in the resolver or store.
            ToolResolverFailedError: If a tool resolver fails.
            TypeError: If a tool resolver returns an invalid type.
            ValueError: If a tool resolver is not defined.
        """
        out: PromptMetadata[ModelConfigT] = metadata.model_copy()
        if out.tools is None:
            return out

        # Resolve tools that are already registered into toolDefs, leave
        # unregistered tools alone.
        unregistered_names: list[str] = []
        out.tool_defs = out.tool_defs or []

        # Collect all the tools:
        # 1. Already registered go into toolDefs.
        # 2. If we have a tool resolver, add the names to the list to resolve.
        # 3. Otherwise, add the names to the list of unregistered tools.
        to_resolve: list[str] = []
        have_resolver = self._tool_resolver is not None
        for name in out.tools:
            if name in self._tools:
                # Found locally.
                out.tool_defs.append(self._tools[name])
            elif have_resolver:
                # Resolve from the tool resolver.
                to_resolve.append(name)
            else:
                # Unregistered tool.
                unregistered_names.append(name)

        if to_resolve:
            async def resolve_and_append(name: str) -> None:
                """Resolve a tool and append it to the list of tools.

                Args:
                    name: The name of the tool to resolve.

                Raises:
                    ToolNotFoundError: If a tool is not found in the resolver or store.
                    ToolResolverFailedError: If a tool resolver fails.
                    TypeError: If a tool resolver returns an invalid type.
                    ValueError: If a tool resolver is not defined.
                """

                tool = await resolve_tool(name, self._tool_resolver)

                if out.tool_defs is not None:
                    out.tool_defs.append(tool)

            try:
                async with anyio.create_task_group() as tg:
                    for name in to_resolve:
                        tg.start_soon(resolve_and_append, name)
            except ExceptionGroup:
                raise Exception(
                    f"Dotprompt: Unable to resolve tool '{name}' to a recognized tool definition."
                )

        out.tools = unregistered_names
        return out

    async def render_metadata(self, source: str | ParsedPrompt, additional_metadata: PromptMetadata | None = None) -> PromptMetadata:
        """Processes and resolves all metadata for a prompt template.

        Args:
            source: The template source or parsed prompt.
            additional_metadata: Additional metadata to include.

        Returns:
            A future or awaitable resolving to the fully processed metadata.
        """
        breakpoint()
        match source:
            case str():
                parsed_source = self.parse(source)
            case _:
                parsed_source = source

        selected_model = (
            (additional_metadata and additional_metadata.model) or
            parsed_source.model or
            self._default_model
        )

        model_config = self._model_configs.get(selected_model, None)

        metadata = [
            additional_metadata
        ]

        _ = PromptMetadata.model_validate(parsed_source.model_dump(exclude={"template"}), from_attributes=True)
        breakpoint()

        param_merge = _.model_copy(update={"config": model_config} if model_config else {})
        result = await self._resolve_metadata(param_merge, *metadata)
        breakpoint()

        return result

    async def _resolve_metadata(self, base: PromptMetadata, *merges: PromptMetadata) -> PromptMetadata:
        """Merges multiple metadata objects together, resolving tools and schemas.

        Args:
            base: The base metadata object.
            merges: Additional metadata objects to merge into the base.

        Returns:
            The merged and processed metadata.
        """
        out = base
        for merge in merges:
            if merge is None:
                continue
            out = PromptMetadata(**{**out.model_dump(), **merge.model_dump(exclude_none=True, exclude={'config'})})

            if merge.config is not None:
                out.config.update(merge.config or {})

        out = await self._resolve_tools(out)
        out = self._render_pico_schema(out)
        return out


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
                schema=meta.input.model_dump(exclude_none=True),
                schema_resolver=self._wrapped_schema_resolver
            )
            new_meta.input = schema

        if meta.output.schema_:
            schema = picoschema(
                schema=meta.output.model_dump(exclude_none=True),
                schema_resolver=self._wrapped_schema_resolver
            )
            new_meta.output = schema

        return new_meta

    async def _wrapped_schema_resolver(self, name ) -> JsonSchema | None | Awaitable[JsonSchema | None]:
        """Wraps a schema resolver to return a callable that resolves schema name to its definition, using registered schemas or a schema resolver.

            Returns:
                Callable[[str], JsonSchema | None | Awaitable[JsonSchema | None]]:
                    A function that takes a schema name and returns either the resolved schema,
                    an awaitable resolving to it, or `None` if not found.
        """

        if schema := self._schemas.get(name):
            return schema

        if self._schema_resolver and iscoroutinefunction(self._schema_resolver):
            return await self._schema_resolver(name)

        if self._schema_resolver:
            return self._schema_resolver(name)

        return None






