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
from typing import Any

import anyio

from dotpromptz.helpers import register_all_helpers
from dotpromptz.parse import parse_document, to_messages
from dotpromptz.picoschema import picoschema_to_json_schema
from dotpromptz.resolvers import resolve_json_schema, resolve_partial, resolve_tool
from dotpromptz.typing import (
    JsonSchema,
    ModelConfigT,
    ParsedPrompt,
    PartialResolver,
    PromptMetadata,
    PromptStore,
    SchemaResolver,
    ToolDefinition,
    ToolResolver, DataArgument, PromptFunction,
)
from dotpromptz.util import remove_undefined_fields
from handlebarrz import EscapeFunction, Handlebars, HelperFn

# Pre-compiled regex for finding partial references in handlebars templates

# Since the handlebars-rust implementation doesn't expose a visitor
# to walk the AST to find partial nodes, we're using a crude regular expression
# to find partials.
_PARTIAL_PATTERN = re.compile(r'{{\s*>\s*([a-zA-Z0-9_.-]+)\s*}}')


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
            escape_fn: escape function to use for the template.
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

    async def render_metadata(
        self,
        source: str | ParsedPrompt[ModelConfigT],
        additional_metadata: PromptMetadata[ModelConfigT] | None = None,
    ) -> PromptMetadata[ModelConfigT]:
        """Render metadata for a prompt.

        Args:
            source: The source code for the prompt or a parsed prompt.
            additional_metadata: Additional metadata to be used to render the prompt.

        Returns:
            The rendered metadata.
        """
        prompt = self.parse(source) if isinstance(source, str) else source

        default_model = prompt.model or self._default_model
        model = additional_metadata.model if additional_metadata else default_model

        config: ModelConfigT | None = None
        if model is not None and self._model_configs.get(model) is not None:
            config = self._model_configs.get(model)

        return await self._resolve_metadata(
            PromptMetadata[ModelConfigT](
                config=config,
            )
            if config is not None
            else PromptMetadata[ModelConfigT](),
            prompt,
            additional_metadata,
        )

    async def _resolve_metadata(
        self, base: PromptMetadata[ModelConfigT], *merges: PromptMetadata[ModelConfigT] | None
    ) -> PromptMetadata[ModelConfigT]:
        """Merges multiple metadata objects together, resolving tools and schemas.

        Later metadata objects override earlier ones.

        Args:
            base: The base metadata object.
            merges: Additional metadata objects to merge into base.

        Returns:
            Merged metadata.
        """
        out = base.model_copy(deep=True)

        for merge in (m for m in merges if m is not None):
            out = self._merge_metadata(out, merge)

        # Remove the template attribute if it exists (TS does this).
        if hasattr(out, 'template'):
            delattr(out, 'template')

        out = remove_undefined_fields(out)
        # TODO: can this be done concurrently?
        out = await self._resolve_tools(out)
        out = await self._render_picoschema(out)
        return out

    def _merge_metadata(
        self,
        current: PromptMetadata[ModelConfigT],
        merge: PromptMetadata[ModelConfigT],
    ) -> PromptMetadata[ModelConfigT]:
        """Merges a single metadata object into the current one.

        Args:
            current: The current metadata object.
            merge: The metadata object to merge into the current one.

        Returns:
            The merged metadata object.
        """
        # Convert Pydantic models to raw dicts by alias first. Skip None values.
        merge_dict = merge.model_dump(exclude_none=True, by_alias=True)
        current_dict = current.model_dump(exclude_none=True, by_alias=True)

        # Keep a reference to the original config.
        original_config = current_dict.get('config', {})
        new_config = merge_dict.get('config', {})

        # Merge the new metadata.
        current_dict.update(merge_dict)

        # Merge the configs and set the resulting config.
        current_dict['config'] = {**original_config, **new_config}

        # Recreate the Pydantic model from the merged dict and validate it.
        return PromptMetadata[ModelConfigT].model_validate(current_dict)

    async def _render_picoschema(self, meta: PromptMetadata[ModelConfigT]) -> PromptMetadata[ModelConfigT]:
        """Render a Picoschema prompt.

        Args:
            meta: The prompt metadata.

        Returns:
            The rendered prompt metadata.
        """
        needs_input_processing = meta.input is not None and meta.input.schema_ is not None
        needs_output_processing = meta.output is not None and meta.output.schema_ is not None

        if not needs_input_processing and not needs_output_processing:
            return meta

        new_meta = meta.model_copy(deep=True)

        async def _process_input_schema(schema_to_process: Any) -> None:
            if new_meta.input is not None:
                new_meta.input.schema_ = await picoschema_to_json_schema(
                    schema_to_process,
                    self._wrapped_schema_resolver,
                )

        async def _process_output_schema(schema_to_process: Any) -> None:
            if new_meta.output is not None:
                new_meta.output.schema_ = await picoschema_to_json_schema(
                    schema_to_process,
                    self._wrapped_schema_resolver,
                )

        async with anyio.create_task_group() as tg:
            if needs_input_processing and meta.input is not None:
                # TODO: use meta.input.model_dump(exclude_none=True)?
                tg.start_soon(_process_input_schema, meta.input.schema_)
            if needs_output_processing and meta.output is not None:
                # TODO: use meta.output.model_dump(exclude_none=True)?
                tg.start_soon(_process_output_schema, meta.output.schema_)

        return new_meta

    def _identify_partials(self, template: str) -> set[str]:
        """Identify all unique partial references in a template.

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
        out: PromptMetadata[ModelConfigT] = metadata.model_copy(deep=True)
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

            async with anyio.create_task_group() as tg:
                for name in to_resolve:
                    tg.start_soon(resolve_and_append, name)

        out.tools = unregistered_names
        return out

    async def _resolve_partials(self, template: str) -> None:
        """Resolve all partials in a template.

        Args:
            template: The template to resolve partials in.

        Returns:
            None
        """
        if self._partial_resolver is None and self._store is None:
            return

        names = self._identify_partials(template)
        unregistered_names: list[str] = [name for name in names if not self._handlebars.has_partial(name)]

        async def resolve_and_register(name: str) -> None:
            """Resolve a partial from the resolver or store and register it.

            The partial resolver is preferred, and the store is used as a
            fallback. If neither is available, the partial is not registered.

            Args:
                name: The name of the partial to resolve.

            Returns:
                None.
            """
            content: str | None = None

            if self._partial_resolver is not None:
                content = await resolve_partial(name, self._partial_resolver)

            if content is None and self._store is not None:
                partial = await self._store.load_partial(name)
                if partial is not None:
                    content = partial.source

            if content is not None:
                self.define_partial(name, content)

                # Recursively resolve partials in the content.
                await self._resolve_partials(content)

        async with anyio.create_task_group() as tg:
            for name in unregistered_names:
                tg.start_soon(resolve_and_register, name)

    async def _wrapped_schema_resolver(self, name: str) -> JsonSchema | None:
        """Resolve a schema from either instance local mapping or the resolver.

        Args:
            name: The name of the schema to resolve.

        Returns:
            The resolved schema or None if it is not found.
        """
        if name in self._schemas:
            return self._schemas[name]

        if self._schema_resolver is None:
            return None

        # TODO: Should we cache the resolved schema in self._schemas?
        return await resolve_json_schema(name, self._schema_resolver)


    async def compile(
            self,
            source: str | ParsedPrompt[ModelConfigT],
            additional_metadata: PromptMetadata[ModelConfigT] | None = None
        ) -> PromptFunction:

        parsed_source = self.parse(source) if isinstance(source, str) else source



        await self._resolve_partials(parsed_source.template)

        render_string = self._handlebars.compile(parsed_source.template)


        async def render_func(
            data: DataArgument,
            options: PromptMetadata[ModelConfigT] | None
        ):
            merged_metadata = await self.render_metadata(parsed_source)


            rendered_string = render_string(
                {
                    **((options and options.input and options.input.default) or {}),
                    **(data.input or {})
                },
                {
                    "data": {
                        "metadata": {
                            "prompt": merged_metadata.metadata,
                            "docs": data.docs,
                            "messages": data.messages,
                        },
                        **(data.context or {})
                    }
                }
            )

            return PromptMetadata.model_validate(
                {
                    **merged_metadata.model_dump(),
                    "messages": to_messages(rendered_string, data),
                }
            )

