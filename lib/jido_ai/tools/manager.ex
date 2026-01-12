defmodule Jido.AI.Tools.Manager do
  @moduledoc """
  Manages tool execution and the tool calling loop for LLM conversations.

  The ToolsManager handles:
  - Converting Jido.Action modules to LLM tool schemas
  - Executing tool calls returned by LLM responses
  - Managing the iterative tool loop until completion
  - Supporting both streaming and non-streaming modes

  ## Usage

      # Process a message with tools (non-streaming)
      {:ok, response} = ToolsManager.process(
        conversation_id,
        "What's the weather in Paris?",
        [WeatherAction],
        max_iterations: 5
      )

      # Process with streaming
      {:ok, stream} = ToolsManager.process_stream(
        conversation_id,
        "Search for restaurants",
        [SearchAction]
      )

      # Execute tool calls directly
      results = ToolsManager.execute_tool_calls(tool_calls, [WeatherAction])
  """

  require Logger

  alias Jido.AI.Actions.ReqLlm.ChatCompletion
  alias Jido.AI.Conversation.Manager, as: ConversationManager
  alias Jido.AI.Tools.SchemaConverter

  @default_max_iterations 10
  @default_timeout 60_000

  @type tool_call :: %{
          id: String.t(),
          name: String.t(),
          arguments: map()
        }

  @type tool_result :: %{
          tool_call_id: String.t(),
          name: String.t(),
          output: term(),
          error: term() | nil
        }

  # =============================================================================
  # Public API - High Level
  # =============================================================================

  @doc """
  Processes a user message with tool support (non-streaming).

  This function:
  1. Adds the user message to the conversation
  2. Calls the LLM with available tools
  3. Executes any tool calls
  4. Loops until no more tool calls or max iterations reached
  5. Returns the final response

  ## Options

    * `:max_iterations` - Maximum tool loop iterations (default: 10)
    * `:timeout` - Timeout per LLM call in ms (default: 60000)
    * `:context` - Context map passed to action execution

  ## Examples

      {:ok, response} = ToolsManager.process(conv_id, "What's the weather?", [WeatherAction])
  """
  @spec process(String.t(), String.t(), [module()], keyword()) ::
          {:ok, map()} | {:error, term()}
  def process(conversation_id, user_message, action_modules, opts \\ []) do
    max_iterations = Keyword.get(opts, :max_iterations, @default_max_iterations)
    context = Keyword.get(opts, :context, %{})

    # Add user message to conversation
    :ok = ConversationManager.add_message(conversation_id, :user, user_message)

    # Get conversation state
    case ConversationManager.get(conversation_id) do
      {:ok, conversation} ->
        # Pass action modules to ChatCompletion, which will translate them to provider-specific tool schemas.
        tools = action_modules
        action_map = SchemaConverter.build_action_map(action_modules)

        # Run the tool loop
        process_loop(conversation, tools, action_map, context, max_iterations, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Processes a user message with tool support (streaming).

  Returns a stream that yields chunks as they arrive. Tool calls are
  executed between chunks when detected, and the loop continues until
  no more tool calls are needed.

  ## Options

    * `:max_iterations` - Maximum tool loop iterations (default: 10)
    * `:timeout` - Timeout per LLM call in ms (default: 60000)
    * `:context` - Context map passed to action execution
    * `:on_tool_call` - Callback function called when a tool is executed

  ## Examples

      {:ok, stream} = ToolsManager.process_stream(conv_id, "Search for X", [SearchAction])

      stream
      |> Stream.each(fn chunk ->
        case chunk do
          {:content, text} -> IO.write(text)
          {:tool_call, call} -> IO.puts("Calling tool: \#{call.name}")
          {:tool_result, result} -> IO.puts("Tool result: \#{inspect(result)}")
          {:done, response} -> IO.puts("Done!")
        end
      end)
      |> Stream.run()
  """
  @spec process_stream(String.t(), String.t(), [module()], keyword()) ::
          {:ok, Enumerable.t()} | {:error, term()}
  def process_stream(conversation_id, user_message, action_modules, opts \\ []) do
    max_iterations = Keyword.get(opts, :max_iterations, @default_max_iterations)
    context = Keyword.get(opts, :context, %{})

    # Add user message to conversation
    :ok = ConversationManager.add_message(conversation_id, :user, user_message)

    # Get conversation state
    case ConversationManager.get(conversation_id) do
      {:ok, conversation} ->
        # Pass action modules to ChatCompletion, which will translate them to provider-specific tool schemas.
        tools = action_modules
        action_map = SchemaConverter.build_action_map(action_modules)

        # Return a stream that processes the tool loop
        stream =
          Stream.resource(
            fn -> init_stream_state(conversation, tools, action_map, context, max_iterations, opts) end,
            &stream_next/1,
            fn _state -> :ok end
          )

        {:ok, stream}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Executes tool calls and returns results.

  ## Examples

      results = ToolsManager.execute_tool_calls(tool_calls, [WeatherAction])
      # => [%{tool_call_id: "...", name: "get_weather", output: %{temp: 20}}]
  """
  @spec execute_tool_calls([tool_call()], [module()], map()) :: [tool_result()]
  def execute_tool_calls(tool_calls, action_modules, context \\ %{}) do
    action_map = SchemaConverter.build_action_map(action_modules)
    execute_tools(tool_calls, action_map, context)
  end

  @doc """
  Converts action modules to tool schemas.

  ## Examples

      tools = ToolsManager.actions_to_tools([WeatherAction, SearchAction])
  """
  @spec actions_to_tools([module()]) :: [map()]
  defdelegate actions_to_tools(action_modules), to: SchemaConverter

  # =============================================================================
  # Private - Non-Streaming Loop
  # =============================================================================

  defp process_loop(_conversation, _tools, _action_map, _context, 0, _opts) do
    {:error, :max_iterations_exceeded}
  end

  defp process_loop(conversation, tools, action_map, context, iterations_left, opts) do
    # Get messages for LLM
    {:ok, messages} = ConversationManager.get_messages_for_llm(conversation.id)

    # Call LLM
    case call_llm(conversation.model, messages, tools, opts) do
      {:ok, response} ->
        handle_llm_response(
          response,
          conversation,
          tools,
          action_map,
          context,
          iterations_left,
          opts
        )

      {:error, reason} ->
        {:error, {:llm_error, reason}}
    end
  end

  defp handle_llm_response(
         response,
         conversation,
         tools,
         action_map,
         context,
         iterations_left,
         opts
       ) do
    content = Map.get(response, :content, "")
    tool_calls = extract_tool_calls(response)

    if tool_calls != [] do
      # Has tool calls - execute and continue loop
      # Save assistant message with tool calls
      :ok =
        ConversationManager.add_message(
          conversation.id,
          :assistant,
          content,
          tool_calls: tool_calls
        )

      # Execute tools
      tool_results = execute_tools(tool_calls, action_map, context)

      # Add tool results to conversation
      Enum.each(tool_results, fn result ->
        output =
          case result.output do
            s when is_binary(s) -> s
            other -> Jason.encode!(other)
          end

        :ok =
          ConversationManager.add_message(
            conversation.id,
            :tool,
            output,
            tool_call_id: result.tool_call_id,
            name: result.name
          )
      end)

      # Get updated conversation and continue loop
      {:ok, updated_conversation} = ConversationManager.get(conversation.id)
      process_loop(updated_conversation, tools, action_map, context, iterations_left - 1, opts)
    else
      # No tool calls - final response
      # Save final assistant message
      :ok = ConversationManager.add_message(conversation.id, :assistant, content)

      max_iterations = Keyword.get(opts, :max_iterations, @default_max_iterations)

      {:ok,
       %{
         content: content,
         conversation_id: conversation.id,
         tool_calls_made: max_iterations - iterations_left
       }}
    end
  end

  # =============================================================================
  # Private - Streaming Loop
  # =============================================================================

  defp init_stream_state(conversation, tools, action_map, context, max_iterations, opts) do
    llm_stream? = Keyword.get(opts, :llm_stream, true)
    emit_thinking? = Keyword.get(opts, :emit_thinking, false)

    %{
      conversation: conversation,
      tools: tools,
      action_map: action_map,
      context: context,
      opts: opts,
      llm_stream?: llm_stream?,
      emit_thinking?: emit_thinking?,
      iterations_left: max_iterations,
      phase: :call_llm,
      accumulated_content: "",
      accumulated_tool_calls: [],
      pending_tool_calls: [],
      stream_response: nil,
      stream_cursor: nil,
      stream_buffer: [],
      done: false
    }
  end

  defp stream_next(%{done: true} = state) do
    {:halt, state}
  end

  defp stream_next(%{phase: :call_llm} = state) do
    {:ok, messages} = ConversationManager.get_messages_for_llm(state.conversation.id)

    if state.llm_stream? do
      case call_llm_stream(state.conversation.model, messages, state.tools, state.opts) do
        {:ok, llm_stream} ->
          {stream_response, cursor} =
            case llm_stream do
              %ReqLLM.StreamResponse{stream: inner_stream} = sr ->
                {sr, {:start, inner_stream}}

              other ->
                {nil, {:start, other}}
            end

          new_state = %{
            state
            | phase: :streaming,
              stream_response: stream_response,
              stream_cursor: cursor,
              stream_buffer: [],
              accumulated_content: "",
              accumulated_tool_calls: []
          }

          stream_next(new_state)

        {:error, reason} ->
          {[{:error, reason}], %{state | done: true}}
      end
    else
      case call_llm(state.conversation.model, messages, state.tools, state.opts) do
        {:ok, response} ->
          handle_non_stream_response(response, state)

        {:error, reason} ->
          {[{:error, reason}], %{state | done: true}}
      end
    end
  end

  defp stream_next(%{phase: :streaming, stream_cursor: cursor} = state) when not is_nil(cursor) do
    case cursor_step(cursor) do
      {:chunk, raw_chunk, next_cursor} ->
        chunk = normalize_chunk(raw_chunk)
        content = Map.get(chunk, :content, "")
        thinking = Map.get(chunk, :thinking, "")
        chunk_tool_calls = Map.get(chunk, :tool_calls, [])
        new_content = state.accumulated_content <> content
        new_tool_calls = state.accumulated_tool_calls ++ chunk_tool_calls

        new_buffer =
          if is_struct(raw_chunk, ReqLLM.StreamChunk) do
            [raw_chunk | state.stream_buffer]
          else
            state.stream_buffer
          end

        new_state = %{
          state
          | stream_cursor: next_cursor,
            stream_buffer: new_buffer,
            accumulated_content: new_content,
            accumulated_tool_calls: new_tool_calls
        }

        cond do
          thinking != "" and state.emit_thinking? ->
            {[{:status, %{status: :thinking, message: thinking}}], new_state}

          content != "" ->
            {[{:content, content}], new_state}

          true ->
            stream_next(new_state)
        end

      {:done, _reason} ->
        final_response = finalize_streaming_response(state)
        handle_stream_response(final_response, state)
    end
  end

  defp stream_next(%{phase: :execute_tools, pending_tool_calls: tool_calls} = state) do
    # Execute tools and emit results
    Logger.debug("[Tools.Manager] Executing tools: #{inspect(tool_calls, limit: 3)}")
    results = execute_tools(tool_calls, state.action_map, state.context)

    # Add tool results to conversation
    Enum.each(results, fn result ->
      Logger.debug("[Tools.Manager] Tool result: #{result.name}, tool_call_id: #{inspect(result.tool_call_id)}")
      output =
        case result.output do
          s when is_binary(s) -> s
          other -> Jason.encode!(other)
        end

      :ok =
        ConversationManager.add_message(
          state.conversation.id,
          :tool,
          output,
          tool_call_id: result.tool_call_id,
          name: result.name
        )
    end)

    # Get updated conversation
    {:ok, updated_conversation} = ConversationManager.get(state.conversation.id)

    # Emit tool results and continue
    result_chunks = Enum.map(results, fn r -> {:tool_result, r} end)

    new_state = %{
      state
      | conversation: updated_conversation,
        phase: :call_llm,
        iterations_left: state.iterations_left - 1
    }

    if state.iterations_left <= 1 do
      {result_chunks ++ [{:error, :max_iterations_exceeded}], %{new_state | done: true}}
    else
      {result_chunks, new_state}
    end
  end

  defp handle_stream_response(response, state) do
    # Use accumulated content if response content is nil or empty
    response_content = Map.get(response, :content)
    content = if response_content && response_content != "", do: response_content, else: state.accumulated_content
    tool_calls = extract_tool_calls(response)

    if tool_calls != [] do
      # Save assistant message with tool calls
      :ok =
        ConversationManager.add_message(
          state.conversation.id,
          :assistant,
          content,
          tool_calls: tool_calls
        )

      # Emit tool call notifications
      tool_chunks = Enum.map(tool_calls, fn tc -> {:tool_call, tc} end)

      new_state = %{
        state
        | phase: :execute_tools,
          pending_tool_calls: tool_calls,
          stream_response: nil,
          stream_cursor: nil,
          stream_buffer: []
      }

      {tool_chunks, new_state}
    else
      # Final response
      :ok = ConversationManager.add_message(state.conversation.id, :assistant, content)

      final = %{
        content: content,
        conversation_id: state.conversation.id
      }

      {[{:done, final}], %{state | done: true}}
    end
  end

  defp handle_non_stream_response(response, state) do
    content = Map.get(response, :content, "") || ""
    tool_calls = extract_tool_calls(response)

    if tool_calls != [] do
      :ok =
        ConversationManager.add_message(
          state.conversation.id,
          :assistant,
          content,
          tool_calls: tool_calls
        )

      thinking_chunks =
        if state.emit_thinking? and content != "" do
          [{:status, %{status: :thinking, message: content}}]
        else
          []
        end

      tool_chunks = Enum.map(tool_calls, fn tc -> {:tool_call, tc} end)

      new_state = %{
        state
        | phase: :execute_tools,
          pending_tool_calls: tool_calls,
          stream_response: nil,
          stream_cursor: nil,
          stream_buffer: []
      }

      {thinking_chunks ++ tool_chunks, new_state}
    else
      :ok = ConversationManager.add_message(state.conversation.id, :assistant, content)

      final = %{
        content: content,
        conversation_id: state.conversation.id
      }

      # In non-stream mode we emit the full answer as a single :content chunk so
      # renderers can display it before :done.
      {[{:content, content}, {:done, final}], %{state | done: true}}
    end
  end

  defp cursor_step({:start, enumerable}) do
    reducer = fn chunk, _acc -> {:suspend, chunk} end

    case Enumerable.reduce(enumerable, {:cont, nil}, reducer) do
      {:suspended, chunk, cont} -> {:chunk, chunk, {:cont, cont}}
      {:done, _} -> {:done, :eos}
      {:halted, _} -> {:done, :halted}
    end
  rescue
    _ -> {:done, :error}
  end

  defp cursor_step({:cont, cont}) when is_function(cont, 1) do
    case cont.({:cont, nil}) do
      {:suspended, chunk, cont2} -> {:chunk, chunk, {:cont, cont2}}
      {:done, _} -> {:done, :eos}
      {:halted, _} -> {:done, :halted}
    end
  rescue
    _ -> {:done, :error}
  end

  defp finalize_streaming_response(%{stream_response: %ReqLLM.StreamResponse{} = sr} = state) do
    # NOTE: ReqLLM.StreamResponse streams are one-shot. We buffer chunks as they arrive and
    # reconstruct a legacy Response at the end (so tool_call args fragments are merged).
    sr = %{sr | stream: Enum.reverse(state.stream_buffer)}

    case ReqLLM.StreamResponse.to_response(sr) do
      {:ok, response} ->
        response_content = ReqLLM.Response.text(response) || ""

        content =
          if response_content != "" do
            response_content
          else
            state.accumulated_content
          end

        %{content: content, tool_calls: ReqLLM.Response.tool_calls(response) || []}

      {:error, _reason} ->
        %{content: state.accumulated_content, tool_calls: state.accumulated_tool_calls}
    end
  end

  defp finalize_streaming_response(state) do
    %{content: state.accumulated_content, tool_calls: state.accumulated_tool_calls}
  end

  # Normalize different chunk formats to a consistent map
  defp normalize_chunk(%ReqLLM.StreamChunk{type: :content} = chunk) do
    %{content: chunk.text || "", tool_calls: []}
  end

  defp normalize_chunk(%ReqLLM.StreamChunk{type: :thinking} = chunk) do
    %{thinking: chunk.text || "", content: "", tool_calls: []}
  end

  defp normalize_chunk(%ReqLLM.StreamChunk{type: :tool_call} = chunk) do
    tool_call = %{
      id: chunk.metadata[:id] || "call_#{:erlang.unique_integer([:positive])}",
      function: %{
        name: chunk.name,
        arguments: if(is_binary(chunk.arguments), do: chunk.arguments, else: Jason.encode!(chunk.arguments || %{}))
      }
    }
    %{content: "", tool_calls: [tool_call]}
  end

  defp normalize_chunk(%ReqLLM.StreamChunk{type: :meta}) do
    %{content: "", tool_calls: []}
  end

  defp normalize_chunk(%ReqLLM.StreamChunk{} = _chunk) do
    %{content: "", tool_calls: []}
  end

  defp normalize_chunk(chunk) when is_map(chunk), do: chunk
  defp normalize_chunk(_), do: %{content: ""}

  # =============================================================================
  # Private - Tool Execution
  # =============================================================================

  defp execute_tools(tool_calls, action_map, context) do
    Enum.map(tool_calls, fn tool_call ->
      name = get_tool_call_name(tool_call)
      args = get_tool_call_arguments(tool_call)
      id = get_tool_call_id(tool_call)

      case Map.get(action_map, name) do
        nil ->
          Logger.warning("[Tools.Manager] Unknown tool: #{name}")

          %{
            tool_call_id: id,
            name: name,
            output: %{error: "Unknown tool: #{name}"},
            error: :unknown_tool
          }

        action_module ->
          execute_single_tool(action_module, args, context, id, name)
      end
    end)
  end

  defp execute_single_tool(action_module, args, context, id, name) do
    # Convert string keys to atoms for the action
    atom_args = atomize_keys(args)

    case action_module.run(atom_args, context) do
      {:ok, result} ->
        %{
          tool_call_id: id,
          name: name,
          output: result,
          error: nil
        }

      {:error, reason} ->
        Logger.warning("[Tools.Manager] Tool #{name} failed: #{inspect(reason)}")

        %{
          tool_call_id: id,
          name: name,
          output: %{error: inspect(reason)},
          error: reason
        }
    end
  rescue
    e ->
      Logger.error("[Tools.Manager] Tool #{name} raised: #{Exception.message(e)}")

      %{
        tool_call_id: id,
        name: name,
        output: %{error: Exception.message(e)},
        error: e
      }
  end

  # =============================================================================
  # Private - LLM Calls
  # =============================================================================

  defp call_llm(model, messages, tools, opts) do
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    api_key = Keyword.get(opts, :api_key)

    params = %{
      model: model,
      messages: messages,
      tools: tools,
      timeout: timeout,
      api_key: api_key
    }

    # Use ChatCompletion action
    case ChatCompletion.run(params, %{}) do
      {:ok, response} -> {:ok, response}
      {:error, reason} -> {:error, reason}
    end
  end

  defp call_llm_stream(model, messages, tools, opts) do
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    api_key = Keyword.get(opts, :api_key)

    params = %{
      model: model,
      messages: messages,
      tools: tools,
      stream: true,
      timeout: timeout,
      api_key: api_key
    }

    case ChatCompletion.run(params, %{}) do
      {:ok, stream} -> {:ok, stream}
      {:error, reason} -> {:error, reason}
    end
  end

  # =============================================================================
  # Private - Helpers
  # =============================================================================

  defp extract_tool_calls(response) do
    # Handle different response formats
    cond do
      # tool_results from our ChatCompletion action
      is_list(response[:tool_results]) and response[:tool_results] != [] ->
        response[:tool_results]

      # tool_calls from raw LLM response
      is_list(response[:tool_calls]) and response[:tool_calls] != [] ->
        response[:tool_calls]

      # Nested in choices (OpenAI format)
      is_list(response[:choices]) ->
        response[:choices]
        |> List.first(%{})
        |> get_in([:message, :tool_calls])
        |> Kernel.||([])

      true ->
        []
    end
  end

  defp get_tool_call_name(%{name: name}), do: name
  defp get_tool_call_name(%{function: %{name: name}}), do: name
  defp get_tool_call_name(%{"name" => name}), do: name
  defp get_tool_call_name(%{"function" => %{"name" => name}}), do: name
  defp get_tool_call_name(_), do: "unknown"

  defp get_tool_call_arguments(%{arguments: args}) when is_map(args), do: args

  defp get_tool_call_arguments(%{arguments: args}) when is_binary(args) do
    case Jason.decode(args) do
      {:ok, decoded} -> decoded
      {:error, _} -> %{}
    end
  end

  defp get_tool_call_arguments(%{function: %{arguments: args}}),
    do: get_tool_call_arguments(%{arguments: args})

  defp get_tool_call_arguments(%{"arguments" => args}),
    do: get_tool_call_arguments(%{arguments: args})

  defp get_tool_call_arguments(%{"function" => f}),
    do: get_tool_call_arguments(%{function: atomize_keys(f)})

  defp get_tool_call_arguments(_), do: %{}

  defp get_tool_call_id(%{id: id}), do: id
  defp get_tool_call_id(%{"id" => id}), do: id
  defp get_tool_call_id(_), do: generate_tool_call_id()

  defp generate_tool_call_id do
    "call_" <> (:crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false))
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {String.to_existing_atom(k), atomize_keys(v)}
      {k, v} when is_atom(k) -> {k, atomize_keys(v)}
      {k, v} -> {k, atomize_keys(v)}
    end)
  rescue
    ArgumentError ->
      # If atom doesn't exist, keep string key
      Map.new(map, fn
        {k, v} when is_binary(k) ->
          atom_key =
            try do
              String.to_existing_atom(k)
            rescue
              _ -> String.to_atom(k)
            end

          {atom_key, atomize_keys(v)}

        {k, v} ->
          {k, atomize_keys(v)}
      end)
  end

  defp atomize_keys(list) when is_list(list), do: Enum.map(list, &atomize_keys/1)
  defp atomize_keys(value), do: value
end
