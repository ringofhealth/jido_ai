defmodule Jido.AI.Actions.ReqLlm.ChatCompletion do
  @moduledoc """
  Chat completion action using ReqLLM for multi-provider support.

  This action provides direct access to chat completion functionality across
  57+ providers through ReqLLM, replacing the LangChain-based implementation
  with lighter dependencies and broader provider support.

  ## Features

  - Multi-provider support (57+ providers via ReqLLM)
  - Tool/function calling capabilities
  - Response quality control with retry mechanisms
  - Support for various LLM parameters (temperature, top_p, etc.)
  - Structured error handling and logging
  - Streaming support (when provider allows)

  ## Usage

  ```elixir
  # Basic usage
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :anthropic, model: "claude-3-sonnet-20240229"},
    prompt: Jido.AI.Prompt.new(:user, "What's the weather in Tokyo?")
  })

  # With function calling / tools
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :openai, model: "gpt-4o"},
    prompt: prompt,
    tools: [Jido.Actions.Weather.GetWeather, Jido.Actions.Search.WebSearch],
    temperature: 0.2
  })

  # Streaming responses
  {:ok, stream} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: model,
    prompt: prompt,
    stream: true
  })

  Enum.each(stream, fn chunk ->
    IO.puts(chunk.content)
  end)
  ```

  ## Support Matrix

  Supports all providers available in ReqLLM (57+), including:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Mistral, Cohere, Groq, and many more

  See ReqLLM documentation for full provider list.
  """
  use Jido.Action,
    name: "reqllm_chat_completion",
    description: "Chat completion action using ReqLLM",
    schema: [
      model: [
        type: {:custom, Jido.AI.Model, :validate_model_opts, []},
        required: true,
        doc:
          "The AI model to use (e.g., {:anthropic, [model: \"claude-3-sonnet-20240229\"]} or %Jido.AI.Model{})"
      ],
      prompt: [
        type: {:custom, Jido.AI.Prompt, :validate_prompt_opts, []},
        required: false,
        doc: "The prompt to use for the response (alternative to :messages)"
      ],
      messages: [
        type: :any,
        required: false,
        doc:
          "A list of chat messages in ReqLLM format (alternative to :prompt). Supports tool calling fields like :tool_calls, :tool_call_id, :name."
      ],
      tools: [
        type: {:list, :any},
        required: false,
        doc:
          "Tools for function calling. Accepts a list of Jido.Action modules, ReqLLM.Tool structs, or OpenAI-style tool schema maps."
      ],
      api_key: [
        type: :string,
        required: false,
        doc:
          "Provider API key override for ReqLLM (useful for local testing). Prefer env vars in production."
      ],
      max_retries: [
        type: :integer,
        default: 0,
        doc: "Number of retries for validation failures"
      ],
      temperature: [type: :float, default: 0.7, doc: "Temperature for response randomness"],
      max_tokens: [type: :integer, default: 1000, doc: "Maximum tokens in response"],
      top_p: [type: :float, doc: "Top p sampling parameter"],
      stop: [type: {:list, :string}, doc: "Stop sequences"],
      timeout: [type: :integer, default: 60_000, doc: "Request timeout in milliseconds"],
      stream: [type: :boolean, default: false, doc: "Enable streaming responses"],
      frequency_penalty: [type: :float, doc: "Frequency penalty parameter"],
      presence_penalty: [type: :float, doc: "Presence penalty parameter"],
      json_mode: [
        type: :boolean,
        default: false,
        doc: "Forces model to output valid JSON (provider-dependent)"
      ],
      verbose: [
        type: :boolean,
        default: false,
        doc: "Enable verbose logging"
      ]
    ]

  require Logger
  alias Jido.AI.Model
  alias Jido.AI.Prompt

  @impl true
  def on_before_validate_params(params) do
    with {:ok, model} <- validate_model(params.model),
         {:ok, params} <- validate_prompt_or_messages(params) do
      {:ok, %{params | model: model}}
    else
      {:error, reason} ->
        Logger.error("ChatCompletion validation failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def run(params, _context) do
    # Validate required parameters exist
    with :ok <- validate_required_param(params, :model, "model"),
         :ok <- validate_required_prompt_or_messages(params) do
      run_with_validated_params(params)
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp run_with_validated_params(params) do
    # Extract options from prompt if available
    prompt_opts =
      case params[:prompt] do
        %Prompt{options: options} when is_list(options) and length(options) > 0 ->
          Map.new(options)

        _ ->
          %{}
      end

    # Keep required parameters
    required_params = Map.take(params, [:model, :prompt, :messages, :tools])

    # Create a map with all optional parameters set to defaults
    # Priority: explicit params > prompt options > defaults
    params_with_defaults =
      %{
        temperature: 0.7,
        max_tokens: 1000,
        top_p: nil,
        stop: nil,
        timeout: 60_000,
        stream: false,
        max_retries: 0,
        frequency_penalty: nil,
        presence_penalty: nil,
        json_mode: false,
        verbose: false
      }
      # Apply prompt options over defaults
      |> Map.merge(prompt_opts)
      # Apply explicit params over prompt options
      |> Map.merge(
        Map.take(params, [
          :temperature,
          :max_tokens,
          :top_p,
          :stop,
          :timeout,
          :stream,
          :api_key,
          :max_retries,
          :frequency_penalty,
          :presence_penalty,
          :json_mode,
          :verbose
        ])
      )
      # Always keep required params
      |> Map.merge(required_params)

    if params_with_defaults.verbose do
      Logger.info(
        "Running ReqLLM chat completion with params: #{inspect(redact_for_log(params_with_defaults), pretty: true)}"
      )
    end

    with {:ok, model} <- validate_model(params_with_defaults.model),
         {:ok, messages} <- normalize_messages(params_with_defaults),
         {:ok, req_options} <- build_req_llm_options(model, params_with_defaults),
         result <- call_reqllm(model, messages, req_options, params_with_defaults) do
      result
    else
      {:error, reason} ->
        Logger.error("Chat completion failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Private functions

  defp validate_required_param(params, key, name) do
    if Map.has_key?(params, key) do
      :ok
    else
      {:error, "Missing required parameter: #{name}"}
    end
  end

  defp validate_required_prompt_or_messages(params) do
    has_prompt? = Map.has_key?(params, :prompt) && !is_nil(Map.get(params, :prompt))
    has_messages? = Map.has_key?(params, :messages) && is_list(Map.get(params, :messages))

    if has_prompt? or has_messages? do
      :ok
    else
      {:error, "Missing required parameter: prompt or messages"}
    end
  end

  defp validate_model(%ReqLLM.Model{} = model), do: {:ok, model}
  defp validate_model(%Model{} = model), do: Model.from(model)
  defp validate_model(spec) when is_tuple(spec), do: Model.from(spec)

  defp validate_model(other) do
    Logger.error("Invalid model specification: #{inspect(other)}")
    {:error, "Invalid model specification: #{inspect(other)}"}
  end

  defp validate_prompt_or_messages(params) do
    cond do
      Map.has_key?(params, :prompt) && !is_nil(Map.get(params, :prompt)) ->
        with {:ok, prompt} <- Prompt.validate_prompt_opts(params.prompt) do
          {:ok, %{params | prompt: prompt}}
        end

      Map.has_key?(params, :messages) && is_list(Map.get(params, :messages)) ->
        {:ok, params}

      true ->
        {:error, "Missing required parameter: prompt or messages"}
    end
  end

  defp normalize_messages(params) do
    cond do
      Map.has_key?(params, :messages) && is_list(Map.get(params, :messages)) ->
        normalize_message_list(params.messages)

      Map.has_key?(params, :prompt) && !is_nil(Map.get(params, :prompt)) ->
        messages =
          params.prompt
          |> Prompt.render()
          |> Enum.map(fn msg ->
            %{role: Atom.to_string(msg.role), content: msg.content}
          end)

        {:ok, messages}

      true ->
        {:error, "Missing required parameter: prompt or messages"}
    end
  end

  defp normalize_message_list(messages) when is_list(messages) do
    normalized =
      Enum.map(messages, fn
        msg when is_binary(msg) ->
          %{role: "user", content: msg}

        msg when is_map(msg) ->
          role = Map.get(msg, :role) || Map.get(msg, "role") || "user"
          content = Map.get(msg, :content) || Map.get(msg, "content") || ""

          role =
            case role do
              r when is_atom(r) -> Atom.to_string(r)
              r when is_binary(r) -> r
              other -> to_string(other)
            end

          %{
            role: role,
            content: content
          }
          |> maybe_put(:tool_calls, Map.get(msg, :tool_calls) || Map.get(msg, "tool_calls"))
          |> maybe_put(:tool_call_id, Map.get(msg, :tool_call_id) || Map.get(msg, "tool_call_id"))
          |> maybe_put(:name, Map.get(msg, :name) || Map.get(msg, "name"))

        other ->
          %{
            role: "user",
            content: inspect(other)
          }
      end)

    {:ok, normalized}
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  defp build_req_llm_options(_model, params) do
    # Build base options
    base_opts =
      []
      |> add_opt_if_present(:temperature, params.temperature)
      |> add_opt_if_present(:max_tokens, params.max_tokens)
      |> add_opt_if_present(:top_p, params.top_p)
      |> add_opt_if_present(:stop, params.stop)
      |> add_opt_if_present(:receive_timeout, params.timeout)
      |> add_opt_if_present(:api_key, params[:api_key])
      |> add_opt_if_present(:frequency_penalty, params.frequency_penalty)
      |> add_opt_if_present(:presence_penalty, params.presence_penalty)

    # Add tools if provided
    opts_with_tools =
      case params[:tools] do
        tools when is_list(tools) and length(tools) > 0 ->
          with {:ok, tool_specs} <- normalize_tools(tools) do
            Keyword.put(base_opts, :tools, tool_specs)
          end

        _ ->
          base_opts
      end

    # ReqLLM handles authentication internally via environment variables
    case opts_with_tools do
      opts when is_list(opts) -> {:ok, opts}
      {:ok, opts} when is_list(opts) -> {:ok, opts}
      {:error, reason} -> {:error, reason}
    end
  end

  defp add_opt_if_present(opts, _key, nil), do: opts
  defp add_opt_if_present(opts, key, value), do: Keyword.put(opts, key, value)

  defp redact_for_log(params) when is_map(params) do
    case Map.get(params, :api_key) do
      nil -> params
      _ -> Map.put(params, :api_key, "[REDACTED]")
    end
  end

  defp call_reqllm(model, messages, req_options, params) do
    # Build model spec string from ReqLLM.Model
    model_spec = "#{model.provider}:#{model.model}"

    if params.stream do
      call_streaming(model_spec, messages, req_options)
    else
      call_standard(model_spec, messages, req_options)
    end
  end

  defp call_standard(model_id, messages, req_options) do
    case ReqLLM.generate_text(model_id, messages, req_options) do
      {:ok, response} ->
        # Use ReqLLM response directly
        format_response(response)

      {:error, error} ->
        {:error, error}
    end
  end

  defp call_streaming(model_id, messages, req_options) do
    opts_with_stream = Keyword.put(req_options, :stream, true)

    case ReqLLM.stream_text(model_id, messages, opts_with_stream) do
      {:ok, stream} ->
        # Return the stream wrapped in :ok tuple
        {:ok, stream}

      {:error, error} ->
        {:error, error}
    end
  end

  defp format_response(%ReqLLM.Response{} = response) do
    format_response(%{
      content: ReqLLM.Response.text(response) || "",
      tool_calls: ReqLLM.Response.tool_calls(response) || []
    })
  end

  defp format_response(%{content: content, tool_calls: tool_calls}) when is_list(tool_calls) do
    formatted_tool_calls =
      Enum.map(tool_calls, fn
        %ReqLLM.ToolCall{} = tool_call ->
          %{
            id: tool_call.id,
            function: %{
              name: tool_call.function.name,
              arguments: tool_call.function.arguments
            }
          }

        %{id: _id, function: %{name: _name, arguments: _arguments}} = tool_call ->
          tool_call

        %{"id" => _id, "function" => %{"name" => _name, "arguments" => _arguments}} = tool_call ->
          tool_call

        %{name: name, arguments: arguments} ->
          %{
            id: nil,
            function: %{
              name: name,
              arguments:
                if is_binary(arguments) do
                  arguments
                else
                  Jason.encode!(arguments)
                end
            }
          }

        %{"name" => name, "arguments" => arguments} ->
          %{
            id: nil,
            function: %{
              name: name,
              arguments:
                if is_binary(arguments) do
                  arguments
                else
                  Jason.encode!(arguments)
                end
            }
          }

        other ->
          %{
            id: nil,
            function: %{
              name: "unknown",
              arguments: Jason.encode!(%{raw: inspect(other)})
            }
          }
      end)

    {:ok, %{content: content, tool_results: formatted_tool_calls}}
  end

  defp format_response(%{content: content}) do
    {:ok, %{content: content, tool_results: []}}
  end

  defp format_response(response) when is_map(response) do
    # Fallback for other response formats
    content = response[:content] || response["content"] || ""
    {:ok, %{content: content, tool_results: []}}
  end

  defp normalize_tools(tools) when is_list(tools) do
    tool_specs = Enum.map(tools, &to_req_llm_tool/1)

    if Enum.any?(tool_specs, &match?({:error, _}, &1)) do
      {:error, :invalid_tools}
    else
      {:ok, Enum.map(tool_specs, fn {:ok, tool} -> tool end)}
    end
  end

  defp to_req_llm_tool(%ReqLLM.Tool{} = tool), do: {:ok, tool}

  defp to_req_llm_tool(tool) when is_atom(tool) do
    _ = Code.ensure_loaded?(tool)

    name =
      if function_exported?(tool, :name, 0) do
        tool.name()
      else
        tool |> Module.split() |> List.last() |> Macro.underscore()
      end

    description =
      if function_exported?(tool, :description, 0) do
        tool.description()
      else
        "No description available"
      end

    parameter_schema =
      if function_exported?(tool, :schema, 0) do
        tool.schema()
      else
        []
      end

    {:ok,
     ReqLLM.Tool.new!(
       name: name,
       description: description,
       parameter_schema: parameter_schema,
       callback: fn _args -> {:error, :tool_execution_not_supported} end
     )}
  end

  defp to_req_llm_tool(%{type: "function", function: function} = _tool) when is_map(function) do
    to_req_llm_tool(function)
  end

  defp to_req_llm_tool(%{"type" => "function", "function" => function} = _tool) when is_map(function) do
    to_req_llm_tool(function)
  end

  defp to_req_llm_tool(%{name: name} = function) when is_binary(name) do
    {:ok,
     ReqLLM.Tool.new!(
       name: name,
       description: Map.get(function, :description, ""),
       parameter_schema: Map.get(function, :parameters, %{}),
       callback: fn _args -> {:error, :tool_execution_not_supported} end
     )}
  end

  defp to_req_llm_tool(%{"name" => name} = function) when is_binary(name) do
    {:ok,
     ReqLLM.Tool.new!(
       name: name,
       description: Map.get(function, "description", ""),
       parameter_schema: Map.get(function, "parameters", %{}),
       callback: fn _args -> {:error, :tool_execution_not_supported} end
     )}
  end

  defp to_req_llm_tool(_other), do: {:error, :invalid_tool}
end
