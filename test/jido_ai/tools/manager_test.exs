defmodule Jido.AI.Tools.ManagerTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Actions.ReqLlm.ChatCompletion
  alias Jido.AI.Conversation.Manager, as: ConversationManager
  alias Jido.AI.TestActions.EchoAction
  alias Jido.AI.Tools.Manager, as: ToolsManager

  setup :verify_on_exit!

  defp stream_response(model, chunks, finish_reason) do
    %ReqLLM.StreamResponse{
      stream: chunks,
      metadata_task:
        Task.async(fn ->
          %{
            usage: %{input_tokens: 0, output_tokens: 0},
            finish_reason: finish_reason,
            provider_meta: %{}
          }
        end),
      cancel: fn -> :ok end,
      model: model,
      context: ReqLLM.Context.new([])
    }
  end

  test "process/4 drives a tool loop via :messages and action modules" do
    model = ReqLLM.Model.new(:openai, "gpt-4o")
    {:ok, conversation_id} = ConversationManager.create(model)

    ReqLLM
    |> stub(:generate_text, fn _model_id, messages, opts ->
      tools = Keyword.get(opts, :tools, [])
      assert [%ReqLLM.Tool{name: "echo"} | _] = tools
      assert Keyword.get(opts, :api_key) == "test-key"

      tool_message =
        Enum.find(messages, fn msg ->
          role = Map.get(msg, :role) || Map.get(msg, "role")
          role in [:tool, "tool"]
        end)

      if tool_message do
        content =
          case Map.get(tool_message, :content) do
            s when is_binary(s) ->
              s

            parts when is_list(parts) ->
              Enum.map_join(parts, "", fn part -> Map.get(part, :text, "") end)

            other ->
              inspect(other)
          end

        assert content =~ "\"echo\":\"hi\""
        {:ok, %{content: "done"}}
      else
        tool_call = %{
          "id" => "call_1",
          "type" => "function",
          "function" => %{"name" => "echo", "arguments" => ~s({"text":"hi"})}
        }

        {:ok, %{content: "", tool_calls: [tool_call]}}
      end
    end)

    {:ok, response} =
      ToolsManager.process(conversation_id, "hello", [EchoAction],
        max_iterations: 3,
        timeout: 5_000,
        api_key: "test-key"
      )

    assert response.content == "done"
    assert response.tool_calls_made == 1

    {:ok, messages} = ConversationManager.get_messages(conversation_id)
    assert Enum.any?(messages, &(&1.role == :tool))
  end

  test "ChatCompletion accepts :messages without :prompt" do
    model = ReqLLM.Model.new(:openai, "gpt-4o")

    ReqLLM
    |> stub(:generate_text, fn _model_id, _messages, _opts ->
      {:ok, %{content: "ok"}}
    end)

    assert {:ok, %{content: "ok", tool_results: []}} =
             ChatCompletion.run(%{model: model, messages: [%{role: "user", content: "hi"}]}, %{})
  end

  test "process_stream/4 streams content and executes tool calls from ReqLLM.StreamResponse" do
    model = ReqLLM.Model.new(:openai, "gpt-4o")
    {:ok, conversation_id} = ConversationManager.create(model)

    Process.put(:stream_call_n, 0)

    ReqLLM
    |> stub(:stream_text, fn _model_id, _messages, opts ->
      assert Keyword.get(opts, :api_key) == "test-key"

      n = Process.get(:stream_call_n, 0) + 1
      Process.put(:stream_call_n, n)

      case n do
        1 ->
          {:ok,
           stream_response(
             model,
             [
               ReqLLM.StreamChunk.text("hello "),
               ReqLLM.StreamChunk.tool_call("echo", %{}, %{id: "call_1", index: 0}),
               ReqLLM.StreamChunk.meta(%{tool_call_args: %{index: 0, fragment: ~s({"text":"hi"})}})
             ],
             "tool_use"
           )}

        _ ->
          {:ok, stream_response(model, [ReqLLM.StreamChunk.text("done")], "stop")}
      end
    end)

    {:ok, stream} =
      ToolsManager.process_stream(conversation_id, "hello", [EchoAction],
        max_iterations: 3,
        timeout: 5_000,
        api_key: "test-key"
      )

    events = Enum.to_list(stream)

    assert {:content, "hello "} in events

    assert Enum.any?(events, fn
             {:tool_call, %{id: "call_1", name: "echo", arguments: %{"text" => "hi"}}} -> true
             _ -> false
           end)

    assert Enum.any?(events, fn
             {:tool_result, %{name: "echo", output: %{echo: "hi"}}} -> true
             _ -> false
           end)

    assert {:content, "done"} in events
    assert Enum.any?(events, &match?({:done, %{content: "done"}}, &1))

    {:ok, messages} = ConversationManager.get_messages(conversation_id)
    assert Enum.any?(messages, &(&1.role == :tool))
  end

  test "process_stream/4 supports non-streaming LLM mode and can emit thinking status" do
    model = ReqLLM.Model.new(:openai, "gpt-4o")
    {:ok, conversation_id} = ConversationManager.create(model)

    ReqLLM
    |> stub(:generate_text, fn _model_id, messages, opts ->
      tools = Keyword.get(opts, :tools, [])
      assert [%ReqLLM.Tool{name: "echo"} | _] = tools
      assert Keyword.get(opts, :api_key) == "test-key"

      tool_message =
        Enum.find(messages, fn msg ->
          role = Map.get(msg, :role) || Map.get(msg, "role")
          role in [:tool, "tool"]
        end)

      if tool_message do
        {:ok, %{content: "done"}}
      else
        tool_call = %{
          "id" => "call_1",
          "type" => "function",
          "function" => %{"name" => "echo", "arguments" => ~s({"text":"hi"})}
        }

        {:ok, %{content: "Planning: call echo", tool_calls: [tool_call]}}
      end
    end)

    {:ok, stream} =
      ToolsManager.process_stream(conversation_id, "hello", [EchoAction],
        max_iterations: 3,
        timeout: 5_000,
        api_key: "test-key",
        llm_stream: false,
        emit_thinking: true
      )

    events = Enum.to_list(stream)

    assert Enum.any?(events, fn
             {:status, %{status: :thinking, message: "Planning: call echo"}} -> true
             _ -> false
           end)

    assert Enum.any?(events, fn
             {:tool_call, %{id: "call_1", name: "echo", arguments: %{"text" => "hi"}}} -> true
             _ -> false
           end)

    assert Enum.any?(events, fn
             {:tool_result, %{name: "echo", output: %{echo: "hi"}}} -> true
             _ -> false
           end)

    assert {:content, "done"} in events
    assert Enum.any?(events, &match?({:done, %{content: "done"}}, &1))
  end
end
