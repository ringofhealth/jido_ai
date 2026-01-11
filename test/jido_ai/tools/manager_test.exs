defmodule Jido.AI.Tools.ManagerTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Actions.ReqLlm.ChatCompletion
  alias Jido.AI.Conversation.Manager, as: ConversationManager
  alias Jido.AI.TestActions.EchoAction
  alias Jido.AI.Tools.Manager, as: ToolsManager

  setup :verify_on_exit!

  test "process/4 drives a tool loop via :messages and action modules" do
    model = ReqLLM.Model.new(:openai, "gpt-4o")
    {:ok, conversation_id} = ConversationManager.create(model)

    ReqLLM
    |> stub(:generate_text, fn _model_id, messages, opts ->
      tools = Keyword.get(opts, :tools, [])
      assert [%ReqLLM.Tool{name: "echo"} | _] = tools
      assert Keyword.get(opts, :api_key) == "test-key"

      if Enum.any?(messages, &match?(%{role: "tool"}, &1)) do
        tool_message = Enum.find(messages, &match?(%{role: "tool"}, &1))
        assert Map.get(tool_message, :content) =~ "\"echo\":\"hi\""
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
end
