defmodule Jido.AI.TestActions.EchoAction do
  use Jido.Action,
    name: "echo",
    description: "Echo back text",
    schema: [
      text: [type: :string, required: true, doc: "Text to echo"]
    ]

  @impl true
  def run(%{text: text}, _context) do
    {:ok, %{echo: text}}
  end
end

