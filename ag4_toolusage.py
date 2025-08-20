from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="gpt-4.1-nano",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY env variable set.
    # Disable parallel tool calls for this example.
    parallel_tool_calls=False,  # type: ignore
)


# Create a tool for incrementing a number.
def increment_number(number: int) -> int:
    """Increment a number by 1."""
    return number + 1


# Create a tool agent that uses the increment_number function.
looped_assistant = AssistantAgent(
    "looped_assistant",
    model_client=model_client,
    tools=[increment_number],  # Register the tool.
    system_message="You are a helpful AI assistant, use the tool to increment the number. Say 'TARGET REACHED' when the target number is reached.",
)

# Termination condition that stops the task if the agent responds with a text message.
termination_condition = TextMentionTermination("TARGET REACHED")

# Create a team with the looped assistant agent and the termination condition.
team = RoundRobinGroupChat(
    [looped_assistant],
    termination_condition=termination_condition,
)

async def main():
    # Run the team with a task and print the messages to the console.
    async for message in team.run_stream(task="Increment the number 5 to 10."):  # type: ignore
        print(type(message).__name__, message)

    await model_client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())