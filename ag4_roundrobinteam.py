import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


# from rich.console import Console
from rich.markdown import Markdown

# console = Console()
# markdown_content = """
# # Heading 1
# ## Heading 2
# - Bullet point 1
# - Bullet point 2
# **Bold text** and *italic text*.
# """

# md = Markdown(markdown_content)
# console.print(md)


# display(Markdown(markdown_content))


load_dotenv()

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="gpt-4.1-nano",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY env variable set.
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide critical feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

# # Use `asyncio.run(...)` when running in a script.
# result = await team.run(task="Write a short poem about the fall season.")
# print(result)

# When running inside a script, use a async main function and call it from `asyncio.run(...)`.
async def main():
    await team.reset()  # Reset the team for a new task.
    await Console(team.run_stream(task="Make a business plan to utilize Gen AI."))
    # async for message in team.run_stream(task="Make a business plan to utilize Gen AI."):  # type: ignore
    #     if isinstance(message, TaskResult):
    #         print("Stop Reason:", message.stop_reason)
    #     else:
    #         # print(message.source, ":")
    #         Console(message)

if __name__ == "__main__":
    asyncio.run(main())
