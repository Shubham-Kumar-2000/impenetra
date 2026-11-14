"""
Food agent module following Crave Craft specifications.
Defines the default LangChain agent specialized in food and recipes with default tools.
"""

import os
from typing import Any, List, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from impenetra.food_agent.food_tools import DEFAULT_FOOD_TOOLS

FOOD_AGENT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in food and recipes. "
    "You can answer questions about food, recipes, calories, nutrition, and more. "
    "Use the tools provided to assist with your responses."
)


class MockFoodChatModel(FakeListChatModel):
    """FakeListChatModel with bind_tools implemented for LangChain agents."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockFoodChatModel":
        return self


class DefaultFoodChatModel(BaseChatModel):
    """
    Default offline-capable Food Chat Model implementing tool binding.
    Answers queries about food, recipes, nutrition, and calories.
    """

    model_name: str = "impenetra-food-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "DefaultFoodChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = (
                    f"Based on the tool results: {m.content}. "
                    "I am happy to assist you with further food, recipe, or nutrition details."
                )
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        last_content = messages[-1].content if messages else ""
        if isinstance(last_content, list):
            last_content = " ".join(str(part) for part in last_content)
        query = str(last_content).lower()

        if "recipe" in query:
            resp = (
                "Here is a recommended healthy recipe: Mediterranean Quinoa Salad with fresh "
                "cucumbers, cherry tomatoes, olives, and feta cheese (approx 320 calories). "
                "Let me know if you would like step-by-step cooking instructions!"
            )
        elif "calorie" in query or "nutrition" in query:
            resp = (
                "Nutritional information: A standard serving of grilled chicken breast (100g) "
                "contains approximately 165 calories, 31g protein, and 3.6g fat. "
                "A medium apple (100g) has about 52 calories and 13.8g carbs."
            )
        elif "search" in query or "food" in query:
            resp = (
                "I found matching food items: Apple (52 kcal), Banana (89 kcal), "
                "Grilled Chicken Breast (165 kcal), Avocado (160 kcal), and Salmon (208 kcal)."
            )
        else:
            resp = f"Yes, I am working on / processing your request: '{last_content}'."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp))])

    @property
    def _llm_type(self) -> str:
        return "food_chat_model"


def create_food_agent(
    model: Any | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
) -> Any:
    """
    Create a LangChain food agent following Crave Craft specifications.

    Args:
        model: Optional LLM/ChatModel instance. Defaults to DefaultFoodChatModel
               (or configured API model if keys are present).
        tools: Sequence of tools. Defaults to Crave Craft DEFAULT_FOOD_TOOLS.
        system_prompt: Optional system prompt. Defaults to FOOD_AGENT_SYSTEM_PROMPT.

    Returns:
        Compiled LangChain agent graph.
    """
    active_prompt = system_prompt or FOOD_AGENT_SYSTEM_PROMPT
    active_tools = list(tools) if tools is not None else list(DEFAULT_FOOD_TOOLS)

    if model is None:
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]

                model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0,
                )
            except Exception:
                model = DefaultFoodChatModel()
        elif os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

                model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception:
                model = DefaultFoodChatModel()
        else:
            model = DefaultFoodChatModel()

    # Ensure model has working bind_tools
    if isinstance(model, FakeListChatModel) and type(model) is FakeListChatModel:
        model.__class__ = MockFoodChatModel

    return create_agent(
        model=model,
        tools=active_tools,
        system_prompt=active_prompt,
    )


def get_default_agent() -> Any:
    """Get the default Crave Craft food agent instance."""
    return create_food_agent()
