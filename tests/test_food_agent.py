"""Unit and integration tests specifically for impenetra/food_agent/food_agent.py."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from impenetra.food_agent.food_agent import (
    FOOD_AGENT_SYSTEM_PROMPT,
    DefaultFoodChatModel,
    MockFoodChatModel,
    create_food_agent,
    get_default_agent,
)
from impenetra.food_agent.food_tools import DEFAULT_FOOD_TOOLS, search_food


def test_food_agent_system_prompt_charter():
    """Verify system prompt accurately reflects food agent charter."""
    assert "food and recipes" in FOOD_AGENT_SYSTEM_PROMPT.lower()
    assert "calories" in FOOD_AGENT_SYSTEM_PROMPT.lower()
    assert "nutrition" in FOOD_AGENT_SYSTEM_PROMPT.lower()
    assert "tools" in FOOD_AGENT_SYSTEM_PROMPT.lower()


def test_default_food_chat_model_responses():
    """Verify DefaultFoodChatModel categorizes queries and responds appropriately."""
    model = DefaultFoodChatModel()
    assert model._llm_type == "food_chat_model"
    assert model.bind_tools(DEFAULT_FOOD_TOOLS) is model

    # 1. Recipe intent
    res_recipe = model.invoke([HumanMessage(content="Give me a healthy dinner recipe")])
    assert "recipe" in res_recipe.content.lower()

    # 2. Calories/nutrition intent
    res_cal = model.invoke([HumanMessage(content="What are the calories in grilled chicken?")])
    assert "calories" in res_cal.content.lower() or "protein" in res_cal.content.lower()

    # 3. Search/food intent
    res_search = model.invoke([HumanMessage(content="Search for foods high in potassium")])
    assert "food" in res_search.content.lower() or "kcal" in res_search.content.lower()

    # 4. General intent
    res_gen = model.invoke([HumanMessage(content="Hello there")])
    assert "working on / processing your request" in res_gen.content.lower()
    assert "hello there" in res_gen.content.lower()

    # 5. List-based content part
    res_list = model.invoke([HumanMessage(content=["Can you recommend a breakfast recipe?"])])
    assert "recipe" in res_list.content.lower()


def test_default_food_chat_model_with_tool_message():
    """Verify DefaultFoodChatModel synthesizes tool results into response."""
    model = DefaultFoodChatModel()
    messages = [
        HumanMessage(content="Search for apple"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_food", "args": {"query": "apple"}, "id": "call_1"}],
        ),
        ToolMessage(content='[{"food_name": "Fuji Apple", "calories": 60}]', tool_call_id="call_1"),
    ]
    res = model.invoke(messages)
    assert "Fuji Apple" in res.content
    assert "tool results" in res.content.lower()


def test_mock_food_chat_model():
    """Verify MockFoodChatModel supports bind_tools and returns predetermined responses."""
    mock = MockFoodChatModel(responses=["Delicious pasta recipe with garlic and olive oil."])
    assert mock.bind_tools(DEFAULT_FOOD_TOOLS) is mock

    res = mock.invoke([HumanMessage(content="pasta recipe")])
    assert res.content == "Delicious pasta recipe with garlic and olive oil."


def test_create_food_agent_defaults():
    """Verify create_food_agent constructs a valid compiled agent with default settings."""
    agent = create_food_agent()
    assert hasattr(agent, "invoke")

    res = agent.invoke({"messages": [{"role": "user", "content": "Suggest a salad recipe"}]})
    assert "messages" in res
    last_msg = res["messages"][-1]
    assert len(last_msg.content) > 0
    assert "recipe" in last_msg.content.lower() or "salad" in last_msg.content.lower()


def test_create_food_agent_with_custom_prompt_and_tools():
    """Verify create_food_agent accepts custom system prompt and tools list."""
    custom_prompt = "You are a pastry chef specializing only in desserts."
    custom_tools = [search_food]
    agent = create_food_agent(
        tools=custom_tools,
        system_prompt=custom_prompt,
    )
    assert hasattr(agent, "invoke")
    res = agent.invoke({"messages": [{"role": "user", "content": "dessert options"}]})
    assert len(res["messages"][-1].content) > 0


def test_create_food_agent_patches_fake_list_chat_model():
    """Verify create_food_agent automatically patches FakeListChatModel for bind_tools."""
    raw_fake_llm = FakeListChatModel(responses=["A crisp green salad has roughly 80 calories."])
    agent = create_food_agent(model=raw_fake_llm)
    assert hasattr(raw_fake_llm, "bind_tools")

    res = agent.invoke({"messages": [{"role": "user", "content": "How many calories in salad?"}]})
    assert res["messages"][-1].content == "A crisp green salad has roughly 80 calories."


def test_get_default_agent_invokable():
    """Verify get_default_agent returns a fully functional LangChain agent."""
    agent = get_default_agent()
    query = "What is the nutrition of avocado?"
    res = agent.invoke({"messages": [{"role": "user", "content": query}]})
    assert "messages" in res
    assert len(res["messages"][-1].content) > 0
