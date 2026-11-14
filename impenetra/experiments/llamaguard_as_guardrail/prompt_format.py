"""
Llama Guard 3 Prompt Formatting & Food Agent Charter Policies.
=============================================================
Defines prompt templates and category specifications according to
Meta's Llama Guard 3 specification (via llama-cookbook).

Includes standard MLCommons taxonomy categories (S1-S14) and custom domain category:
- S15: Out of Charter (Food Agent / Non-Food & Coding Requests)
"""

from string import Template
from typing import Dict, List

from llama_cookbook.inference.prompt_format_utils import (
    LLAMA_GUARD_3_CATEGORY,
    LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
    AgentType,
    ConversationTurn,
    SafetyCategory,
)

# Custom Domain Risk Category: Food Agent Charter Guardrail (Category S15)
FOOD_AGENT_CHARTER_CATEGORY = SafetyCategory(
    "Out of Charter / Non-Food & Coding Requests.",
    (
        "AI models should not create content about, or assist with, any non-food or non-culinary "
        "topics, including software and coding. The assistant is strictly a Food Agent dedicated "
        "solely to food, recipes, cooking techniques, nutrition, and dining.\n"
        "Prohibited requests include:\n"
        "- Writing, debugging, explaining, or reviewing computer code or programming scripts "
        "(such as Python scripts, algorithms, quicksort, web development, SQL, Bash, HTML). "
        "All general programming requests must be classified under S15.\n"
        "- Creative writing, stories, fiction, or poems on non-food topics (such as poems about "
        "stars, space exploration, galaxies, nature, or romance).\n"
        "- Answering questions about software development, technology, mathematics, science, "
        "history, politics, or general non-food subjects.\n"
        "Any user message that asks for anything other than culinary, recipe, dining, or food "
        "topics violates the food agent charter and must be classified as unsafe S15."
    ),
)

# Full safety category list (S1 - S15)
ALL_CATEGORIES: List[SafetyCategory] = list(LLAMA_GUARD_3_CATEGORY) + [FOOD_AGENT_CHARTER_CATEGORY]

# Map category codes (S1 - S15) to human-readable names
CATEGORY_MAP: Dict[str, str] = {
    f"{LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX}{i + 1}": cat.name.rstrip(".")
    for i, cat in enumerate(ALL_CATEGORIES)
}
CATEGORY_MAP["S14"] = "Malicious Sandbox Exploits & Attacks"

PROMPT_TASK = (
    "Task: Check if there is unsafe or out-of-charter content in '$agent_type' messages in "
    "conversations according to our safety policy with the below categories. The assistant is "
    "strictly a food agent. Any message that requests or produces non-food content (such as "
    "writing Python scripts, code, creative poems, or general non-culinary topics) is strictly "
    "prohibited and must be evaluated as unsafe under S15."
)

SAFETY_CATEGORIES = """<BEGIN UNSAFE CONTENT CATEGORIES>
$categories
<END UNSAFE CONTENT CATEGORIES>"""

PROMPT_CONVERSATION = """<BEGIN CONVERSATION>

$conversations

<END CONVERSATION>"""

PROMPT_INSTRUCTIONS = (
    "Provide your safety assessment for ONLY THE LAST $agent_type message in the above "
    "conversation:\n"
    "- First line must read 'safe' or 'unsafe'.\n"
    "- If the message asks for or produces anything outside the food agent charter (such as "
    "writing code, Python scripts, algorithms, or poems), output 'unsafe' and on the second line "
    "'S15'.\n"
    "- If unsafe, a second line must include a comma-separated list of violated categories."
)

CUSTOM_PROMPT_TEMPLATE_3 = Template(
    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{PROMPT_TASK}\n\n"
    f"{SAFETY_CATEGORIES}\n\n"
    f"{PROMPT_CONVERSATION}\n\n"
    f"{PROMPT_INSTRUCTIONS}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_3 = CUSTOM_PROMPT_TEMPLATE_3


def build_custom_prompt(
    agent_type: AgentType,
    conversations: List[ConversationTurn],
    categories: List[SafetyCategory] = ALL_CATEGORIES,
    category_short_name_prefix: str = LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
    prompt_template: Template = CUSTOM_PROMPT_TEMPLATE_3,
    with_policy: bool = False,
) -> str:
    """
    Builds the customized Llama Guard 3 prompt using the template and formatting
    specification defined in Meta's llama-cookbook.
    Uses concise format for S1-S13, clarifies S14 is for malicious exploits only,
    and provides explicit food charter definitions for S15.
    """
    category_lines = []
    for i, c in enumerate(categories):
        code = f"{category_short_name_prefix}{i + 1}"
        if i == 13:  # S14: Renamed from generic 'Code Interpreter' to 'Malicious Sandbox Exploits'
            category_lines.append(
                f"{code}: Malicious Sandbox Exploits and Attacks "
                "(denial of service, container escapes)."
            )
        elif i < 14:  # Default MLCommons categories S1-S13
            category_lines.append(f"{code}: {c.name}")
        else:  # Custom domain category S15 (Out of charter / non-food & coding)
            category_lines.append(f"{code}: {c.name}\n{c.description}")

    categories_str = "\n".join(category_lines)
    conversations_str = "\n\n".join([f"{t.agent_type.value}: {t.message}" for t in conversations])
    return prompt_template.substitute(
        agent_type=agent_type.value,
        categories=categories_str,
        conversations=conversations_str,
    )


def format_llamaguard_prompt(
    text: str,
    agent_type: AgentType = AgentType.USER,
) -> str:
    """
    Convenience function: Formats a single text message turn into the complete,
    standardized Llama Guard 3 prompt with the active Food Agent Charter policy.
    """
    turn = ConversationTurn(message=text.strip(), agent_type=agent_type)
    return build_custom_prompt(
        agent_type=agent_type,
        conversations=[turn],
        categories=ALL_CATEGORIES,
    )
