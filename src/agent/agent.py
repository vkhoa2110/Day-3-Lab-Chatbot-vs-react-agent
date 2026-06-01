import re
import json
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        """
        System prompt that instructs the agent to follow ReAct.
        Should include:
        1.  Available tools and their descriptions.
        2.  Format instructions: Thought, Action, Observation.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        return f"""
You are an intelligent assistant. You have access to the following tools:
{tool_descriptions}

Use the following format exactly:
Thought: your line of reasoning.
Action: tool_name(arguments)
Observation: result of the tool call.
... (repeat Thought/Action/Observation if needed)
Final Answer: your final response.

Only call tools listed above. If a final answer is ready, write Final Answer and stop.
"""

    def run(self, user_input: str) -> str:
        """
        ReAct loop logic.
        1. Generate Thought + Action.
        2. Parse Action and execute Tool.
        3. Append Observation to prompt and repeat until Final Answer.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        scratchpad = ""
        steps = 0

        while steps < self.max_steps:
            current_prompt = f"User question: {user_input}\n\n{scratchpad}".strip()
            result = self.llm.generate(
                current_prompt,
                system_prompt=self.get_system_prompt(),
            )
            content = result.get("content", "")
            logger.log_event(
                "AGENT_LLM_STEP",
                {
                    "step": steps + 1,
                    "content": content,
                    "usage": result.get("usage", {}),
                    "cost": result.get("cost", {}),
                    "latency_ms": result.get("latency_ms"),
                },
            )

            final_answer = self._extract_final_answer(content)
            if final_answer:
                logger.log_event("AGENT_END", {"steps": steps + 1})
                return final_answer

            action = self._parse_action(content)
            if not action:
                logger.log_event(
                    "AGENT_PARSE_ERROR",
                    {"step": steps + 1, "content": content},
                )
                return content

            tool_name, args = action
            observation = self._execute_tool(tool_name, args)
            scratchpad += f"\n{content}\nObservation: {observation}\n"
            logger.log_event(
                "AGENT_TOOL_OBSERVATION",
                {"step": steps + 1, "tool": tool_name, "observation": observation},
            )
            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})
        return "Agent stopped because it reached the maximum number of reasoning steps."

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                func = tool.get("func")
                if not callable(func):
                    return f"Tool {tool_name} is registered without a callable function."
                try:
                    parsed_args = self._parse_tool_args(args)
                    result = func(**parsed_args) if parsed_args else func()
                    return json.dumps(result, ensure_ascii=False, default=str)
                except Exception as exc:
                    logger.error(f"Tool execution failed: {tool_name}")
                    return f"Tool {tool_name} failed: {exc}"
        return f"Tool {tool_name} not found."

    @staticmethod
    def _extract_final_answer(content: str) -> Optional[str]:
        match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _parse_action(content: str) -> Optional[tuple[str, str]]:
        match = re.search(
            r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)",
            content,
            re.DOTALL,
        )
        if not match:
            return None
        return match.group(1), match.group(2).strip()

    @staticmethod
    def _parse_tool_args(args: str) -> Dict[str, Any]:
        if not args:
            return {}
        text = args.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        parsed: Dict[str, Any] = {}
        for key, raw_value, single_quoted, double_quoted in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('([^']*)'|\"([^\"]*)\"|[^,]+)",
            text,
        ):
            value = single_quoted or double_quoted or raw_value
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed[key] = value.lower() == "true"
            elif value.isdigit():
                parsed[key] = int(value)
            else:
                parsed[key] = value
        return parsed
