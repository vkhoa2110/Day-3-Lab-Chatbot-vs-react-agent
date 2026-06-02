# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyen Phuc Hieu
- **Student ID**: 2A202600747
- **Date**: 2026-06-02
- **Project**: Vinpearl Room Availability Chatbot
- **Final Test Command**: `./venv/bin/python -m pytest -q`
- **Final Test Result**: `1 passed`

---

## Scoring Coverage Summary

| Individual Rubric Component | Points | Evidence |
| :--- | ---: | :--- |
| I. Technical Contribution | 15 | Implemented the tool-connected baseline chatbot in `src/agent/chatbot.py` and improved the local GGUF provider in `src/core/local_provider.py`. |
| II. Debugging Case Study | 10 | Analyzed missing-field behavior, local-model loading failures, prompt leakage, and fallback handling. |
| III. Personal Insights | 10 | Reflected on the difference between a normal chatbot and a tool-grounded ReAct-style flow. |
| IV. Future Improvements | 5 | Proposed production improvements for scalability, safety, and performance. |

---

## I. Technical Contribution (15 Points)

My main contribution was building the Vinpearl baseline chatbot and connecting it with a local LLM provider. The goal was to make a chatbot that can still behave safely in a hotel-booking domain: it should ask for missing booking fields, use deterministic tools for factual room data, and only use the LLM for final natural-language wording.

### Modules Implemented / Modified

| Module | Contribution |
| :--- | :--- |
| `src/agent/chatbot.py` | Implemented `VinpearlBaselineChatbot`, the Vietnamese system prompt, context enrichment, domain guardrails, location disambiguation, required-field checks, availability tool calls, room-card formatting, trace text, deterministic fallback answers, and LLM-based grounded answer generation. |
| `src/core/local_provider.py` | Implemented `LocalProvider` for GGUF local models through `llama-cpp-python`, including model loading, Phi-3 style prompt formatting, non-streaming generation, streaming output, token usage extraction, latency measurement, GPU/CPU runtime parameters, and clear errors when the model or dependency is missing. |

### Code Highlights

In `chatbot.py`, the chatbot first enriches state from the user message before calling any room-availability tool:

```python
context = self._enrich_context_from_message(self._clean_context(context or {}), message)
```

This allows the chatbot to extract or preserve:

```text
hotel_id / location
checkin
checkout
guests
```

The chatbot then applies a strict missing-field guardrail:

```python
missing_fields = self._missing_fields(context)
if missing_fields:
    return self._response_payload(
        answer=self._missing_info_answer(missing_fields),
        context=context,
        trace_text="\n".join(trace_lines),
    )
```

Only when the request has enough information does it call the structured availability tool:

```python
availability = check_room_availability(
    hotel_id=context["hotel_id"],
    checkin=context["checkin"],
    checkout=context["checkout"],
    guests=context["guests"],
)
```

This means the model does not invent room data. The tool result is converted into `room_cards`, and the final response is either a deterministic answer or an LLM-polished answer grounded in the tool output.

In `local_provider.py`, I implemented local model inference through the shared `LLMProvider` interface:

```python
response = self.llm(
    full_prompt,
    max_tokens=self.max_tokens,
    stop=["<|end|>", "Observation:"],
    temperature=self.temperature,
    top_p=self.top_p,
    repeat_penalty=self.repeat_penalty,
    echo=False,
)
```

The provider returns a consistent dictionary:

```text
content
usage.prompt_tokens
usage.completion_tokens
usage.total_tokens
latency_ms
provider = local
```

Because the output shape matches other providers, the chatbot can track latency and token usage without caring whether the answer came from OpenAI, Gemini, or a local GGUF model.

### Documentation of ReAct Interaction

Although `VinpearlBaselineChatbot` is the baseline chatbot, I designed it to expose a ReAct-like trace:

```text
Action: get_vinpearl_locations(...)
Observation: ...
Action: check_room_availability(...)
Observation: ...
Action: llm.generate(...)
Observation: ...
```

This makes the baseline easier to compare with the ReAct agent because we can inspect which step produced the final answer. The key distinction is that the chatbot does not let the LLM freely choose arbitrary tools. Instead, Python code controls the tool order, and the LLM is only used after tool observations are available.

---

## II. Debugging Case Study (10 Points)

### Case Study 1: Chatbot Answered Before Required Booking Fields Were Complete

#### Problem Description

At first, a user could ask a vague request such as:

```text
Tôi muốn đặt phòng Vinpearl Nha Trang
```

If the chatbot immediately called the LLM, the answer could sound helpful but still miss required booking information such as check-in date, check-out date, and number of guests.

#### Log / Trace Source

The debugging trace helped reveal whether the system asked for missing fields or tried to continue too early:

```text
Action: get_vinpearl_locations(region='Tôi muốn đặt phòng Vinpearl Nha Trang')
Observation: Tìm thấy cơ sở Vinpearl phù hợp trong dữ liệu demo.
```

Then the guardrail logs the missing-field state:

```text
event: TOOL_CHATBOT_GUARDRAIL
reason: missing_required_fields
missing_fields: ["ngày nhận phòng", "ngày trả phòng", "số khách"]
```

#### Diagnosis

Hotel availability cannot be checked safely without location, check-in, check-out, and guest count. A normal chatbot may continue with a fluent but incomplete answer, while a booking assistant must stop and request the missing fields.

#### Solution

I added deterministic context cleaning, message enrichment, and required-field validation before the availability call:

```python
def _missing_fields(self, context):
    missing_fields = []
    if not context.get("location") and not context.get("hotel_id"):
        missing_fields.append("cơ sở hoặc địa chỉ Vinpearl")
    if not context.get("checkin"):
        missing_fields.append("ngày nhận phòng")
    if not context.get("checkout"):
        missing_fields.append("ngày trả phòng")
    if not context.get("guests"):
        missing_fields.append("số khách")
    return missing_fields
```

After this fix, the chatbot only calls `check_room_availability` when it has the minimum required booking fields.

### Case Study 2: Local Model Failure Should Not Break the Chatbot

#### Problem Description

When the local GGUF model or `llama-cpp-python` dependency was missing, the chatbot could fail during LLM generation. This is risky because a booking chatbot should still return a safe answer when tool data is already available.

#### Diagnosis

The local provider depends on both:

```text
llama-cpp-python
models/Phi-3-mini-4k-instruct-q4.gguf
```

If either one is unavailable, local generation should raise a clear error, but the chatbot should not lose the deterministic tool result.

#### Solution

In `local_provider.py`, I added explicit checks and useful error messages:

```python
if Llama is None:
    raise RuntimeError("llama-cpp-python is not installed...")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}...")
```

In `chatbot.py`, I wrapped grounded LLM wording in a fallback path:

```python
try:
    result = llm.generate(prompt, system_prompt=self.system_prompt)
except Exception as exc:
    logger.log_event("LLM_FALLBACK", {"reason": "generation_failed", "error": str(exc)})
    return fallback_answer
```

This preserves reliability: even if the local model fails, the user still receives a deterministic answer based on the availability tool.

### Case Study 3: Prompt Leakage / Unsafe LLM Output

#### Problem Description

Local models can sometimes echo prompt instructions or generate overly long responses. In this project, that would be bad because the user should not see internal rules, JSON-like data, or backend wording.

#### Diagnosis

The issue is caused by weaker instruction following in small local models and by completion truncation when `max_tokens` is reached.

#### Solution

I added a safety filter in `chatbot.py`:

```python
if not answer or hit_token_limit or self._looks_like_prompt_leak(answer):
    logger.log_event("LLM_FALLBACK", {"reason": "unsafe_empty_or_truncated_answer"})
    return fallback_answer
```

The fallback answer is shorter, deterministic, and generated only from verified tool data.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning

A normal chatbot can answer quickly, but it may hide its reasoning and produce facts directly from the model. The ReAct pattern is easier to debug because it separates:

```text
Thought -> Action -> Observation -> Final Answer
```

In my chatbot implementation, I used a simplified version of this idea by recording tool actions and observations in `trace_text`. This helped me see whether the system failed during location lookup, required-field validation, availability checking, local generation, or final formatting.

### 2. Reliability

The tool-grounded chatbot is more reliable than a pure chatbot for room availability because final room names, prices, capacity, and availability come from `check_room_availability`, not from model memory.

However, the baseline chatbot can perform worse than a full ReAct agent when a request requires flexible multi-step reasoning. The baseline follows a fixed Python-controlled flow, while a ReAct agent can decide which tool to call next based on observations.

### 3. Observation Feedback

Observations changed the next step directly:

- If no Vinpearl location is found, the chatbot asks for a clearer property or address.
- If many locations match, it returns location options instead of guessing.
- If required fields are missing, it asks only for those fields.
- If room availability succeeds, it builds room cards and an answer from the tool result.
- If local LLM generation fails or leaks prompt text, it falls back to a deterministic answer.

This made the chatbot much safer than directly sending every user message to the LLM.

### 4. LLM Role

This lab clarified that the LLM should not be the source of truth for booking data. In my implementation:

```text
Python code = control flow and validation
Tools = factual hotel and room data
LocalProvider = natural-language wording
Telemetry = debugging and performance visibility
```

That design is a better fit for hotel booking because factual correctness matters more than creative generation.

---

## IV. Future Improvements (5 Points)

### Scalability

- Replace the demo JSON dataset with a real database or Vinpearl booking API.
- Add caching for repeated location lookup and availability queries.
- Support async tool calls for concurrent users.
- Add session persistence so the chatbot remembers booking context across page refreshes.

### Safety

- Add Pydantic schemas for context and tool responses.
- Add stricter date validation against available dataset dates.
- Add a supervisor check that verifies the final answer only contains facts from tool observations.
- Add more regression tests for out-of-scope questions, ambiguous locations, invalid dates, and prompt leakage.

### Performance

- Load the local model lazily, as the current chatbot already does, and add a warm-up endpoint for production.
- Tune `LOCAL_N_THREADS`, `LOCAL_N_GPU_LAYERS`, `LOCAL_N_BATCH`, and `BASELINE_MAX_TOKENS` for the deployment machine.
- Compare local GGUF latency against OpenAI/Gemini latency and cost.
- Track task-level success rate, not only token count and latency.

### Production Direction

For a production version, I would keep availability and pricing as deterministic tools, use the LLM only for language understanding and response wording, and add ReAct-style planning for more complex booking workflows such as comparing hotels, filtering by amenities, or explaining cancellation policies.
