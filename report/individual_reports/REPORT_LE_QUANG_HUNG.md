# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Lê Quang Hưng
- **Student ID**: 2A202600891
- **Date**: 2026-06-01
- **Project**: Vinpearl Room Agent
- **Final Test Command**: `python -m pytest -q`
- **Final Test Result**: `20 passed`

---

## Scoring Coverage Summary

| Individual Rubric Component | Points | Evidence |
| :--- | ---: | :--- |
| I. Technical Contribution | 15 | Implemented `vinpearl_tools.py` data/search/availability logic and `web_app.py` browser UI, API routes, context handling, room cards, traces, and metrics display. |
| II. Debugging Case Study | 10 | Analyzed follow-up filtering failure and OpenAI key failure using trace/log evidence, then fixed both. |
| III. Personal Insights | 10 | Reflected on reliability, observations, tool use, and why ReAct is different from a normal chatbot. |
| IV. Future Improvements | 5 | Proposed production scaling path with API/database backend, schema validation, RAG, and multi-agent split. |

---

## I. Technical Contribution (15 Points)

### Focused Contribution: `vinpearl_tools.py` and `web_app.py`

My main technical contribution was implementing the connection between the Vinpearl room dataset and the browser-based agent demo. I worked mostly on `src/tools/vinpearl_tools.py`, which prepares reliable hotel and room data for the agent, and `src/web_app.py`, which turns that agent into a usable web application.



### Modules Implemented / Modified

| Module | Contribution |
| :--- | :--- |
| `src/tools/vinpearl_tools.py` | Built the `VinpearlKnowledgeBase` data layer for loading regions, hotels, room types, availability, policies, meal plans, and promotions from the demo dataset. |
| `src/tools/vinpearl_tools.py` | Implemented Vietnamese-friendly normalization and location matching so users can search by hotel name, short name, address, province, region, or alias. |
| `src/tools/vinpearl_tools.py` | Implemented room availability checking with date validation, dataset range checks, guest-capacity filtering, stop-sell/status checks, nightly rates, total price calculation, and promotion discounts. |
| `src/tools/vinpearl_tools.py` | Returned structured room results for the agent, including room name, area, max guests, bed options, amenities, meal plan, cancellation policy, available room count, total price, and formatted VND display. |
| `src/tools/vinpearl_tools.py` | Added helper parsing functions for user text, including date-range extraction, guest-count extraction, and Vinpearl room-request detection. |
| `src/web_app.py` | Built the browser-based UI and HTTP server with routes for `/`, `/api/locations`, and `/api/chat`. |
| `src/web_app.py` | Implemented the sidebar search workflow for hotel/location selection, check-in/check-out dates, guest count, quick hints, and context collection. |
| `src/web_app.py` | Created the chat UI that renders user messages, assistant answers, room cards, location-option cards, ReAct traces, and LLM token/cost metrics. |
| `src/web_app.py` | Added frontend state handling for selected hotel, remembered context, generated search messages, loading status, backend errors, and reset behavior. |
| `src/web_app.py` | Connected environment-based agent setup so the web app can use OpenAI when configured, while still falling back to the default agent path. |

### Code Highlights

The main idea of my implementation was to keep the factual hotel logic in `vinpearl_tools.py`, while `web_app.py` collects user context and displays the agent response:

```text
Dataset -> VinpearlKnowledgeBase
User form/chat -> web_app.py context
Agent -> room availability tool
Browser UI -> room cards, location choices, trace, and metrics
```

Example request sent from the web app:

```text
Find Vinpearl rooms at Bai Dai from 15/07/2026 to 18/07/2026 for 2 guests
```

Example context payload sent to `/api/chat`:

```json
{
  "location": "Bai Dai",
  "checkin": "2026-07-15",
  "checkout": "2026-07-18",
  "guests": 2
}
```

`VinpearlKnowledgeBase.check_room_availability` then checks dates, guests, available inventory, nightly rates, promotions, and total price before returning structured room options to the agent.

### Web App Contribution

The web app makes the agent easier to test because it supports both natural-language chat and form-based search. The sidebar stores the hotel, location, check-in date, check-out date, and number of guests. The chat area then renders the response in a visual format instead of showing only plain text.

The UI renders:

- assistant answer text,
- room cards with price, area, guests, bed options, meal plan, policy, amenities, and promotions,
- location-choice cards when the query is ambiguous,
- ReAct trace output for debugging,
- token and estimated-cost metric pills when LLM usage data is available.

### Evidence of Code Quality

- `vinpearl_tools.py` keeps dataset loading, text normalization, hotel matching, date parsing, guest parsing, availability checking, and promotion calculation in separate helper functions.
- The availability checker handles invalid dates, out-of-range demo dates, unavailable rooms, stop-sell records, guest-capacity limits, and formatted VND output.
- `web_app.py` exposes clear JSON endpoints for location loading and chat requests.
- The frontend escapes dynamic values before rendering them in room cards, location cards, and traces.
- The final regression suite passes:

```text
20 passed
```

---

## II. Debugging Case Study (10 Points)

### Case Study 1: Follow-up Returned the Old Room List

#### Problem Description

After a successful room search, the user asked:

```text
tôi cần phòng trên 80m2
```

The early version returned nearly the same old room list instead of filtering only rooms above 80m2.

#### Log / Trace Source

The ReAct trace showed that the follow-up was not being interpreted as a strict area filter:

```text
Thought: Xem câu hỏi có phải follow-up từ kết quả tìm phòng trước đó hay không.
Action: classify_follow_up(message='toi can phong tren 80m2')
Observation: {"type": "new_search"}
```

#### Diagnosis

The issue was not simply that the LLM gave a weak answer. The agent itself lacked deterministic criteria extraction for area filters.

Specific causes:

- `_extract_option_criteria` did not parse `trên 80m2`.
- `_classify_follow_up` treated area/detail phrases too generally.
- `_select_options_for_response` did not apply `min_area_sqm` or `max_area_sqm`.

Because of that, the agent reused previous room options without enforcing the user's new constraint.

#### Solution

I added area criteria extraction:

```python
criteria["min_area_sqm"] = 80
```

Then applied deterministic filtering:

```python
room["area_sqm"] > criteria["min_area_sqm"]
```

I also added regression tests:

- `test_follow_up_filters_rooms_over_area`
- `test_follow_up_filters_rooms_under_area`

After the fix:

```text
User: tôi cần phòng trên 80m2
Agent: returns only rooms with area_sqm > 80
```

### Case Study 2: OpenAI Key Was Not Actually Used

#### Problem Description

The trace showed:

```text
AuthenticationError: 401 invalid_api_key
```

This meant the app was attempting to call OpenAI, but it was not using the expected key from the project `.env`.

#### Diagnosis

The process already had an old `OPENAI_API_KEY` in the system environment. By default, `load_dotenv()` does not override existing environment variables, so the app used the stale key instead of the project key.

#### Solution

The fix was:

```python
load_dotenv(PROJECT_ROOT / ".env", override=True)
```

After restart, the trace showed successful OpenAI calls:

```text
provider: openai
model: gpt-4o
usage: {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
cost: {"total_cost_usd": ...}
```

This also improved debuggability because the app now exposes whether LLM calls are real and how many tokens/cost they consume.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning

A normal chatbot can produce fluent text, but it does not naturally decide when to call tools or verify structured facts. ReAct forced the system to split work into:

```text
Thought -> Action -> Observation -> Final Answer
```

This made failures easier to locate. If an answer was wrong, I could inspect whether the issue came from intent extraction, location resolution, availability checking, follow-up classification, filtering, or formatting.

### 2. Reliability

The ReAct agent is more reliable than a chatbot for booking data because the final answer is grounded in tool observations. Room availability, room size, price, policies, and promotions come from `VinpearlKnowledgeBase`, not model memory.

The agent can perform worse than a generic chatbot when the user asks broad questions outside the tool scope. For example, a normal chatbot can answer coding or finance questions, but this agent intentionally refuses them because the assignment requires the system to stay within Vinpearl room-search.

### 3. Observation Feedback

The observation step is the core difference. The agent observes:

- whether a Vinpearl location is missing or ambiguous,
- how many room options are available,
- whether a filter returns zero matches,
- whether OpenAI extraction failed,
- which room cards were displayed previously,
- token usage and estimated cost of LLM calls.

The next step is based on those observations instead of only the user text.

### 4. LLM Role

This lab clarified that the LLM does not need to be the final source of truth. In this project:

```text
LLM = language understanding
Tools = data retrieval
Code = correctness and filtering
```

That design is a better fit for hotel booking workflows because factual correctness matters more than creative generation.

---

## IV. Future Improvements (5 Points)

### Scalability

- Replace the JSON dataset with a relational database or real Vinpearl booking API.
- Add caching for repeated location lookups and availability searches.
- Use async request handling for concurrent users.
- Add a scheduled evaluation job that exports success rate, latency, token usage, and estimated cost.

### Safety

- Set `temperature=0`, `max_tokens=300-400`, and `response_format={"type":"json_object"}` for extraction calls.
- Add strict schema validation with Pydantic before merging LLM output into agent state.
- Add a supervisor check for final answers to ensure no room/price/policy is invented.
- Keep out-of-scope refusal and hotline handoff for unknown or unsupported requests.

### Performance

- Reduce prompt size by sending only relevant hotel and room summaries to the LLM.
- Use `gpt-4o-mini` for extraction when accuracy is sufficient, then compare cost and success rate.
- Track cost per successful task, not only cost per API call.

### Production RAG / Multi-Agent Direction

For production, I would keep availability and pricing as structured tools, then add RAG only for unstructured content such as policies, FAQs, transportation, and cancellation details.

A multi-agent version could split responsibilities:

- **Intent Agent**: classify intent and extract booking fields.
- **Booking Tool Agent**: query inventory and pricing APIs.
- **Policy RAG Agent**: retrieve policy/FAQ documents.
- **Supervisor Agent**: verify the final answer against tool observations.

This would preserve the strongest part of the current project: natural-language flexibility without giving the LLM uncontrolled authority over factual booking data.
