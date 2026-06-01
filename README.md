# Lab 3: Chatbot vs ReAct Agent (Industry Edition)

Welcome to Phase 3 of the Agentic AI course! This lab focuses on moving from a simple LLM Chatbot to a sophisticated **ReAct Agent** with industry-standard monitoring.

## 🚀 Getting Started

### 1. Setup Environment
Copy the `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Directory Structure
- `src/tools/`: Extension point for your custom tools.

## 🏠 Running with Local Models (CPU)

If you don't want to use OpenAI or Gemini, you can run open-source models (like Phi-3) directly on your CPU using `llama-cpp-python`.

### 1. Download the Model
Download the **Phi-3-mini-4k-instruct-q4.gguf** (approx 2.2GB) from Hugging Face:
- [Phi-3-mini-4k-instruct-GGUF](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- Direct Download: [phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf)

### 2. Place Model in Project
Create a `models/` folder in the root and move the downloaded `.gguf` file there.

### 3. Update `.env`
Change your `DEFAULT_PROVIDER` and set the path:
```env
DEFAULT_PROVIDER=local
LOCAL_MODEL_PATH=./models/Phi-3-mini-4k-instruct-q4.gguf
```

## 🎯 Lab Objectives

1.  **Baseline Chatbot**: Observe the limitations of a standard LLM when faced with multi-step reasoning.
2.  **ReAct Loop**: Implement the `Thought-Action-Observation` cycle in `src/agent/agent.py`.
3.  **Provider Switching**: Swap between OpenAI and Gemini seamlessly using the `LLMProvider` interface.
4.  **Failure Analysis**: Use the structured logs in `logs/` to identify why the agent fails (hallucinations, parsing errors).
5.  **Grading & Bonus**: Follow the [SCORING.md](file:///Users/tindt/personal/ai-thuc-chien/day03-lab-agent/SCORING.md) to maximize your points and explore bonus metrics.

## 🛠️ How to Use This Baseline
The code is designed as a **Production Prototype**. It includes:
- **Telemetry**: Every action is logged in JSON format for later analysis.
- **Robust Provider Pattern**: Easily extendable to any LLM API.
- **Clean Skeletons**: Focus on the logic that matters—the agent's reasoning process.

## Vinpearl Room Agent Demo

This repo now includes a focused Vinpearl room-availability agent backed by:

- `data/vinpearl_nationwide_agent_demo_dataset.json`
- `src/tools/vinpearl_tools.py`
- `src/agent/vinpearl_agent.py`
- `src/web_app.py`

Run the web UI:

```bash
python src/web_app.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Agent behavior:

- Requires a Vinpearl property/address before checking rooms.
- Asks the user to choose a specific property if the location is ambiguous, e.g. only "Phu Quoc".
- Checks room availability by check-in/check-out dates and guest count.
- Returns available room cards with price, capacity, bed options, meal plan, cancellation policy, amenities, and demo promotions.
- Supports follow-up questions from the latest search, such as asking for other rooms, cheaper/larger options, meal details, cancellation policy, or "details for room 2".
- Refuses unrelated questions and keeps the response inside the room-search domain.
- Exposes the lab format in each response:

```text
Thought: ...
Action: ...
Observation: ...
Final Answer: ...
```

---

*Happy Coding! Let's build agents that actually work.*
