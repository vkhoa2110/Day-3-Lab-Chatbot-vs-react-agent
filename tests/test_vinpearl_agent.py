from src.agent.vinpearl_agent import VinpearlRoomAgent


class FakeOpenAILLM:
    model_name = "fake-openai-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt=None):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return {
            "content": (
                '{"intent":"room_search","location_query":"Bai Dai, Ganh Dau",'
                '"checkin":"2026-07-15","checkout":"2026-07-18","guests":2}'
            ),
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "latency_ms": 1,
            "provider": "openai",
        }


def test_rejects_out_of_scope_question():
    agent = VinpearlRoomAgent()

    result = agent.respond("Viet code python sort list")

    assert result["status"] == "out_of_scope"
    assert not result["room_cards"]


def test_requires_vinpearl_location_before_searching_rooms():
    agent = VinpearlRoomAgent()

    result = agent.respond("Tim phong cho 2 nguoi tu 15/07 den 18/07")

    assert result["status"] == "missing_location"
    assert "địa chỉ" in result["answer"].lower()


def test_region_only_request_is_ambiguous():
    agent = VinpearlRoomAgent()

    result = agent.respond("Tim phong Vinpearl Phu Quoc cho 2 nguoi tu 15/07 den 18/07")

    assert result["status"] == "ambiguous_location"
    assert len(result["location_options"]) >= 2


def test_specific_address_returns_available_room_cards():
    agent = VinpearlRoomAgent()

    result = agent.respond(
        "Tim phong tai Bai Dai, Ganh Dau tu 15/07 den 18/07 cho 2 nguoi"
    )

    assert result["status"] == "ok"
    assert result["room_cards"]
    assert "Vinpearl Resort & Spa Phú Quốc" in result["answer"]
    assert result["trace_text"].startswith("Thought:")
    assert "Final Answer:" in result["trace_text"]


def test_openai_llm_is_called_when_configured():
    llm = FakeOpenAILLM()
    agent = VinpearlRoomAgent(llm=llm)

    result = agent.respond("Co phong nao khong?")

    assert llm.calls
    assert result["status"] == "ok"
    assert result["room_cards"]
    assert "openai_extract_booking_request" in result["trace_text"]


def test_follow_up_returns_room_not_shown_before():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Discovery Greenhill Phu Quoc tu 15/07 den 18/07 cho 1 nguoi"
    )

    follow_up = agent.respond("co phong khac khong?", first["context"])

    first_ids = {room["room_type_id"] for room in first["room_cards"]}
    follow_up_ids = {room["room_type_id"] for room in follow_up["room_cards"]}
    assert follow_up["status"] == "ok"
    assert follow_up_ids
    assert follow_up_ids.isdisjoint(first_ids)


def test_follow_up_can_answer_room_details_by_number():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong tai Bai Dai, Ganh Dau tu 15/07 den 18/07 cho 2 nguoi"
    )

    detail = agent.respond("chi tiet phong 2", first["context"])

    assert detail["status"] == "ok"
    assert len(detail["room_cards"]) == 1
    assert detail["room_cards"][0]["room_type_id"] == first["room_cards"][1]["room_type_id"]


def test_follow_up_can_answer_meal_question_for_displayed_rooms():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong tai Bai Dai, Ganh Dau tu 15/07 den 18/07 cho 2 nguoi"
    )

    meal = agent.respond("cac phong nay co an sang khong?", first["context"])

    assert meal["status"] == "ok"
    assert len(meal["room_cards"]) == len(first["room_cards"])
    assert "buffet" in meal["answer"].lower()
