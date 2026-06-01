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


class FakeFollowUpCriteriaLLM:
    model_name = "fake-openai-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt=None):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt or ""})
        if "follow-up" in (system_prompt or ""):
            content = (
                '{"type":"refine","criteria":{"bed_keywords":["king"],'
                '"amenity_keywords":["buffet"],"sort":"largest"}}'
            )
        else:
            content = (
                '{"intent":"room_search","location_query":null,'
                '"checkin":null,"checkout":null,"guests":null}'
            )
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "latency_ms": 1,
            "provider": "openai",
        }


def test_rejects_out_of_scope_question():
    agent = VinpearlRoomAgent()

    result = agent.respond("Viet code python sort list")

    assert result["status"] == "out_of_scope"
    assert not result["room_cards"]
    assert "1900 56 56 56" in result["answer"]


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


def test_follow_up_other_cheaper_does_not_repeat_same_room():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinholidays Fiesta Phu Quoc tu 05/06 den 17/06 cho 2 nguoi"
    )

    cheaper = agent.respond("toi can phong khac re hon", first["context"])

    assert cheaper["status"] == "no_more_rooms"
    assert not cheaper["room_cards"]
    assert "chưa tìm thấy" in cheaper["answer"].lower()


def test_follow_up_can_change_checkout_by_day_number():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinholidays Fiesta Phu Quoc tu 05/06 den 17/06 cho 2 nguoi"
    )

    changed = agent.respond("toi muon doi sang ngay 18", first["context"])

    assert changed["status"] == "ok"
    assert changed["summary"]["checkout"] == "2026-06-18"
    assert "18/06/2026" in changed["answer"]


def test_follow_up_filters_multiple_room_criteria():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Golf Nha Trang tu 03/06 den 05/06 cho 2 nguoi"
    )

    filtered = agent.respond(
        "toi can phong co giuong King, va co buffet sang, dien tich phong lon",
        first["context"],
    )

    assert filtered["status"] == "ok"
    assert [room["room_code"] for room in filtered["room_cards"]] == ["SUITE", "PREMIER"]
    assert all(
        "king" in " ".join(room["bed_options"]).lower()
        for room in filtered["room_cards"]
    )


def test_follow_up_without_matching_data_points_to_hotline():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Golf Nha Trang tu 03/06 den 05/06 cho 2 nguoi"
    )

    filtered = agent.respond("toi can phong tren 999m2", first["context"])

    assert filtered["status"] == "no_more_rooms"
    assert not filtered["room_cards"]
    assert "1900 56 56 56" in filtered["answer"]


def test_openai_follow_up_criteria_are_used_when_configured():
    llm = FakeFollowUpCriteriaLLM()
    agent = VinpearlRoomAgent(llm=llm)
    first = agent.respond(
        "Tim phong Vinpearl Golf Nha Trang tu 03/06 den 05/06 cho 2 nguoi"
    )

    filtered = agent.respond(
        "toi can phong co giuong King, va co buffet sang, dien tich phong lon",
        first["context"],
    )

    assert any("follow-up" in call["system_prompt"] for call in llm.calls)
    assert "openai_extract_follow_up_criteria" in filtered["trace_text"]
    assert [room["room_code"] for room in filtered["room_cards"]] == ["SUITE", "PREMIER"]


def test_follow_up_moderate_budget_recommends_midrange_room():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Discovery Greenhill Phu Quoc tu 15/07 den 18/07 cho 2 nguoi"
    )

    recommendation = agent.respond(
        "tai chinh vua thi nen chon phong nao",
        first["context"],
    )

    assert recommendation["status"] == "ok"
    assert recommendation["room_cards"]
    assert recommendation["room_cards"][0]["room_name"] == "Phòng Premier"
    assert "tài chính vừa" in recommendation["answer"].lower()
