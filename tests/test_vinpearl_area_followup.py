from src.agent.vinpearl_agent import VinpearlRoomAgent


def test_follow_up_filters_rooms_over_area():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Discovery Greenhill Phu Quoc tu 15/07 den 18/07 cho 2 nguoi"
    )

    filtered = agent.respond("toi can phong tren 80m2", first["context"])

    assert filtered["status"] == "ok"
    assert filtered["room_cards"]
    assert all(room["area_sqm"] > 80 for room in filtered["room_cards"])
    assert {room["room_code"] for room in filtered["room_cards"]} == {
        "FAMILY",
        "VILLA_2BR",
    }


def test_follow_up_filters_rooms_under_area():
    agent = VinpearlRoomAgent()
    first = agent.respond(
        "Tim phong Vinpearl Discovery Greenhill Phu Quoc tu 15/07 den 18/07 cho 2 nguoi"
    )

    filtered = agent.respond("toi can phong duoi 80m2", first["context"])

    assert filtered["status"] == "ok"
    assert filtered["room_cards"]
    assert all(room["area_sqm"] < 80 for room in filtered["room_cards"])
    assert "VILLA_2BR" not in {room["room_code"] for room in filtered["room_cards"]}
    assert "FAMILY" not in {room["room_code"] for room in filtered["room_cards"]}
