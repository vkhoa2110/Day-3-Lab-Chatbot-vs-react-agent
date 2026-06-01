from __future__ import annotations

import json
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "vinpearl.json"


@lru_cache(maxsize=1)
def _load_data() -> Dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(text.split())


def _contains_query(candidates: Iterable[Any], query: str) -> bool:
    if not query:
        return True

    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if normalized and (query in normalized or normalized in query):
            return True
    return False


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + " VND"


def _policy_lookup(data: Dict[str, Any], collection_name: str, id_field: str) -> Dict[str, Dict[str, Any]]:
    return {
        item[id_field]: item
        for item in data.get("policies", {}).get(collection_name, [])
        if id_field in item
    }


def _amenity_names(data: Dict[str, Any], amenity_ids: Iterable[str]) -> List[str]:
    catalog = data.get("amenity_catalog", {})
    return [catalog.get(amenity_id, amenity_id) for amenity_id in amenity_ids]


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value

    text = str(value or "").strip()
    formats = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue

    # Agent prompts often contain dates like "15/07"; the demo dataset is for 2026.
    for date_format in ("%d/%m", "%d-%m"):
        try:
            parsed = datetime.strptime(text, date_format).date()
            dataset_year = _parse_dataset_start_date().year
            return parsed.replace(year=dataset_year)
        except ValueError:
            continue

    raise ValueError("Ngày phải có định dạng YYYY-MM-DD, DD/MM/YYYY hoặc DD-MM-YYYY.")


def _parse_dataset_start_date() -> date:
    data = _load_data()
    min_availability_date = min(item["date"] for item in data.get("availability", []))
    return datetime.strptime(min_availability_date, "%Y-%m-%d").date()


def _dates_between(checkin: date, checkout: date) -> List[str]:
    nights = (checkout - checkin).days
    return [(checkin + timedelta(days=offset)).isoformat() for offset in range(nights)]


def _coerce_guest_count(guests: Any) -> int:
    if isinstance(guests, dict):
        total = 0
        for key in ("adults", "adult", "children", "child", "infants", "infant"):
            value = guests.get(key, 0)
            if value in (None, ""):
                continue
            total += int(value)
        return total

    if isinstance(guests, str):
        guests = guests.strip()

    return int(guests)


def _find_hotel(data: Dict[str, Any], hotel_id: str) -> Optional[Dict[str, Any]]:
    query = _normalize_text(hotel_id)
    for hotel in data.get("hotels", []):
        candidates = [
            hotel.get("hotel_id"),
            hotel.get("hotel_name"),
            hotel.get("short_name"),
            hotel.get("address"),
        ]
        if hotel.get("hotel_id") == hotel_id or _contains_query(candidates, query):
            return hotel
    return None


def get_vinpearl_locations(region: Optional[str] = None) -> Dict[str, Any]:
    """Trả về danh sách cơ sở Vinpearl kèm ID theo khu vực/từ khóa."""
    data = _load_data()
    query = _normalize_text(region)
    regions_by_id = {region_item["region_id"]: region_item for region_item in data.get("regions", [])}

    matched_region_ids = set()
    matched_hotel_ids = set()

    for region_item in data.get("regions", []):
        candidates = [
            region_item.get("region_id"),
            region_item.get("name_vi"),
            region_item.get("name_en"),
            region_item.get("province"),
            region_item.get("airport"),
            *region_item.get("keywords", []),
        ]
        if _contains_query(candidates, query):
            matched_region_ids.add(region_item["region_id"])

    for alias in data.get("lookup_aliases", []):
        if query and _contains_query([alias.get("alias")], query):
            if alias.get("type") == "region":
                matched_region_ids.add(alias["target_id"])
            elif alias.get("type") == "hotel":
                matched_hotel_ids.add(alias["target_id"])

    locations: List[Dict[str, Any]] = []
    for hotel in data.get("hotels", []):
        hotel_query_fields = [
            hotel.get("hotel_id"),
            hotel.get("hotel_name"),
            hotel.get("short_name"),
            hotel.get("address"),
            hotel.get("property_type"),
            *hotel.get("tags", []),
        ]
        hotel_matches = _contains_query(hotel_query_fields, query)
        region_matches = hotel.get("region_id") in matched_region_ids

        if not query or region_matches or hotel_matches or hotel.get("hotel_id") in matched_hotel_ids:
            region_item = regions_by_id.get(hotel["region_id"], {})
            locations.append(
                {
                    "id": hotel["hotel_id"],
                    "name": hotel["hotel_name"],
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "short_name": hotel.get("short_name"),
                    "region_id": hotel["region_id"],
                    "region_name": region_item.get("name_vi"),
                    "province": region_item.get("province"),
                    "address": hotel.get("address"),
                    "star_rating": hotel.get("star_rating"),
                    "property_type": hotel.get("property_type"),
                    "is_beachfront": hotel.get("is_beachfront"),
                    "airport_transfer_available": hotel.get("airport_transfer_available"),
                    "checkin_time": hotel.get("checkin_time"),
                    "checkout_time": hotel.get("checkout_time"),
                    "phone": hotel.get("phone"),
                    "email": hotel.get("email"),
                    "amenities": _amenity_names(data, hotel.get("amenities", [])),
                    "nearby_attractions": hotel.get("nearby_attractions", []),
                }
            )

    matched_regions = []
    for region_id in sorted({location["region_id"] for location in locations}):
        region_item = regions_by_id.get(region_id)
        if region_item:
            matched_regions.append(
                {
                    "region_id": region_item["region_id"],
                    "name_vi": region_item["name_vi"],
                    "name_en": region_item.get("name_en"),
                    "province": region_item.get("province"),
                    "airport": region_item.get("airport"),
                }
            )

    return {
        "status": "success" if locations else "not_found",
        "query": region,
        "count": len(locations),
        "regions": matched_regions,
        "locations": locations,
        "message": (
            "Không tìm thấy cơ sở Vinpearl phù hợp trong dữ liệu demo."
            if not locations
            else "Tìm thấy cơ sở Vinpearl phù hợp trong dữ liệu demo."
        ),
    }


def check_room_availability(
    hotel_id: str,
    checkin: str,
    checkout: str,
    guests: Any,
) -> Dict[str, Any]:
    """Trả về các hạng phòng còn trống và tổng giá theo hotel/ngày/số khách."""
    data = _load_data()

    hotel = _find_hotel(data, hotel_id)
    if not hotel:
        return {
            "status": "not_found",
            "hotel_id": hotel_id,
            "available_rooms": [],
            "message": "Không tìm thấy hotel_id trong dữ liệu demo. Hãy gọi get_vinpearl_locations trước để lấy ID cơ sở.",
        }

    try:
        checkin_date = _parse_date(checkin)
        checkout_date = _parse_date(checkout)
        guest_count = _coerce_guest_count(guests)
    except (TypeError, ValueError) as exc:
        return {
            "status": "invalid_input",
            "hotel_id": hotel["hotel_id"],
            "available_rooms": [],
            "message": str(exc),
        }

    if guest_count <= 0:
        return {
            "status": "invalid_input",
            "hotel_id": hotel["hotel_id"],
            "available_rooms": [],
            "message": "Số khách phải lớn hơn 0.",
        }

    if checkout_date <= checkin_date:
        return {
            "status": "invalid_input",
            "hotel_id": hotel["hotel_id"],
            "available_rooms": [],
            "message": "Ngày checkout phải sau ngày checkin.",
        }

    stay_dates = _dates_between(checkin_date, checkout_date)
    nights = len(stay_dates)
    room_types = [
        room_type
        for room_type in data.get("room_types", [])
        if room_type.get("hotel_id") == hotel["hotel_id"]
        and room_type.get("active", True)
        and int(room_type.get("max_guests", 0)) >= guest_count
    ]
    availability_by_room_and_date = {
        (item["room_type_id"], item["date"]): item
        for item in data.get("availability", [])
        if item.get("hotel_id") == hotel["hotel_id"]
    }
    cancellation_policies = _policy_lookup(data, "cancellation_policies", "policy_id")
    meal_plans = _policy_lookup(data, "meal_plans", "meal_plan_id")

    available_rooms: List[Dict[str, Any]] = []
    unavailable_reasons: List[Dict[str, str]] = []

    for room_type in room_types:
        nightly_rows = [
            availability_by_room_and_date.get((room_type["room_type_id"], stay_date))
            for stay_date in stay_dates
        ]

        if any(row is None for row in nightly_rows):
            unavailable_reasons.append(
                {
                    "room_type_id": room_type["room_type_id"],
                    "reason": "Thiếu dữ liệu tồn phòng cho một hoặc nhiều đêm.",
                }
            )
            continue

        if any(
            row.get("stop_sell")
            or row.get("status") != "available"
            or int(row.get("available_rooms", 0)) <= 0
            for row in nightly_rows
        ):
            unavailable_reasons.append(
                {
                    "room_type_id": room_type["room_type_id"],
                    "reason": "Có ít nhất một đêm đã hết phòng hoặc đang stop-sell.",
                }
            )
            continue

        max_min_stay = max(int(row.get("min_stay_nights", 1)) for row in nightly_rows)
        if nights < max_min_stay:
            unavailable_reasons.append(
                {
                    "room_type_id": room_type["room_type_id"],
                    "reason": f"Yêu cầu lưu trú tối thiểu {max_min_stay} đêm.",
                }
            )
            continue

        total_price = sum(int(row["rate_vnd"]) for row in nightly_rows)
        min_available_rooms = min(int(row["available_rooms"]) for row in nightly_rows)
        cancellation_policy = cancellation_policies.get(room_type.get("cancellation_policy_id"), {})
        meal_plan = meal_plans.get(room_type.get("meal_plan_id"), {})

        available_rooms.append(
            {
                "room_type_id": room_type["room_type_id"],
                "room_name": room_type.get("name_vi"),
                "room_name_en": room_type.get("name_en"),
                "code": room_type.get("code"),
                "max_guests": room_type.get("max_guests"),
                "area_sqm": room_type.get("area_sqm"),
                "bed_options": room_type.get("bed_options", []),
                "available_rooms": min_available_rooms,
                "available_room_count": min_available_rooms,
                "nights": nights,
                "currency": "VND",
                "total_price": total_price,
                "total_price_vnd": total_price,
                "price_total_vnd": total_price,
                "total_price_display": _format_vnd(total_price),
                "average_price_per_night_vnd": round(total_price / nights),
                "amenities": _amenity_names(data, room_type.get("amenities", [])),
                "meal_plan": {
                    "meal_plan_id": room_type.get("meal_plan_id"),
                    "name_vi": meal_plan.get("name_vi"),
                },
                "cancellation_policy": {
                    "policy_id": room_type.get("cancellation_policy_id"),
                    "name_vi": cancellation_policy.get("name_vi"),
                    "description_vi": cancellation_policy.get("description_vi"),
                },
                "nightly_rates": [
                    {
                        "date": row["date"],
                        "rate_vnd": row["rate_vnd"],
                        "rate_display": _format_vnd(int(row["rate_vnd"])),
                        "available_rooms": row["available_rooms"],
                    }
                    for row in nightly_rows
                ],
            }
        )

    available_rooms.sort(key=lambda item: item["total_price_vnd"])

    return {
        "status": "success" if available_rooms else "sold_out",
        "hotel": {
            "hotel_id": hotel["hotel_id"],
            "hotel_name": hotel["hotel_name"],
            "address": hotel.get("address"),
            "checkin_time": hotel.get("checkin_time"),
            "checkout_time": hotel.get("checkout_time"),
            "phone": hotel.get("phone"),
            "email": hotel.get("email"),
        },
        "checkin": checkin_date.isoformat(),
        "checkout": checkout_date.isoformat(),
        "nights": nights,
        "guests": guest_count,
        "available_count": len(available_rooms),
        "available_rooms": available_rooms,
        "unavailable_reasons": unavailable_reasons,
        "message": (
            "Tìm thấy hạng phòng còn trống trong dữ liệu demo."
            if available_rooms
            else "Không có hạng phòng phù hợp còn trống cho điều kiện đã chọn trong dữ liệu demo."
        ),
    }


TOOL_REGISTRY = {
    "get_vinpearl_locations": get_vinpearl_locations,
    "check_room_availability": check_room_availability,
}

TOOLS = [
    {
        "name": "get_vinpearl_locations",
        "description": "Tìm cơ sở Vinpearl theo khu vực/tỉnh/thành phố/từ khóa và trả về hotel_id.",
        "function": get_vinpearl_locations,
    },
    {
        "name": "check_room_availability",
        "description": "Kiểm tra phòng trống theo hotel_id, checkin, checkout, guests và trả về giá tổng.",
        "function": check_room_availability,
    },
]


__all__ = [
    "DATA_PATH",
    "TOOLS",
    "TOOL_REGISTRY",
    "check_room_availability",
    "get_vinpearl_locations",
]
