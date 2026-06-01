import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "vinpearl_nationwide_agent_demo_dataset.json"
)


def normalize_text(value: Optional[str]) -> str:
    """Lowercase, strip accents, and collapse punctuation for Vietnamese matching."""
    if not value:
        return ""
    value = value.lower().replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def format_vnd(amount: int) -> str:
    return f"{amount:,.0f}".replace(",", ".") + " VND"


def daterange(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor < end:
        yield cursor
        cursor += timedelta(days=1)


class VinpearlKnowledgeBase:
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = Path(dataset_path or DEFAULT_DATASET_PATH)
        self.data = json.loads(self.dataset_path.read_text(encoding="utf-8"))

        self.regions_by_id = {item["region_id"]: item for item in self.data["regions"]}
        self.hotels_by_id = {item["hotel_id"]: item for item in self.data["hotels"]}
        self.room_types_by_id = {
            item["room_type_id"]: item for item in self.data["room_types"]
        }
        self.rooms_by_hotel: Dict[str, List[Dict[str, Any]]] = {}
        for room in self.data["room_types"]:
            self.rooms_by_hotel.setdefault(room["hotel_id"], []).append(room)

        self.amenity_catalog = self.data.get("amenity_catalog", {})
        policies = self.data.get("policies", {})
        self.cancellation_policies = {
            item["policy_id"]: item
            for item in policies.get("cancellation_policies", [])
        }
        self.meal_plans = {
            item["meal_plan_id"]: item for item in policies.get("meal_plans", [])
        }

        self.availability_index = {
            (item["hotel_id"], item["room_type_id"], item["date"]): item
            for item in self.data["availability"]
        }
        available_dates = sorted({item["date"] for item in self.data["availability"]})
        self.min_date = self.parse_iso_date(available_dates[0])
        self.max_date = self.parse_iso_date(available_dates[-1])
        self.demo_today = self.min_date

        self._hotel_aliases = self._build_hotel_aliases()

    @staticmethod
    def parse_iso_date(value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()

    def _build_hotel_aliases(self) -> Dict[str, List[Tuple[str, str]]]:
        aliases: Dict[str, List[Tuple[str, str]]] = {}

        def add(alias: str, hotel_id: str, source: str) -> None:
            normalized = normalize_text(alias)
            if not normalized or normalized in {"vinpearl", "hotel", "resort"}:
                return
            aliases.setdefault(hotel_id, []).append((normalized, source))

        for hotel in self.data["hotels"]:
            region = self.regions_by_id[hotel["region_id"]]
            fields = [
                (hotel["hotel_name"], "hotel"),
                (hotel.get("short_name", ""), "hotel"),
                (hotel.get("address", ""), "address"),
                (region["name_vi"], "region"),
                (region["name_en"], "region"),
                (region.get("province", ""), "region"),
            ]
            for value, source in fields:
                add(value, hotel["hotel_id"], source)

        for item in self.data.get("lookup_aliases", []):
            alias = item.get("alias", "")
            target_id = item.get("target_id")
            if item.get("type") == "hotel" and target_id in self.hotels_by_id:
                add(alias, target_id, "alias")
            elif item.get("type") == "region" and target_id in self.regions_by_id:
                for hotel in self.hotels_for_region(target_id):
                    add(alias, hotel["hotel_id"], "region")

        return aliases

    def hotels_for_region(self, region_id: str) -> List[Dict[str, Any]]:
        return [
            hotel for hotel in self.data["hotels"] if hotel["region_id"] == region_id
        ]

    def all_hotels(self) -> List[Dict[str, Any]]:
        return list(self.data["hotels"])

    def location_examples(self, limit: int = 6) -> List[Dict[str, str]]:
        examples = []
        for hotel in self.data["hotels"][:limit]:
            region = self.regions_by_id[hotel["region_id"]]
            examples.append(
                {
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "address": hotel["address"],
                    "region": region["name_vi"],
                }
            )
        return examples

    def find_locations(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        query = normalize_text(keyword)
        if not query:
            return []

        scored: List[Tuple[int, str, str, Dict[str, Any]]] = []
        query_tokens = set(query.split())

        for hotel_id, hotel in self.hotels_by_id.items():
            best_score = 0
            best_source = ""
            for alias, source in self._hotel_aliases.get(hotel_id, []):
                alias_tokens = set(alias.split())
                if alias in query:
                    score = 100 + len(alias)
                elif query in alias and len(query) >= 4:
                    score = 70 + len(query)
                elif alias_tokens and alias_tokens.issubset(query_tokens):
                    score = 60 + len(alias_tokens)
                else:
                    overlap = alias_tokens.intersection(query_tokens)
                    score = len(overlap) * 8 if len(overlap) >= 2 else 0

                if source == "hotel" and score:
                    score += 30
                elif source == "address" and score:
                    score += 25
                elif source == "region" and score:
                    score += 5

                if score > best_score:
                    best_score = score
                    best_source = source

            if best_score > 0:
                scored.append((best_score, best_source, hotel_id, hotel))

        scored.sort(key=lambda item: (-item[0], item[3]["hotel_name"]))
        results = []
        for score, source, _hotel_id, hotel in scored[:limit]:
            region = self.regions_by_id[hotel["region_id"]]
            results.append(
                {
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "short_name": hotel.get("short_name", ""),
                    "address": hotel["address"],
                    "region": region["name_vi"],
                    "score": score,
                    "match_source": source,
                }
            )
        return results

    def resolve_hotel(self, keyword: str) -> Dict[str, Any]:
        matches = self.find_locations(keyword, limit=10)
        if not matches:
            return {"status": "missing", "matches": []}

        top = matches[0]
        close_matches = [
            item for item in matches if item["score"] >= top["score"] - 12
        ]
        if top["match_source"] in {"hotel", "address", "alias"} and top["score"] >= 140:
            return {"status": "found", "hotel": self.hotels_by_id[top["hotel_id"]]}
        if len(close_matches) == 1 and top["match_source"] in {"hotel", "address", "alias"}:
            return {"status": "found", "hotel": self.hotels_by_id[top["hotel_id"]]}

        if len(matches) == 1:
            return {"status": "found", "hotel": self.hotels_by_id[top["hotel_id"]]}

        return {"status": "ambiguous", "matches": close_matches[:6]}

    def check_room_availability(
        self,
        hotel_id: str,
        checkin: str,
        checkout: str,
        guests: Optional[int] = None,
        limit: int = 8,
    ) -> Dict[str, Any]:
        checkin_date = self.parse_iso_date(checkin)
        checkout_date = self.parse_iso_date(checkout)
        if checkout_date <= checkin_date:
            return {
                "status": "invalid_dates",
                "message": "Ngày trả phòng phải sau ngày nhận phòng.",
            }
        if checkin_date < self.min_date or checkout_date > self.max_date + timedelta(days=1):
            return {
                "status": "out_of_range",
                "message": (
                    "Dataset demo chỉ có tồn phòng từ "
                    f"{self.min_date.isoformat()} đến {self.max_date.isoformat()}."
                ),
            }

        hotel = self.hotels_by_id[hotel_id]
        region = self.regions_by_id[hotel["region_id"]]
        nights = (checkout_date - checkin_date).days
        options = []

        for room in self.rooms_by_hotel.get(hotel_id, []):
            if not room.get("active", True):
                continue
            if guests and room["max_guests"] < guests:
                continue

            nightly_rates = []
            min_available = None
            blocked = False
            for night in daterange(checkin_date, checkout_date):
                key = (hotel_id, room["room_type_id"], night.isoformat())
                item = self.availability_index.get(key)
                if (
                    not item
                    or item.get("stop_sell")
                    or item.get("status") != "available"
                    or item.get("available_rooms", 0) <= 0
                ):
                    blocked = True
                    break
                nightly_rates.append(
                    {
                        "date": item["date"],
                        "rate_vnd": item["rate_vnd"],
                        "available_rooms": item["available_rooms"],
                    }
                )
                available = item["available_rooms"]
                min_available = (
                    available if min_available is None else min(min_available, available)
                )

            if blocked or not nightly_rates:
                continue

            gross_total = sum(item["rate_vnd"] for item in nightly_rates)
            promotions = self._applicable_promotions(
                hotel=hotel,
                room=room,
                checkin_date=checkin_date,
                checkout_date=checkout_date,
                guests=guests or 0,
                gross_total=gross_total,
            )
            discount_total = sum(item["discount_vnd"] for item in promotions)
            net_total = max(gross_total - discount_total, 0)

            meal_plan = self.meal_plans.get(room.get("meal_plan_id"), {})
            cancellation = self.cancellation_policies.get(
                room.get("cancellation_policy_id"), {}
            )
            amenities = [
                self.amenity_catalog.get(item, item) for item in room.get("amenities", [])
            ]

            options.append(
                {
                    "hotel_id": hotel_id,
                    "hotel_name": hotel["hotel_name"],
                    "address": hotel["address"],
                    "region": region["name_vi"],
                    "phone": hotel.get("phone", ""),
                    "email": hotel.get("email", ""),
                    "checkin_time": hotel.get("checkin_time", "14:00"),
                    "checkout_time": hotel.get("checkout_time", "12:00"),
                    "room_type_id": room["room_type_id"],
                    "room_name": room["name_vi"],
                    "room_code": room["code"],
                    "max_guests": room["max_guests"],
                    "area_sqm": room["area_sqm"],
                    "bed_options": room.get("bed_options", []),
                    "amenities": amenities,
                    "meal_plan": meal_plan.get("name_vi", room.get("meal_plan_id", "")),
                    "cancellation_policy": cancellation.get(
                        "name_vi", room.get("cancellation_policy_id", "")
                    ),
                    "cancellation_detail": cancellation.get("description_vi", ""),
                    "nightly_rates": nightly_rates,
                    "min_available_rooms": min_available or 0,
                    "nights": nights,
                    "gross_total_vnd": gross_total,
                    "discount_total_vnd": discount_total,
                    "total_vnd": net_total,
                    "promotions": promotions,
                    "total_display": format_vnd(net_total),
                    "gross_total_display": format_vnd(gross_total),
                }
            )

        options.sort(key=lambda item: (item["total_vnd"], -item["max_guests"]))
        return {
            "status": "ok",
            "hotel": hotel,
            "region": region,
            "checkin": checkin,
            "checkout": checkout,
            "nights": nights,
            "guests": guests,
            "options": options[:limit],
            "total_matches": len(options),
        }

    def _applicable_promotions(
        self,
        hotel: Dict[str, Any],
        room: Dict[str, Any],
        checkin_date: date,
        checkout_date: date,
        guests: int,
        gross_total: int,
    ) -> List[Dict[str, Any]]:
        promotions = []
        nights = (checkout_date - checkin_date).days
        room_code = room.get("code", "")
        for promo in self.data.get("promotions", []):
            if hotel["region_id"] not in promo.get("applicable_region_ids", [hotel["region_id"]]):
                continue
            if nights < promo.get("min_nights", 1):
                continue
            if guests and guests < promo.get("min_guests", 0):
                continue
            if "applicable_room_codes" in promo:
                allowed_codes = promo["applicable_room_codes"]
                if room_code not in allowed_codes and not any(
                    token in room_code for token in allowed_codes
                ):
                    continue
            if "stay_date_from" in promo:
                stay_from = self.parse_iso_date(promo["stay_date_from"])
                stay_to = self.parse_iso_date(promo["stay_date_to"])
                if checkin_date < stay_from or checkout_date - timedelta(days=1) > stay_to:
                    continue
            if "min_days_before_checkin" in promo:
                lead_days = (checkin_date - self.demo_today).days
                if lead_days < promo["min_days_before_checkin"]:
                    continue

            if promo["discount_type"] == "percent":
                discount = round(gross_total * promo["discount_value"] / 100)
            elif promo["discount_type"] == "fixed":
                discount = promo.get("discount_value_vnd", 0) * nights
            else:
                discount = 0
            if discount <= 0:
                continue
            promotions.append(
                {
                    "promotion_id": promo["promotion_id"],
                    "name_vi": promo["name_vi"],
                    "description_vi": promo.get("description_vi", ""),
                    "discount_vnd": discount,
                    "discount_display": format_vnd(discount),
                }
            )
        return promotions


DATE_RE = re.compile(
    r"(?<!\d)(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)(?!\d)"
)


def parse_user_date(raw: str, default_year: int = 2026) -> Optional[str]:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    for separator in ("/", "-"):
        pieces = raw.split(separator)
        if len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
            day, month = [int(piece) for piece in pieces]
            try:
                return date(default_year, month, day).isoformat()
            except ValueError:
                return None
    return None


def extract_date_range(text: str, default_year: int = 2026) -> Tuple[Optional[str], Optional[str]]:
    matches = DATE_RE.findall(text or "")
    if not matches:
        return None, None
    parsed = [parse_user_date(item, default_year=default_year) for item in matches[:2]]
    if len(parsed) == 1:
        return parsed[0], None
    return parsed[0], parsed[1]


def extract_guests(text: str) -> Optional[int]:
    normalized = normalize_text(text)
    patterns = [
        r"(\d{1,2})\s*(?:nguoi|khach|pax|adult|adults)",
        r"cho\s*(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            value = int(match.group(1))
            return value if value > 0 else None
    return None


def looks_like_vinpearl_room_request(text: str) -> bool:
    normalized = normalize_text(text)
    keywords = {
        "vinpearl",
        "vinholidays",
        "vinwonders",
        "phong",
        "room",
        "khach san",
        "resort",
        "dat phong",
        "con phong",
        "trong",
        "checkin",
        "checkout",
        "nhan phong",
        "tra phong",
        "gia phong",
    }
    return any(keyword in normalized for keyword in keywords)
