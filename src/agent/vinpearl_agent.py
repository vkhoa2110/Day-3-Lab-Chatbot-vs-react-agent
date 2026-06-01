import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.tools.vinpearl_tools import (
    VinpearlKnowledgeBase,
    extract_date_range,
    extract_guests,
    format_vnd,
    looks_like_vinpearl_room_request,
    normalize_text,
)


OFF_TOPIC_HINTS = {
    "thoi tiet",
    "du bao",
    "bong da",
    "chung khoan",
    "crypto",
    "bitcoin",
    "lap trinh",
    "code",
    "python",
    "javascript",
    "tin tuc",
    "chinh tri",
    "nha hang",
    "mon an",
    "dia diem an",
}


class VinpearlRoomAgent:
    """Domain-specific ReAct agent for Vinpearl room availability questions."""

    def __init__(
        self,
        knowledge_base: Optional[VinpearlKnowledgeBase] = None,
        llm: Optional[LLMProvider] = None,
    ):
        self.kb = knowledge_base or VinpearlKnowledgeBase()
        self.llm = llm

    def respond(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        trace: List[Dict[str, str]] = []
        message = (message or "").strip()
        logger.log_event("VINPEARL_AGENT_START", {"input": message, "context": context})

        def add_trace(thought: str, action: str, observation: Any) -> None:
            trace.append(
                {
                    "thought": thought,
                    "action": action,
                    "observation": (
                        observation
                        if isinstance(observation, str)
                        else json.dumps(observation, ensure_ascii=False, default=str)
                    ),
                }
            )

        llm_extraction = self._extract_request_with_llm(message, context)
        if llm_extraction:
            add_trace(
                "Dùng OpenAI để hiểu ý định và trích xuất tham số đặt phòng từ câu hỏi người dùng.",
                "openai_extract_booking_request(message=...)",
                {
                    "provider": llm_extraction.get("_provider"),
                    "model": llm_extraction.get("_model"),
                    "error": llm_extraction.get("_error"),
                    "usage": llm_extraction.get("_usage"),
                    "intent": llm_extraction.get("intent"),
                    "location_query": llm_extraction.get("location_query"),
                    "checkin": llm_extraction.get("checkin"),
                    "checkout": llm_extraction.get("checkout"),
                    "guests": llm_extraction.get("guests"),
                },
            )

        location_probe = (
            (llm_extraction or {}).get("location_query")
            or context.get("location")
            or message
        )
        location_matches = self.kb.find_locations(location_probe, limit=4)
        in_scope = self._is_in_scope(
            message,
            context,
            bool(location_matches),
            llm_intent=(llm_extraction or {}).get("intent"),
        )
        add_trace(
            "Kiểm tra câu hỏi có thuộc phạm vi tìm phòng Vinpearl hay không.",
            f"classify_intent(message={message!r})",
            {"in_scope": in_scope},
        )
        if not in_scope:
            answer = (
                "Mình chỉ hỗ trợ tìm phòng trống tại các cơ sở Vinpearl trong "
                "dataset demo. Vui lòng gửi địa chỉ/cơ sở Vinpearl, ngày nhận "
                "phòng, ngày trả phòng và số khách."
            )
            return self._response(answer, trace, context=context, status="out_of_scope")

        request = self._extract_request(message, context, llm_extraction)
        add_trace(
            "Trích xuất địa chỉ/cơ sở, ngày lưu trú và số khách từ câu hỏi.",
            f"extract_booking_request(message={message!r})",
            request,
        )

        hotel = None
        if request.get("hotel_id") in self.kb.hotels_by_id:
            hotel = self.kb.hotels_by_id[request["hotel_id"]]
            add_trace(
                "Dùng cơ sở Vinpearl đã được xác định trong ngữ cảnh chat.",
                f"get_vinpearl_locations(hotel_id={request['hotel_id']!r})",
                {
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "address": hotel["address"],
                },
            )
        else:
            location_query = request.get("location_query") or message
            if not normalize_text(location_query):
                examples = self.kb.location_examples()
                add_trace(
                    "Chưa có địa chỉ/cơ sở Vinpearl nên cần hỏi lại người dùng.",
                    "get_vinpearl_locations(keyword='')",
                    {"examples": examples},
                )
                answer = self._ask_for_location(examples)
                return self._response(
                    answer,
                    trace,
                    context=self._merge_context(context, request),
                    status="missing_location",
                    location_options=examples,
                )

            resolution = self.kb.resolve_hotel(location_query)
            add_trace(
                "Tìm cơ sở Vinpearl khớp với địa chỉ/cơ sở người dùng cung cấp.",
                f"get_vinpearl_locations(keyword={location_query!r})",
                resolution if resolution["status"] != "found" else {
                    "status": "found",
                    "hotel_id": resolution["hotel"]["hotel_id"],
                    "hotel_name": resolution["hotel"]["hotel_name"],
                    "address": resolution["hotel"]["address"],
                },
            )
            if resolution["status"] == "missing":
                examples = self.kb.location_examples()
                answer = (
                    "Mình chưa nhận diện được địa chỉ/cơ sở Vinpearl trong yêu cầu. "
                    "Vui lòng nhập rõ tên cơ sở hoặc địa chỉ, ví dụ: "
                    + "; ".join(
                        f"{item['hotel_name']} - {item['address']}"
                        for item in examples[:3]
                    )
                    + "."
                )
                return self._response(
                    answer,
                    trace,
                    context=self._merge_context(context, request),
                    status="missing_location",
                    location_options=examples,
                )
            if resolution["status"] == "ambiguous":
                answer = self._ask_to_disambiguate(resolution["matches"])
                return self._response(
                    answer,
                    trace,
                    context=self._merge_context(context, request),
                    status="ambiguous_location",
                    location_options=resolution["matches"],
                )
            hotel = resolution["hotel"]

        request["hotel_id"] = hotel["hotel_id"]
        request["location"] = f"{hotel['hotel_name']} - {hotel['address']}"

        if not request.get("checkin") or not request.get("checkout"):
            add_trace(
                "Chưa đủ ngày nhận/trả phòng để kiểm tra tồn phòng theo ngày.",
                "check_required_fields(fields=['checkin', 'checkout'])",
                {
                    "checkin": request.get("checkin"),
                    "checkout": request.get("checkout"),
                    "available_date_range": (
                        f"{self.kb.min_date.isoformat()} to {self.kb.max_date.isoformat()}"
                    ),
                },
            )
            answer = (
                f"Mình đã xác định cơ sở {hotel['hotel_name']} ({hotel['address']}). "
                "Vui lòng cung cấp ngày nhận phòng và ngày trả phòng để mình kiểm tra "
                f"tồn phòng. Dataset demo có dữ liệu từ {self.kb.min_date.strftime('%d/%m/%Y')} "
                f"đến {self.kb.max_date.strftime('%d/%m/%Y')}."
            )
            return self._response(
                answer,
                trace,
                context=self._merge_context(context, request),
                status="missing_dates",
            )

        guests = request.get("guests") or 2
        request["guests"] = guests
        availability = self.kb.check_room_availability(
            hotel_id=hotel["hotel_id"],
            checkin=request["checkin"],
            checkout=request["checkout"],
            guests=guests,
            limit=8,
        )
        add_trace(
            "Kiểm tra phòng còn trống và tổng giá theo từng đêm.",
            (
                "check_room_availability("
                f"hotel_id={hotel['hotel_id']!r}, checkin={request['checkin']!r}, "
                f"checkout={request['checkout']!r}, guests={guests})"
            ),
            {
                "status": availability["status"],
                "total_matches": availability.get("total_matches", 0),
            },
        )

        if availability["status"] != "ok":
            return self._response(
                availability.get("message", "Không thể kiểm tra tồn phòng."),
                trace,
                context=self._merge_context(context, request),
                status=availability["status"],
            )

        if not availability["options"]:
            answer = (
                f"Không tìm thấy phòng trống phù hợp tại {hotel['hotel_name']} "
                f"({hotel['address']}) cho {guests} khách từ "
                f"{self._display_date(request['checkin'])} đến {self._display_date(request['checkout'])}. "
                "Bạn có thể thử đổi ngày, giảm số khách hoặc chọn cơ sở Vinpearl khác."
            )
            return self._response(
                answer,
                trace,
                context=self._merge_context(context, request),
                status="no_rooms",
            )

        answer = self._format_room_answer(availability)
        response_context = self._merge_context(context, request)
        logger.log_event(
            "VINPEARL_AGENT_END",
            {
                "status": "ok",
                "hotel_id": hotel["hotel_id"],
                "checkin": request["checkin"],
                "checkout": request["checkout"],
                "guests": guests,
                "returned_options": len(availability["options"]),
            },
        )
        return self._response(
            answer,
            trace,
            context=response_context,
            status="ok",
            room_cards=availability["options"],
            summary={
                "hotel_name": hotel["hotel_name"],
                "address": hotel["address"],
                "checkin": request["checkin"],
                "checkout": request["checkout"],
                "guests": guests,
                "nights": availability["nights"],
                "total_matches": availability["total_matches"],
            },
        )

    def locations_for_ui(self) -> List[Dict[str, Any]]:
        locations = []
        for hotel in self.kb.all_hotels():
            region = self.kb.regions_by_id[hotel["region_id"]]
            locations.append(
                {
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "address": hotel["address"],
                    "region": region["name_vi"],
                    "label": f"{hotel['hotel_name']} - {hotel['address']}",
                }
            )
        return locations

    def _is_in_scope(
        self,
        message: str,
        context: Dict[str, Any],
        has_location_match: bool,
        llm_intent: Optional[str] = None,
    ) -> bool:
        if llm_intent == "room_search":
            return True
        if llm_intent == "out_of_scope":
            return False
        normalized = normalize_text(message)
        if not normalized:
            return True
        has_room_intent = looks_like_vinpearl_room_request(message)
        has_context = any(
            context.get(key) for key in ("location", "hotel_id", "checkin", "checkout")
        )
        has_off_topic_hint = any(hint in normalized for hint in OFF_TOPIC_HINTS)
        if has_off_topic_hint and not has_room_intent:
            return False
        return has_room_intent or has_context or has_location_match

    def _extract_request(
        self,
        message: str,
        context: Dict[str, Any],
        llm_extraction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        llm_extraction = llm_extraction or {}
        default_year = self.kb.min_date.year
        parsed_checkin, parsed_checkout = extract_date_range(
            message,
            default_year=default_year,
        )
        checkin = (
            parsed_checkin
            or self._clean_iso_date(llm_extraction.get("checkin"))
            or context.get("checkin")
        )
        checkout = (
            parsed_checkout
            or self._clean_iso_date(llm_extraction.get("checkout"))
            or context.get("checkout")
        )
        guests = (
            extract_guests(message)
            or llm_extraction.get("guests")
            or context.get("guests")
        )
        try:
            guests = int(guests) if guests else None
        except (TypeError, ValueError):
            guests = None

        location_query = (
            llm_extraction.get("location_query")
            or context.get("location")
            or message
        )
        if context.get("hotel_id") and not self.kb.find_locations(message, limit=1):
            location_query = context.get("location") or ""

        return {
            "hotel_id": context.get("hotel_id"),
            "location": context.get("location"),
            "location_query": location_query,
            "checkin": checkin,
            "checkout": checkout,
            "guests": guests,
        }

    def _extract_request_with_llm(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None

        hotels = [
            {
                "hotel_id": hotel["hotel_id"],
                "hotel_name": hotel["hotel_name"],
                "address": hotel["address"],
                "region": self.kb.regions_by_id[hotel["region_id"]]["name_vi"],
            }
            for hotel in self.kb.all_hotels()
        ]
        system_prompt = (
            "You extract structured booking-search parameters for a Vinpearl room "
            "availability agent. Return only a valid JSON object. Do not answer the "
            "user. Use intent='room_search' only for Vinpearl hotel/resort room "
            "availability, price, or booking-date questions. Use intent='out_of_scope' "
            "for unrelated topics. Convert dates without year to 2026. If a value is "
            "missing, use null. JSON keys: intent, location_query, checkin, checkout, guests."
        )
        prompt = json.dumps(
            {
                "message": message,
                "context": context,
                "demo_availability_date_range": {
                    "from": self.kb.min_date.isoformat(),
                    "to": self.kb.max_date.isoformat(),
                },
                "known_vinpearl_locations": hotels,
            },
            ensure_ascii=False,
        )

        try:
            result = self.llm.generate(prompt, system_prompt=system_prompt)
            logger.log_event(
                "OPENAI_EXTRACT_REQUEST",
                {
                    "provider": result.get("provider"),
                    "model": self.llm.model_name,
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms"),
                },
            )
            parsed = self._parse_json_object(result.get("content", ""))
            parsed["_provider"] = result.get("provider")
            parsed["_model"] = self.llm.model_name
            parsed["_usage"] = result.get("usage", {})
            return parsed
        except Exception as exc:
            error_message = self._safe_error_message(str(exc))
            logger.log_event(
                "OPENAI_EXTRACT_REQUEST_FAILED",
                {
                    "model": self.llm.model_name,
                    "error_type": exc.__class__.__name__,
                    "message": error_message,
                },
            )
            return {
                "_provider": "openai",
                "_model": self.llm.model_name,
                "_error": f"{exc.__class__.__name__}: {error_message}",
            }

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _clean_iso_date(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else None

    @staticmethod
    def _safe_error_message(message: str) -> str:
        return re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***", message)

    def _merge_context(
        self,
        context: Dict[str, Any],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(context)
        for key in ("hotel_id", "location", "checkin", "checkout", "guests"):
            if request.get(key):
                merged[key] = request[key]
        return merged

    @staticmethod
    def _response(
        answer: str,
        trace: List[Dict[str, str]],
        context: Dict[str, Any],
        status: str,
        room_cards: Optional[List[Dict[str, Any]]] = None,
        location_options: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "status": status,
            "trace": trace,
            "trace_text": VinpearlRoomAgent._trace_to_text(trace, answer),
            "context": context,
            "room_cards": room_cards or [],
            "location_options": location_options or [],
            "summary": summary or {},
        }

    @staticmethod
    def _trace_to_text(trace: List[Dict[str, str]], final_answer: str) -> str:
        blocks = []
        for item in trace:
            blocks.append(
                "\n".join(
                    [
                        f"Thought: {item['thought']}",
                        f"Action: {item['action']}",
                        f"Observation: {item['observation']}",
                    ]
                )
            )
        blocks.append(f"Final Answer: {final_answer}")
        return "\n".join(blocks)

    def _ask_for_location(self, examples: List[Dict[str, str]]) -> str:
        sample = "; ".join(
            f"{item['hotel_name']} - {item['address']}" for item in examples[:3]
        )
        return (
            "Vui lòng cung cấp địa chỉ hoặc tên cơ sở Vinpearl cần kiểm tra phòng. "
            f"Ví dụ: {sample}."
        )

    @staticmethod
    def _ask_to_disambiguate(matches: List[Dict[str, Any]]) -> str:
        lines = [
            "Mình tìm thấy nhiều cơ sở Vinpearl khớp với thông tin bạn nhập. "
            "Vui lòng chọn đúng địa chỉ/cơ sở:"
        ]
        for index, item in enumerate(matches, start=1):
            lines.append(
                f"{index}. {item['hotel_name']} - {item['address']} ({item['region']})"
            )
        return "\n".join(lines)

    def _format_room_answer(self, availability: Dict[str, Any]) -> str:
        hotel = availability["hotel"]
        checkin = self._display_date(availability["checkin"])
        checkout = self._display_date(availability["checkout"])
        guests = availability.get("guests") or 2
        options = availability["options"]
        lines = [
            (
                f"Mình tìm thấy {availability['total_matches']} hạng phòng còn trống tại "
                f"{hotel['hotel_name']} ({hotel['address']}) cho {guests} khách, "
                f"từ {checkin} đến {checkout} ({availability['nights']} đêm)."
            ),
            "Các lựa chọn nổi bật:",
        ]

        for index, room in enumerate(options[:5], start=1):
            promo_text = ""
            if room["discount_total_vnd"] > 0:
                promo_names = ", ".join(item["name_vi"] for item in room["promotions"])
                promo_text = (
                    f"; giá gốc {format_vnd(room['gross_total_vnd'])}, "
                    f"đã trừ {format_vnd(room['discount_total_vnd'])} ({promo_names})"
                )
            bed_text = ", ".join(room["bed_options"])
            amenities = ", ".join(room["amenities"][:4])
            lines.append(
                (
                    f"{index}. {room['room_name']}: còn tối thiểu "
                    f"{room['min_available_rooms']} phòng; {room['area_sqm']}m2; "
                    f"tối đa {room['max_guests']} khách; giường {bed_text}; "
                    f"{room['meal_plan']}; tổng {format_vnd(room['total_vnd'])}{promo_text}. "
                    f"Tiện ích: {amenities}."
                )
            )

        lines.append(
            "Lưu ý: đây là dữ liệu giả lập phục vụ demo/đào tạo, không phải giá "
            "và tồn phòng thực tế của Vinpearl."
        )
        return "\n".join(lines)

    @staticmethod
    def _display_date(value: str) -> str:
        year, month, day = value.split("-")
        return f"{day}/{month}/{year}"
