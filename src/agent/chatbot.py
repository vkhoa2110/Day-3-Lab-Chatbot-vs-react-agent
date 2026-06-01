from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools.tools import check_room_availability, get_vinpearl_locations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "vinpearl.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Phi-3-mini-4k-instruct-q4.gguf"


BASELINE_SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn tìm phòng trống Vinpearl có kết nối tool nội bộ.

Nhiệm vụ duy nhất:
- Tư vấn bước tiếp theo để tìm phòng trống tại cơ sở Vinpearl.
- Chỉ nói về Vinpearl, cơ sở/địa chỉ, ngày nhận phòng, ngày trả phòng, số khách,
  hạng phòng, giá tham khảo và chính sách đặt/hủy phòng.
- Khi backend đã cung cấp kết quả tool, chỉ tóm tắt đúng dữ liệu đó.

Quy tắc bắt buộc:
- Nếu thiếu cơ sở/địa chỉ Vinpearl, ngày nhận phòng, ngày trả phòng hoặc số khách,
  chỉ hỏi lại đúng thông tin còn thiếu.
- Nếu người dùng hỏi ngoài chủ đề, từ chối ngắn gọn và mời họ nhập yêu cầu tìm phòng Vinpearl.
- Không bịa danh sách phòng trống, không bịa giá, không bịa mã phòng.
- Trả lời bằng tiếng Việt tự nhiên, tối đa 5 câu, không lan man.
""".strip()


class VinpearlDemoKnowledgeBase:
    """Small read-only helper for UI metadata; the baseline LLM does not query it."""

    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.data = self._load_data()
        availability_dates = [item["date"] for item in self.data.get("availability", [])]
        self.min_date = datetime.strptime(min(availability_dates), "%Y-%m-%d").date()
        self.max_date = datetime.strptime(max(availability_dates), "%Y-%m-%d").date()

    def _load_data(self) -> Dict[str, Any]:
        with self.data_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def locations_for_ui(self) -> List[Dict[str, Any]]:
        regions = {item["region_id"]: item for item in self.data.get("regions", [])}
        hotels = []
        for hotel in self.data.get("hotels", []):
            region = regions.get(hotel["region_id"], {})
            region_name = region.get("name_vi", "")
            hotels.append(
                {
                    "hotel_id": hotel["hotel_id"],
                    "hotel_name": hotel["hotel_name"],
                    "address": hotel.get("address", ""),
                    "region": region_name,
                    "label": f"{hotel['hotel_name']} - {hotel.get('address', '')}",
                }
            )
        return hotels


class VinpearlBaselineChatbot:
    """
    Tool-connected Vinpearl chatbot.
    It uses deterministic Python tools for lookup/availability and keeps the LLM for wording.
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        system_prompt: str = BASELINE_SYSTEM_PROMPT,
        kb: Optional[VinpearlDemoKnowledgeBase] = None,
    ):
        self.kb = kb or VinpearlDemoKnowledgeBase()
        self._llm = llm
        self._llm_was_provided = llm is not None
        self.system_prompt = system_prompt

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = self._build_local_llm()
        return self._llm

    def _build_local_llm(self) -> LLMProvider:
        from src.core.local_provider import LocalProvider

        model_path = Path(os.getenv("LOCAL_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        n_ctx = int(os.getenv("LOCAL_N_CTX", "4096"))
        n_threads_env = os.getenv("LOCAL_N_THREADS")
        n_threads = int(n_threads_env) if n_threads_env else None
        max_tokens = int(os.getenv("BASELINE_MAX_TOKENS", os.getenv("LOCAL_MAX_TOKENS", "256")))
        n_gpu_layers = int(os.getenv("LOCAL_N_GPU_LAYERS", "-1"))
        n_batch = int(os.getenv("LOCAL_N_BATCH", "512"))
        main_gpu = int(os.getenv("LOCAL_MAIN_GPU", "0"))
        temperature = float(os.getenv("LOCAL_TEMPERATURE", "0.2"))
        top_p = float(os.getenv("LOCAL_TOP_P", "0.9"))
        repeat_penalty = float(os.getenv("LOCAL_REPEAT_PENALTY", "1.1"))

        return LocalProvider(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            max_tokens=max_tokens,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            main_gpu=main_gpu,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )

    def locations_for_ui(self) -> List[Dict[str, Any]]:
        return self.kb.locations_for_ui()

    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = self._enrich_context_from_message(self._clean_context(context or {}), message)
        trace_lines: List[str] = []

        if not self._is_in_scope(message, context):
            return self._response_payload(
                answer="Mình chỉ hỗ trợ các yêu cầu tìm phòng trống tại cơ sở Vinpearl. Anh/chị vui lòng nhập cơ sở Vinpearl, ngày lưu trú và số khách.",
                context=context,
            )

        locations_result: Optional[Dict[str, Any]] = None
        selected_location: Optional[Dict[str, Any]] = None
        if not context.get("hotel_id"):
            location_query = context.get("location") or message
            locations_result = get_vinpearl_locations(location_query)
            trace_lines.append(f"Action: get_vinpearl_locations(region={location_query!r})")
            trace_lines.append(f"Observation: {locations_result.get('message')}")

            locations = locations_result.get("locations", [])
            if not locations:
                return self._response_payload(
                    answer="Mình chưa tìm thấy cơ sở Vinpearl phù hợp. Anh/chị vui lòng cung cấp rõ tên cơ sở hoặc địa chỉ Vinpearl.",
                    context=context,
                    trace_text="\n".join(trace_lines),
                )

            selected_location = self._pick_single_location(locations, location_query)
            if not selected_location:
                return self._response_payload(
                    answer="Mình tìm thấy nhiều cơ sở Vinpearl phù hợp. Anh/chị vui lòng chọn đúng cơ sở hoặc địa chỉ trước khi kiểm tra phòng trống.",
                    context=context,
                    location_options=self._location_options(locations),
                    trace_text="\n".join(trace_lines),
                )

            context["hotel_id"] = selected_location["hotel_id"]
            context["location"] = f"{selected_location['hotel_name']} - {selected_location.get('address', '')}"

        missing_fields = self._missing_fields(context)
        if missing_fields:
            logger.log_event(
                "TOOL_CHATBOT_GUARDRAIL",
                {"reason": "missing_required_fields", "missing_fields": missing_fields},
            )
            return self._response_payload(
                answer=self._missing_info_answer(missing_fields),
                context=context,
                trace_text="\n".join(trace_lines),
            )

        availability = check_room_availability(
            hotel_id=context["hotel_id"],
            checkin=context["checkin"],
            checkout=context["checkout"],
            guests=context["guests"],
        )
        trace_lines.append(
            "Action: check_room_availability("
            f"hotel_id={context['hotel_id']!r}, checkin={context['checkin']!r}, "
            f"checkout={context['checkout']!r}, guests={context['guests']!r})"
        )
        trace_lines.append(f"Observation: {availability.get('message')}")

        room_cards = self._room_cards(availability.get("available_rooms", []))
        fallback_answer = self._availability_answer(availability, room_cards)
        answer = self._grounded_answer_with_llm(
            message=message,
            context=context,
            availability=availability,
            room_cards=room_cards,
            fallback_answer=fallback_answer,
            trace_lines=trace_lines,
        )

        return self._response_payload(
            answer=answer,
            context=context,
            room_cards=room_cards,
            trace_text="\n".join(trace_lines),
        )

    def _response_payload(
        self,
        answer: str,
        context: Dict[str, Any],
        room_cards: Optional[List[Dict[str, Any]]] = None,
        location_options: Optional[List[Dict[str, Any]]] = None,
        trace_text: str = "",
    ) -> Dict[str, Any]:
        return {
            "mode": "tool_chatbot",
            "answer": answer,
            "context": context,
            "room_cards": room_cards or [],
            "location_options": location_options or [],
            "trace_text": trace_text,
        }

    def _enrich_context_from_message(self, context: Dict[str, Any], message: str) -> Dict[str, Any]:
        if not context.get("location"):
            locations_result = get_vinpearl_locations(message)
            locations = locations_result.get("locations", [])
            selected_location = self._pick_single_location(locations, message)
            if selected_location:
                context["hotel_id"] = selected_location["hotel_id"]
                context["location"] = f"{selected_location['hotel_name']} - {selected_location.get('address', '')}"
            elif locations:
                context["location"] = message

        if not context.get("checkin") or not context.get("checkout"):
            dates = self._extract_dates(message)
            if dates and not context.get("checkin"):
                context["checkin"] = dates[0]
            if len(dates) > 1 and not context.get("checkout"):
                context["checkout"] = dates[1]

        if not context.get("guests"):
            guests = self._extract_guests(message)
            if guests:
                context["guests"] = guests

        return context

    def _is_in_scope(self, message: str, context: Dict[str, Any]) -> bool:
        if context.get("hotel_id") or context.get("location"):
            return True

        normalized = self._normalize_for_match(message)
        keywords = (
            "vinpearl",
            "vinholidays",
            "vinwonders",
            "phong",
            "hotel",
            "resort",
            "checkin",
            "checkout",
            "nhan phong",
            "tra phong",
            "khach",
        )
        return any(keyword in normalized for keyword in keywords)

    def _extract_dates(self, message: str) -> List[str]:
        pattern = r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
        return re.findall(pattern, message or "")[:2]

    def _extract_guests(self, message: str) -> Optional[int]:
        match = re.search(r"\b(\d{1,2})\s*(?:khach|khách|nguoi|người|pax)\b", message or "", re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r"\bcho\s+(\d{1,2})\b", message or "", re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _pick_single_location(self, locations: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        if len(locations) == 1:
            return locations[0]

        normalized_query = self._normalize_for_match(query)
        exact_matches = []
        for location in locations:
            hotel_name = self._normalize_for_match(location.get("hotel_name"))
            address = self._normalize_for_match(location.get("address"))
            if hotel_name and hotel_name in normalized_query:
                exact_matches.append(location)
            elif address and address in normalized_query:
                exact_matches.append(location)

        if len(exact_matches) == 1:
            return exact_matches[0]
        return None

    def _location_options(self, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "hotel_id": location["hotel_id"],
                "hotel_name": location["hotel_name"],
                "address": location.get("address", ""),
                "region": location.get("region_name") or location.get("province", ""),
            }
            for location in locations[:8]
        ]

    def _room_cards(self, available_rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cards = []
        for room in available_rooms:
            meal_plan = room.get("meal_plan") or {}
            cancellation_policy = room.get("cancellation_policy") or {}
            cards.append(
                {
                    "room_type_id": room.get("room_type_id"),
                    "room_name": room.get("room_name"),
                    "total_display": room.get("total_price_display"),
                    "total_price_vnd": room.get("total_price_vnd"),
                    "min_available_rooms": room.get("available_rooms"),
                    "area_sqm": room.get("area_sqm"),
                    "max_guests": room.get("max_guests"),
                    "bed_options": room.get("bed_options", []),
                    "meal_plan": meal_plan.get("name_vi") or meal_plan.get("meal_plan_id") or "",
                    "cancellation_policy": cancellation_policy.get("name_vi") or "",
                    "amenities": room.get("amenities", []),
                    "promotions": [],
                }
            )
        return cards

    def _availability_answer(self, availability: Dict[str, Any], room_cards: List[Dict[str, Any]]) -> str:
        if availability.get("status") != "success":
            return availability.get("message") or "Không tìm thấy phòng trống phù hợp trong dữ liệu demo."

        hotel = availability.get("hotel", {})
        lines = [
            f"Mình đã kiểm tra dữ liệu demo cho {hotel.get('hotel_name')} ({hotel.get('address')}).",
            f"Thời gian: {availability.get('checkin')} đến {availability.get('checkout')} - {availability.get('nights')} đêm, {availability.get('guests')} khách.",
            f"Có {availability.get('available_count')} hạng phòng còn trống. Một số lựa chọn giá tốt:",
        ]

        for room in room_cards[:3]:
            lines.append(
                f"- {room.get('room_name')}: {room.get('total_display')}, còn {room.get('min_available_rooms')} phòng, tối đa {room.get('max_guests')} khách."
            )

        return "\n".join(lines)

    def _grounded_answer_with_llm(
        self,
        message: str,
        context: Dict[str, Any],
        availability: Dict[str, Any],
        room_cards: List[Dict[str, Any]],
        fallback_answer: str,
        trace_lines: List[str],
    ) -> str:
        llm_mode = os.getenv("BASELINE_USE_LLM", "auto").strip().lower()
        if llm_mode in {"0", "false", "no", "off"}:
            trace_lines.append("Action: llm.generate(skipped)")
            trace_lines.append("Observation: BASELINE_USE_LLM disabled; used deterministic answer.")
            return fallback_answer
        if llm_mode in {"", "auto"} and not self._llm_was_provided:
            trace_lines.append("Action: llm.generate(skipped)")
            trace_lines.append("Observation: BASELINE_USE_LLM auto skipped local wording; used deterministic answer.")
            return fallback_answer

        prompt = self._build_grounded_answer_prompt(
            message=message,
            context=context,
            availability=availability,
            room_cards=room_cards,
            fallback_answer=fallback_answer,
        )

        try:
            llm = self.llm
            trace_lines.append(f"Action: llm.generate(model={llm.model_name!r})")
            result = llm.generate(prompt, system_prompt=self.system_prompt)
        except Exception as exc:
            logger.log_event(
                "LLM_FALLBACK",
                {"reason": "generation_failed", "error": str(exc)},
            )
            trace_lines.append(f"Observation: LLM fallback because generation failed: {exc}")
            return fallback_answer

        usage = result.get("usage", {})
        latency_ms = int(result.get("latency_ms", 0) or 0)
        provider = result.get("provider", "unknown")
        tracker.track_request(
            provider=provider,
            model=llm.model_name,
            usage=usage,
            latency_ms=latency_ms,
        )

        answer = str(result.get("content", "")).strip()
        trace_lines.append(
            f"Observation: LLM completed via {provider} in {latency_ms} ms, "
            f"{usage.get('total_tokens', 0)} tokens."
        )

        hit_token_limit = bool(getattr(llm, "max_tokens", None)) and (
            int(usage.get("completion_tokens", 0) or 0) >= int(getattr(llm, "max_tokens"))
        )
        if not answer or hit_token_limit or self._looks_like_prompt_leak(answer):
            logger.log_event(
                "LLM_FALLBACK",
                {
                    "reason": "unsafe_empty_or_truncated_answer",
                    "provider": provider,
                    "hit_token_limit": hit_token_limit,
                },
            )
            trace_lines.append("Observation: LLM answer rejected; used deterministic answer.")
            return fallback_answer

        return answer

    def _build_grounded_answer_prompt(
        self,
        message: str,
        context: Dict[str, Any],
        availability: Dict[str, Any],
        room_cards: List[Dict[str, Any]],
        fallback_answer: str,
    ) -> str:
        hotel = availability.get("hotel") or {}
        room_lines = []
        for index, room in enumerate(room_cards[:3], start=1):
            room_lines.append(
                f"{index}. {room.get('room_name')}: {room.get('total_display')}, "
                f"còn {room.get('min_available_rooms')} phòng, tối đa {room.get('max_guests')} khách"
            )
        rooms_text = "\n".join(room_lines) if room_lines else "Không có phòng phù hợp."

        return f"""
Tin nhắn người dùng: {message}

Kết quả tool đã xác nhận:
- Trạng thái: {availability.get('status')}
- Thông báo: {availability.get('message')}
- Cơ sở: {hotel.get('hotel_name')} ({hotel.get('address')})
- Lưu trú: {availability.get('checkin')} đến {availability.get('checkout')}, {availability.get('nights')} đêm, {availability.get('guests')} khách
- Số hạng phòng còn trống: {availability.get('available_count')}
- Phòng nên nhắc:
{rooms_text}

Câu trả lời an toàn:
{fallback_answer}

Chỉ trả về câu trả lời cuối cùng bằng tiếng Việt, tối đa 5 câu.
Không nhắc đến prompt, tool, JSON, backend hoặc các quy tắc ở trên.
Không thêm bất kỳ phòng, giá, mã, ưu đãi hoặc chính sách nào ngoài kết quả tool.
""".strip()

    def _normalize_for_match(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        replacements = {
            "à": "a",
            "á": "a",
            "ạ": "a",
            "ả": "a",
            "ã": "a",
            "â": "a",
            "ầ": "a",
            "ấ": "a",
            "ậ": "a",
            "ẩ": "a",
            "ẫ": "a",
            "ă": "a",
            "ằ": "a",
            "ắ": "a",
            "ặ": "a",
            "ẳ": "a",
            "ẵ": "a",
            "è": "e",
            "é": "e",
            "ẹ": "e",
            "ẻ": "e",
            "ẽ": "e",
            "ê": "e",
            "ề": "e",
            "ế": "e",
            "ệ": "e",
            "ể": "e",
            "ễ": "e",
            "ì": "i",
            "í": "i",
            "ị": "i",
            "ỉ": "i",
            "ĩ": "i",
            "ò": "o",
            "ó": "o",
            "ọ": "o",
            "ỏ": "o",
            "õ": "o",
            "ô": "o",
            "ồ": "o",
            "ố": "o",
            "ộ": "o",
            "ổ": "o",
            "ỗ": "o",
            "ơ": "o",
            "ờ": "o",
            "ớ": "o",
            "ợ": "o",
            "ở": "o",
            "ỡ": "o",
            "ù": "u",
            "ú": "u",
            "ụ": "u",
            "ủ": "u",
            "ũ": "u",
            "ư": "u",
            "ừ": "u",
            "ứ": "u",
            "ự": "u",
            "ử": "u",
            "ữ": "u",
            "ỳ": "y",
            "ý": "y",
            "ỵ": "y",
            "ỷ": "y",
            "ỹ": "y",
            "đ": "d",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return " ".join(text.split())

    def _clean_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key in ("hotel_id", "location", "checkin", "checkout", "guests"):
            value = context.get(key)
            if value not in (None, ""):
                cleaned[key] = value
        return cleaned

    def _missing_fields(self, context: Dict[str, Any]) -> List[str]:
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

    def _missing_info_answer(self, missing_fields: List[str]) -> str:
        fields = ", ".join(missing_fields)
        return (
            "Để kiểm tra phòng trống Vinpearl, anh/chị vui lòng cung cấp thêm: "
            f"{fields}. Sau đó mình sẽ dùng tool kiểm tra tồn phòng và trả danh sách hạng phòng còn trống."
        )

    def _safe_baseline_reply(self, context: Dict[str, Any]) -> str:
        location = context.get("location") or context.get("hotel_id") or "cơ sở Vinpearl đã chọn"
        return (
            f"Mình đã ghi nhận yêu cầu tìm phòng tại {location}. "
            "Vì đây là chatbot baseline không có tool kiểm tra tồn phòng, mình chưa thể xác nhận danh sách phòng trống hoặc giá chính xác. "
            "Hãy dùng bản agent/tool để kiểm tra availability và trả về chi tiết từng hạng phòng."
        )

    def _looks_like_prompt_leak(self, answer: str) -> bool:
        lowered = answer.lower()
        leak_markers = (
            "== ai ==",
            "nhiệm vụ:",
            "quy tắc:",
            "system prompt",
            "thông tin còn thiếu:",
            "dữ liệu backend",
            "kết quả tool",
            "tool_result",
            "safe_fallback",
            "written",
            "written in english",
            "only use",
            "no more than",
            "json",
        )
        return len(answer) > 1200 or any(marker in lowered for marker in leak_markers)

    def _build_prompt(self, message: str, context: Dict[str, Any]) -> str:
        context_lines = []
        labels = {
            "hotel_id": "Hotel ID",
            "location": "Cơ sở/địa chỉ",
            "checkin": "Ngày nhận phòng",
            "checkout": "Ngày trả phòng",
            "guests": "Số khách",
        }
        for key, label in labels.items():
            if key in context:
                context_lines.append(f"- {label}: {context[key]}")

        if not context_lines:
            context_text = "Không có ngữ cảnh từ UI."
        else:
            context_text = "\n".join(context_lines)

        missing_fields = self._missing_fields(context)

        missing_text = ", ".join(missing_fields) if missing_fields else "Không thiếu thông tin chính."

        return f"""
Tin nhắn người dùng:
{message}

Ngữ cảnh từ UI:
{context_text}

Thông tin còn thiếu:
{missing_text}

Hãy trả lời như chatbot baseline theo system prompt.
""".strip()


VinpearlRoomAgent = VinpearlBaselineChatbot


__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "VinpearlBaselineChatbot",
    "VinpearlDemoKnowledgeBase",
    "VinpearlRoomAgent",
]
