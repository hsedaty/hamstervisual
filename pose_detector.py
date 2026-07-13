import base64
import math
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    FACE_DEBUG_INDICES = [10, 152, 1, 13, 14, 159, 145, 386, 374, 468, 473]
    HAND_DEBUG_INDICES = list(range(21))

    def __init__(self) -> None:
        self.hand_landmark = mp.solutions.hands.HandLandmark
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_from_base64(self, image_data: str) -> dict[str, Any]:
        if "," not in image_data:
            raise ValueError("Expected a data URL payload.")

        _, encoded = image_data.split(",", 1)

        try:
            binary = base64.b64decode(encoded)
        except ValueError as exc:
            raise ValueError("Could not decode the webcam frame.") from exc

        frame_array = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not parse the webcam frame.")

        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self.face_mesh.process(rgb_frame)
        hand_result = self.hands.process(rgb_frame)

        face = self._extract_face(face_result.multi_face_landmarks, frame.shape)
        hands = self._extract_hands(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness,
            frame.shape,
        )

        pose_name, confidence = self._classify_pose(face, hands)
        metrics = self._build_metrics(face, hands)
        overlay = self._build_overlay(face, hands)

        return {
            "pose": pose_name,
            "confidence": round(confidence, 2),
            "metrics": metrics,
            "overlay": overlay,
            "handCount": len(hands),
            "faceDetected": bool(face),
        }

    def _extract_face(self, multi_face_landmarks, frame_shape):
        if not multi_face_landmarks:
            return None

        height, width = frame_shape[:2]
        landmarks = multi_face_landmarks[0].landmark

        def point(index: int) -> tuple[float, float]:
            landmark = landmarks[index]
            return landmark.x * width, landmark.y * height

        top = point(10)
        chin = point(152)
        left_eye_top = point(159)
        left_eye_bottom = point(145)
        right_eye_top = point(386)
        right_eye_bottom = point(374)
        mouth_upper = point(13)
        mouth_lower = point(14)
        nose = point(1)
        left_eye_center = point(468)
        right_eye_center = point(473)

        face_height = self._distance(top, chin)
        debug_points = [point(index) for index in self.FACE_DEBUG_INDICES]

        return {
            "face_height": max(face_height, 1.0),
            "nose": nose,
            "mouth_center": self._midpoint(mouth_upper, mouth_lower),
            "mouth_open": self._distance(mouth_upper, mouth_lower),
            "left_eye_open": self._distance(left_eye_top, left_eye_bottom),
            "right_eye_open": self._distance(right_eye_top, right_eye_bottom),
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
            "debug_points": debug_points,
        }

    def _extract_hands(self, multi_hand_landmarks, multi_handedness, frame_shape):
        if not multi_hand_landmarks:
            return []

        handedness_list = multi_handedness or []
        hands = []

        for index, hand_landmarks in enumerate(multi_hand_landmarks):
            label = "Unknown"
            if index < len(handedness_list):
                label = handedness_list[index].classification[0].label

            points = self._extract_hand_points(hand_landmarks, frame_shape)
            wrist = points["wrist"]
            thumb = points["thumb"]
            index_finger = points["index"]
            middle_finger = points["middle"]
            ring_finger = points["ring"]
            pinky = points["pinky"]
            debug_points = points["debug_points"]
            palm_center = self._average_point(
                [
                    wrist,
                    index_finger["mcp"],
                    middle_finger["mcp"],
                    ring_finger["mcp"],
                    pinky["mcp"],
                ]
            )

            hands.append(
                {
                    "label": label,
                    "wrist": wrist,
                    "thumb": thumb,
                    "index": index_finger,
                    "middle": middle_finger,
                    "ring": ring_finger,
                    "pinky": pinky,
                    "thumb_mcp": thumb["mcp"],
                    "thumb_ip": thumb["ip"],
                    "thumb_tip": thumb["tip"],
                    "index_mcp": index_finger["mcp"],
                    "index_pip": index_finger["pip"],
                    "index_tip": index_finger["tip"],
                    "middle_mcp": middle_finger["mcp"],
                    "middle_pip": middle_finger["pip"],
                    "middle_tip": middle_finger["tip"],
                    "ring_mcp": ring_finger["mcp"],
                    "ring_pip": ring_finger["pip"],
                    "ring_tip": ring_finger["tip"],
                    "pinky_mcp": pinky["mcp"],
                    "pinky_pip": pinky["pip"],
                    "pinky_tip": pinky["tip"],
                    "center": palm_center,
                    "palm_width": self._distance(index_finger["mcp"], pinky["mcp"]),
                    "debug_points": debug_points,
                }
            )

        return hands

    def _extract_hand_points(self, hand_landmarks, frame_shape) -> dict[str, Any]:
        point = self._build_hand_point_getter(hand_landmarks, frame_shape)

        thumb = {
            "cmc": point(self.hand_landmark.THUMB_CMC),
            "mcp": point(self.hand_landmark.THUMB_MCP),
            "ip": point(self.hand_landmark.THUMB_IP),
            "tip": point(self.hand_landmark.THUMB_TIP),
        }
        index_finger = {
            "mcp": point(self.hand_landmark.INDEX_FINGER_MCP),
            "pip": point(self.hand_landmark.INDEX_FINGER_PIP),
            "dip": point(self.hand_landmark.INDEX_FINGER_DIP),
            "tip": point(self.hand_landmark.INDEX_FINGER_TIP),
        }
        middle_finger = {
            "mcp": point(self.hand_landmark.MIDDLE_FINGER_MCP),
            "pip": point(self.hand_landmark.MIDDLE_FINGER_PIP),
            "dip": point(self.hand_landmark.MIDDLE_FINGER_DIP),
            "tip": point(self.hand_landmark.MIDDLE_FINGER_TIP),
        }
        ring_finger = {
            "mcp": point(self.hand_landmark.RING_FINGER_MCP),
            "pip": point(self.hand_landmark.RING_FINGER_PIP),
            "dip": point(self.hand_landmark.RING_FINGER_DIP),
            "tip": point(self.hand_landmark.RING_FINGER_TIP),
        }
        pinky = {
            "mcp": point(self.hand_landmark.PINKY_MCP),
            "pip": point(self.hand_landmark.PINKY_PIP),
            "dip": point(self.hand_landmark.PINKY_DIP),
            "tip": point(self.hand_landmark.PINKY_TIP),
        }

        return {
            "wrist": point(self.hand_landmark.WRIST),
            "thumb": thumb,
            "index": index_finger,
            "middle": middle_finger,
            "ring": ring_finger,
            "pinky": pinky,
            "debug_points": [point(landmark_index) for landmark_index in self.HAND_DEBUG_INDICES],
        }

    def _build_hand_point_getter(self, hand_landmarks, frame_shape):
        height, width = frame_shape[:2]

        def point(landmark_index) -> tuple[float, float]:
            landmark = hand_landmarks.landmark[int(landmark_index)]
            return landmark.x * width, landmark.y * height

        return point

    def _classify_pose(self, face, hands) -> tuple[str, float]:
        if self._is_love_pose(hands):
            return "love", 0.93

        if self._is_peace_pose(hands):
            return "peace", 0.9

        thumb_up_hand = self._find_thumbs_up_hand(hands)
        if thumb_up_hand:
            return "thumb up", 0.88

        thumb_down_hand = self._find_thumbs_down_hand(hands)
        if thumb_down_hand:
            return "thumb down", 0.88

        if face and self._is_shh_pose(face, hands):
            return "shh", 0.9

        if face and self._is_strong_pose(face, hands):
            return "strong", 0.86

        if face and self._is_nerd_pose(face, hands):
            return "nerd", 0.78

        if face:
            return "neutral", 0.66

        return "no-face", 0.0

    def _is_peace_pose(self, hands) -> bool:
        if len(hands) < 2:
            return False

        return all(self._is_peace_hand(hand) for hand in hands[:2])

    def _is_shh_pose(self, face, hands) -> bool:
        mouth = face["mouth_center"]
        face_height = face["face_height"]

        for hand in hands:
            if not self._is_pointing_hand(hand):
                continue

            index_finger = hand["index"]
            finger_distance = self._distance(index_finger["tip"], mouth)
            vertical_extension = abs(index_finger["tip"][1] - index_finger["pip"][1])
            horizontal_extension = abs(index_finger["tip"][0] - index_finger["pip"][0])
            if finger_distance < face_height * 0.28 and vertical_extension > horizontal_extension * 0.75:
                return True

        return False

    def _is_nerd_pose(self, face, hands) -> bool:
        face_height = face["face_height"]
        face_targets = [
            face["mouth_center"],
            face["nose"],
            face["left_eye_center"],
            face["right_eye_center"],
        ]

        for hand in hands:
            if not self._is_index_only_hand(hand):
                continue

            index_finger = hand["index"]
            distances = [self._distance(index_finger["tip"], target) for target in face_targets]
            hand_distance = self._distance(hand["center"], face["nose"])
            if min(distances) > face_height * 0.55 and hand_distance > face_height * 0.75:
                return True

        return False

    def _is_love_pose(self, hands) -> bool:
        details = self._love_pose_details(hands)
        return (
            details["thumbTipsClose"]
            and details["thumbsBelowOtherTips"]
            and details["pinkyCurled"]
        )

    def _is_strong_pose(self, face, hands) -> bool:
        face_height = face["face_height"]
        mouth_x, mouth_y = face["mouth_center"]

        for hand in hands:
            if not self._is_closed_fist(hand):
                continue

            center_x, center_y = hand["center"]
            if center_y < mouth_y and abs(center_x - mouth_x) < face_height * 0.9:
                return True

        return False

    def _build_metrics(self, face, hands) -> dict[str, Any]:
        checks = self._build_checks(face, hands)
        if not face:
            return {
                "message": "No face detected",
                "hands": self._summarize_hands(hands),
                "checks": checks,
            }

        face_height = face["face_height"]
        metrics = {
            "mouthOpen": round(face["mouth_open"] / face_height, 3),
            "leftEyeOpen": round(face["left_eye_open"] / face_height, 3),
            "rightEyeOpen": round(face["right_eye_open"] / face_height, 3),
            "hands": self._summarize_hands(hands),
            "checks": checks,
        }
        return metrics

    def _build_overlay(self, face, hands) -> dict[str, Any]:
        return {
            "face": [] if not face else [self._serialize_point(point) for point in face["debug_points"]],
            "hands": [
                {
                    "label": hand["label"],
                    "points": [self._serialize_point(point) for point in hand["debug_points"]],
                }
                for hand in hands
            ],
        }

    def _build_checks(self, face, hands) -> dict[str, Any]:
        checks = {
            "peaceHandsDetected": len(hands),
            "peaceReadyHands": sum(1 for hand in hands if self._is_peace_hand(hand)),
            "thumbUpDetected": any(self._is_thumb_up_hand(hand) for hand in hands),
            "thumbDownDetected": any(self._is_thumb_down_hand(hand) for hand in hands),
            "loveReady": self._is_love_pose(hands),
            "loveHandsReady": sum(1 for hand in hands if self._is_love_hand(hand)),
            "loveThumbGap": None,
            "loveThumbsBelowTips": None,
            "lovePinkyCurled": None,
            "shhDistance": None,
            "nerdDistance": None,
            "strongFistAboveMouth": None,
        }

        love_details = self._love_pose_details(hands)
        checks["loveThumbGap"] = love_details["thumbGap"]
        checks["loveThumbsBelowTips"] = love_details["thumbsBelowOtherTips"]
        checks["lovePinkyCurled"] = love_details["pinkyCurled"]

        if not face:
            return checks

        face_height = face["face_height"]
        pointing_hands = [hand for hand in hands if self._is_pointing_hand(hand)]
        if pointing_hands:
            checks["shhDistance"] = round(
                min(
                    self._distance(hand["index"]["tip"], face["mouth_center"]) / face_height
                    for hand in pointing_hands
                ),
                3,
            )
            checks["nerdDistance"] = round(
                min(
                    min(
                        self._distance(hand["index"]["tip"], face["mouth_center"]),
                        self._distance(hand["index"]["tip"], face["nose"]),
                    )
                    / face_height
                    for hand in pointing_hands
                ),
                3,
            )

        checks["strongFistAboveMouth"] = any(
            self._is_closed_fist(hand)
            and hand["center"][1] < face["mouth_center"][1]
            and abs(hand["center"][0] - face["mouth_center"][0]) < face_height * 0.9
            for hand in hands
        )
        return checks

    def _summarize_hands(self, hands) -> list[dict[str, Any]]:
        summaries = []
        for hand in hands:
            finger_states = self._finger_states(hand)
            summaries.append(
                {
                    "label": hand["label"],
                    "x": round(hand["center"][0], 1),
                    "y": round(hand["center"][1], 1),
                    "indexExtended": finger_states["index"],
                    "middleExtended": finger_states["middle"],
                    "ringExtended": finger_states["ring"],
                    "pinkyExtended": finger_states["pinky"],
                    "thumbUp": self._is_thumb_up_hand(hand),
                    "thumbDown": self._is_thumb_down_hand(hand),
                    "peaceReady": self._is_peace_hand(hand),
                    "loveReady": self._is_love_hand(hand),
                    "closedFist": self._is_closed_fist(hand),
                }
            )
        return summaries

    def _find_thumbs_up_hand(self, hands):
        return next((hand for hand in hands if self._is_thumb_up_hand(hand)), None)

    def _find_thumbs_down_hand(self, hands):
        return next((hand for hand in hands if self._is_thumb_down_hand(hand)), None)

    def _is_thumb_up_hand(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        if not self._are_non_thumb_fingers_folded(hand):
            return False

        thumb = hand["thumb"]
        thumb_dx = abs(thumb["tip"][0] - thumb["mcp"][0])
        thumb_dy = thumb["mcp"][1] - thumb["tip"][1]
        thumb_reach = self._distance(thumb["tip"], hand["center"])
        thumb_straight = self._joint_angle(thumb["mcp"], thumb["ip"], thumb["tip"])

        return (
            thumb_dy > thumb_dx * 0.9
            and thumb_reach > hand["palm_width"] * 0.8
            and thumb_straight > 145
        )

    def _is_thumb_down_hand(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        if not self._are_non_thumb_fingers_folded(hand):
            return False

        thumb = hand["thumb"]
        thumb_dx = abs(thumb["tip"][0] - thumb["mcp"][0])
        thumb_dy = thumb["tip"][1] - thumb["mcp"][1]
        thumb_reach = self._distance(thumb["tip"], hand["center"])
        thumb_straight = self._joint_angle(thumb["mcp"], thumb["ip"], thumb["tip"])

        return (
            thumb_dy > thumb_dx * 0.9
            and thumb_reach > hand["palm_width"] * 0.8
            and thumb_straight > 145
        )

    def _is_peace_hand(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        finger_gap = self._distance(hand["index"]["tip"], hand["middle"]["tip"])
        index_middle_diverge = self._distance(hand["index"]["tip"], hand["middle"]["pip"])
        return (
            finger_states["index"]
            and finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
            and finger_gap > hand["palm_width"] * 0.32
            and index_middle_diverge > hand["palm_width"] * 0.42
            and hand["wrist"][1] > min(hand["index"]["tip"][1], hand["middle"]["tip"][1])
        )

    def _is_index_only_hand(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        return (
            finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
            and self._is_thumb_folded(hand)
        )

    def _is_pointing_hand(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        return (
            finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        )

    def _is_love_hand(self, hand) -> bool:
        return self._thumb_below_other_fingertips(hand) and self._is_pinky_curled_for_love(hand)

    def _is_closed_fist(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        if any(finger_states.values()) or not self._is_thumb_folded(hand):
            return False

        curled_tips = [
            hand["index"]["tip"],
            hand["middle"]["tip"],
            hand["ring"]["tip"],
            hand["pinky"]["tip"],
        ]
        return all(
            self._distance(point, hand["center"]) < hand["palm_width"] * 0.85
            for point in curled_tips
        )

    def _finger_states(self, hand) -> dict[str, bool]:
        return {
            "index": self._is_extended_finger(hand["index"], hand["wrist"]),
            "middle": self._is_extended_finger(hand["middle"], hand["wrist"]),
            "ring": self._is_extended_finger(hand["ring"], hand["wrist"]),
            "pinky": self._is_extended_finger(hand["pinky"], hand["wrist"]),
        }

    def _is_extended_finger(
        self,
        finger: dict[str, tuple[float, float]],
        wrist: tuple[float, float],
    ) -> bool:
        tip = finger["tip"]
        dip = finger["dip"]
        pip = finger["pip"]
        mcp = finger["mcp"]
        tip_reach = self._distance(tip, mcp)
        dip_reach = self._distance(dip, mcp)
        pip_reach = self._distance(pip, mcp)
        from_wrist = (
            self._distance(tip, wrist),
            self._distance(dip, wrist),
            self._distance(pip, wrist),
            self._distance(mcp, wrist),
        )
        joint_chain = tip_reach > dip_reach > pip_reach
        wrist_chain = from_wrist[0] > from_wrist[1] > from_wrist[2] > from_wrist[3] * 0.95
        straight_angle = self._joint_angle(mcp, pip, tip)
        return joint_chain and wrist_chain and straight_angle > 150 and tip_reach > pip_reach * 1.28

    def _is_thumb_extended(self, hand) -> bool:
        thumb = hand["thumb"]
        thumb_reach = self._distance(thumb["tip"], thumb["mcp"])
        thumb_straight = self._joint_angle(thumb["mcp"], thumb["ip"], thumb["tip"])
        return thumb_reach > hand["palm_width"] * 0.6 and thumb_straight > 140

    def _is_thumb_folded(self, hand) -> bool:
        thumb = hand["thumb"]
        return (
            self._distance(thumb["tip"], hand["center"]) < hand["palm_width"] * 1.1
            and self._joint_angle(thumb["mcp"], thumb["ip"], thumb["tip"]) < 150
        )

    def _is_thumb_available_for_love(self, hand) -> bool:
        thumb = hand["thumb"]
        thumb_reach = self._distance(thumb["tip"], hand["center"])
        thumb_angle = self._joint_angle(thumb["mcp"], thumb["ip"], thumb["tip"])
        return thumb_reach > hand["palm_width"] * 0.45 and thumb_angle > 115

    def _love_pose_details(self, hands) -> dict[str, Any]:
        if len(hands) < 2:
            return {
                "thumbGap": None,
                "thumbTipsClose": False,
                "thumbsBelowOtherTips": False,
                "pinkyCurled": False,
            }

        first, second = hands[:2]
        hand_scale = max((first["palm_width"] + second["palm_width"]) / 2, 1.0)
        thumb_gap = self._distance(first["thumb"]["tip"], second["thumb"]["tip"]) / hand_scale
        thumbs_below_other_tips = (
            self._thumb_below_other_fingertips(first)
            and self._thumb_below_other_fingertips(second)
        )
        pinky_curled = (
            self._is_pinky_curled_for_love(first)
            and self._is_pinky_curled_for_love(second)
        )

        return {
            "thumbGap": round(thumb_gap, 3),
            "thumbTipsClose": thumb_gap < 0.9,
            "thumbsBelowOtherTips": thumbs_below_other_tips,
            "pinkyCurled": pinky_curled,
        }

    def _thumb_below_other_fingertips(self, hand) -> bool:
        thumb_tip_y = hand["thumb"]["tip"][1]
        other_tip_ys = [
            hand["index"]["tip"][1],
            hand["middle"]["tip"][1],
            hand["ring"]["tip"][1],
            hand["pinky"]["tip"][1],
        ]
        return all(thumb_tip_y > tip_y for tip_y in other_tip_ys)

    def _is_pinky_curled_for_love(self, hand) -> bool:
        return hand["pinky"]["tip"][1] > hand["pinky"]["pip"][1]

    def _are_non_thumb_fingers_folded(self, hand) -> bool:
        finger_states = self._finger_states(hand)
        return not any(finger_states[finger] for finger in ("index", "middle", "ring", "pinky"))

    @staticmethod
    def _joint_angle(
        first: tuple[float, float],
        joint: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        first_vector = (first[0] - joint[0], first[1] - joint[1])
        third_vector = (third[0] - joint[0], third[1] - joint[1])
        first_norm = math.hypot(*first_vector)
        third_norm = math.hypot(*third_vector)
        if first_norm == 0 or third_norm == 0:
            return 0.0
        dot_product = first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1]
        cosine = max(-1.0, min(1.0, dot_product / (first_norm * third_norm)))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.dist(first, second)

    @staticmethod
    def _midpoint(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
        return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)

    @staticmethod
    def _average_point(points: list[tuple[float, float]]) -> tuple[float, float]:
        xs, ys = zip(*points)
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @staticmethod
    def _serialize_point(point: tuple[float, float]) -> dict[str, float]:
        return {"x": round(point[0], 1), "y": round(point[1], 1)}
