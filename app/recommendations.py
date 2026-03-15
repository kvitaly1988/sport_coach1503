import numpy as np

GROUP_KEYS = {
    "hip": ["hip_left", "hip_right"],
    "knee": ["knee_left", "knee_right"],
    "ankle": ["ankle_left", "ankle_right"],
    "torso": ["torso"],
    "shoulder": ["shoulder_left", "shoulder_right"],
    "elbow": ["elbow_left", "elbow_right"],
}

GROUP_NAMES = {
    "hip": "таз/бедра",
    "knee": "колени",
    "ankle": "лодыжки",
    "torso": "корпус",
    "shoulder": "плечи",
    "elbow": "локти",
}

PHASE_NAMES = {"start": "начальная фаза", "mid": "средняя фаза", "end": "финальная фаза"}

def _phase_label(pos: float) -> str:
    if pos < 0.33:
        return "start"
    if pos < 0.66:
        return "mid"
    return "end"

def _avg(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return np.nan
    return float(np.mean(values))

def summarize_group_errors(angle_mae: dict, important_joints: dict):
    stats = []
    for group, keys in GROUP_KEYS.items():
        err = _avg([angle_mae.get(k, np.nan) for k in keys])
        if np.isfinite(err):
            w = float(important_joints.get(group, 1.0))
            stats.append({"group": group, "name": GROUP_NAMES[group], "err": err, "weight": w, "priority": err * w})
    stats.sort(key=lambda x: x["priority"], reverse=True)
    return stats

def worst_phase_by_group(user_angles: dict, ref_angles: dict, idx_user: list, idx_ref: list):
    result = {}
    if not idx_user or not idx_ref:
        return result
    T = len(idx_user)
    for group, keys in GROUP_KEYS.items():
        per_frame = []
        for i in range(T):
            vals = []
            for k in keys:
                if k not in user_angles or k not in ref_angles:
                    continue
                u = user_angles[k][idx_user][i]
                r = ref_angles[k][idx_ref][i]
                if np.isfinite(u) and np.isfinite(r):
                    vals.append(abs(float(u) - float(r)))
            per_frame.append(float(np.mean(vals)) if vals else np.nan)
        arr = np.asarray(per_frame, dtype=np.float32)
        if np.isfinite(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
            worst_i = int(np.argmax(arr))
            pos = worst_i / max(1, T - 1)
            result[group] = {"index": worst_i, "phase": _phase_label(pos), "phase_ru": PHASE_NAMES[_phase_label(pos)], "err": float(arr[worst_i])}
    return result

def generate_ai_recommendations(angle_mae: dict, cfg_el: dict, user_angles: dict, ref_angles: dict, idx_user: list, idx_ref: list):
    if not angle_mae:
        return []
    title = cfg_el.get("title", "упражнение")
    thresholds = cfg_el.get("tips_thresholds_deg", {"minor": 10, "major": 20})
    minor = float(thresholds.get("minor", 10))
    major = float(thresholds.get("major", 20))
    important = cfg_el.get("important_joints", {})

    stats = summarize_group_errors(angle_mae, important)
    phases = worst_phase_by_group(user_angles, ref_angles, idx_user, idx_ref)

    tips = []
    for item in stats:
        group = item["group"]
        err = item["err"]
        weight = item["weight"]
        phase_ru = phases.get(group, {}).get("phase_ru", "ключевая фаза")
        if err < minor:
            continue
        if group == "knee":
            if err >= major:
                tips.append(f"В упражнении «{title}» заметная ошибка в зоне **коленей** (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**. Сконцентрируйтесь на одинаковой глубине сгибания коленей и не выводите их из траектории эталона.")
            else:
                tips.append(f"Немного подправьте **угол коленей** в фазе: **{phase_ru}**. Текущее отклонение ≈ {err:.1f}°.")
        elif group == "hip":
            if err >= major:
                tips.append(f"Ключевая неточность — **таз/бедра** (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**. Попробуйте лучше контролировать посадку таза и положение бедер относительно корпуса.")
            else:
                tips.append(f"Чуть скорректируйте **положение таза/бедер** в фазе: **{phase_ru}**. Отклонение ≈ {err:.1f}°.")
        elif group == "torso":
            if err >= major:
                tips.append(f"**Корпус** отклоняется от эталона (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**. Старайтесь держать линию корпуса стабильнее и не заваливаться вперед или назад.")
            else:
                tips.append(f"Сделайте **корпус** более стабильным в фазе: **{phase_ru}**. Отклонение ≈ {err:.1f}°.")
        elif group == "ankle":
            tips.append(f"Есть отличие в работе **лодыжек/стопы** (≈ {err:.1f}°), сильнее в фазе: **{phase_ru}**. Проверьте опору стопы и распределение веса.")
        elif group == "shoulder":
            tips.append(f"Зона **плеч** отличается от эталона (≈ {err:.1f}°), особенно в фазе: **{phase_ru}**. Держите плечевой пояс более собранным и повторяйте траекторию эталона.")
        elif group == "elbow":
            tips.append(f"Есть ошибка в работе **локтей** (≈ {err:.1f}°), сильнее в фазе: **{phase_ru}**. Проконтролируйте момент сгибания/разгибания рук.")
    if not tips:
        tips.append("Техника близка к эталону. Основной резерв — улучшить синхронность и чистоту переходов между фазами.")
    return tips[:5]
