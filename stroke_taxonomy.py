"""Shared mainland Chinese vocabulary; preserve source labels separately."""
VERSION = 'mainland-v3'
LABELS = {
    'clear': '高远球', 'clear_flat': '平高球', 'lift': '挑球',
    'drop': '吊球', 'smash': '杀球', 'drive': '平抽球',
    'drive_rear_flat': '后场抽平', 'drive_rear_high': '后场抽高',
    'push': '推球', 'net_kill': '扑球', 'net': '放网',
    'net_spin': '搓球', 'net_cross': '勾对角',
    'transition': '过渡（未细分）',
    'transition_hard': '硬过渡', 'transition_soft': '软过渡',
    'defense_deep_net': '挡远网',
    'defense': '挡网', 'defense_rear': '接杀弹后场',
    'serve_short': '发网前球', 'serve_long': '发后场球',
    'serve': '发球', 'other': '其他击球',
    'no_stroke': '没有击球', 'unknown': '看不清 / 无法判断',
}
AUTHOR_MAP = {
    '長球': 'clear', '挑球': 'lift', '切球': 'drop', '殺球': 'smash',
    '平球': 'drive', '推球': 'push', '撲球': 'net_kill',
    '放小球': 'net', '勾球': 'net_cross', '擋小球': 'defense',
    '發短球': 'serve_short', '發長球': 'serve_long', '未知球種': 'unknown',
}
TRAINABLE = [k for k in LABELS if k not in ['unknown', 'other', 'serve']]
PRIMARY = ['clear', 'lift', 'push', 'net_kill', 'drop', 'smash', 'drive', 'net']
TERMS = [
    ('clear', '高远球', ['高远球', '高远']),
    ('clear_flat', '平高球', ['平高球', '平高']),
    ('lift', '挑球', ['挑球', '挑高球', '挑高']),
    ('drop', '吊球', ['吊球', '劈吊', '滑板吊']),
    ('smash', '杀球', ['杀球', '扣杀', '重杀', '点杀', '跳杀']),
    ('drive', '平抽球', ['平抽', '抽球', '抽挡']),
    ('drive_rear_flat', '后场抽平', ['后场抽平', '后场平抽']),
    ('drive_rear_high', '后场抽高', ['后场抽高', '后场高抽']),
    ('push', '推球', ['推球']),
    ('net_kill', '扑球', ['扑球', '扑杀']),
    ('net', '放网', ['放网']),
    ('net_spin', '搓球', ['搓球', '搓网']),
    ('net_cross', '勾对角', ['勾对角', '勾球']),
    ('transition', '过渡', ['过渡球', '过渡']),
    ('transition_hard', '硬过渡', ['硬过渡']),
    ('transition_soft', '软过渡', ['软过渡']),
    ('defense_deep_net', '挡远网', ['挡远网', '接杀挡远网']),
    ('defense', '挡网', ['挡网', '接杀', '挡杀', '防守']),
    ('defense_rear', '接杀弹后场', ['接杀弹后场', '接杀弹到后场', '接杀弹球到后场']),
    ('serve_short', '发网前球', ['发网前球', '发小球', '发短球']),
    ('serve_long', '发后场球', ['发后场球', '发高球', '发高远球', '发平高球']),
    ('serve', '发球', ['发球']),
]


def normalise_event(event):
    value = dict(event)
    predictions = value.get('specialist_predictions') or {}
    ranked = predictions.get('bst_full') or predictions.get('bst0') or []
    if ranked:
        raw = ranked[0]['label']
        value['source_model_label'] = raw
        value['stroke_type'] = AUTHOR_MAP.get(raw.split('_', 1)[-1], 'unknown')
    value['stroke_label'] = LABELS.get(value.get('stroke_type'), '看不清 / 无法判断')
    value['taxonomy_version'] = VERSION
    return value


def query_matches(prediction, requested):
    children = {
        'serve': ['serve_short', 'serve_long'],
        'transition': ['transition_hard', 'transition_soft'],
        'drive': ['drive_rear_flat'],
    }
    return prediction in requested or any(
        prediction in children.get(parent, []) for parent in requested)
