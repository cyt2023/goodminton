"""Court-line choices shared by the stdlib server and vision workers."""

LINE_COLORS = {'white': '白色', 'dark_blue': '深蓝色', 'blue': '蓝色',
               'yellow': '黄色', 'red': '红色', 'black': '黑色',
               'green': '绿色', 'unknown': '不确定'}


def line_color(value=None):
    value = 'unknown' if value is None else value
    if value not in LINE_COLORS:
        raise ValueError('请选择有效的羽毛球场地线颜色')
    return value
