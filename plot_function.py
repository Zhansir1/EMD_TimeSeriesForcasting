import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Microsoft Yahei'
plt.rcParams['axes.unicode_minus'] = False


def line_plot(
              fig_size: tuple,  # 图形大小
              line_args: list[dict],  # 图形参数列表 [{'x':value, 'y':value, 'label':value, 'color':value}]
              y_lim: tuple,  # y轴上下界
              y_label: str,  # y轴标签
              title: str,  # 图片标题
              x_ticks: dict = {},  # 图形x轴参数列表 {'point_range':value, 'point':value, 'rotation':value}
              box_on: bool = False
             ):

    plt.figure(figsize=fig_size, facecolor='white')

    for line_dict in line_args:
        plt.plot(
                 line_dict.get('x'),
                 line_dict.get('y'),
                 label=line_dict.get('label'),
                 color=line_dict.get('color')
                )

    if x_ticks != {}:
        plt.xticks(x_ticks.get('point_range'),
                   x_ticks.get('point'),
                   rotation=x_ticks.get('rotation', 45))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.box(on=box_on)
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
