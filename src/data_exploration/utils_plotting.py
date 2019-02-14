import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import plotly.plotly as py
import plotly.graph_objs as go


import plotly


def init_plotly():
    pass


def prepare_plot_dict(df, col, main_count):
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col]==1, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values()))/ main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0
    return plot_dict


def make_count_plot(df, x, hue='AdoptionSpeed', tittle='', ):
    main_count = df["AdoptionSpeed"].value_counts(normalize=True).sort_index()
    print('Main adoption speed count', main_count)

    g = sns.countplot(x=x, data=df, hue=hue)
#    plt.title(f'Adoption speed {title}')
    plt.title('Adoption speed {0}'.format(tittle))
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    plt.show()



def adoption_trends_plot(df, feature):
    if not adoption_trends_plot.plotly_init:
        print('Initing fucking plotly...')
        init_plotly()
        adoption_trends_plot.plotly_init = True
    data = []
    for i in range(0, 5):
        adopt_speed = df.loc[df['AdoptionSpeed'] == i]
        data.append(go.Scatter(
            x=adopt_speed[feature].value_counts().sort_index().index,
            y=adopt_speed[feature].value_counts().sort_index().values,
            name=str(i))
        )

    layout = go.Layout(dict(title='Adoption speed by {0}'.format(feature),
                            xaxis=dict(title=feature),
                            yaxis=dict(title='Counts'),
                            )
                       )
    py.iplot(dict(data=data, layout=layout), filename='basic-line')


adoption_trends_plot.plotly_init = False