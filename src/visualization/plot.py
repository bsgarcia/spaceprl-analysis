import seaborn as sns
import ptitprince as pt
import numpy as np
import matplotlib.pyplot as plt

def raincloud2(x, y, df, markersize=None, order=None, hue=None):
    f, ax = plt.subplots()
    ax=RainCloud(x =x, y = y, hue = hue, data = df, bw = .25, 
                 width_viol = .53, ax = ax, orient ='v' , alpha = .65, dodge = True, order=order, linewidth=0,
                 width_box=.34, point_size=markersize, point_border="white")
    return ax

def raincloud(x, y, markersize, df, order=None, hue=None):
    # adding color
    pal = sns.color_palette()
    if df is not None:
        means = df.groupby([x])[y].mean().reindex(order)
    else:
        if len(order) > 2:
            raise Exception('Too many unique x values for this custom plot')

        mean_1 = np.mean([i[0] for i in zip(y, x) if i[1] == order[0]])
        mean_2 = np.mean([i[0] for i in zip(y, x) if i[1] == order[1]])
        means = [mean_1, mean_2]

    dodge = None
    if hue:
        means = df.groupby([x, hue])[y].mean().reindex(order)
        dodge = True
        
    sns.scatterplot(
        x=means.index,
        y=means,
        color='white',
        edgecolor='black',
        s=150/2, linewidth=1.5, zorder=4)

    ax = pt.half_violinplot(x=x, y=y, data=df, palette=pal, bw=.25, cut=0., linewidth=0,
                            scale="area", width=.7, inner=None, orient='v', zorder=1, order=order, hue=hue, dodge=dodge)
    # pt.RainCloud(x=x, y=y, data=df, palette=pal, bw=.25, width_viol=.6, orient='v', zorder=2, order=order, hue=hue, dodge=dodge)

    ax = sns.stripplot(x=x, y=y, data=df, palette=pal, edgecolor="white",  linewidth=1, order=order,
                       size=markersize, orient='v', zorder=2, jitter=1, alpha=0.6, hue=hue, dodge=dodge)

    ax2 = sns.pointplot(x=x, y=y, data=df, color='black',  join=False, errorbar='se', linewidth=.8,
     edgecolor='black', capsize=.08, zorder=3, order=order, hue=hue, dodge=dodge)
    

def RainCloud(x = None, y = None, hue = None, data = None,
              order = None, hue_order = None,
              orient = "v", width_viol = .7, width_box = .15,
              palette = "Set2", bw = .2, linewidth = 1, cut = 0.,
              scale = "area", jitter = 1, move = 0., offset = None,
              point_size = 3, ax = None, pointplot = False,
              alpha = None, dodge = False, linecolor = 'red', point_border='white', point_lw=1, **kwargs):

    '''Draw a Raincloud plot of measure `y` of different categories `x`. Here `x` and `y` different columns of the pandas dataframe `data`.

    A raincloud is made of:

        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)

    Main inputs:
        x           categorical data. Iterable, np.array, or dataframe column name if 'data' is specified
        y           measure data. Iterable, np.array, or dataframe column name if 'data' is specified
        hue         a second categorical data. Use it to obtain different clouds and rainpoints
        data        input pandas dataframe
        order       list, order of the categorical data
        hue_order   list, order of the hue
        orient      string, vertical if "v" (default), horizontal if "h"
        width_viol  float, width of the cloud
        width_box   float, width of the boxplot
        move        float, adjusts rain position to the x-axis (default value 0.)
        offset      float, adjusts cloud position to the x-axis

    kwargs can be passed to the [cloud (default), boxplot, rain/stripplot, pointplot]
    by preponing [cloud_, box_, rain_ point_] to the argument name.
    '''

    if orient == 'h':  # swap x and y
        x, y = y, x
    if ax is None:
        ax = plt.gca()
        # f, ax = plt.subplots(figsize = figsize) old version had this

    if offset is None:
        offset = max(width_box/1.8, .15) + .05
    n_plots = 3
    split = False
    boxcolor = "black"
    boxprops = {'facecolor': 'none', "zorder": 10}
    if hue is not None:
        split = True
        boxcolor = palette
        boxprops = {"zorder": 10}

    kwcloud = dict()
    kwbox   = dict(saturation = 1, whiskerprops = {'linewidth': 2, "zorder": 10})
    kwrain  = dict(zorder = 0) #edgecolor = "white")
    kwpoint = dict(capsize = 0., errwidth = 0., zorder = 20)
    for key, value in kwargs.items():
        if "cloud_" in key:
            kwcloud[key.replace("cloud_", "")] = value
        elif "box_" in key:
            kwbox[key.replace("box_", "")] = value
        elif "rain_" in key:
            kwrain[key.replace("rain_", "")] = value
        elif "point_" in key:
            kwpoint[key.replace("point_", "")] = value
        else:
            kwcloud[key] = value

    # Draw cloud/half-violin
    pt.half_violinplot(x = x, y = y, hue = hue, data = data,
                    order = order, hue_order = hue_order,
                    orient = orient, width = width_viol,
                    inner = None, palette = palette, bw = bw,  linewidth = linewidth,
                    cut = cut, scale = scale, split = split, offset = offset, ax = ax, **kwcloud)

    # Draw umberella/boxplot
    # sns.boxplot   (x = x, y = y, hue = hue, data = data, orient = orient, width = width_box,
                        #  order = order, hue_order = hue_order,
                        #  color = boxcolor, showcaps = True, boxprops = boxprops,
                        #  palette = palette, dodge = dodge, ax =ax, **kwbox)

    # Set alpha of the two
    if not alpha is None:
        _ = plt.setp(ax.collections + ax.artists, alpha = alpha)

    # Draw rain/stripplot
    ax =  pt.stripplot (x = x, y = y, hue = hue, data = data, orient = orient,
                    order = order, hue_order = hue_order, palette = palette, edgecolor = point_border, linewidth=point_lw,
                    move = move, size = point_size, jitter = jitter, dodge = dodge, alpha=alpha+alpha/4,
                    width = width_box, ax = ax, **kwrain)
    # Add pointplot over stripplot
   

    if pointplot:
        n_plots = 4
        if not hue is None:
            sns.pointplot(x = x, y = y, hue = hue, data = data,
                          orient = orient, order = order, hue_order = hue_order,
                          dodge = width_box/2., palette = palette, ax = ax, **kwpoint)
        else:
            sns.pointplot(x = x, y = y, hue = hue, data = data, color = linecolor,
                           orient = orient, order = order, hue_order = hue_order,
                           dodge = width_box/2., ax = ax, **kwpoint)

    # Prune the legend, add legend title
    if not hue is None:
        handles, labels = ax.get_legend_handles_labels()
        _ = plt.legend(handles[0:len(labels)//n_plots], labels[0:len(labels)//n_plots], \
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., \
                       title = str(hue))#, title_fontsize = 25)

    # Adjust the ylim to fit (if needed)
    if orient == "h":
        ylim = list(ax.get_ylim())
        ylim[-1]  -= (width_box + width_viol)/4.
        _ = ax.set_ylim(ylim)
    elif orient == "v":
        xlim = list(ax.get_xlim())
        xlim[-1]  -= (width_box + width_viol)/4.
        _ = ax.set_xlim(xlim)

    return ax
