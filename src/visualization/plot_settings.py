import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(rc={'figure.figsize':(6,5), 'figure.dpi':120})
sns.set_context("notebook", font_scale=1)
sns.set_style("ticks")
sns.set_palette("Set2")
# increase x and y AXIS width not tick width
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5