# ---
# title: This is a Knowledge Template Header
# authors:
# - sally_smarts 
# - wesley_wisdom
# tags:
# - knowledge
# - example
# created_at: 2016-06-29
# updated_at: 2016-06-30
# tldr: This is short description of the content and findings of the post.
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: cs109b-project
#     language: python
#     name: cs109b-project
# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ## Instructions:
#
# ## Todo:
#
# ### Inputs:
#
# ### Outputs:
#

# + [markdown] nteract={"transient": {"deleting": false}}
# <a name='index'></a>
#
# ## Notebook Index
#
# 1. <a href=#imports>Imports</a>
#
#
# 2. <a href=#read>Read Dataset</a>
#
#
# 3. <a href=#functions>Define data generator functions and default parameters</a>
#
#
# 4. <a href=#analyses>Analyses</a>

# +
from IPython.display import HTML, Image, IFrame

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').show();
 } else {
 $('div.input').hide();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# -

# ---

# <a name='imports'></a>
# ## Imports
# Imports for function used in this notebook.
#
# <a href=#index>index</a>

# ### Visualization

# +
import plotly.io as pio
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
pio.renderers.keys()
pio.renderers.default = 'jupyterlab'

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# Improve resolution of output graphcis
# %config InlineBackend.figure_format ='retina'
# -

# ### Data Wrangling

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from pandas_profiling import ProfileReport

# ### ML

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix, plot_confusion_matrix, classification_report
#import xgboost as xgb
import sklearn
import shap
shap.initjs()

# ### Clustering

from umap import UMAP
import hdbscan

# ### Utils

# +
# %load_ext autoreload
# %autoreload 1

#autoreload utils module to reload everytime it's modified
# %aimport utils
# -

from pickle import dump, load
from tqdm.auto import tqdm
import glob


# ---

# <a name='read'></a>
# ## Data Loading
# All the data loaded from disk and used in this notebook
#
# <a href=#index>index</a>

data_dict = {}
for file in sorted(glob.glob("../data/interim/*.csv")):
    file_name = file.split("/")[-1].split(".")[0]
    if file_name.startswith("NYC"):
        date_cols = [
    'Design_Start',
    'Final_Change_Date',
    'Schedule_Start',
    'Schedule_End',
]
    else:
        date_cols = []
    data_dict[file_name] = pd.read_csv(file, parse_dates=date_cols)
data_dict.keys()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for name, df in data_dict.items():
        print(name)
        print("Shape:",df.shape)
        display(df.head())
        print("-------")

# ---

df = data_dict["NYC_capital_projects_3yr"]
df.head()

df.info()

df_to_transform = df.select_dtypes(["int64", "float64"]).drop(columns=["PID"])
transformer = sklearn.preprocessing.RobustScaler().fit(df_to_transform)
transformed_columns = pd.DataFrame(transformer.transform(df_to_transform), columns = df_to_transform.columns)
scaled_df = (df[df.columns.difference(transformed_columns.columns)]).join(transformed_columns)

df.describe()

scaled_df.describe()

profile = ProfileReport(scaled_df)

profile.to_file("report_after_robust_scaling.html")

del profile

# + [markdown] nteract={"transient": {"deleting": false}}
# <a name='analyses'></a>
# ## Analyses
#
# <a href=#index>index</a>
# -

# #### TL,DR

# *NOTE: In the TL,DR, optimize for **clarity** and **comprehensiveness**. The goal is to convey the post with the least amount of friction, especially since ipython/beakers require much more scrolling than blog posts. Make the reader get a correct understanding of the post's takeaway, and the points supporting that takeaway without having to strain through paragraphs and tons of prose. Bullet points are great here, but are up to you. Try to avoid academic paper style abstracts.*
#
#  - Having a specific title will help avoid having someone browse posts and only finding vague, similar sounding titles
#  - Having an itemized, short, and clear tl,dr will help readers understand your content
#  - Setting the reader's context with a motivation section makes someone understand how to judge your choices
#  - Visualizations that can stand alone, via legends, labels, and captions are more understandable and powerful
#

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Dimensionality Reduction
# -

# #### Motivation

# *NOTE: optimize in this section for **context setting**, as specifically as you can. For instance, this post is generally a set of standards for work in the repo. The specific motivation is to have least friction to current workflow while being able to painlessly aggregate it later.*
#
# The knowledge repo was created to consolidate research work that is currently scattered in emails, blogposts, and presentations, so that people didn't redo their work.

# ### This Section Says Exactly This Takeaway

identifier_columns = ["PID",  "Project_Name", "Description"]

# + jupyter={"outputs_hidden": false}
dummified = pd.get_dummies(scaled_df.drop(columns=identifier_columns))
dummified[identifier_columns] = scaled_df[identifier_columns]
dummified= dummified[identifier_columns + list(dummified.columns.difference(identifier_columns))]
dummified.head()
# -

dummified.info()

# +
budget_cols = ['Budget_Abs_Per_Error',
 'Budget_Change',
 'Budget_Change_Ratio',
 'Budget_End',
 'Budget_End_Ratio',
 'Budget_Ratio_Inv',
 'Budget_Rel_Per_Error',
 #'Budget_Start',
]

schedule_cols = ['Schedule_Change',
                 'Schedule_Change_Ratio',
                 'Duration_End', 
                 'Duration_End_Ratio',
                 'Duration_Ratio_Inv',
                 'Final_Change_Years'
                 #'Schedule_End', #not int or float
                 #'Schedule_Start' #not int or float
                ]
# -

dummified["PID"] = dummified["PID"].astype("category")

umap_df = dummified.select_dtypes([int, float, "uint8"]).drop(columns = budget_cols + schedule_cols)
umap_df.columns

dummified['Budget_Change']

# +
cmap = sns.diverging_palette(10, 220, sep=80, as_cmap=True) #sns.cubehelix_palette(as_cmap=True)

for n_neighbor in [2,5,10,20,50,100, 200]:
    utils.draw_umap(umap_df, n_neighbors=n_neighbor, c=dummified['Budget_Change'], cmap=cmap, title = f"UMAP, neighbors=2, colored by {dummified['Budget_Change'].name}")
# -

dummified['Budget_Change'].plot.hist()

#remove outliers
for n_neighbor in [2,5,10,20,50,100, 200]:
    filter_cond = dummified['Budget_Change']< 20
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=dummified['Budget_Change'][filter_cond], title = f"UMAP, neighbors={n_neighbor}, colored by {dummified['Budget_Change'].name}")

scaled_df['Schedule_Change'][filter_cond]

#use plotly
for n_neighbor in [5,10]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Budget_Change'][filter_cond]
    size = scaled_df['Schedule_Change'][filter_cond].abs()# size can't be negative
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})


for n_neighbor in [5,10]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Schedule_Change'][filter_cond]
    size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})

umap_df.columns[umap_df.columns.str.lower().str.contains("budget")]

len(umap_df.drop(columns="Budget_Start").loc[filter_cond,:].columns)

umap_df.drop(columns="Budget_Start").loc[filter_cond,:].columns

# +
#drop budget_start

for n_neighbor in [5,10]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Budget_Change'][filter_cond]
    size = scaled_df['Schedule_Change'][filter_cond].abs()# size can't be negative
    
    utils.draw_umap(umap_df.drop(columns="Budget_Start").loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})
                                
# -

for n_neighbor in [5,10,20,50,100, 200]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Schedule_Change'][filter_cond]
    size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
    hover_name = scaled_df["PID"][filter_cond]
    symbol = scaled_df["Category"][filter_cond]
    
    utils.draw_umap(umap_df.drop(columns="Budget_Start").loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})


interesting_group = [812, 673, 813, 680, 817, 815, 816, 866, 864, 868]
df.query("PID in @interesting_group")

# +
filter_cond = dummified['Budget_Change']< 20
color = dummified['Schedule_Change'][filter_cond]
size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
hover_name = scaled_df["PID"][filter_cond]
symbol = scaled_df["Category"][filter_cond]

utils.draw_umap(umap_df.drop(columns="Budget_Start").loc[filter_cond,:], n_neighbors=5, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"});

# -

# *NOTE: in graphs, optimize for being able to **stand alone**. When aggregating and putting things in presentations, you won't have to recreate and add code to each plot to make it understandable without the entire post around it. Will it be understandable without several paragraphs?*

# ### Putting Big Bold Headers with Clear Takeaways Will Help Us Aggregate Later

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Clustering
# -

# #### TL,DR

# *NOTE: In the TL,DR, optimize for **clarity** and **comprehensiveness**. The goal is to convey the post with the least amount of friction, especially since ipython/beakers require much more scrolling than blog posts. Make the reader get a correct understanding of the post's takeaway, and the points supporting that takeaway without having to strain through paragraphs and tons of prose. Bullet points are great here, but are up to you. Try to avoid academic paper style abstracts.*
#
#  - Having a specific title will help avoid having someone browse posts and only finding vague, similar sounding titles
#  - Having an itemized, short, and clear tl,dr will help readers understand your content
#  - Setting the reader's context with a motivation section makes someone understand how to judge your choices
#  - Visualizations that can stand alone, via legends, labels, and captions are more understandable and powerful
#

# ### Motivation

# *NOTE: optimize in this section for **context setting**, as specifically as you can. For instance, this post is generally a set of standards for work in the repo. The specific motivation is to have least friction to current workflow while being able to painlessly aggregate it later.*
#
# The knowledge repo was created to consolidate research work that is currently scattered in emails, blogposts, and presentations, so that people didn't redo their work.

# #### Create different embeddings to visualize the clustering on

viz_embedding_5, mapper = utils.draw_umap(umap_df.drop(columns="Budget_Start"), plot=False, n_neighbors=5, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})
viz_embedding_10, mapper = utils.draw_umap(umap_df.drop(columns="Budget_Start"), plot=False, n_neighbors=10, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})
viz_embedding_20, mapper = utils.draw_umap(umap_df.drop(columns="Budget_Start"), plot=False, n_neighbors=20, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})


# #### Clustering on the raw data

for min_cluster_size in [2,5,10]:
    utils.cluster_hdbscan(umap_df.drop(columns="Budget_Start"), min_cluster_size, [viz_embedding_5, viz_embedding_10, viz_embedding_20 ])
    print("---------------")

# #### Clustering on output of UMAP 5 neighbors

for min_cluster_size in [2,5,10]:
    utils.cluster_hdbscan(viz_embedding_5, min_cluster_size, [viz_embedding_5, viz_embedding_10, viz_embedding_20 ])
    print("---------------")

# #### Clustering on UMAP output 10 neighbors:

# +

for min_cluster_size in [2,5,10]:
    utils.cluster_hdbscan(viz_embedding_10, min_cluster_size, [viz_embedding_5, viz_embedding_10, viz_embedding_20 ])
    print("---------------")
# -

# #### Final clustering

labels = utils.cluster_hdbscan(viz_embedding_5, 5, [viz_embedding_5, viz_embedding_10, viz_embedding_20 ])

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Find clusters' characteristics
# -

clustering_X = umap_df.drop(columns="Budget_Start").loc[labels[labels!= -1],:]
clustering_X["label"] = labels[labels!= -1]

from IPython.utils import io


# +
#long output and will crash your notebook

min_size5_dict = get_cluster_defining_features(umap_df.drop(columns="Budget_Start").loc[labels[labels!= -1],:], labels[labels!= -1], "min_cluster_size5") 
# -

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(clustering_X.groupby("label").mean().var().sort_values(ascending=False))

min_size5_dict.keys()

min_size5_dict['combined_cluster_info_mean_peaks_per_cluster']

min_size5_dict[0].keys()

min_size5_dict[0]["AUC_fig"]

cluster_features = list(clustering_X.groupby("label").mean().var()[clustering_X.groupby("label").mean().var() != 0].index)
cluster_features

radar_plots_df = clustering_X[cluster_features + ["label"]].groupby("label").mean().reset_index().rename(columns={"label":"group"})
radar_plots_df

# +
from math import pi
def make_spider(mean_peaks_per_cluster, row, title, color):
    # number of variable
    categories=list(mean_peaks_per_cluster)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(4,4,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    #plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    #plt.ylim(0,40)

    # Ind1
    scaled = mean_peaks_per_cluster.loc[row].drop('group').values
    values=mean_peaks_per_cluster.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)

    
# ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=70
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi + 50), dpi=my_dpi + 40)

# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(radar_plots_df.index))

# Loop to plot
for row in range(0, len(radar_plots_df.index)):
    make_spider( radar_plots_df, row=row, title='group '+radar_plots_df['group'][row].astype("str"), color=my_palette(row))

# +
# ------- PART 1: Create background
 
# number of variable
categories=list(radar_plots_df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
#plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
#plt.ylim(0,40)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable

# Ind1
values=radar_plots_df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 1")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=radar_plots_df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 2")
ax.fill(angles, values, 'r', alpha=0.1)

# Ind3
values=radar_plots_df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 3")
ax.fill(angles, values, 'g', alpha=0.1)

# Ind3
values=radar_plots_df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 4")
ax.fill(angles, values, 'o', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# -

# #### TL,DR

# *NOTE: In the TL,DR, optimize for **clarity** and **comprehensiveness**. The goal is to convey the post with the least amount of friction, especially since ipython/beakers require much more scrolling than blog posts. Make the reader get a correct understanding of the post's takeaway, and the points supporting that takeaway without having to strain through paragraphs and tons of prose. Bullet points are great here, but are up to you. Try to avoid academic paper style abstracts.*
#
#  - Having a specific title will help avoid having someone browse posts and only finding vague, similar sounding titles
#  - Having an itemized, short, and clear tl,dr will help readers understand your content
#  - Setting the reader's context with a motivation section makes someone understand how to judge your choices
#  - Visualizations that can stand alone, via legends, labels, and captions are more understandable and powerful
#

# ### Motivation

# *NOTE: optimize in this section for **context setting**, as specifically as you can. For instance, this post is generally a set of standards for work in the repo. The specific motivation is to have least friction to current workflow while being able to painlessly aggregate it later.*
#
# The knowledge repo was created to consolidate research work that is currently scattered in emails, blogposts, and presentations, so that people didn't redo their work.

# ### This Section Says Exactly This Takeaway

# + jupyter={"outputs_hidden": false}

# %matplotlib inline

x = np.linspace(0, 3*np.pi, 500)
plot_df = pd.DataFrame()
plot_df["x"] = x
plot_df["y"] = np.sin(x**2)


plot_df.plot('x', 'y', 
             color='lightblue',
             figsize=(15,10))
plt.title("Put enough labeling in your graph to be understood on its own", size=25)
plt.xlabel('you definitely need axis labels', size=20)
plt.ylabel('both of them', size=20)
# -

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Point 1
# -

# #### TL,DR

# *NOTE: In the TL,DR, optimize for **clarity** and **comprehensiveness**. The goal is to convey the post with the least amount of friction, especially since ipython/beakers require much more scrolling than blog posts. Make the reader get a correct understanding of the post's takeaway, and the points supporting that takeaway without having to strain through paragraphs and tons of prose. Bullet points are great here, but are up to you. Try to avoid academic paper style abstracts.*
#
#  - Having a specific title will help avoid having someone browse posts and only finding vague, similar sounding titles
#  - Having an itemized, short, and clear tl,dr will help readers understand your content
#  - Setting the reader's context with a motivation section makes someone understand how to judge your choices
#  - Visualizations that can stand alone, via legends, labels, and captions are more understandable and powerful
#

# ### Motivation

# *NOTE: optimize in this section for **context setting**, as specifically as you can. For instance, this post is generally a set of standards for work in the repo. The specific motivation is to have least friction to current workflow while being able to painlessly aggregate it later.*
#
# The knowledge repo was created to consolidate research work that is currently scattered in emails, blogposts, and presentations, so that people didn't redo their work.

# ### This Section Says Exactly This Takeaway

# + jupyter={"outputs_hidden": false}

# %matplotlib inline

x = np.linspace(0, 3*np.pi, 500)
plot_df = pd.DataFrame()
plot_df["x"] = x
plot_df["y"] = np.sin(x**2)


plot_df.plot('x', 'y', 
             color='lightblue',
             figsize=(15,10))
plt.title("Put enough labeling in your graph to be understood on its own", size=25)
plt.xlabel('you definitely need axis labels', size=20)
plt.ylabel('both of them', size=20)
# -

# *NOTE: in graphs, optimize for being able to **stand alone**. When aggregating and putting things in presentations, you won't have to recreate and add code to each plot to make it understandable without the entire post around it. Will it be understandable without several paragraphs?*

# ### Putting Big Bold Headers with Clear Takeaways Will Help Us Aggregate Later

# ### Appendix

# Put all the stuff here that is not necessary for supporting the points above. Good place for documentation without distraction.
