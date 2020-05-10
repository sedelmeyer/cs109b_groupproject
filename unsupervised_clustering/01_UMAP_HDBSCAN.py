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

# ### Clustering

from umap import UMAP
import hdbscan

# ### ML

import sklearn

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

# ---

columns= ['PID', 'Project_Name', 'Description', 'Category', 'Borough',
       'Managing_Agency', 'Client_Agency', 'Phase_Start',
       'Current_Project_Years', 'Current_Project_Year', 'Design_Start',
       'Budget_Start', 'Schedule_Start', 'Final_Change_Date',
       'Final_Change_Years', 'Phase_End', 'Budget_End', 'Schedule_End',
       'Number_Changes', 'Duration_Start', 'Duration_End', 'Schedule_Change',
       'Budget_Change', 'Schedule_Change_Ratio', 'Budget_Change_Ratio',
       'Budget_Abs_Per_Error', 'Budget_Rel_Per_Error', 'Duration_End_Ratio',
       'Budget_End_Ratio', 'Duration_Ratio_Inv', 'Budget_Ratio_Inv']

# + tags=["parameters"]
use_entire_data = False
data_name = "../data/processed/NYC_capital_projects_3yr_train.csv" #"NYC_capital_projects_3yr"
embedding_file = "../data/processed/embeddings_uncased_L-2_H-128_A-2.csv"
# -

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

if use_entire_data:
    df = data_dict[data_name]
else:
    date_cols = [
    'Design_Start',
    'Final_Change_Date',
    'Schedule_Start',
    'Schedule_End',
]
    df = pd.read_csv(data_name, parse_dates=date_cols)[columns + ["Category_Old"]]
    df = df.drop(columns="Category").rename(columns={"Category_Old": "Category"})
df.head()

df.info()

# **Read BERT embedding of project descriptions**

embedding = pd.read_csv(embedding_file)
embedding
embedding_expanded = embedding["embedding"].str.split(",", expand=True)
embedding_expanded.columns = [f"description_embedding_original" for num in embedding_expanded.columns]
embedding = embedding.join(embedding_expanded)
embedding = embedding.drop_duplicates("PID")
embedding

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

# #### Step 1: Scaling numeric columns

df_to_transform = df.select_dtypes(["int64", "float64"]).drop(columns=["PID"])
transformer = sklearn.preprocessing.RobustScaler().fit(df_to_transform)
transformed_columns = pd.DataFrame(transformer.transform(df_to_transform), columns = df_to_transform.columns)
scaled_df = (df[df.columns.difference(transformed_columns.columns)]).join(transformed_columns)

# **Columns that need to be scaled**

df_to_transform.columns

# **Compare before and after scaling**

df.describe()

scaled_df.describe()

# #### Step 2: Dropping columns that are identifiers or unknown at inference time

identifier_columns = ["PID",  "Project_Name", "Description"]
drop_cols = ["Current_Project_Year",                          
            "Number_Changes",                          
            "Current_Project_Years",
            "Phase_End"]


# + jupyter={"outputs_hidden": false}
dummified = pd.get_dummies(scaled_df.drop(columns=identifier_columns + drop_cols))
dummified[identifier_columns] = scaled_df[identifier_columns]
dummified= dummified[identifier_columns + list(dummified.columns.difference(identifier_columns + drop_cols))]
dummified.head()
# -

dummified.info()

# #### Drop columns that are related to either budget or schedule that can cause data leakage

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


other_dropped_cols = [
    
]


#Number_Changes, Phase_End_3-Construction Procurement
# -

# **COLUMNS THAT WERE ONE-HOT ENCODED**

columns_before_dummified = scaled_df.drop(columns=identifier_columns + drop_cols + budget_cols + schedule_cols +["Schedule_End", "Schedule_Start", "Design_Start", "Final_Change_Date"]).columns
columns_before_dummified

# #### Step 3: Unify one-hot-encoded columns

# **Since we need to make sure that we can use the UMAP mapping function on test samples, and since our data is sparse, test and train might not have the same Categories/Boroughs. We need to take the superset of all possible one hot encoding columns and fill them with 0s**

# +
all_cols_all_datasets = set()
for year in [1,2,3,4]:
    new_cols = set(pd.get_dummies(data_dict[f"NYC_capital_projects_{year}yr"][columns_before_dummified]).columns)
    all_cols_all_datasets = all_cols_all_datasets | new_cols
    
all_cols_all_datasets
# -

dummified["PID"] = dummified["PID"].astype("category")

umap_df = dummified.select_dtypes([int, float, "uint8"]).drop(columns = budget_cols + schedule_cols)
umap_df.columns

# Columns that we will add into the dataframe to harmonize one hot encoding:

added_cols = all_cols_all_datasets - set(umap_df.columns)
added_cols = {col: 0 for col in added_cols}
added_cols

umap_df = umap_df.assign(**added_cols)

# #### Step 4: Try out a couple of UMAP configurations

# The mos

# +
cmap = sns.diverging_palette(10, 220, sep=80, as_cmap=True) #sns.cubehelix_palette(as_cmap=True)

for n_neighbor in [2,5,10,20,50,100, 200]:
    utils.draw_umap(umap_df, n_neighbors=n_neighbor, c=dummified['Budget_Change'], cmap=cmap, title = f"UMAP, neighbors={n_neighbor}, colored by {dummified['Budget_Change'].name}")

# +
cmap = sns.diverging_palette(10, 220, sep=80, as_cmap=True) #sns.cubehelix_palette(as_cmap=True)

for n_neighbor in [2,5,10,20,50,100, 200]:
    utils.draw_umap(umap_df, n_neighbors=n_neighbor, c=dummified['Budget_Change'], cmap=cmap, title = f"UMAP, neighbors={n_neighbor}, colored by {dummified['Budget_Change'].name}")
# -

dummified['Budget_Change'].plot.hist()

#remove outliers
for n_neighbor in [2,5,10,20,50,100, 200]:
    filter_cond = dummified['Budget_Change']< 20
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=dummified['Budget_Change'][filter_cond], title = f"UMAP, neighbors={n_neighbor}, colored by {dummified['Budget_Change'].name}")

#use plotly
for n_neighbor in [5,10,20,50]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Budget_Change'][filter_cond]
    size = scaled_df['Schedule_Change'][filter_cond].abs()# size can't be negative
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})


for n_neighbor in [5,10, 20, 50]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Schedule_Change'][filter_cond]
    size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
    px
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})

umap_df.columns[umap_df.columns.str.lower().str.contains("budget")]

len(umap_df.drop(columns="Budget_Start").loc[filter_cond,:].columns)

umap_df.columns

new_cols

umap_df

umap_df.drop(columns="Budget_Start").loc[filter_cond,:].columns

# +
#drop the high value ones to visualize better

for n_neighbor in [5,10]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Budget_Change'][filter_cond]
    size = scaled_df['Schedule_Change'][filter_cond].abs()# size can't be negative
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, labels={"color":color.name, "size": size.name})
                                
# -

#switch size and color
for n_neighbor in [5,10,20,50,100, 200]:
    filter_cond = dummified['Budget_Change'].notnull()#< 20
    color = dummified['Schedule_Change'][filter_cond]
    size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
    hover_name = scaled_df["PID"][filter_cond]
    symbol = scaled_df["Category"][filter_cond]
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =False, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})

for n_neighbor in [5,10,20,50,100, 200]:
    filter_cond = dummified['Budget_Change']< 20
    color = dummified['Schedule_Change'][filter_cond]
    size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
    hover_name = scaled_df["PID"][filter_cond]
    symbol = scaled_df["Category"][filter_cond]
    
    utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})


interesting_group = [812, 673, 813, 680, 817, 815, 816, 866, 864, 868]
df.query("PID in @interesting_group")

umap_df.columns

# +
filter_cond = dummified['Budget_Change']< 20
color = dummified['Schedule_Change'][filter_cond]
size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
hover_name = scaled_df["PID"][filter_cond]
symbol = scaled_df["Category"][filter_cond]

utils.draw_umap(umap_df.drop(columns=["Budget_Start", "Duration_Start"]).loc[filter_cond,:], n_neighbors=5, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"});


# +
filter_cond = dummified['Budget_Change']< 20
color = dummified['Schedule_Change'][filter_cond]
size = scaled_df['Budget_Change'][filter_cond].abs()# size can't be negative
hover_name = scaled_df["PID"][filter_cond]
symbol = scaled_df["Category"][filter_cond]

utils.draw_umap(umap_df.loc[filter_cond,:], n_neighbors=5, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"});

# -

# *NOTE: in graphs, optimize for being able to **stand alone**. When aggregating and putting things in presentations, you won't have to recreate and add code to each plot to make it understandable without the entire post around it. Will it be understandable without several paragraphs?*

# **Output embeddings**

# +
mapping_df_list =[]
mapper_dict= {}
mapper_dict["attributes"] = {}

for dim in [2,5,10]:
    mapping, mapper = utils.draw_umap(umap_df,n_neighbors=5,n_components=dim, plot=False)
    mapping_df_list.append(pd.DataFrame(mapping, columns=[f"attributes_embedding_{dim}D"]* dim))
    mapper_dict["attributes"][f"{dim}D"] = mapper
# -

final_embedding_df = pd.concat(mapping_df_list, axis=1)
final_embedding_df

# +
# dump(mapper_dict, open("../data/interim/attributes_UMAP_all_mappers.pkl", "wb"))
# -

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

viz_embedding_5, mapper = utils.draw_umap(umap_df, plot=False, n_neighbors=5, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})
viz_embedding_10, mapper = utils.draw_umap(umap_df, plot=False, n_neighbors=10, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})
viz_embedding_20, mapper = utils.draw_umap(umap_df, plot=False, n_neighbors=20, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"})


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

labels, clusterer= utils.cluster_hdbscan(viz_embedding_5, 5, [viz_embedding_5, viz_embedding_10, viz_embedding_20 ])

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Find clusters' characteristics
# -

clustering_X = umap_df.loc[labels[labels!= -1],:]
clustering_X["label"] = labels[labels!= -1]

from IPython.utils import io


# +
#long output and will crash your notebook

#min_size5_dict = utils.get_cluster_defining_features(umap_df.loc[labels[labels!= -1],:], labels[labels!= -1], "min_cluster_size5") 
# -

clustering_X.groupby("label").mean()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(clustering_X.groupby("label").mean().var().sort_values(ascending=False))

# +
#min_size5_dict.keys()

# +
#min_size5_dict['combined_cluster_info_mean_peaks_per_cluster']

# +
#min_size5_dict[0].keys()

# +
#min_size5_dict[0]["AUC_fig"]
# -

cluster_features = list(clustering_X.groupby("label").mean().var()[clustering_X.groupby("label").mean().var() != 0].index)
cluster_features

radar_plots_df = clustering_X[cluster_features + ["label"]].groupby("label").mean().reset_index().rename(columns={"label":"group"})
radar_plots_df

# +
test = clustering_X.copy()
test[budget_cols + ["Budget_Start"] + schedule_cols + ["Schedule_End"]] = scaled_df[budget_cols + ["Budget_Start"] + schedule_cols + ["Schedule_End"]]
test

dump(test, open("../data/interim/clustering_X.pkl", "wb"))

# +
test = clustering_X.copy()
test[budget_cols + ["Budget_Start"] + schedule_cols + ["Schedule_End"]] = scaled_df[budget_cols + ["Budget_Start"] + schedule_cols + ["Schedule_End"]]

test = test.melt(id_vars = test.columns.difference(budget_cols + ["Budget_Start"]), value_vars = budget_cols + ["Budget_Start"], var_name = "Budget_metric", value_name="Budget_metric_value")
test = test.melt(id_vars = test.columns.difference(schedule_cols + ["Schedule_End"]), value_vars = schedule_cols + ["Schedule_End"], var_name = "Schedule_metric", value_name= "Schedule_metric_value")
# -

px.histogram(test,x="Budget_metric_value",color = "label", facet_row = "Budget_metric", height=2000)

px.histogram(test,x="Schedule_metric_value",color = "label", facet_row = "Schedule_metric", height=2000)

clustering_X.label.value_counts()

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
my_dpi=50
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi + 40)

# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(radar_plots_df.index))

# Loop to plot
for row in range(0, len(radar_plots_df.index)):
    make_spider( radar_plots_df, row=row, title='group '+radar_plots_df['group'][row].astype("str"), color=my_palette(row))
    
plt.tight_layout()

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


# **Output clustering label**

np.unique(labels)

final_embedding_df["clustering_label"] = labels

final_embedding_df.shape

# ---

# + [markdown] nteract={"transient": {"deleting": false}}
# ### Clustering Embeddings
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

final_embedding_df["PID"] = df["PID"]

final_embedding_df = final_embedding_df.merge(embedding, on="PID", how="left")

final_embedding_df.isnull().sum().sum()

# + jupyter={"outputs_hidden": false}

for n_neighbor in [2,3, 4, 5,10,20,50,100, 200]:
    utils.draw_umap(embedding.drop(columns = ["PID", "embedding"]), n_neighbors=n_neighbor, title=f"{n_neighbor}")
# -

project_info = embedding.merge(data_dict['Capital_Projects_pid'], on="PID", how="left")
project_info

project_info.isnull().sum()

project_info.notnull().all(axis=1)

embedding

for n_neighbor in [3,10]:
    filter_cond = project_info.notnull().all(axis=1)#no filter condition # dummified['Budget_Change']< 20
    color = project_info["Category"][filter_cond]#project_info['Original_Duration'][filter_cond]
    size = project_info['Original_Budget'][filter_cond].abs()# size can't be negative
    hover_name = project_info["PID"][filter_cond]
    symbol = project_info["Category"][filter_cond]

    utils.draw_umap(embedding.reset_index()[filter_cond].drop(columns = ["PID", "embedding", "index"]), n_neighbors=n_neighbor, c=color, cmap=cmap, use_plotly =True, title = f"UMAP, neighbors={n_neighbor}, colored by {color.name}", size=size, color_continuous_scale=px.colors.diverging.Portland_r, hover_name = hover_name, symbol = symbol, labels={"color":color.name, "size": size.name, "hover_name":"PID", "symbol":"Category"}, height=1000);


# +
description_mapping_df_list =[]

mapper_dict["description"] = {}

for dim in [2,5,10,50, 100]:
    mapping, mapper =  utils.draw_umap(embedding.drop(columns = ["PID", "embedding"]), n_neighbors=3, plot=False, n_components=dim)
    description_mapping_df_list.append(pd.DataFrame(mapping, columns=[f"description_embedding_{dim}D"]* dim))
    mapper_dict["description"][f"{dim}D"] = mapper
# -

embedding.shape

final_embedding = final_embedding_df.merge(pd.concat(description_mapping_df_list, axis=1).assign(PID= embedding["PID"].values), on= "PID", how="left")

final_embedding_df[final_embedding_df.isnull().any(axis=1)].PID

combined_embedding_df = (umap_df.assign(PID=df["PID"])).merge(embedding, on = "PID", how = "left").drop(columns=["PID", "embedding"])

for n_neighbor in [2,5, 5,10,20,50,100]:
    utils.draw_umap(combined_embedding_df, n_neighbor=n_neighbor, title=str(n_neighbor))

combined_embedding_df

# +
combined_mapping_df_list =[]
mapper_dict["combined"] = {}

for dim in [2,5,10,50, 100]:
    mapping, mapper =  utils.draw_umap(combined_embedding_df, n_neighbor=5, plot=False, n_components=dim) 
    combined_mapping_df_list.append(pd.DataFrame(mapping, columns=[f"description_embedding_{dim}D"]* dim))
    mapper_dict["combined"][f"{dim}D"] = mapper
# -

df_to_transform.columns

type(mapper_dict["combined"]["2D"])

columns

len(all_cols_all_datasets)

len(umap_df.columns.symmetric_difference(all_cols_all_datasets))

umap_df.columns.symmetric_difference(all_cols_all_datasets)

columns

df_to_transform.columns

columns_before_dummified

umap_df.columns

embedding.head()

columns_before_dummified

# +
from dataclasses import dataclass


class UMAP_embedder(object):
    def __init__(self):
        self.initial_columns = columns
        #self.initial_columns = ['PID', 'Project_Name', 'Description', 'Category', 'Borough', 'Managing_Agency', 'Client_Agency', 'Phase_Start', 'Current_Project_Years', 'Current_Project_Year', 'Design_Start', 'Budget_Start', 'Schedule_Start', 'Final_Change_Date', 'Final_Change_Years', 'Phase_End', 'Budget_End', 'Schedule_End', 'Number_Changes', 'Duration_Start', 'Duration_End', 'Schedule_Change', 'Budget_Change', 'Schedule_Change_Ratio', 'Budget_Change_Ratio', 'Budget_Abs_Per_Error', 'Budget_Rel_Per_Error', 'Duration_End_Ratio', 'Budget_End_Ratio', 'Duration_Ratio_Inv', 'Budget_Ratio_Inv']
        self.scale_cols = df_to_transform.columns
#         self.scale_cols = ['Current_Project_Years', 'Current_Project_Year', 'Budget_Start',
#        'Final_Change_Years', 'Budget_End', 'Number_Changes', 'Duration_Start',
#        'Duration_End', 'Schedule_Change', 'Budget_Change',
#        'Schedule_Change_Ratio', 'Budget_Change_Ratio', 'Budget_Abs_Per_Error',
#        'Budget_Rel_Per_Error', 'Duration_End_Ratio', 'Budget_End_Ratio',
#        'Duration_Ratio_Inv', 'Budget_Ratio_Inv']
        self.scaler = transformer
#         self.cols_to_dummify = ['Borough', 'Category', 'Client_Agency', 'Managing_Agency',
#        'Phase_Start', 'Budget_Start', 'Duration_Start'] 
        self.cols_to_dummify = columns_before_dummified
        self.final_cols = list(umap_df.columns)
        self.mapper_dict = mapper_dict
        self.clusterer = clusterer
        self.embedding = embedding.drop(columns="embedding").reset_index().drop(columns="index")
        
    def get_mapping_attributes(self,df, return_extra=False, ):
        """
        if return extra = True, returns 3 objects:
            0. mapping
            1. columns needed to be added to harmonize with entire data
            2. dummified df before adding columns of [1]
        """
        raw_df = df[test_class.initial_columns]
        df_to_transform = df[self.scale_cols]#.drop(columns=["PID"])
        transformed_columns = pd.DataFrame(self.scaler.transform(df_to_transform), columns = df_to_transform.columns)
        scaled_df = (df[df.columns.difference(transformed_columns.columns)]).join(transformed_columns)
        dummified = pd.get_dummies(scaled_df[self.cols_to_dummify])
        
        added_cols = all_cols_all_datasets - set(dummified.columns)
        added_cols = {col: 0 for col in added_cols}
        
        dummified_full = dummified.assign(**added_cols)
        dummified_full = dummified_full[self.final_cols]
        mapping_df_list =[]
        
        for mapper in mapper_dict["attributes"].values():
            
            mapping = mapper.transform(dummified_full)
            mapping_df = pd.DataFrame(mapping, columns= [f"umap_attributes_{mapping.shape[1]}D_embed_{col+1}" for col in range(mapping.shape[1])])
            mapping_df_list.append(mapping_df)
            
        final_df = pd.concat(mapping_df_list, axis=1)
        final_df["PID"] = scaled_df["PID"]
        
        if return_extra:
            return final_df, added_cols, scaled_df, dummified
        else:
            return final_df
       
    def get_mapping_description(self, df):
        
        merged = df[["PID"]].merge(self.embedding, on = "PID", how="left").drop(columns="PID")
#         print(df.shape)
#         print(merged.shape)
#         display(merged)
        mapping_df_list =[merged]
        #mapping_columns = [list(self.embedding.columns.copy())]
        for mapper in mapper_dict["description"].values():
            mapping = mapper.transform(merged)
            mapping_df = pd.DataFrame(mapping, columns= [f"umap_descr_{mapping.shape[1]}D_embed_{col+1}" for col in range(mapping.shape[1])])
            mapping_df_list.append(mapping_df)
           # mapping_columns += list(mapping_df.columns.copy())
                                   
        final_df = pd.concat(mapping_df_list, axis=1)
        #final_df.columns = mapping_columns
#         print(final_df.shape)
#         print(df["PID"].shape)
        final_df["PID"] = df["PID"].values
        
        return final_df
    
    def get_full_df(self, df):
        attribute_df = self.get_mapping_attributes(df)
        description_df = self.get_mapping_description(df)
        labels, probabilities = self.get_clustering(attribute_df[["umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2"]])
        full_df = description_df.merge(attribute_df, on = "PID", how="left")
        full_df["PID"] = attribute_df["PID"].values
        full_df["attribute_clustering_label"] = labels
        return full_df
    
    def get_clustering(self, attributes_2D_mapping):
        assert attributes_2D_mapping.shape[1] ==2
        new_labels = hdbscan.approximate_predict(clusterer, attributes_2D_mapping)
        return new_labels
        
test_class = UMAP_embedder()
test_class

# +
dump(transformer, open("../data/interim/UMAP_fitted_robust_scaler.pkl", "wb"))

dump(clusterer, open("../data/interim/UMAP_fitted_HDBSCAN.pkl", "wb"))

dump(umap_df.columns, open("../data/interim/UMAP_final_cols", "wb"))

dump(mapper_dict, open("../data/interim/UMAP_mapper_dict", "wb"))

dump(umap_df.columns, open("../data/interim/UMAP_final_cols", "wb"))

# +
# test_class.get_mapping_attributes(data_dict["NYC_capital_projects_3yr"])["PID"]

# test_class.get_mapping_attributes(pd.read_excel("../data/interim/NYC_capital_projects_3yr_test.xlsx").drop(columns="Category").rename(columns={"Category_Old": "Category"}))

# test_class.get_mapping_description(data_dict["NYC_capital_projects_3yr"])

# test_class.get_clustering((test_class.get_mapping_attributes(data_dict["NYC_capital_projects_3yr"])[["attributes_2D"]]))

# test_set_embeddings = test_class.get_full_df(pd.read_csv("../data/processed/NYC_capital_projects_3yr_test.csv").drop(columns="Category").rename(columns={"Category_Old": "Category"}))
# test_set_embeddings

# test_set_embeddings.to_csv("../data/interim/UMAP_embeddings_NYC_capital_projects_3yr_test.csv", index=False)

# train_set_embeddings = test_class.get_full_df(pd.read_csv("../data/processed/NYC_capital_projects_3yr_train.csv").drop(columns="Category").rename(columns={"Category_Old": "Category"}))

# train_set_embeddings[["umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2"]].plot.scatter("umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2")

# train_set_embeddings = test_class.get_full_df(pd.read_csv("../data/processed/NYC_capital_projects_3yr_train.csv").drop(columns="Category").rename(columns={"Category_Old": "Category"}))

# train_set_embeddings.to_csv("../data/processed/UMAP_embeddings_NYC_capital_projects_3yr_train.csv", index=False)

# +
# df_to_transform.info()

# pd.read_csv("../data/interim/all_years_df")[test_class.scale_cols].info()#.isnull().sum().

# test_class.get_mapping_attributes(pd.read_csv("../data/interim/all_years_df"))[["umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2"]].plot.scatter("umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2")

# UMAP_all_years_embedding = test_class.get_f(pd.read_csv("../data/interim/all_years_df"))

# UMAP_all_years_embedding.to_csv("../data/interim/UMAP_embeddings_all_years", index= False)

# pd.read_csv(f"../data/interim/NYC_capital_projects_{year}yr.csv").repace

# df_to_embed[~df_to_embed['Budget_Abs_Per_Error'].replace([np.inf, -np.inf], np.nan).isna()].dropna()

# df_to_embed.isnull().sum().sum()

# for year in [1,2,4]:
#     df_to_embed = pd.read_csv(f"../data/interim/NYC_capital_projects_{year}yr.csv")
#     print(df_to_embed.shape)
#     df_to_embed = df_to_embed[~df_to_embed['Budget_Abs_Per_Error'].replace([np.inf, -np.inf], np.nan).isna()].dropna()
#     df_to_embed.to_csv(f"../data/interim/fixed_inf_NYC_capital_projects_{year}yr.csv")
#     embedding_df_year = test_class.get_full_df(df_to_embed)
#     embedding_df_year.to_csv(f"../data/interim/UMAP_embeddings_NYC_capital_projects_{year}yr.csv")
# -


