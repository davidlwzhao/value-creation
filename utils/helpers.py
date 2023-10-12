import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import math

def percent_stacked_plot_overall(df, target_name):
    # create a figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # proportion of observation of each class
    prop_response = df[target_name].value_counts(normalize=True)

    # create a bar plot showing the percentage of churn
    prop_response.plot(kind='bar',
                       ax=ax,
                       color=['springgreen', 'salmon'])

    # set title and labels
    ax.set_title('Proportion of observations of the response variable',
                 fontsize=18, loc='left')
    ax.set_xlabel('churn',
                  fontsize=14)
    ax.set_ylabel('proportion of observations',
                  fontsize=14)
    ax.tick_params(rotation='auto')

    # eliminate the frame from the plot
    spine_names = ('top', 'right', 'bottom', 'left')
    for spine_name in spine_names:
        ax.spines[spine_name].set_visible(False)



def histogram_plots(df, target_name, columns_to_plot, super_title, responses=('No', 'Yes')):
    '''
    Prints a histogram for each independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''

    # set number of rows and number of columns
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    fig.suptitle(super_title, fontsize=22, y=.95)

    # loop to each demographic column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # histograms for each class (normalized histogram)
        df[df[target_name] == responses[0]][column].plot(kind='hist', ax=ax, density=True,
                                                         alpha=0.5, color='springgreen', label='No')
        df[df[target_name] == responses[1]][column].plot(kind='hist', ax=ax, density=True,
                                                         alpha=0.5, color='salmon', label='Yes')

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# function that computes the mutual infomation score between a categorical serie and the column Churn
def mutual_information_plot(df, target_name):
    from sklearn.metrics import mutual_info_score
    # select categorial variables excluding the response variable
    categorical_variables = df.select_dtypes(include=object).drop(target_name, axis=1)

    # compute the mutual information score between each categorical variable and the target
    feature_importance = categorical_variables.apply(lambda x: mutual_info_score(x, df[target_name])).sort_values(
        ascending=False)

    # visualize feature importance
    feature_importance.plot(kind='barh', title='Feature Importance')


def corr_matrix(df):
    corr = df.corr('pearson')
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    return (corr
     .style
     .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
     #.highlight_null()  # Color NaNs grey
     .format(precision=2))


def pairplot(df, target_name, sample=0.01):
    Num_feats = df.select_dtypes(include=[np.number]).copy()

    sample_for_pair_plot = Num_feats.groupby(target_name, group_keys=False).apply(
        lambda x: x.sample(frac=sample)
    )

    sns.pairplot(
        sample_for_pair_plot,
        hue=target_name,
        kind="scatter",
        diag_kind="kde",
        palette=sns.color_palette(["#e42256", "#00b1b0"]),
        height=1.5,
        aspect=1,
        plot_kws=dict(s=10),
    )
    plt.show()


def counts_plot(df, y_var, col="w", ax=None):
    y_var_counts = (
        df.loc[:, y_var]
        .value_counts()
        .reset_index()
        .rename(columns={"index": y_var, y_var: "counts"})
        .assign(
            percent=lambda df_: (df_["counts"] / df_["counts"].sum()).round(2) * 100
        )
    )
    sns.set_context("paper")
    ax0 = sns.barplot(
        data=y_var_counts,
        x="percent",
        y=y_var,
        color=col,
        ax=ax,
        order=y_var_counts[y_var],
    )
    values1 = ax0.containers[0].datavalues
    labels = ["{:g}%".format(val) for val in values1]
    ax0.bar_label(ax0.containers[0], labels=labels, fontsize=9, color="#740405")
    ax0.set_ylabel("")
    ax0.set_xlabel("Percent", fontsize=10)
    ax0.set_title(str.title(y_var) + " | proportions ", fontsize=10)
    return

def num_distributions(df, target_name, var_1, var_2):
    age_dur = df[[var_1, var_2, target_name]]
    target_color = sns.color_palette(["#e42256", "#00b1b0"])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))

    ## HistPlot1

    sns.histplot(
        stat='percent',
        data=age_dur,
        kde=True,
        line_kws={"lw": 1.5, "alpha": 0.6},
        common_norm=False,
        x=var_1,
        bins=20,
        hue=target_name,
        palette=target_color,
        alpha=0.6,
        ax=ax1,
    )
    ax1.legend(
        title="Subscribed?",
        loc="upper right",
        labels=["YES", "NO"],
        ncol=2,
        frameon=True,
        shadow=True,
        title_fontsize=8,
        prop={"size": 7},
        bbox_to_anchor=(1.18, 1.25),
    )
    ax1.set_xlabel(str.title(var_1), fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title(str.title(var_1) + " distributions", fontsize=12)
    ax1.yaxis.set_major_formatter(ticker.EngFormatter())

    ## Scatter plot

    sns.scatterplot(
        data=age_dur,
        x=var_1,
        y=var_2,
        hue=target_name,
        ax=ax2,
        palette=target_color,
        legend=False,
        alpha=0.6,
    )
    ax2.yaxis.set_major_formatter(ticker.EngFormatter())
    ax2.set_title(str.title(var_2) + " distributions", fontsize=12)
    ax2.set_ylabel(str.title(var_2), fontsize=10)
    ax2.set_xlabel(str.title(var_1), fontsize=10)

    ## HistPlot3

    sns.histplot(
        stat='percent',
        data=age_dur,
        kde=True,
        line_kws={"lw": 1.5, "alpha": 0.6},
        common_norm=False,
        x=var_2,
        bins=20,
        hue=target_name,
        palette=target_color,
        alpha=0.6,
        ax=ax3,
    )
    ax3.legend(
        title="Subscribed?",
        loc="upper right",
        labels=["YES", "NO"],
        ncol=2,
        frameon=True,
        shadow=True,
        title_fontsize=8,
        prop={"size": 7},
        bbox_to_anchor=(1.18, 1.25),
    )
    ax3.set_xlabel(str.title(var_2), fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title(str.title(var_2) + " distributions", fontsize=12)
    ax3.yaxis.set_major_formatter(ticker.EngFormatter())
    return