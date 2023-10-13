import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import math


class EDAHelper:
    def __init__(self, df, target_name, no_is_good=False):
        """
        Assumes that target_name col will already be 1/0 transformed - 1 always taken as 'treatment'
        """
        self.data = df
        self.target_name = target_name
        self.numerical_cols = df.select_dtypes(np.number).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if no_is_good:
            self.color = sns.color_palette(["#e42256", "#00b1b0"])
        else:
            self.color = sns.color_palette(["#e42256", "#00b1b0"])

    def init_diagnostic(self, exclude=None):
        numerical_cols = self.numerical_cols
        cat_cols = self.cat_cols

        if exclude is not None:
            numerical_cols = [x for x in numerical_cols if x not in exclude]
            cat_cols = [x for x in cat_cols if x not in exclude]

        # class balance
        self.percent_stacked_plot_overall()  # FIX COLOURING HERE

        # loop through categorical dimensions
        for col in cat_cols:
            # instantiate subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
            fig.suptitle(f"{col.capitalize()} Counts & Cut with {self.target_name}", fontsize=12, y=.95)

            # counts
            self.counts_plot(col, ax=ax1, col="#35a79c") # NEED TO FIGURE OUT HOW TO MAKE THIS SHARED AXIS

            # 100% stacked
            self.percentage_stacked_plot(col, ax=ax2)

        # loop through dimensions
        for col in numerical_cols:
            if col == self.target_name:
                continue

            # instantiate subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
            fig.suptitle(f"{col.capitalize()} Distribution", fontsize=12, y=.95)

            # distribution
            self.numerical_dist(col, ax=ax1)

            # distribution (not-normalized)
            self. numerical_dist(col, normalize=False, ax=ax2)

            # box and whisker

        # correlations and multi-feature explorations
        self.pairplot()
        self.mutual_information_plot() # need to re-convert this into a category...
        # feature importance
        return self.corr_matrix()

    def percent_stacked_plot_overall(self):
        # create a figure
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111)

        # proportion of observation of each class
        prop_response = self.data[self.target_name].value_counts(normalize=True)

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

    def percentage_stacked_plot(self, column, ax=None):
        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(self.data[column], self.data[self.target_name]).apply(lambda x: x / x.sum() * 100, axis=1)

        prop_by_independent.plot(kind='barh', ax=ax, stacked=True,
                                 rot=0, color=self.color)

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title=self.target_name, fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=10, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

    def numerical_dist(self, col, ax=None, normalize=True):
        selected = self.data[[col, self.target_name]]

        ## HistPlot1
        sns.histplot(
            stat='percent',
            data=selected,
            kde=True,
            line_kws={"lw": 1.5, "alpha": 0.6},
            common_norm=not normalize,
            x=col,
            bins=20,
            hue=self.target_name,
            palette=self.color,
            alpha=0.6,
            ax=ax,
        )
        ax.legend(
            title=self.target_name,
            loc="upper right",
            labels=self.data[self.target_name].unique().tolist(),
            ncol=2,
            frameon=True,
            shadow=True,
            title_fontsize=8,
            prop={"size": 7},
            bbox_to_anchor=(1.18, 1.25),
        )
        ax.set_xlabel(str.title(col), fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(str.title(col) + " distributions", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

    # function that computes the mutual infomation score between a categorical serie and the column Churn
    def mutual_information_plot(self):
        from sklearn.metrics import mutual_info_score
        # select categorial variables excluding the response variable
        # categorical_variables = df.select_dtypes(include=object).drop(target_name, axis=1)

        # compute the mutual information score between each categorical variable and the target
        feature_importance = (self.data[self.cat_cols]
                                .apply(lambda x: mutual_info_score(x, self.data[self.target_name]))
                                .sort_values(
                                    ascending=False))

        # visualize feature importance
        feature_importance.plot(kind='barh', title='Feature Importance')

    def corr_matrix(self):
        corr = self.data.corr('pearson')
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr[mask] = np.nan
        return (corr
                .style
                .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
                # .highlight_null()  # Color NaNs grey
                .format(precision=2))

    def pairplot(self, sample=0.01):
        Num_feats = self.data[self.numerical_cols]

        sample_for_pair_plot = Num_feats.groupby(self.target_name, group_keys=False).apply(
            lambda x: x.sample(frac=sample)
        )

        sns.pairplot(
            sample_for_pair_plot,
            hue=self.target_name,
            kind="scatter",
            diag_kind="kde",
            palette=sns.color_palette(["#e42256", "#00b1b0"]),
            height=1.5,
            aspect=1,
            plot_kws=dict(s=10),
        )
        plt.show()

    def counts_plot(self, y_var, col="w", ax=None):
        y_var_counts = (
            self.data.loc[:, y_var]
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

if __name__ == '__main__':
    df = pd.read_csv("../data/raw/bank-full.csv", sep=';')
    cols_to_category = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    df[cols_to_category] = df[cols_to_category].astype("category")
    df["y"] = np.where(df["y"] == "no", 0, 1)
    target_name = 'y'
    outcome_str = 'uptake'
    no_is_good = True
    target_color = sns.color_palette(["#e42256", "#00b1b0"])

    helper = EDAHelper(df, target_name)
    helper.init_diagnostic()