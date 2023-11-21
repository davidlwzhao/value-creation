import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import math
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, mutual_info_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

from yellowbrick.classifier import ROCAUC, ClassificationReport, ClassPredictionError, PrecisionRecallCurve
from yellowbrick.model_selection import FeatureImportances

class ModelEvaluator:
    def __init__(self, clf, classes, X_train, X_test, y_train, y_test, model_name, voting=False):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.y_train_pred = clf.predict(X_train)
        self.y_test_pred = clf.predict(X_test)
        self.y_scores = clf.predict_proba(X_test)[:, 1]

        self.classes = classes
        self.results = clf.cv_results_
        self.model_step_name = model_name
        self.voting = voting

        #self.transformed_data = ppl[:-1].fit_transform(X_train, y_train)
        self.transformed_feats = clf.best_estimator_[:-1].get_feature_names_out()

    def init_eval(self):
        print("Best parameter (CV score=%0.3f):" % self.clf.best_score_)
        print(self.clf.best_params_)

        # confusion matrix
        self.plot_classification_report(support=True)
        #self.plot_confusion_matrix()

        # feature importance
        self.check_feature_importance()

        # ROC AUC
        self.plot_roc_auc()

        # look at precision vs. recall
        p, r, thresholds = precision_recall_curve(self.y_test, self.y_scores)
        self.plot_precision_recall_threshold(p, r, thresholds, 0.30)
        #self.plot_precision_recall()

    def plot_confusion_matrix(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            confusion_matrix(self.y_train, self.y_train_pred),
            annot=True,
            fmt="g",
            cbar=False,
            cmap="Greens",
            annot_kws={"size": 15},
            ax=ax1
        )
        ax1.set_title("Confusion matrix (Train set)", fontsize=16)
        ax1.tick_params(rotation=0)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")

        sns.heatmap(
            confusion_matrix(self.y_test, self.y_test_pred),
            annot=True,
            fmt="g",
            cbar=False,
            cmap="Reds",
            annot_kws={"size": 15},
            ax=ax2
        )
        ax2.set_title("Confusion matrix (Test set)", fontsize=16)
        ax2.tick_params(rotation=0)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        print("Train", classification_report(self.y_train, self.y_train_pred))
        print("Test", classification_report(self.y_test, self.y_test_pred))

    def plot_classification_report(self, support=True):
        cr = ClassificationReport(self.clf, classes=self.classes, support=support)
        cr.fit(self.X_train, self.y_train)
        cr.score(self.X_test, self.y_test)
        cr.show()

    def plot_roc_auc(self):
        cr = ROCAUC(self.clf, classes=self.classes)
        cr.fit(self.X_train, self.y_train)
        cr.score(self.X_test, self.y_test)
        cr.show()

    def plot_precision_recall(self):
        cr = PrecisionRecallCurve(self.clf)
        cr.fit(self.X_train, self.y_train)
        cr.score(self.X_test, self.y_test)
        cr.show()

    @staticmethod
    def adjusted_classes(y_scores, t):
        """
        This function adjusts class predictions based on the prediction threshold (t).
        Will only work for binary classification problems.
        """
        return [1 if y >= t else 0 for y in y_scores]

    def plot_precision_recall_threshold(self, p, r, thresholds, t=0.5):
        """
        plots the precision recall curve and shows the current value for each
        by identifying the classifier's threshold (t).
        """

        # generate new class predictions based on the adjusted_classes
        # function above and view the resulting confusion matrix.
        y_pred_adj = self.adjusted_classes(self.y_scores, t)
        print(pd.DataFrame(confusion_matrix(self.y_test, y_pred_adj),
                           columns=['pred_neg', 'pred_pos'],
                           index=['neg', 'pos']))

        # plot the curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Precision and Recall curve; ^ = current threshold", fontsize=14)

        # plot precision vs. recall
        ax1.step(r, p, color='b', alpha=0.2,
                 where='post')
        ax1.fill_between(r, p, step='post', alpha=0.2,
                         color='b')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')

        # plot the current threshold on the line
        for t_plot, color in zip([t, 0.5], ['c', 'k']):
            close_default_clf = np.argmin(np.abs(thresholds - t_plot))
            ax1.plot(r[close_default_clf], p[close_default_clf], '^', c=color,
                     markersize=15)
            ax1.annotate(f"t={t_plot}", (r[close_default_clf], p[close_default_clf]))
            ax2.axvline(x=t_plot, color=color)

        # plot both as a function of threshold
        ax2.plot(thresholds, p[:-1], "b--", label="Precision")
        ax2.plot(thresholds, r[:-1], "g-", label="Recall")
        ax2.set_ylabel("Score")
        ax2.set_xlabel("Decision Threshold")
        ax2.legend(loc='best')

        plt.show()

    def plot_param_tuning_scores(results, param_name, scoring):
        plt.figure(figsize=(13, 13))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

        plt.xlabel(param_name)
        plt.ylabel("Score")

        ax = plt.gca()
        # ax.set_xlim(0, 402)
        # ax.set_ylim(0.00, 1)

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results[f"param_{param_name}"].data, dtype=float)

        for scorer, color in zip(sorted(scoring), ["g", "k", "r", "b", "m", "c", "y"]):
            for sample, style in (("train", "--"), ("test", "-")):
                sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
                sample_score_std = results["std_%s_%s" % (sample, scorer)]
                ax.fill_between(
                    X_axis,
                    sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == "test" else 0,
                    color=color,
                )
                ax.plot(
                    X_axis,
                    sample_score_mean,
                    style,
                    color=color,
                    alpha=1 if sample == "test" else 0.7,
                    label="%s (%s)" % (scorer, sample),
                )

            best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
            best_score = results["mean_test_%s" % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot(
                [
                    X_axis[best_index],
                ]
                * 2,
                [0, best_score],
                linestyle="-.",
                color=color,
                marker="x",
                markeredgewidth=3,
                ms=8,
            )

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()

    def _get_importances(self):
        model = self.clf.best_estimator_.named_steps[self.model_step_name]
        # check if type of voting classifier
        if not self.voting:
            return model.feature_importances_

        imps = dict()
        # is voting classifier
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                imps[str(est)] = est.feature_importances_
            elif hasattr(est, 'coef_'):
                #imps[str(est)] = est.coef_[0] # need to scale this?
                continue
            else:
                print("no importance or coef found")
                continue

        return np.mean(list(imps.values()), axis=0)



    def check_feature_importance(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Feature Importances", fontsize=14, y=.95)

        # check feature importances
        importances = self._get_importances()
        (pd.DataFrame(importances, index=self.transformed_feats, columns=['Importances'])
         .sort_values('Importances')
         .plot(kind='barh', ax=ax1))
        ax1.set_title("Model Feature Importance")

        # permutation importance
        perm_importance_result_train = permutation_importance(self.clf, self.X_train, self.y_train, random_state=42)
        feat_name = self.X_train.columns

        indices = perm_importance_result_train['importances_mean'].argsort()
        ax2.barh(range(len(indices)),
                 perm_importance_result_train['importances_mean'][indices],
                 xerr=perm_importance_result_train['importances_std'][indices])
        ax2.set_yticks(range(len(indices)))
        ax2.set_title("Permutation importance")

        tmp = np.array(feat_name)
        _ = ax2.set_yticklabels(tmp[indices])
        plt.show()


class EDAHelper:
    def __init__(self, df, target_name):
        """
        Assumes that target_name col will already be 1/0 transformed - 1 always taken as 'treatment'
        Add cat count deltas
        ADD OTHER BARH- need exception if row doesn't exist in order??

        """
        self.data = df
        self.target_name = target_name
        self.numerical_cols = df.select_dtypes(np.number).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        self.color = sns.color_palette(["#00b1b0", "#e42256"])


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
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(f"{col.capitalize()} Counts & Cut with {self.target_name}", fontsize=12, y=.95)

            # counts
            order = self.counts_plot(col,
                                     ax=ax1,
                                     title=str.title(col) + " | Overall Proportions ",
                                     col="#2e5090")
            target_val = self.data[self.target_name].unique()

            # counts
            _ = self.counts_plot(col,
                                 order=order,
                                 hide_label=True,
                                 target_filter=target_val[1],
                                 ax=ax2,
                                 title=f"{self.target_name}={target_val[1]}",
                                 col=self.color[1])

            # counts
            _ = self.counts_plot(col,
                                 order=order,
                                 hide_label=True,
                                 delta=True,
                                 ax=ax3,
                                 title=f"{self.target_name}={target_val[1]} - {self.target_name}={target_val[0]}")

            # 100% stacked
            self.percentage_stacked_plot(col, order, ax=ax4)

        # loop through dimensions
        for col in numerical_cols:
            if col == self.target_name:
                continue

            # instantiate subplot
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(f"{col.capitalize()} Distribution", fontsize=12, y=.95)

            # distribution
            self.numerical_dist(col, title=f"{col.capitalize()} Normalized", ax=ax1)

            # distribution (not-normalized)
            self. numerical_dist(col, title=f"{col.capitalize()} Not Normalized", normalize=False, ax=ax2)
            #self.plot_scatter(col, title="", ax=ax2)

            # distribution (log transform)
            self.numerical_dist(col, title=f"{col.capitalize()} Log Transformed", log=True, normalize=False, ax=ax3)

            # box and whisker
            self.box_whisker(col, ax=ax4)

        # correlations and multi-feature explorations
        self.pairplot()
        self.mutual_information_plot() # need to re-convert this into a category...
        # feature importance
        return self.corr_matrix()

    # def plot_scatter(self, col, ax=None, title=None):
    #     sns.scatterplot(
    #         data=self.data,
    #         x=var_1,
    #         y=var_2,
    #         hue="y",
    #         ax=ax2,
    #         palette=target_color,
    #         legend=False,
    #         alpha=0.6,
    #     )
    #     ax2.yaxis.set_major_formatter(ticker.EngFormatter())
    #     ax2.set_title(str.title(var_2) + " distributions", fontsize=12)
    #     ax2.set_ylabel(str.title(var_2), fontsize=10)
    #     ax2.set_xlabel(str.title(var_1), fontsize=10)

    def percent_stacked_plot_overall(self):
        # create a figure
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111)

        # proportion of observation of each class
        prop_response = self.data[self.target_name].value_counts(normalize=True)

        # create a bar plot showing the percentage of churn
        prop_response.plot(kind='bar',
                           ax=ax,
                           color=self.color)

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

    def percentage_stacked_plot(self, column, order, ax=None):
        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(self.data[column], self.data[self.target_name]).apply(lambda x: x / x.sum() * 100, axis=1)

        prop_by_independent.loc[order.tolist()[::-1], :].plot(kind='barh', ax=ax, stacked=True,
                                 rot=0, color=self.color)

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title=self.target_name, fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=10, loc='left')

        # ax.tick_params(rotation='auto')
        ax.set(yticklabels=[])
        ax.set(ylabel=None)  # remove the y-axis label
        ax.tick_params(left=False)  # remove the ticks

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

    def numerical_dist(self, col, ax=None, log=False, title=None, normalize=True):
        selected = (self.data
        .loc[:, [col, self.target_name]]
        .assign(
            feature=(lambda x: np.log(x[col]) if log else x[col])
        ))

        ## HistPlot1
        sns.histplot(
            stat='percent',
            data=selected,
            kde=True,
            line_kws={"lw": 1.5, "alpha": 0.6},
            common_norm=not normalize,
            x='feature',
            bins=20,
            hue=self.target_name,
            palette=self.color,
            alpha=0.6,
            ax=ax,
        )
        ax.set_xlabel(str.title(col), fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

    def box_whisker(self, col, ax=None):
        selected = self.data[[col, self.target_name]]

        sns.boxplot(
            x=self.target_name,
            y=col,
            #hue='kind',
            data=selected,
            palette=self.color,
            ax=ax)

        ax.set_title(str.title(col) + " distributions", fontsize=12)

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

    def counts_plot(self, y_var, order=None, title=None, delta=False, hide_label=False, target_filter=None, col="w", ax=None):
        if target_filter is not None:
            row_mask = self.data[self.target_name] == target_filter
        else:
            row_mask = np.repeat(True, self.data.shape[0])

        y_var_counts = (
            self.data.loc[row_mask, y_var]
            .value_counts()
            .reset_index()
            .rename(columns={"index": y_var, y_var: "counts"})
            .assign(
                percent=lambda df_: (df_["counts"] / df_["counts"].sum()).round(2) * 100
            )
        )

        if order is None:
            order = y_var_counts[y_var]

        if delta:
            y_var_counts = (
                pd.crosstab(self.data[y_var], self.data[self.target_name])
                .apply(lambda x: x/x.sum() *100, axis=0)
                .assign(
                    percent=lambda df_: (df_[1] - df_[0]).round(2),
                    col=lambda df_: np.where(df_['percent'] <= 0, 'r', 'g')
                )
                .reset_index()
            )
            col = y_var_counts.set_index(y_var).loc[order,'col'].values.tolist()
            sns.set_context("paper")
            ax0 = sns.barplot(
                data=y_var_counts,
                x="percent",
                y=y_var,
                palette=col,
                ax=ax,
                order=order,
            )
        else:
            sns.set_context("paper")
            ax0 = sns.barplot(
                data=y_var_counts,
                x="percent",
                y=y_var,
                color=col,
                ax=ax,
                order=order,
            )

        values1 = y_var_counts.set_index(y_var).loc[order, "percent"].values
        labels = ["{:g}%".format(val) for val in values1]
        ax0.bar_label(ax0.containers[0], labels=labels, fontsize=9, color="#740405")

        if hide_label:
            ax0.set(yticklabels=[])
            ax0.set(ylabel=None)  # remove the y-axis label
            ax0.tick_params(left=False)  # remove the ticks

        ax0.set_xlabel("Percent", fontsize=10)
        ax0.set(title=title)
        return y_var_counts[y_var]


class OutlierHelper:
    def __init__(self, df, strategy):
        self.data = df.copy()
        self.strategy = strategy

        self.OUTLIER_OVER = 'over'
        self.OUTLIER_UNDER = 'under'
        self.OUTLIER_NOT = 'included'
        self.outlier_labels = [self.OUTLIER_UNDER, self.OUTLIER_NOT , self.OUTLIER_OVER]

        self.score_implementations = ['percentile', 'iqr', 'zscore']

    def flag_outliers(self):
        df_num_only = self.data.select_dtypes(np.number)
        iso = IsolationForest(contamination=0.1)
        outliers = iso.fit_predict(df_num_only)
        return outliers

    def score_outlier(self, series, kind='percentile', threshold=(2, 98)):
        '''
        Returns bool mask indicating where outliers are
        '''

        if kind not in self.score_implementations:
            return series == series

        if kind == self.score_implementations[0]:
            lower_limit, upper_limit = np.percentile(a=series, q=threshold)

        elif kind == self.score_implementations[1]:
            Q1, Q3 = np.percentile(a=series, q=[25, 75])
            IQR = Q3 - Q1
            upper_limit = Q3 + (1.5 * IQR)
            lower_limit = Q1 - (1.5 * IQR)

        elif kind == self.score_implementations[2]:
            mean = np.mean(series)
            std = np.std(series)

            upper_limit = mean + std * 3
            lower_limit = mean - std * 3

        return pd.cut(series,
                      [-np.inf, lower_limit, upper_limit, np.inf],
                      labels=self.outlier_labels,
                      retbins=True)

    def cap_outlier(self, series, kind='percentile', threshold=(2, 98)):
        index, bins = self.score_outlier(series, kind, threshold)
        _, lower_value, upper_value, _ = bins

        treated = np.where(index == self.OUTLIER_OVER, upper_value, series)
        treated = np.where(index == self.OUTLIER_UNDER, lower_value, treated)

        return treated

    def treat_outliers(self):
        '''
        Strategy is a series of actions to do ie

        Multi-column - isolation forest train/ flag
        Then single column additional treatment
        '''
        self.data['outlier'] = False
        counter = 0

        if self.strategy['iso']:
            # flagbased on iso forest
            self.data['iso_filter'] = np.where(self.flag_outliers() == -1, True, False)
            self.data['outlier'] = self.data['iso_filter']
            counter += self.data.outlier.sum()
            print(f"{counter} outliers identified by isolation forest")

        for col, strat, params in self.strategy['cols']:
            if strat == 'cap':
                self.data[f"{col}_capped"] = self.cap_outlier(self.data[col], **params)

            if strat == 'filter':
                self.data[f'{col}_filter'] = np.where(self.score_outlier(self.data[col], **params)[0] == self.OUTLIER_NOT, False, True)
                self.data['outlier'] = self.data['outlier'] | self.data[f'{col}_filter']
                new_count = self.data.outlier.sum() - counter
                print(f"{new_count} outliers identified in col={col} based on {params}")
                counter += new_count

        print(f"{counter} outliers in total identified")
        return self.data


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