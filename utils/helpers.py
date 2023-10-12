import pandas as pd
import matplotlib.pyplot as plt
import math

def percent_stacked_plot_overall(target):
    # create a figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # proportion of observation of each class
    prop_response = target.value_counts(normalize=True)

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


def percentage_stacked_plot(df, target_name, columns_to_plot, super_title):
    '''
    Prints a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''

    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    fig.suptitle(super_title, fontsize=22, y=.95)

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(df[column], df[target_name]).apply(lambda x: x / x.sum() * 100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['springgreen', 'salmon'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=16, loc='left')

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