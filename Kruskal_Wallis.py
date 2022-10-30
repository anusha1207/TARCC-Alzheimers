from scipy import stats
import matplotlib.pyplot as plt

def kruskal_wallis(input_dataset, output_dataset, p_value=0):
    input_columns = []
    p_values = []
    # Go through each column/feature to perform the Kruskal Wallis Test
    for col in input_dataset.columns:
        feature = input_dataset[col]
        # Calculate Kruskal Wallis test on the column/feature
        # Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
        result = stats.kruskal(list(feature), list(output_dataset))
        # reject null hypothesis if p <= p_value, else fail to reject null hypothesis and accept the column
        if result.pvalue > p_value:
            input_columns.append(col)
            p_values.append(result.pvalue)
    return (input_columns, p_values)

def kruskal_wallis_data_visualization(input_columns, column_p_values):
    # Graph a horizontal bar plot which clearly shows the statistically
    # significant columns and their corresponding p-values
    plt.barh(input_columns, column_p_values)
    plt.title("P-Values of Kruskal Wallis Statistically Significant Input Columns")
    plt.xlabel("p-value")
    plt.ylabel("Column")
    plt.show()
    return