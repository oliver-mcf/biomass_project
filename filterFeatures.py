
# Filter Predictor Variables / Model Features

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def matrix(csv_file):
    df = pd.read_csv(csv_file)
    df = df.iloc[:, 4:]
    correlation_matrix = df.corr()
    mask = np.eye(len(correlation_matrix), dtype = bool)
    plt.figure(figsize = (24, 20))
    sns.heatmap(correlation_matrix, annot = True, fmt = ".1f", cmap = 'coolwarm', square = True, cbar_kws = {"shrink": .5}, mask = mask)
    for i in range(len(correlation_matrix)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill = True, color = 'black', lw = 0))
    plt.title('Correlation Matrix Heatmap')
    plt.show()



# Code #############################################################################################################
if __name__ == '__main__':

    csv_file = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv'
    matrix(csv_file)
