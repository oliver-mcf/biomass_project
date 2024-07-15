
# Assess Available Training Data by Year and Site


# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def sample_year(file):       
    df = pd.read_csv(file)
    df = df.dropna()
    source_counts = df['Source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source}: {count}")


# Code #############################################################################################################
if __name__ == '__main__':

    # Determine command line arguments
    parser = argparse.ArgumentParser(description = 'Assess data stored in csv for filtered training data')
    parser.add_argument('--file', help = 'Filename of csv with directory, to return n footprints by unique value (site_year)')
    args = parser.parse_args()
    
    #file = '/home/s1949330/data/diss_data/pred_vars/input_merge/All_MODEL_INPUT_GEO.csv'
    sample_year(args.file)