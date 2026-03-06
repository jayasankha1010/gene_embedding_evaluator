# from huggingface_hub import snapshot_download

# # This pulls the entire 'gene_classification' folder without downloading the massive 30M dataset
# snapshot_download(
#     repo_id="ctheodoris/Genecorpus-30M",
#     repo_type="dataset",
#     allow_patterns="example_input_files/gene_classification/*",
#     local_dir="./gene_labels"
# )

import os
import pickle
import pandas as pd
import urllib.request

def convert_pickle_to_csv():
    # 1. Download the Long/Short Range TF CSV from Nature Communications
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16106-x/MediaObjects/41467_2020_16106_MOESM4_ESM.csv"
    print("Downloading 41467_2020_16106_MOESM4_ESM.csv...")
    try:
        # Save to the current directory
        urllib.request.urlretrieve(url, "41467_2020_16106_MOESM4_ESM.csv")
        print(" -> Success!\n")
    except Exception as e:
        print(f" -> Failed to download TF file: {e}\n")

    # 2. Find and convert all .pickle files in the downloaded Hugging Face folders
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pickle"):
                pkl_path = os.path.join(root, file)
                print(f"Reading {file}...")
                
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Geneformer target dicts are usually structured as {gene_id: label}
                if isinstance(data, dict):
                    # Convert the dictionary to a pandas DataFrame and save as CSV
                    df = pd.DataFrame(list(data.items()), columns=['gene_id', 'label'])
                    
                    # Save it in the current working directory for easy access
                    csv_filename = file.replace('.pickle', '.csv')
                    df.to_csv(csv_filename, index=False)
                    
                    print(f" -> Converted and saved as {csv_filename}")
                    print(f" -> Data Preview:\n{df.head(3)}\n")
                else:
                    print(f" -> Unrecognized pickle format. Skipping.\n")

if __name__ == "__main__":
    convert_pickle_to_csv()