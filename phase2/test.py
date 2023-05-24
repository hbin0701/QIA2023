##################################################################
import pandas as pd
import csv
from tqdm.auto import tqdm
import json

fieldnames = ['User_ID', 'I/E', 'S/N', 'T/F', 'J/P']
fname = "0522_final_test_phase2.csv"

test_df = pd.read_csv("/workspace/final_QIA/phase2/test.csv")
users = test_df.User_ID.unique()

li1 = json.load(open("/workspace/final_QIA/phase2/phase2_0.json"))
li2 = json.load(open("/workspace/final_QIA/phase2/phase2_1.json"))
li3 = json.load(open("/workspace/final_QIA/phase2/phase2_2.json"))
li4 = json.load(open("/workspace/final_QIA/phase2/phase2_3.json"))

# open the CSV file for writing
with open(f"/workspace/final_QIA/{fname}", 'w', newline='') as csvfile:
    # create a writer object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # write the header row
    writer.writeheader()

    for idx in tqdm(range(120)):
        
        # Majority voting
        ie = li1[idx]
        sn = li2[idx]
        tf = li3[idx]
        jp = li4[idx]

        writer.writerow({
            'User_ID': users[idx],
            'I/E': ie,
            'S/N': sn,
            'T/F': tf,
            'J/P': jp
        })