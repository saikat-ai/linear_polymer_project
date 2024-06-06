##import important libraries

import pandas as pd
import numpy as np
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import contacts

u = mda.Universe('/data/carbon.gro','/data/all_carbon.xtc')
sel_basic = "name C*"
sel_acidic = "name C*"
acidic = u.select_atoms(sel_acidic)
basic = u.select_atoms(sel_basic)

import numpy as np
from MDAnalysis.analysis import distances

def contacts_within_cutoff(u, group_a, group_b, radius=5.0):
    timeseries = []
    for ts in u.trajectory:
        # Calculate distances between group_a and group_b
        dist = distances.distance_array(group_a.positions, group_b.positions)

        # Remove diagonal entries to avoid self-contacts
        np.fill_diagonal(dist, np.inf)

        # Determine which distances <= radius
        contact_matrix = np.where(dist <= radius, 1, 0)

        # Remove lower triangle entries to avoid overcounting
        contact_matrix = np.triu(contact_matrix)

        # Calculate the number of contacts
        n_contacts = int(contact_matrix.sum())

        timeseries.append([ts.frame, n_contacts])
        #print(contact_matrix)
    return np.array(timeseries)

ca = contacts_within_cutoff(u, acidic, basic, radius=5.0)
ca_df = pd.DataFrame(ca, columns=['Frame',
                                  'Contacts'])
ca_df1=ca_df.iloc[:,[1]]
ca_df1.to_csv('total number of contacts.csv')
