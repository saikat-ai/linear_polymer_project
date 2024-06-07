## import important libraries
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyemma
from pyemma.util.contexts import settings
## finding optimal CVs from TICA
tica = coor.tica(tsne_df1,dim=2, lag=15) 
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)
a_norm = np.linalg.norm(tica.eigenvectors)
a_normalized = tica.eigenvectors/a_norm
#print(f"a = {tica.eigenvectors}")
#print(f"L2 norm of a = {a_norm}")
print(f"normalized a = {a_normalized}") ## Contribution of OPs in TICA framework
