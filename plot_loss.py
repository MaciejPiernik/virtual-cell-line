import pandas as pd

pd.read_csv('model_losses_cellxgene_colon_20240827084220.txt').plot().get_figure().savefig('model_losses_cellxgene_colon_20240827084220.png')