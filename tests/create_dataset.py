from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer

import logging
from rdkit import rdBase
from io import StringIO

# import sys

# path = Path(__file__).parents[1]
# sys.path.append(str(path))

# from chempred.preprocessing import MissingValuesRemover


def get_bbbp_data():
    # read base file (raw)
    filepath = "/home/freddy/Downloads/BBBP.csv"
    bbbp = pd.read_csv(filepath)

    # Tell the RDKit's C++ backend to log to use the python logger:
    rdBase.LogToPythonLogger()
    logger = logging.getLogger("rdkit")
    # set the log level for the default log handler (the one which sense output
    # to the console/notebook):
    logger.handlers[0].setLevel(logging.WARN)

    # create a handler that uses the StringIO and set its log level:
    logger_sio = StringIO()
    handler = logging.StreamHandler(logger_sio)
    handler.setLevel(logging.INFO)
    # add the handler to the Python logger:
    logger.addHandler(handler)

    # create pipeline to check molecules
    pipe = make_pipeline(
        SmilesToMolTransformer(), MolecularDescriptorTransformer()
    )
    # Iterate over molecules
    smis = []
    perm = []
    messages = []
    for i, smi in enumerate(bbbp.smiles):
        try:
            # Process smiles
            desc = pipe.fit_transform([smi])
        except ValueError:
            continue
        # check there are not NaNs or infinite values in desc
        # also check no warning related to the structure
        if not np.isnan(desc).any() and not np.isinf(desc).any():
            smis.append(smi)
            perm.append(bbbp.p_np[i])
            # collect messages (possible warnings)
            text = logger_sio.getvalue()
            messages.append(text)

        # reset StringIO object
        logger_sio.truncate(0)
        logger_sio.seek(0)

    return pd.DataFrame(
        [(smi, p, m) for smi, p, m in zip(smis, perm, messages)],
        columns=["smiles", "perm", "Warnings"]
    )


data = get_bbbp_data()
# print(data.shape)
data = data.query("Warnings == ''").copy()
data.drop(columns="Warnings", inplace=True)
print(data.shape)
pathfile = Path(__file__).parent / "data" / "BBBP.csv"
data.to_csv(pathfile, index=False)
