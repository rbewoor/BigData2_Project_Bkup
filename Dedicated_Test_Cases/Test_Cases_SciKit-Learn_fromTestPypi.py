## The environment has been setup by compiling from source from here:
##     https://test.pypi.org/project/scikit-learn-VAL-TestPypi/
## --------------------------------------------------------------------
## --------------------------------------------------------------------
## --------------------------------------------------------------------
## Test Pypi does not allow a simple linux wheel to be uploaded.
##      Seems to be an existing issue.
##      Supposedly, a many linux wheel version can be uploaded. But we are unable to create it.
##      So we were forced to upload only the source.
##      Thus, while creating an environment, one needs to compile from source (See steps below)
## --------------------------------------------------------------------
## --------------------------------------------------------------------
## --------------------------------------------------------------------
## TO CREATE NEW ENV using the TEST PYPI -- with compiling from source
##    The dependencies are not picked up automatically for some reason and need to be installed manually.
## --------------------------------------------------------------------
## Create the environment and activate it
## cd /home/rohit/.venvPython
## python3 -m venv venv11BigDataPgm2_from_testPypi
## source ~/.venvPython/venv11BigDataPgm2_from_testPypi/bin/activate
## 
## pip3 install --upgrade setuptools wheel
## pip3 install Cython>=0.28.5
## 
## pip3 install numpy>=1.14.0
## pip3 install scipy>=1.1.0
## 
## pip3 install joblib>=0.11
## pip3 install threadpoolctl>=2.0.0
## pip3 install -i https://test.pypi.org/simple/ scikit-learn-VAL-TestPypi
## 
## Note: pandas is only required for this particular test script
## pip3 install pandas

# This is our own test script where each possible use case is part of a tuple.¶
# They are executed one at a time using the exec statement.
# --------------------------
# NOTE: Exception handling is only to prevent script from breaking with a use case that should fail;
# and thus the new use case should be picked up and continue executing.
# Verify the error message thrown for the use case that should fail.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Test_Data.csv")
print(f"data.shape = {data.shape}")

#Slicing of data to devide it into target and feature set
X=data.drop(columns=['Class'])
y=data['Class'].to_frame()
print(f"X.shape = {X.shape}")
print(f"y.shaep = {y.shape}")

# Test our all combinations:¶
# with the new parameter and thus expect 3 outputs for each input array (train, test, validation)
# without specifying new parameter and thus expect 2 outputs for each input array (train, test) - backward compatible
# with/ without SHUFFLE, STRATIFY

commandTuple = (
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, train_size=20, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=80, train_size=30, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, train_size=20, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.2, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, train_size=0.3, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.2, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, validation_size=0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, validation_size=0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, validation_size=0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, train_size=20, validation_size=0, random_state=42)", 
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=80, train_size=30, validation_size=0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, validation_size=0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, validation_size=0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, validation_size=0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, train_size=20, validation_size=0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, validation_size=0.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, validation_size=0.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, validation_size=0.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.2, validation_size=0.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, train_size=0.3, validation_size=0.0, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, validation_size=0.0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, validation_size=0.0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, validation_size=0.0, stratify=y, random_state=42)",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.2, validation_size=0.0, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=80, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=90, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=90, validation_size=15, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=20, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=90, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=90, validation_size=15, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, train_size=20, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, train_size=2, validation_size=10, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=10, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, validation_size=10, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=20, validation_size=10, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, train_size=20, validation_size=10, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=70, train_size=2, validation_size=10, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=0.1, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=0.9, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.7, validation_size=0.1, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=0.2, validation_size=0.1, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.7, train_size=0.2, validation_size=0.1, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, validation_size=0.1, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.7, validation_size=0.1, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=0.2, validation_size=0.1, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, train_size=0.2, validation_size=0.02, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.02, train_size=0.2, validation_size=0.1, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.7, train_size=0.2, validation_size=0.1, stratify=y, random_state=42)",
    "X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X, y, test_size=0.7, train_size=0.2, validation_size=0.2, stratify=y, random_state=42)"
)

useCaseNo = 0

for eachCmd in commandTuple:
    try:
        #run_usecase(eachCmd)
        useCaseNo += 1
        print(f"\n\n--- Use case {useCaseNo} ---")
        print(f"executing command statement =\n{eachCmd}")
        
        X_train, X_test, X_val, y_train, y_test, y_val = [None]*6
        exec(eachCmd)
        
        if X_val is None: # trying an old functionality with only train + test subset return
            print(f"\nOutput =\nX_train=\n{X_train.shape}\nX_test=\n{X_test.shape}\n\ny_train=\n{y_train.shape}\ny_test=\n{y_test.shape}\n\n")
        else: # trying an old functionality with only train + test + validation subset return
            print(f"\nOutput =\nX_train=\n{X_train.shape}\nX_test=\n{X_test.shape}\nX_val=\n{X_val.shape}\n\ny_train=\n{y_train.shape}\ny_test=\n{y_test.shape}\ny_val=\n{y_val.shape}\n\n")
    
    ## the exception handling is only to prevent script from breaking with a use case that should fail.
    ##     and thus the new use case should be picked up and continue executing
    except Exception as e:
        print(f"\n{e}\n")
        print(f"\nRECEIVED ERROR - CHECK THE ABOVE MESSAGE --- MOVING TO NEXT USE CASE\n")

print(f"\n\nDone\n\n")