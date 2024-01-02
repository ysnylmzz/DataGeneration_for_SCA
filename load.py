import h5py

def load_database(database):
    return h5py.File(database, "r")


db = load_database("MachineLearningBasedSideChannelAttackonEdDSA/databaseEdDSA.h5")



a_label = list(db["Attack_traces"]["label"])

a_traces = list(db["Attack_traces"]["traces"])


p_label =  list(db["Profiling_traces"]["label"])
p_traces = list(db["Profiling_traces"]["traces"])


------------------------------------

# cond - ( 0 - 15 ) output 1x1000

-------------------------------------





print("--")