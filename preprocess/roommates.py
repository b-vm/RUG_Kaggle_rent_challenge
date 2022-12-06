
def roommate_helper(x):
    if x == "None":
        return 0
    if x == "More than 8":
        return 10
    return x

def preprocess_roommates(df):
    df["roommates"] = df["roommates"].apply(lambda x: roommate_helper(x))
    return df
