import pandas as pd
from langdetect import detect, LangDetectException

#This function returns the language that is idenitfied for the given sentence
def detect_language(sentence):
    if isinstance(sentence, str):
        try:
            return detect(sentence)
        except LangDetectException:
            print(f"Unknown: {sentence}")
            return "unknown"
    else:
        print(f"Numbers: {sentence}")
        return "unknown"

#This function returns a new data frame with only the requested language (usually English)
def keep_only_language(df, language):  
    df["language"] = df["claimReviewed"].apply(detect_language)
    new_df = df[df["language"] == language]
    new_df = new_df.drop(columns="language")
    return new_df

#Mapping from the review's rating names in the dataset and the ones we have chosen to standardize
word_mapping = {
    "FALSE": "False",
    "Misleading": "Mostly False",
    "Pants on Fire": "False",
    "MISLEADING": "Mostly False",
    "Unproven": "Mixture",
    "Miscaptioned": "Mostly False",
    "Half True": "Mixture",
    "Labeled Satire": "False",
    "False.": "Mixture",
    "Fake": "False",
    "Correct Attribution": "True",
    "Missing Context": "Mixture",
    "Satire": "False",
    "Scam": "False",
    "Missing context": "Mixture",
    "Legend": "False",
    "Half true": "Mixture",
    "Four Pinocchios" : "False"
}

df = pd.read_csv("Datasets/claim_review.csv")

df = keep_only_language(df, "en")
df["reviewRating.alternateName"] = df["reviewRating.alternateName"].replace(word_mapping)
df = df[df["reviewRating.alternateName"].isin(word_mapping.values())]

#Final dataset with only the claims in english and the review's rating name standardized
df.to_csv("Datasets/new_dataset.csv", index = False, encoding="utf-16", sep="\t")