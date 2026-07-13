import pandas as pd
import json

class LabelLoader:
    def __init__(self, file):
        with open(file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.removed_labels = None
        self.removed_images = None
    
    def _vectorize_labels(self):
        """
        Truns annotation JSON into binary feature dataframe
        """
        df = pd.DataFrame(self.data).rename(columns={"image": "image_id"})
        df = df.explode("annotations").rename(columns={"annotations": "label"})

        binary = (
            df.assign(val=1)
            .pivot_table(index="image_id", columns="label", values="val", aggfunc="sum", fill_value=0)
            .clip(upper=1)
            .astype(int)
        )
        all_labels = sorted(df["label"].unique())
        binary = binary.reindex(columns=all_labels, fill_value=0)

        binary.index = binary.index.astype(str).str.removesuffix('.jpg')

        return binary
    
    def load(self, min_count, remove_meta=True, remove_unusable=True, remove_empty = False, custom=[]):
        """
        Removes all images that are labeld as "unbrauchbar" and 
        drops all features which only occur 'min_count' times.
        If 'remove_meta' is ture, metadata labels are removed

        Returns:
            pandas DataFrame with binary feature vectors
        """
        
        if remove_meta:
            custom = custom + ["faecher", "tuer", "gesamter_kuehlschrank", "schublade"]

        label_df = self._vectorize_labels()

        if remove_unusable:
            unusable = label_df[label_df["unbrauchbar"] == 1].index.to_list()
            label_df = label_df.drop(columns=["unbrauchbar"])
        else:
            unusable = []

        col_counts = label_df.sum(axis=0)
        rare_cols = col_counts[col_counts < min_count].index.tolist()
        drop_cols = rare_cols + custom
        reduced_df = label_df.drop(drop_cols, axis=1)
        
        count_rows = reduced_df.sum(axis=1)

        if remove_empty:
            empty_rows = count_rows[count_rows == 0].index.tolist()
        else:
            empty_rows = []
        
        drop_rows = empty_rows + unusable
        cleaned_df = reduced_df.drop(drop_rows)


        self.removed_labels = drop_cols
        self.removed_images = drop_rows
        return cleaned_df
    
    def load_two(self):
        df = self.load(0, remove_meta=False)

        cols = ["faecher", "tuer", "gesamter_kuehlschrank", "schublade"]

        area_df = df[cols]
        drop_df = df.drop(columns=cols)

        faecher_df = drop_df[df["faecher"] == 1]
        tuer_df = drop_df[df["tuer"] == 1]
        gesammt_df = drop_df[df["gesamter_kuehlschrank"] == 1]
        schublade_df = drop_df[df["schublade"] == 1]

        faecher_df = faecher_df.loc[:, (faecher_df != 0).any(axis=0)]
        tuer_df = tuer_df.loc[:, (tuer_df != 0).any(axis=0)]
        gesammt_df = gesammt_df.loc[:, (gesammt_df != 0).any(axis=0)]
        schublade_df = schublade_df.loc[:, (schublade_df != 0).any(axis=0)]

        return area_df, faecher_df, tuer_df, gesammt_df, schublade_df

    def get_removed_labels(self):
        return self.removed_labels

    def get_removed_images(self):
        return self.removed_images

    
class MetaLoader:
    def __init__(self, file, label_file):
        self.file = file
        self.label_df = LabelLoader(label_file).load(0, False, False)
        self.null_entries = None
    
    def load(self):
        """
        Parses file from survey to return a binary dataframe where each row includes which
        answers were given for a specific image.
        """
        df = pd.read_csv(self.file)

        # split image links
        df["Bilderupload"] = (
            df["Bilderupload"]
            .fillna("")
            .str.split(r",\s*")
        )


        df = df.explode("Bilderupload").reset_index(drop=True)
        df = df[df["Bilderupload"].str.strip() != ""]
        df = df.rename(columns={"Bilderupload": "image_link"})

        df["image_link"] = df["image_link"].str.extract(
            r"id=([A-Za-z0-9_-]+)"
        )

        # columns with multiple answers
        multi_cols = [
            "Anzahl der Kühlschranknutzer:innen",
            "Geschlecht",
            "Alter",
            "Ernährungsform",
            "Wie hast du von der Umfrage erfahren?"
        ]

        encoded_parts = []

        for col in multi_cols:

            # split answers by comma
            split_series = (
                df[col]
                .fillna("")
                .astype(str)
                .str.split(r",\s*")
            )

            # create binary matrix
            dummies = split_series.explode().str.get_dummies().groupby(level=0).max()

            encoded_parts.append(dummies)

        encoded_df = pd.concat(encoded_parts, axis=1)

        encoded_df = encoded_df.reset_index(drop=True)

        final_df = pd.concat(
            [
                df[["Zeitstempel", "image_link"]].reset_index(drop=True),
                encoded_df
            ],
            axis=1
        )

        final_df["Zeitstempel"] = pd.to_datetime(
            final_df["Zeitstempel"],
            format="%d.%m.%Y %H:%M:%S"
        )
        final_df = final_df.set_index("image_link")
        return final_df
    
    def add_meta(self):
        """
        Adds Metadata to the labels, removes all NaN rows
        """
        meta_df = self.load()
        joined = meta_df.join(self.label_df)
        self.null_entries = joined[joined.isna().any(axis=1)]
        joined = joined.dropna()
        return joined


    def get_unlabled_images(self):
        """
        Get all image ids that have not yet been labeld
        """
        return self.null_entries.index.to_list()
    
    def add_food_type(self, map_file, add=True):
        df = self.add_meta()

        original_cols = self.load().columns.tolist()

        with open(map_file, "r") as f:
            mapping = json.load(f)

        new_cols = []

        for category, foods in mapping.items():

            existing_cols = [f for f in foods if f in df.columns]

            df[category] = df[existing_cols].any(axis=1).astype(int)

            new_cols.append(category)

        if not add:
            df = df[original_cols + new_cols]

        return df
        



if __name__ == "__main__":
    loader = MetaLoader("./images/umfrage_antworten_11_05.csv", "./images/data/labels.json")
    join = loader.add_meta()
    null = loader.get_unlabled_images()
    print(null)
    print(len(join))




            