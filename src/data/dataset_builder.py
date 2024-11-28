import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.simulated_data import SimpleSimulatedDataset
from data.warfarin_data import WarfarinDataset
from data.simulated_dataset_complex import SimulatedDatasetComplex
from data.simulated_non_linear import SimulatedDatasetNonLinear
from data.simulated_confounded import SimulatedDatasetBetaConfounded

class DatasetBuilder:
    """
    A factory class for creating and managing different types of datasets.
    
    Attributes:
        dataset: The internal dataset object (WarfarinDataset or SimulatedDataset)
    """
    
    def __init__(self, name: str, sim: bool = True):
        """
        Initialize the DatasetBuilder with a specific dataset type.
        
        Args:
            name (str): Type of dataset to create
            sim (bool): Whether dataset is simulated or not
            
        Raises:
            ValueError: If an invalid dataset name is provided
        """
        self.sim = sim
        self.dataset = None
        self.name = name
        self.updated = False

        self.fair_variables = ["estimated_y0", "estimated_y1"]

        if self.sim:
            if name.lower() == "warfarin":
                self.dataset = WarfarinDataset()
            elif name.lower() == "sim":
                self.dataset = SimpleSimulatedDataset()
            elif name.lower() == "sim_complex":
                self.dataset = SimulatedDatasetComplex()
            elif name.lower() == "sim_non_linear":
                self.dataset = SimulatedDatasetNonLinear()
            elif name.lower() == "sim_confounding":
                self.dataset = SimulatedDatasetBetaConfounded()
            else:
                raise ValueError(f"Unknown dataset type: {name}. Available options are 'warfarin' or 'sim'")
        
    def load(self, **kwargs):
        """
        Load the dataset
        """

        if self.sim:
            if self.dataset is None:
                raise RuntimeError("Dataset not initialized")
        
            self.data = self.dataset.load(**kwargs)
            columns = kwargs.get('columns', self.data.columns)

            self.columns = columns
            self.decision = 'D'
            self.protected_attribute_name = 'A'
            self.priviledged_class = [0]
            self.outcome = 'Y'

            self.data, self.true_outcomes = self.dataset.preprocess_data()

            return self.data, self.true_outcomes
        
        else:

            self.data = kwargs['dataset']
            self.data = self.data.drop(['Z'], axis=1)
            columns = kwargs.get('columns', self.data.columns)
            decision = kwargs.get('decision', 'D')
            protected_attribute_name = kwargs.get('protected_attribute_name', 'A')
            priviledged_class = kwargs.get('priviledged_class', [0])
            outcome = kwargs.get('outcome', 'Y')
            self.data = self.data[columns + [decision] + [outcome]]

            self.columns = columns
            self.decision = decision
            self.protected_attribute_name = protected_attribute_name
            self.priviledged_class = priviledged_class
            self.outcome = outcome

            self.data['D'] = (self.data['D'] >= 1).astype(int)

            return self.data
        
    def unupdate(self):
        self.data = self.data.drop(columns=("estimated_y0", "estimated_y1", "result"))
        self.updated = False
        return self.data
        
    def update(self, new_cols: pd.DataFrame, result_method: bool = True, estimates: bool = True):
        for col in new_cols.columns:
            self.data[col] = new_cols[col]
        
        if estimates:
            self.data['estimated_y0'] = np.where(self.data['D'] == 0, self.data['Y'], self.data['estimated_y0'])
            self.data['estimated_y1'] = np.where(self.data['D'] == 1, self.data['Y'], self.data['estimated_y1'])

        if result_method and estimates:
            self.data['result'] = np.where(
                self.data[self.decision] == 1,
                np.where((self.true_outcomes['Y0'] if self.sim else self.data['estimated_y0']) < self.data[self.outcome], 1, 0),
                np.where(self.data[self.outcome] == 1, 1, 0)
            )

        elif estimates:
            self.data['result'] = np.where(
                self.data[self.decision] == 0,
                np.where((self.true_outcomes['Y1'] if self.sim else self.data['estimated_y1']) > self.data[self.outcome], 1, 0),
                np.where(self.data[self.outcome] == 1, 1, 0)
            )
        else:
            self.data['result'] = self.data[self.outcome]
        
        self.updated = True
        return self.data
    
    def train_test_split(self, split = 0.2):

        if not self.sim:
            temp, self.test = train_test_split(
                self.data, test_size=split, random_state=123
            )

            self.train, self.val = train_test_split(
                temp, test_size=split, random_state=123
            )
        else:
            temp, self.test, true_outcomes_temp, self.true_outcomes_test = train_test_split(
                self.data, self.true_outcomes, test_size=split, random_state=123
            )

            self.train, self.val, self.true_outcomes_train, self.true_outcomes_val = train_test_split(
                self.data, self.true_outcomes, test_size=split, random_state=123
            )


    # Adapted from https://github.com/windxrz/DCFR/blob/master/dcfr/datasets/standard_dataset.py
    def process(self, categorical_features = [], normalize = False):
        
        cols = [x for x in self.train.columns if x not in (
            categorical_features
            + [self.decision, self.outcome]
            + [self.protected_attribute_name] 
            + ["result"])
            ]
        
        result = []
        for df in [self.train, self.val, self.test]:
            df = df.drop(columns = [self.decision, self.outcome])

            df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")
            pos = np.logical_or.reduce(
                np.equal.outer(self.priviledged_class, df[self.protected_attribute_name].values)
            )

            df.loc[pos, self.protected_attribute_name] = 1
            df.loc[~pos, self.protected_attribute_name] = 0
            df[self.protected_attribute_name] = df[self.protected_attribute_name].astype(int)

            df["result"] = df["result"].astype(int)

            result.append(df)

        if normalize:
            for col in cols:
                data = result[0][col].tolist()
                mean, std = np.mean(data), np.std(data)
                result[0][col] = (result[0][col] - mean) / std
                result[0][col] = (result[0][col] - mean) / std

        self.train = result[0]
        test = result[2]
        for col in self.train.columns:
            if col not in test.columns:
                test[col] = 0
        
        cols = self.train.columns
        self.test = test[cols]
        self.val = result[1]
        self.val = self.val[cols]

        return self.train, self.test
    
    # Adapted from https://github.com/windxrz/DCFR/blob/master/dcfr/datasets/standard_dataset.py
    def analyze(self, df_old, y=None, log=True):

        df = df_old.copy()
        if y is not None:
            df["y hat"] = (y > 0.5).astype(int)

        s = self.protected_attribute_name
        res = dict()
        n = df.shape[0]
        y1 = df.loc[df["result"] == 1].shape[0]/n

        if "y hat" in df.columns:
            yh1s0 = (
                df.loc[(df[s] == 0.0) & (df["y hat"] == 1.0)].shape[0]
                / df.loc[df[s] == 0.0].shape[0]
            )
            yh1s1 = (
                df.loc[(df[s] == 1.0) & (df["y hat"] == 1.0)].shape[0]
                / df.loc[df[s] == 1.0].shape[0]
            )
            yh1y1s0 = (
                df.loc[(df["y hat"] == 1.0) & (df["result"] == 1.0) & (df[s] == 0.0)].shape[0]
                / df.loc[(df["result"] == 1.0) & (df[s] == 0.0)].shape[0]
            )
            yh1y1s1 = (
                df.loc[(df["y hat"] == 1) & (df["result"] == 1) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 1) & (df[s] == 1)].shape[0]
            )
            yh0y0s0 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 0)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 0)].shape[0]
            )
            yh0y0s1 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 1)].shape[0]
            )

            correct_decisions = (df["y hat"] == df["result"])
            fair_variables = ["estimated_y0", "estimated_y1"]

            res["acc"] = correct_decisions.mean()
            res["DP"] = np.abs(yh1s1 - yh1s0)
            tpr = yh1y1s0 - yh1y1s1
            fpr = yh0y0s0 - yh0y0s1
            res["EO"] = np.abs(tpr) * y1 + np.abs(fpr) * (1 - y1)

        if self.sim:
            df_rows = len(df)
            if df_rows == len(self.true_outcomes_train):
                true_outcomes = self.true_outcomes_train
            elif df_rows == len(self.true_outcomes_val):
                true_outcomes = self.true_outcomes_val
            elif df_rows == len(self.true_outcomes_test):
                true_outcomes = self.true_outcomes_test
            else:
                raise ValueError("DataFrame size doesn't match any true outcomes dataset")
            
            fair_variables = ["Y0", "Y1"]
            df_for_grouping = df.copy()
            df_for_grouping[['Y0', 'Y1']] = true_outcomes[['Y0', 'Y1']]
        else:
            df_for_grouping = df

        count = (
            df_for_grouping.groupby(fair_variables + [s])
            .count()["y hat"]
            .reset_index()
            .rename(columns={"y hat": "count"})
        )
        count_y = (
            df_for_grouping.groupby(fair_variables + [s])
            .sum()["y hat"]
            .reset_index()
            .rename(columns={"y hat": "count_y"})
        )
        count_merge = pd.merge(count, count_y, how="outer", on=fair_variables + [s])
        count_merge["ratio"] = count_merge["count_y"] / count_merge["count"]
        count_merge = count_merge.drop(columns=["count", "count_y"])
        count_merge["ratio"] = (2 * count_merge[s] - 1) * count_merge["ratio"]
        
        if len(self.fair_variables) > 0:
            result = (
                count_merge.groupby(fair_variables)
                .sum()["ratio"]
                .reset_index(drop=True)
                .values
            )
        else:
            result = count_merge.sum()["ratio"]

        if len(fair_variables) > 0:
            fairs = (
                df_for_grouping.groupby(fair_variables).count()[s].reset_index(drop=True).values
            )
            fairs = fairs / np.sum(fairs)
        else:
            fairs = 1
        res["CF"] = np.sum(np.abs(result) * fairs)

        if log:
            for key, value in res.items():
                print(key, "=", value)
        return res


    
