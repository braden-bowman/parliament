"""
adapted from 
    https://www.kaggle.com/code/jordandelbar/titanic-with-polars-and-pytorch
    
    
    
polars better for ETL
    - bigger datasets
    - speed comparable to CuDF
"""


# need to set up requirements.txt

# polars
# !pip install polars, rustworkx, feature-engineering-polars==0.2.0

# cuda
# !pip install cugraph


#### PyTorch ####
# on a MacBook
# https://pytorch.org/get-started/locally/
# !pip3 install torch torchvision torchaudio

# ib windows

# Import our libraries
import logging
from typing import List, Union, Tuple

import numpy
import polars
import torch
from fe_polars.encoding.target_encoding import TargetEncoder
from fe_polars.imputing.base_imputing import Imputer

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# Define a fonction to load our data with Polars
def data_loader() -> Tuple[polars.DataFrame, polars.DataFrame, polars.DataFrame]:
    """Load the data from titanic files.

    Returns:
        train (polars.DataFrame): train data without the target column
        target (polars.DataFrame): target columns of our train dataframe.
        test (polars.DataFrame): test data (to submit to the Titanic competition).
    """
    train = polars.read_csv("../input/titanic/train.csv")
    test = polars.read_csv("../input/titanic/test.csv")

    target = train.select("Survived")
    train = train.drop("Survived")

    return train, target, test



# Define a Preprocessor class to create new features
class Preprocessor():

    def __init__(self, features_to_drop: List):
        """Init.
        
        Args:
            feature_to_drop (list): list of feature to drop
        """
        self.features_to_drop = features_to_drop

    def transform(self, x: polars.DataFrame) -> polars.DataFrame:
        """Transform our dataset with new features
        
        Args:
            x (polars.DataFrame): dataframe with our features
        
        Returns:
            polars.DataFrame: preprocessed dataframe
        """
        # Is the passenger a baby
        x = x.with_columns(
            polars.when(polars.col("Age") < 5).then(1).otherwise(0).alias("is_baby")
        )

        # Was the passenger travelling alone
        x = x.with_columns(
            polars.when((polars.col("SibSp") == 0) & (polars.col("Parch") == 0))
            .then(1)
            .otherwise(0)
            .alias("alone")
        )

        # Family member total
        x = x.with_columns(
            (polars.col("SibSp") + polars.col("Parch")).alias("family"))

        # Create a title column
        x = x.with_columns(
            polars.col("Name")
            .str.extract("([A-Za-z]+)\.")
            .str.replace("Mlle", "Miss")
            .str.replace("Ms", "Miss")
            .str.replace("Mme", "Mrs")
            .str.replace("Don", "Mr")
            .str.replace("Dona", "Mrs")
            .alias("title")
        )
        # Drop features not useful anymore
        x = x.drop(self.features_to_drop)

        return x
    
# Define our PyTorch model
class LogisticRegression(torch.nn.Module):
    """Logistic Regression in PyTorch."""

    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 1,
        epochs: int = 5000,
        loss_func=torch.nn.BCELoss(),
    ):
        """Init.
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            epochs (int, optional): Number of training epochs. 
                                        Defaults to 5000.
            loss_func: Loss function.
        """
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_func = loss_func
        self.epochs = epochs
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        """Forward pass."""
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def fit(self, x: polars.DataFrame, y: polars.Series):
        """Fit.
            
        Args:
            x (polars.DataFrame): training dataframe
            y (polars.Series): target series
            
        Returns:
            None
        """
        y = y.to_numpy().squeeze()

        x = torch.from_numpy(x.astype(numpy.float32))
        y = torch.from_numpy(y.astype(numpy.float32))[:, None]

        iter = 0
        epochs = self.epochs
        for epoch in range(0, epochs):
            pred_y = self.forward(x)

            # Compute and print loss
            loss = self.loss_func(pred_y, y)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            iter += 1
            if iter % 500 == 0:
                print(f"epoch {epoch}, loss {loss.item()}")
        return None

    def predict_proba(self, x: numpy.ndarray):
        """Return probability of class.
            
        Args:
            x (numpy.ndarray): dataframe to infer
            
        Returns:
            numpy.ndarray: probability of survival
            """
        x = torch.from_numpy(x.astype(numpy.float32))

        y_proba = self.forward(x)
        return y_proba.flatten().detach().numpy()

    def predict(self, x: numpy.ndarray, threshold: float = 0.5):
        """Predict survival score.
        Args:
            x (numpy.ndarray): dataframe to infer

        Returns:
            numpy.ndarray: score prediction
        """
        y_pred = self.predict_proba(x)
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        return y_pred