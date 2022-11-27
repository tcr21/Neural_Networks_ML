import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd

from sklearn import preprocessing

# from skorch import NeuralNetRegressor
# from sklearn.model_selection import GridSearchCV


class HousingDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.features = x_tensor
        self.labels = y_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class NeuralNetwork(nn.Module):
    def __init__(self, n_input_vars, n_output_vars=1):
        super().__init__()  # call constructor of superclass
        self.layers = nn.Sequential(
            nn.Linear(n_input_vars, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, n_output_vars),
        )

    def forward(self, x):
        return self.layers(x)


class Regressor:
    def __init__(self, x, nb_epoch=10):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.lb = None
        self.ocean_proximity_mode = None
        self.binarized_labels = []
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.mean_values = None
        self.std_values = None
        self.model = None

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Set the mode with training data
        if training:
            self.ocean_proximity_mode = x["ocean_proximity"].mode().values[0]
        x["ocean_proximity"].fillna(value=self.ocean_proximity_mode, inplace=True)

        # Initialise LabelBinarizer and fit to dataset (should only be used with train)
        if training:
            self.lb = preprocessing.LabelBinarizer()
            self.lb.fit(x["ocean_proximity"].values)

        # Binarize ocean proximity labels
        binarized_labels = self.lb.transform(x["ocean_proximity"].values)

        labels_df = pd.DataFrame(binarized_labels, columns=self.lb.classes_)
        # Drop strings column from dataframe
        x_dropped = x.drop(
            labels="ocean_proximity",
            axis=1,
        )

        # Fill all NA values with mean of their column
        mean_values = x_dropped.mean()
        x_dropped.fillna(x_dropped.mean(), inplace=True)

        # Standardise the values with mean/std_dev
        std_values = x_dropped.std()
        x_dropped = (x_dropped - mean_values) / std_values

        # Merge the binarized labels onto the original dataframe
        x_dropped.reset_index(inplace=True)
        labels_df.reset_index(inplace=True)

        x_dropped_merged = pd.concat([x_dropped, labels_df], axis=1)
        x_dropped_merged.drop(labels="index", axis=1, inplace=True)

        # Return preprocessed x and y, return None for y if it was None
        x_tensor = torch.from_numpy(x_dropped_merged.values).float()
        return x_tensor, (
            torch.from_numpy(y.values).float() if isinstance(y, pd.DataFrame) else None
        )

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x_train_tensor, y_train_tensor = self._preprocessor(
            x, y=y, training=True
        )  # Do not forget

        # Ask PyTorch to store any computed gradients so that we can examine them
        x_train_tensor.requires_grad_(True)

        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Construct the model
        model = NeuralNetwork(self.input_size, self.output_size)
        # Define the loss function and optimizer
        criterion = nn.L1Loss()
        optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

        # Load data using dataset loader for mini-batching
        train_dataset = HousingDataset(x_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        for epoch in range(self.nb_epoch):
            for x, y in train_dataloader:
                # Reset the gradients
                optimiser.zero_grad()

                # forward pass
                y_hat = model(x)

                # compute loss
                loss = criterion(y_hat, y)

                # Backward pass (compute the gradients)
                loss.backward()

                # update parameters
                optimiser.step()
            print("Epoch: ", epoch)
        self.model = model
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_test_tensor, _ = self._preprocessor(x, training=False)  # Do not forget
        y_predictions = self.model.forward(x_test_tensor)
        y_predictions_np = y_predictions.cpu().detach().numpy()
        return y_predictions_np

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_test_tensor, y_test_tensor = self._preprocessor(x, y=y, training=False)
        loss = nn.MSELoss()
        y_predictions = self.model.forward(x_test_tensor)
        output = torch.sqrt(loss(y_predictions, y_test_tensor))

        return output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    return trained_model


def RegressorHyperParameterSearch(model, x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # net = NeuralNetRegressor(
    #     model,
    #     max_epochs=10,
    #     lr=0.001,
    #     optimizer=torch.optim.SGD,
    #     criterion=nn.L1Loss(),
    # )

    # params = {
    #     "lr": [0.01, 0.001, 0.0001],
    #     "max_epochs": [10, 50, 100],
    #     "batch_size": [16, 32, 64],
    # }

    # gs = GridSearchCV(
    #     net,
    #     params,
    #     scoring="neg_root_mean_squared_error",
    #     n_jobs=16,
    #     verbose=3,
    #     cv=10,
    # )

    # gs.fit(x_train, y_train)

    # print("Best Score  : {}".format(gs.best_score_))
    # print("Best Params : {}".format(gs.best_params_))
    # print("Results {}".format(gs.cv_results_))

    # return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    x_train = pd.DataFrame(x)
    x = pd.DataFrame(x)
    x_test = pd.DataFrame(x[int(len(x) * 0.8) :])
    y_train = pd.DataFrame(y)
    y = pd.DataFrame(y)
    y_test = pd.DataFrame(y[int(len(y) * 0.8) :])
    regressor = Regressor(x_train, nb_epoch=100)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    regressor = load_regressor()
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # For HyperParameter Optimisation
    # x_train_tensor, y_train_tensor = regressor._preprocessor(
    #     x, y=y, training=True
    # )  # Do not forget
    # RegressorHyperParameterSearch(
    #     NeuralNetwork(x_train_tensor.shape[1], 1), x_train_tensor, y_train_tensor
    # )


if __name__ == "__main__":
    example_main()
