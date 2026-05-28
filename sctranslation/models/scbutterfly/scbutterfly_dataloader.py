import torch
from torch.utils.data import Dataset, DataLoader
from sctranslation.data import scData
import numpy as np
from typing import Optional


class scButterflyDataset(Dataset):
    """A Dataset to handle the multi-omics data for the Butterfly model."""

    def __init__(self, data: scData, data_type: str = "both", modal2='a'):
        """
        Initializes the dataset with a preprocessed Polars DataFrame.

        Parameters:
        ----------
        dataframe : pl.DataFrame
            The DataFrame containing the spectrum data.
        """
        super().__init__()
        RNA_data = data.ad["r"]
        ATAC_data = data.ad[modal2]
        if RNA_data.shape[0] != ATAC_data.shape[0]:
            common_obs = RNA_data.obs_names.intersection(ATAC_data.obs_names)
            RNA_data = RNA_data[common_obs, :]
            ATAC_data = ATAC_data[common_obs, :]
        assert RNA_data.shape[0] > 0

        if data_type == "both":
            RNA_dataset = RNA_data.X.A
            ATAC_dataset = ATAC_data.X.A
            self.data = np.concatenate([RNA_dataset, ATAC_dataset], axis=1)
            self.count = self.data.shape[0]
        elif data_type == "r":
            self.data = RNA_data.X.A
            self.count = self.data.shape[0]
        elif data_type == modal2:
            self.data = ATAC_data.X.A
            self.count = self.data.shape[0]
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        self.data_type = data_type

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx):
        x = self.data[idx, :]
        return torch.from_numpy(x).float()


class scButterflyDataModule:
    """
    A simplified data loader for the Butterfly model.
    """

    def __init__(
        self,
        ad: scData,
        batch_size: int = 128,
        n_workers: Optional[int] = 0,
        data_type: str = "both",
        modal2: str='a'
    ):
        """
        Initializes the data module with a scData object.
        Parameters:
        ----------
        ad : scData
            The scData object containing the multi-omics data.
        batch_size : int
            The batch size for the DataLoader.
        n_workers : int
            The number of workers for the DataLoader.
        data_type : str
            The type of data to use. Can be 'both', 'r', or 'a'.
            'r' uses only the RNA data, 'a' uses only the ATAC data,
            and 'both' uses both RNA and ATAC data.
        """
        self.adata = ad
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = scButterflyDataset(ad, data_type=data_type, modal2=modal2)

    def get_dataloader(self, shuffle=False) -> DataLoader:
        """
        Create and return a PyTorch DataLoader.

        Returns:
        -------
        DataLoader: A PyTorch DataLoader for the multi-omics data.
        """
        if len(self.dataset) % self.batch_size == 1 and shuffle:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=shuffle,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=shuffle,
                drop_last=False,
            )
