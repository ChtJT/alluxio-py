import datasets # pip install datasets
import lance

class LoadDataset():
    def __init__(self, source):
        self.source = source
    


if __name__ == "__main__":
    hf_ds = datasets.load_dataset(
        "poloclub/diffusiondb",
        split="train",
        # name="2m_first_1k",
    )
    lance.write_dataset(hf_ds, "diffusiondb_train.lance")

