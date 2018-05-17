import aglaia
import glob
from aglaia.dimensionality_reduction import VAE

if __name__ == "__main__":
    filenames = glob.glob("aglaia/hexane/*.xyz")[7500:7800]

    model = VAE(layer_sizes = [20], batch_size = 100, filenames = filenames)
    model.fit()

