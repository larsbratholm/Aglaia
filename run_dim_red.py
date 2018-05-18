import aglaia
import glob
from aglaia.dimensionality_reduction import VAE

if __name__ == "__main__":
    filenames = sorted(glob.glob("aglaia/hexane/*.xyz"))[6500:8650]

    model = VAE(layer_sizes = [20], batch_size = 100, filenames = filenames)
    model.fit()

