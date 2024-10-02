import numpy as np

if __name__ == "__main__":
    data = np.load("./processed_vid/cut_pats_lec_short_longer.mp4.npz", allow_pickle=True)
    print(len(data["segments"]))