import numpy as np
import typing as ty
import csv
from matplotlib import pyplot as plt
import time
import datetime
import math


TList = ty.TypeVar(ty.List[ty.Union[float, int]])


def smooth(x, window_len=11, window="hanning"):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    methods = {
        "flat": lambda win_len: np.ones(window_len, "d"),  # moving average
        "hanning": lambda win_len: np.hanning(win_len),
        "hamming": lambda win_len: np.hamming(win_len),
        "bartlett": lambda win_len: np.bartlett(win_len),
        "blackman": lambda win_len: np.blackman(win_len),
    }

    w = methods[window](window_len)

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def load_csv(path_to_file, with_time=False) -> ty.Dict[str, ty.Any]:
    data = {}
    with open(path_to_file, newline="") as ifile:
        reader = csv.reader(ifile)
        columns = next(reader)
        data.update({key: [] for key in columns if key != ""})
        for idx, line in enumerate(reader):
            for index, item in enumerate(line):
                if index:
                    try:
                        data[columns[index]].append(float(item))
                    except ValueError:
                        data[columns[index]].append(item)
                elif with_time and not index:
                    timestamp = time.mktime(
                        datetime.datetime.strptime(
                            item, "%Y-%m-%d"
                        ).timetuple()
                    )
                    data[columns[index]].append(timestamp)
    return data


class SSA:
    def __init__(self, source_data: TList, source_times: TList):
        self.__source = np.array(list(zip(source_times, source_data)))

    def getValues(self) -> TList:
        return self.__source[:, 1]

    def getKeys(self) -> TList:
        return self.__source[:, 0]

    @staticmethod
    def normalize_data(data: TList) -> ty.Dict[str, ty.Any]:
        norm_res = np.array(list())
        mean = np.mean(data)
        std = np.std(data)
        for d in data:
            norm_res = np.append(norm_res, (d - mean) / std)

        return {"data": norm_res, "mean": mean, "std": std}

    def deconstruct(self, t=None) -> ty.Dict[str, ty.Any]:
        normalized = SSA.normalize_data(self.getValues())
        values = normalized["data"]
        N = len(values)
        if not t or t > (N + 1) / 2:
            t = (N + 1) / 2
            t = math.floor(t)

        n = N - t
        if N % 2 != 0:
            n += 1

        new_sample = []
        for i in range(n):
            new_sample.append(values[i: t + i])

        return {
            "source": values,
            "mean": normalized["mean"],
            "std": normalized["std"],
            "matrix": np.array(new_sample),
            "n": n,
            "N": N,
        }

    def recovered_data(self, t=None) -> ty.Dict[str, ty.Any]:
        data = self.deconstruct(t)
        cov_data = np.cov(data["matrix"].T)
        lambdas, vectors = np.linalg.eigh(cov_data)

        # get non negative lambdas
        first_non_negative = -1
        for idx, lam in enumerate(lambdas):
            if lam >= 0:
                first_non_negative = idx
                break

        lambdas, vectors = (
            lambdas[first_non_negative:],
            vectors[first_non_negative:],
        )

        sum_lambdas = sum(lambdas)
        variance_explained = []
        for i in lambdas:
            variance_explained.append((i / sum_lambdas) * 100)

        # find count of main_components
        cumulative_variance_explained = np.cumsum(variance_explained)
        count_of_main_comp = -1
        for idx, num in enumerate(cumulative_variance_explained):
            if num >= 80:
                count_of_main_comp = idx
                break

        v_r = (vectors.T[:][:count_of_main_comp]).T

        pca = (
            v_r.T @ data["matrix"][first_non_negative:]
        )  # data["matrix"][first_non_negative:].T @ v_r

        _, columns = pca.shape
        recovered_matrix = v_r[:columns] @ pca

        # t = data['N'] - data['n']
        t, n = recovered_matrix.shape
        N = t + n
        recovered_time_series = []
        mean, std = data["mean"], data["std"]
        for s in range(1, t):
            t_sum = []

            for i in range(s):
                t_sum.append(recovered_matrix[i][s - i] * std + mean)

            element = 1.0 / s * sum(t_sum)
            recovered_time_series.append(element)

        for s in range(t, n + 1):
            t_sum = []

            for i in range(t):
                t_sum.append(recovered_matrix[i][s - i - 1] * std + mean)

            element = 1.0 / t * sum(t_sum)
            recovered_time_series.append(element)

        for s in range(n + 1, N):
            t_sum = []

            for i in range(N - s + 1):
                t_sum.append(
                    recovered_matrix[i + (s - n) - 1][n - i - 1] * std + mean
                )

            element = 1 / (N - s + 1) * sum(t_sum)
            recovered_time_series.append(element)

        len_start = len(self.getValues()) - N
        return {
            "data": np.array(recovered_time_series),
            "N": N,
            "t": t,
            "n": n,
            "v_r": v_r,
            "lambdas": lambdas,
            "dates": self.getKeys()[len_start:],
        }

    def getPrediction(
        self, recovered_data: ty.Dict[str, ty.Any], horizon: int
    ):
        v_r = recovered_data["v_r"]
        v_r_1 = v_r[:, :-1]
        v_r_row = v_r[:, -1]
        v_r_inv = 1 - v_r_row @ v_r_row.T
        coeff = (v_r_row @ v_r_1) / v_r_inv
        (length,) = coeff.shape
        q = recovered_data["data"][-length:]
        predict = np.array([])
        right_side = horizon
        for i in range(right_side):
            elem = coeff @ q
            predict = np.append(predict, elem)
            q = np.append(q, elem)
            q = q[1:]

        return predict


def main():
    # initialize starting data
    d = load_csv("./AAPL.csv", True)
    # d = {
    #     'Date': np.array([]),
    #     'Volume': np.array([])
    # }
    # for i in range(2000):
    #     d['Date'] = np.append(d['Date'], i)
    #     d['Volume'] = np.append(d['Volume'], math.sin(math.pi * i / 16))

    predict_length = 0.2
    data_length = len(d["Volume"])
    predict_length = int(data_length * predict_length)
    predict_data_length = data_length - predict_length
    data = d["Volume"][: predict_data_length + 1]
    data_times = d["Date"][: predict_data_length + 1]
    # start calculating ssa
    ssa = SSA(data, data_times)

    # recovered by pca
    recovered = ssa.recovered_data()
    start_from = abs(len(data) - recovered["N"])
    length = recovered["N"] + start_from - 1
    # show plot with data
    plt.plot(ssa.getKeys(), ssa.getValues())
    plt.plot(d["Date"][start_from:length], recovered["data"])
    plt.legend(["recovered data", "actual data"])
    plt.show()

    # get prediction by predict_length
    prediction = ssa.getPrediction(recovered, predict_length)
    prediction = smooth(prediction)
    end_index = len(d["Date"][length:])
    end_index_date = len(prediction[:end_index])
    plt.plot(
        d["Date"][length: length + end_index_date], prediction[:end_index]
    )
    plt.plot(d["Date"][length:], d["Volume"][length:])
    plt.legend(["predicted data", "actual data"])
    plt.show()


if __name__ == "__main__":
    main()
