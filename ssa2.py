import numpy as np
import argparse
import typing as ty
import csv
from matplotlib import pyplot as plt
import time
import datetime
import math
import random as rnd


class DataSSA:
    """
        Result of self.getEigentripleMatrixByIndex
        and getEigentripleMatrixByIndex1 equal by SSA method
    """

    _x = None
    _s = None
    _u = None
    _lambdas = None
    _v = None
    _X = {}

    def __init__(self, x, s, u, lambdas, v):
        self._x = x
        self._s = s
        self._u = u
        self._lambdas = lambdas
        self._v = v

    def getXByIndex(self, index):
        sqrt_lI, uI, vI = self.getEigentripleByIndex(index)

        if not self._X.get(index):
            result = np.dot(uI, vI) * sqrt_lI
            self._X[index] = result

        return self._X[index]
    
    def getEigentripleByIndex(self, index) -> ty.Tuple[float, float, float]:
        sqrt_lI = math.sqrt(self._lambdas[index])
        uI = self._u[index]
        vI = self._v[index]
        return (sqrt_lI, uI, vI)
    
    def getEigentripleMatrixByIndex(self, index, matrixT) -> np.array:
        uI = self._u[index]
        return np.dot(matrixT, uI)
    
    def getEigentripleMatrixByIndex1(self, index):
        sqrt_lI, _, vI = self.getEigentripleByIndex(index)
        return sqrt_lI * vI
    
    def getGroups(self, max_value=5):
        max_n = len(self._lambdas)
        groups = []
        current_n = 0
        while current_n < max_n:
            l = rnd.randint(1, max_value)
            next_n = current_n + l
            if max_n - next_n >= 0:
                groups.append(tuple(range(current_n, next_n)))
                current_n = next_n
            else:
                groups.append(tuple(range(current_n, max_n)))
                current_n = max_n
        
        return groups


class Matrix:
    _data = None
    def __init__(self, data):
        if data is np.array:
            self._data = data
        else:
            self._data = np.array(data)
    
    @property
    def data(self):
        return self._data

    def getTranspose(self) -> 'Matrix':
        return Matrix(self._data.transpose())

    def getOutcom(self, n=1) -> 'Matrix':
        result = self._data @ self._data.transpose()
        result *= 1.0/n
        return Matrix(result)
    
    def getSelfMetrics(self) -> ty.Tuple[np.array, np.array]:
        return np.linalg.eigh(self._data)

    def __str__(self):
        return str(self._data)

    def getLinear(self):
        linear = self.data[:][0]
        linear = np.append(linear, self.data[-1][1:])
        return linear

    def dot(self, matrix):
        if isinstance(matrix, np.array):
            return Matrix(self._data @ matrix)
        elif isinstance(matrix, Matrix):
            return Matrix(self._data @ matrix._data)

    def __repr__(self):
        return f"<Matrix({self._data})>"


class SSA:
    _data = None
    # _min_value = None

    def __init__(self, data: list, normalize=False):
        self._data = np.array(data)
        if normalize:
            self._data, _ = self.normalize_data()
    
    def getKeys(self) -> np.array:
        return self._data[:,0]
    
    def getValues(self) -> np.array:
        return self._data[:,1]

    def normalize_data(self):
        data = self.getValues()
        keys = self.getKeys()
        max_value = np.array(data).max()
        normalized = data / max_value
        normalized = np.array(list(zip(keys, normalized)))
        return normalized, max_value

    def restrict(self): # -> Matrix:
        values = self.getValues()
        N = len(values)
        m = (N + 1) / 2
        m = math.floor(m)
        n = N - m
        if N % 2 != 0:
            n += 1
        new_sample = []
        for i in range(n):
            new_sample.append(values[i:m+i])
        
        return Matrix(np.array(new_sample)), n, N

    @classmethod
    def createSSA(cls, data):
        # create ssa class
        ssa = cls(data)

        # get restricted matrix 
        matrix, n, N = ssa.restrict()
        s_matrix = matrix.getOutcom(n)
        lambdas, vectors = s_matrix.getSelfMetrics()

        lambdas_vectors = list(zip(lambdas, vectors))
        lambdas_vectors = np.array(sorted(lambdas_vectors, key=lambda item: item[0], reverse=True))
        lambdas, vectors = lambdas_vectors[:,0], lambdas_vectors[:,1]
        
        lambdas = np.array(list(lambdas))
        vectors = np.array(list(vectors))
        y_main = vectors @ matrix._data
        r = len(list(filter(lambda item: item >= 0, lambdas)))
        
        equal_matrix = vectors @ y_main
        # r = equal_matrix.shape[0] - 1 
        # r = 31
        
        recovered_time_series = []
        for s in range(1, r): # 1/s * sum([equal_matrix[i, s - i + 1] for i in range(s + 1)]) -> 0<=s<=r-1
            recovered_time_series.append(1.0/s * sum([equal_matrix[i][s - i] for i in range(s)]))
        
        for s in range(r - 1, n): # 1/r * sum([equal_matrix[i, s - i + 1] for i in range(r)]) -> r-1<=s<=n-1
            recovered_time_series.append(1.0/r * sum([equal_matrix[i][s - i] for i in range(r)]))
        
        for s in range(n - 1, N - 1): # 1/(N - s + 1) * sum([equal_matrix[i + s - n, n - i + 1] for i in range(1, N - s + 1)]) -> n-1<=s<=N-1
            recovered_time_series.append(1.0/(N - s + 1) * sum([equal_matrix[i + s - (n - 1)][(n - 1) - i] for i in range((N - 1) - s)]))

        return (r, y_main, recovered_time_series, lambdas, vectors)

    @classmethod
    def calculateNextValues(cls, source_data, main_vecs, r, horizon):
        v_star = main_vecs[:r, :]
        down_score = v_star.T @ v_star
        donw_score_inverse = np.linalg.inv(1 - down_score)
        coeff = main_vecs @ donw_score_inverse @ v_star.T
        len_source = len(source_data)
        q_vector = source_data[len_source - r:] 
        
        prog = coeff * q_vector
        y_ = np.array([])

        for _ in range(horizon):
            prog = np.average([sum(prog_i) for prog_i in prog]) # 1/r * sum([sum(prog_i) for prog_i in prog]) # ???
            y_ = np.append(y_, prog)
            q_vector = np.append(q_vector, prog)
            q_vector = q_vector[1:]
            prog = coeff * q_vector

        return y_

    @classmethod
    def smoothData(cls, data, window_width):
        data = np.array(data)
        smooth_data = []
        for index in range(len(data) - window_width):
            mean_window = np.mean(data[index:index+window_width])
            smooth_data.append(mean_window)
        
        return np.array(smooth_data)


    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"<SSA({self._data})>"

def standardize_data(X):
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = X.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    mean_std = []

    for column in range(columns):
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        mean_std.append((mean, std))
        tempArray = np.empty(0)
        for element in X[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
        standardizedArray[:,column] = tempArray
    
    return standardizedArray, mean_std


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--draw")
    parser.add_argument("--percent")
    return parser.parse_args()


def load_data(infile, with_time=False):
    data = {}
    
    with open(infile, newline="") as ifile:
        reader = csv.reader(ifile)
        columns = next(reader)
        data.update({
            key: []
            for key in columns if key != ''
        })
        for idx, line in enumerate(reader):
            for index, item in enumerate(line):
                if index:
                    try:
                        data[columns[index]].append(float(item))
                    except:
                        data[columns[index]].append(item)
                elif with_time and not index:
                    timestamp = time\
                        .mktime(
                                datetime\
                                    .datetime\
                                    .strptime(item, "%Y-%m-%d")\
                                    .timetuple()
                        )
                    data[columns[index]].append(timestamp)
            if idx == 500:
                break
    return data


def main():
    args = parse_arguments()

    infile = None
    if hasattr(args, 'infile') and args.infile:
        infile = args.infile
    
    percent = 10
    if hasattr(args, 'percent') and args.percent:
        percent = args.percent

    percent_prog = int(percent) / 100
    percent_test = 1 - percent_prog

    other_file = False

    data = {}
    if infile:
        data = load_data(infile, other_file)

    if not data:
        print("data not loaded")
    
    if other_file:
        len_train_data = int(len(data['Volume']) * percent_test)
        horizon = len(data['Volume']) - len_train_data
        train_data = data['Volume'][:len_train_data]

        t_data = SSA(list(zip(data['Date'], train_data)))
    else:
        # data = {
        #     'Volume': [d if d != '' else 0.0 for d in data['LateAircraftDelay']],
        #     'Date': list(range(len(data['LateAircraftDelay'])))
        # }
        # len_train_data = int(len(data['Volume']) * percent_test)
        # horizon = len(data['Volume']) - len_train_data
        # train_data = data['Volume'][:len_train_data]
        # t_data = SSA(list(zip(data['Date'], train_data)))

        import math
        data = {'Date': [], 'Volume': []}
        for i in range(800):
            data['Volume'].append(math.sin(math.pi/32 * i))
            data['Date'].append(i)
        len_train_data = 320 # int(len(data['Volume']) * percent_test)
        horizon = len(data['Volume']) - len_train_data
        train_data = data['Volume'][:len_train_data]
        t_data = SSA(list(zip(data['Date'], train_data)))

    mat_data, n, N = t_data.restrict()
    
    mat_data, mean_std = standardize_data(mat_data.data)
    cov_mat = Matrix(np.cov(mat_data.T))

    lambdas, vectors = cov_mat.getSelfMetrics()

    # first_non_negative = -1
    # for idx, l in enumerate(lambdas):
    #     if l >= 0:
    #         first_non_negative = idx
    #         break
    
    # lambdas, vectors = lambdas[first_non_negative:], vectors[first_non_negative:]

    sum_lambdas = sum(lambdas)
    variance_explained = []
    for i in lambdas:
        variance_explained.append((i/sum_lambdas)*100)

    cumulative_variance_explained = np.cumsum(variance_explained)

    count_of_main_comp = -1
    for idx, num in enumerate(cumulative_variance_explained):
        if num >= 80:
            count_of_main_comp = idx
            break

    v_r = (vectors.T[:][:count_of_main_comp]).T # [:][:count_of_main_comp + 1]

    pca = mat_data @ v_r
    r = N - n

    # equal_matrix = pca @ v_r.T #v_r @ pca.T # pca @ v_r.T # TODO
    equal_matrix = v_r @ pca.T

    recovered_time_series = []
    st = min(n, r)
    for s in range(1, st): # 1/s * sum([equal_matrix[i, s - i + 1] for i in range(s + 1)]) -> 0<=s<=r-1
        t_sum = []
        for i in range(1, s):
            mean, std = mean_std[i]
            t_sum.append(equal_matrix[i][s - i] * std + mean)

        element = 1.0/s * sum(t_sum)
        recovered_time_series.append(element)
    

    en = max(n, r)

    for s in range(st, en + 1): # 1/r * sum([equal_matrix[i, s - i + 1] for i in range(r)]) -> r-1<=s<=n-1
        t_sum = []
        for i in range(1, st):
            mean, std = mean_std[i]
            t_sum.append(equal_matrix[i][s - i] * std + mean)
        
        element = 1.0/st * sum(t_sum)
        recovered_time_series.append(element)
    
    for s in range(en + 1, N + 1): # 1/(N - s + 1) * sum([equal_matrix[i + s - n, n - i + 1] for i in range(1, N - s + 1)]) -> n-1<=s<=N-1
        t_sum = []
        for i in range(1, N - s + 1):
            mean, std = mean_std[i + (s - n) - 1]
            t_sum.append(equal_matrix[i + (s - n) - 1][n - i] * std + mean)
        
        element = 1/(N - s + 1) * sum(t_sum)
        recovered_time_series.append(element)

    plt.plot(t_data.getKeys(), t_data.getValues(), color="red")
    plt.plot(t_data.getKeys(), np.array(recovered_time_series), color="blue")
    plt.show()

    predict = []
    q = recovered_time_series[N - r + 1:]
    v_r_1 = v_r[:-1][:]
    # p1 = v_r @ v_r.T
    # p2 = np.linalg.inv(1 - v_r @ v_r[:-1].T)
    v_r_inv = 1 - np.linalg.inv(v_r_1.T @ v_r_1)
    coeff = v_r @ v_r_inv @ v_r_1.T # p1 @ p2
    ln_lambdas = np.array(np.log(lambdas))
    plt.plot(list(range(len(ln_lambdas))), ln_lambdas)
    plt.show()

    # plt.plot(t_data.getKeys(), t_data.getValues(), "r-")
    

    start_index = N - r + 1 if N % 2 == 0 else N - r
    
    while True:
        q = np.array(recovered_time_series[start_index:])
        predict = []
        try:
            t = int(input("t=")) # 132/124/101     400/20 sin
                                 # 34/43/46/66/82/113              400/20 cos
            # t = 180 #count_of_main_comp - 61 # ????? # 97 for sin 400 func
        except:
            break
        print(count_of_main_comp)
        right_side = horizon
        for i in range(right_side):
            elem = coeff @ q
            element = elem[t] # elem[count_of_main_comp]
            predict.append(element)
            q = np.append(q, element)
            q = q[1:]
        predict = np.array(predict)
        plt.plot(t_data.getKeys(), t_data.getValues(), "r-")
        plt.plot(data['Date'][:len_train_data], np.array(recovered_time_series), "b--")
        plt.plot(data['Date'][len_train_data:len_train_data + right_side], predict, "b--")
        plt.plot(data['Date'][len_train_data:], data['Volume'][len_train_data:], "r--")    
        plt.show()
    # norm_data, _ = t_data.normalize_data()
    # norm_data = np.array(norm_data)
    # plt.plot(norm_data[:,0], norm_data[:,1], color="red")
    # plt.plot(norm_data[:,0], np.array(recovered_time_series), color="blue")
    # plt.plot(t_data.getKeys(), t_data.getValues(), "r-")
    # plt.plot(t_data.getKeys(), recovered_time_series)
    # plt.plot(t_data.getKeys(), t_data.getValues(), "r-")
    # plt.plot(data['Date'][len_train_data - 1:], predict, "b--")
    # plt.plot(data['Date'][len_train_data - 1:], data['Volume'][len_train_data - 1:], "r--")
    # plt.show()
main()