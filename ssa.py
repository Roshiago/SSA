import numpy as np
import argparse
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

    def getXByIndex(self, index) -> list:
        sqrt_lI, uI, vI = self.getEigentripleByIndex(index)

        if not self._X.get(index):
            result = np.dot(uI, vI) * sqrt_lI
            self._X[index] = result

        return self._X[index]
    
    def getEigentripleByIndex(self, index):
        sqrt_lI = math.sqrt(self._lambdas[index])
        uI = self._u[index]
        vI = self._v[index]
        return (sqrt_lI, uI, vI)
    
    def getEigentripleMatrixByIndex(self, index, matrixT):
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
    
    def getTranspose(self):
        return Matrix(self._data.transpose())

    def getOutcom(self, n=1):
        result = self._data @ self._data.transpose()
        result *= 1.0/n
        return Matrix(result)
    
    def getSelfMetrics(self):
        return np.linalg.eigh(self._data)

    def __str__(self):
        return str(self._data)

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
        self._max_value = None
        if normalize:
            self._data, _ = self.normalize_data()
    
    def getKeys(self) -> np.array:
        return self._data[:,0]
    
    def getValues(self) -> np.array:
        return self._data[:,1]

    def normalize_data(self):
        data = self.getValues()
        keys = self.getKeys()
        self._max_value = np.array(data).max()
        normalized = data / self._max_value
        normalized = np.array(list(zip(keys, normalized)))
        return normalized, self._max_value

    def restrict(self): # -> Matrix:
        values = self.getValues()
        N = len(values)
        m = (N + 1) / 2
        m = int(math.floor(m))
        n = N - m
        new_sample = []
        for i in range(m):
            new_sample.append(values[i:n+i])
        
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--draw")
    parser.add_argument("--percent")
    return parser.parse_args()


def load_data(infile):
    data = {}
    
    with open(infile, newline="") as ifile:
        reader = csv.reader(ifile)
        columns = next(reader)
        data.update({
            key: []
            for key in columns
        })
        for line in reader:
            for index, item in enumerate(line):
                if index:
                    data[columns[index]].append(float(item))
                else:
                    timestamp = time\
                        .mktime(
                                datetime\
                                    .datetime\
                                    .strptime(item, "%Y-%m-%d")\
                                    .timetuple()
                        )
                    data[columns[index]].append(timestamp)
    
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
    # outfile = None
    # if hasattr(args, 'outfile') and args.outfile:
    #     outfile = args.outfile

    data = {}
    if infile:
        data = load_data(infile)

    draw = False
    if hasattr(args, 'draw') and args.draw:
        draw = bool(args.draw)
    
    
    ssa = None
    source_data = None
    len_source = None
    last_index = None
    real_data = None
    norm_pred = None
    mean_pred = None
    if data:
        len_source = len(data['Volume'])
        last_index = int(len_source * percent_test)
        data_pred = np.array(data['Volume'][:last_index])
        real_data = np.array(list(zip(data['Date'], data['Volume'])))

        # import math
        # sin_value = [math.sin(i/32*math.pi) for i in range(500)]
        # real_data = np.array(list(zip(data['Date'], sin_value)))
        # len_source = len(real_data)
        # last_index = int(len_source * percent_test)
        # data_pred = np.array(sin_value[:last_index])

        norm_pred = np.linalg.norm(data_pred)
        mean_pred = np.mean(data_pred)
        data_pred -= mean_pred
        data_pred /= norm_pred

        source_data = list(zip(data['Date'], data_pred))
        # len_source = len(source_data)
        # last_index = int(len_source * percent_test)
        # ssa = SSA(source_data[:last_index])
        ssa = SSA(source_data)
        normed_data = SSA(real_data)
        source_data = SSA(source_data)
    
    # def createVi(matrixT, uI, lambdaI):
    #     return np.dot(matrixT, uI) / math.sqrt(lambdaI)
    
    recovered_time_series = None
    n = None
    down_score = None
    if ssa:
        matrix, n, N = ssa.restrict()
        s_matrix = matrix.getOutcom(n)
        # matrixT = matrix.getTranspose()
        lambdas, vectors = s_matrix.getSelfMetrics()
        # filtered_lambdas = [l if l >= 0 else -1 for l in lambdas]
        # r = np.where(filtered_lambdas == np.amax(filtered_lambdas))
        # if r:
        #     r = r[0][0] + 1
        # else:
        #     raise Exception("lambdas not found")
        
        lambdas_vectors = list(zip(lambdas, vectors))
        lambdas_vectors = np.array(sorted(lambdas_vectors, key=lambda item: item[0], reverse=True))
        lambdas, vectors = lambdas_vectors[:,0], lambdas_vectors[:,1]
        lambdas = np.array(list(lambdas))
        vectors = np.array(list(vectors))
        ln_lambdas = sorted(np.log(lambdas), reverse=True)
        last_positive_index = len(list(filter(lambda item: item >= 0, lambdas))) - 1
        plt.plot(list(range(len(ln_lambdas))),ln_lambdas)
        plt.show()

        # sqrt_lambdas = np.sqrt(np.array(list(lambdas)))
        y_main = vectors @ matrix._data
        # y_main = (y_main.T * sqrt_lambdas).T
        equal_matrix = vectors @ y_main
        # filtered_lambdas = len(filter(lambda item: item >= 0, lambdas))
        # r = len(list(filter(lambda item: item >= 0, lambdas)))
        r = equal_matrix.shape[0] - 1 
        # r = 31
        
        recovered_time_series = []
        for s in range(1, r): # 1/s * sum([equal_matrix[i, s - i + 1] for i in range(s + 1)]) -> 0<=s<=r-1
            recovered_time_series.append(1.0/s * sum([equal_matrix[i][s - i] for i in range(s)]))
        
        for s in range(r - 1, n): # 1/r * sum([equal_matrix[i, s - i + 1] for i in range(r)]) -> r-1<=s<=n-1
            recovered_time_series.append(1.0/r * sum([equal_matrix[i][s - i] for i in range(r)]))
        
        for s in range(n - 1, N - 1): # 1/(N - s + 1) * sum([equal_matrix[i + s - n, n - i + 1] for i in range(1, N - s + 1)]) -> n-1<=s<=N-1
            recovered_time_series.append(1.0/(N - s + 1) * sum([equal_matrix[i + s - (n - 1)][(n - 1) - i] for i in range((N - 1) - s)]))

        # vectors_r_minus_one = vectors[:-1, :]
        # upper_score = vectors @ vectors_r_minus_one.T
        # down_score = 1 - vectors @ vectors.T
        # coeff = upper_score.T @ down_score

        # -------win condition-------
        v_star = y_main[:r, :]
        down_score = v_star.T @ v_star
        donw_score_inverse = np.linalg.inv(down_score)
        coeff = y_main @ donw_score_inverse @ v_star.T
        # ----------------------------
        # v_star = y_main[:main_vectors_num, :]
        # down_score = v_star.T @ v_star
        # donw_score_inverse = np.linalg.inv(down_score)
        # coeff = y_main @ donw_score_inverse @ v_star.T

        # v_vectors = []
        # for uI, lI in zip(vectors, lambdas):
        #     v_vectors.append(createVi(matrixT._data, uI, lI))
        
        # dataSSA = DataSSA(
        #     matrix._data, s_matrix, vectors, lambdas, v_vectors
        # )
        # groupsX = dataSSA.getGroups()
        # print(groupsX)
        # x_groups = {}
        # for group in groupsX:
        #     result = sum([dataSSA.getEigentripleMatrixByIndex1(i) for i in group])
        #     x_groups[group] = result # SSA(list(zip(range(len(result)), result)))
    
        # keys = list(x_groups.keys())
        # count_group = len(keys) # m - length of set splited indexes
        # print(count_group)
        # print(x_groups[keys[0]])
        # E = np.eye(index_d + 1)
        # result_mat = np.dot(x_groups[keys[0]], E)
        # for i in range(1, len(keys)):
        #     result_mat += np.dot(x_groups[keys[i]], E)
        # for i in range(5):
        #     print(x_groups[keys[i]])
        # print(x_groups[keys[0]])
        # ssa_matrix = x_groups[keys[0]].restrict()
        # print(ssa_matrix)
        # print(len(ssa_matrix._data[0]))
        # print(result_mat)
        # print(ssa.getValues()[:index_d+1])
        # print(matrix._data)
        # print(dataSSA.getXByIndex(0))
        # print(dataSSA.getXByIndex(1))

    if draw:
        x_ = ssa.getKeys()
        # y = ssa.getValues()
        recover_time_series = np.array(recovered_time_series) * norm_pred
        recover_time_series += mean_pred
        y_ = recover_time_series  # np.array(recover_time_series)
        # plt.plot(x_, y)
        plt.plot(x_, y_, color="red")
        #plt.show()
        src_data = normed_data.getValues()
        prod_src_data = source_data.getValues()
        # prog_data = prod_src_data[last_index - 1:] # source_data[last_index:]
        # q_vector = np.array(src_data[len_source - r - 1:])[:-1] # np.array(source_data[len_source - r - 1:])[:-1,1]
        # q_vector = np.array(prod_src_data[len_source - r:])# np.array(src_data[len_source - r:])
        q_vector = np.array(prod_src_data[-r:])
        q_vector *= norm_pred
        q_vector += mean_pred
        # for i in range(len_prog_data):
        prog = coeff * q_vector # np.dot(upper_score, q_vector) / down_score 
        # prog_data_vec = prog[-1]

        # prog_data = np.array(prog_data)
        x = normed_data.getKeys()
        x_ = x[last_index - 1:] # prog_data[:, 0]
        len_x = len(x_)
        # y = prog_data[:, 1]
        # y_ = prog[-1]
        # # draw predict
        # plt.plot(x_, y)
        # plt.plot(x_, y_[:len_x])
        # plt.show()
        swith_mode = 1
        last_index_prog = len(y_) - 1
        if swith_mode == 1:
            offset = 0 # r // 4
            # disp = sorted([(index, np.std(prog_i)) for index, prog_i in enumerate(coeff) if index <= last_positive_index], key=lambda i: i[1])
            # n_components = disp[0][0]
            # print(n_components)
            for i in range(len_source - last_index + offset):
                # prog = 1/r * sum(prog[-1])
                n_components = 54 # 54 # 43
                # prog = sum([sum(prog_i) for prog_i in prog[:n_components]])
                prog = sum(prog[n_components])
                #prog = np.mean([sum(prog_i) for prog_i in prog]) # ???
                
                y_prog = prog
                

                y_ = np.append(y_, y_prog)

                q_vector = np.append(q_vector, y_prog)
                q_vector = q_vector[1:]
                prog = coeff * q_vector

            len_y_ = len(y_)
            # print(y_[last_index] + src_data[last_index])
            # print(mean_pred)
            plt.plot(x, src_data, color="black")
            
            # plt.plot(x[last_index - 1:len_y_], y_[last_index:], color="blue", linestyle='dashed')
            prog_series = y_[last_index_prog:]
            len_prog_series = len(prog_series)
            # offset = 5
            x_ = x[-len_prog_series:]
            plt.plot(x_[:], prog_series, color="blue", linestyle='dashed')
        else:
            y_ = prog[-1]
            len_y_ = len(y_)
            #source_data = np.array(source_data)
            #plt.plot(source_data[:, 0], source_data[:, 1], color="black")
            plt.plot(x, src_data, color="black")
            if len_x > len_y_:
                plt.plot(x_[:len_y_], y_, color="blue", linestyle='dashed')
            else:
                plt.plot(x_, y_[:len_x], color="blue", linestyle='dashed')
        plt.show()


def main1():
    args = parse_arguments()

    infile = None
    if hasattr(args, 'infile') and args.infile:
        infile = args.infile
    
    percent = 10
    if hasattr(args, 'percent') and args.percent:
        percent = args.percent

    percent_prog = int(percent) / 100
    percent_test = 1 - percent_prog

    data = {}
    if infile:
        data = load_data(infile)

    draw = False
    if hasattr(args, 'draw') and args.draw:
        draw = bool(args.draw)
    
    
    data_pred = []
    if data:
        data_pred = np.array(data['Volume'])
    else:
        import math
        data_pred = np.array([math.sin(i/32*math.pi) for i in range(500)])
    
    norm_pred = np.linalg.norm(data_pred)
    data_pred -= np.mean(data_pred)
    data_pred /= norm_pred
    if data:
        source_data = np.array(list(zip(data['Date'], data_pred)))
    else:
        source_data = np.array(list(zip(range(len(data_pred)), data_pred)))

    len_source = len(source_data)
    last_index = int(len_source * percent_test)
    x = source_data[:, 0]

    r, y_main, recovered_time_series, lambdas, vectors = SSA.createSSA(source_data[:last_index])
    len_of_recovered = len(recovered_time_series)
    
    recovered_series_smooth = SSA.smoothData(recovered_time_series, 10)
    len_of_smooth_series = len(recovered_series_smooth)
    

    if draw:
        plt.plot(x[:len_of_smooth_series],recovered_series_smooth, color='red')

    prog_ssa = SSA.calculateNextValues(source_data[:, 1], y_main, r, len_source - last_index)

    prog_ssa = SSA.smoothData(prog_ssa, 10)

    y_ = np.array(recovered_series_smooth[:])

    y_ = np.append(y_, prog_ssa)
    src_data = source_data[:, 1] 

    len_y_ = len(y_)
    if draw:
        plt.plot(x, src_data, color="black")
        plt.plot(x[last_index:len_y_], y_[last_index:], color="blue", linestyle='dashed')
        plt.show()

        


if __name__ == "__main__":
    main()