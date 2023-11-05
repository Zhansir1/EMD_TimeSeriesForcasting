import numpy as np


def SE(
        nums: list,  # 传入的时间序列对象,以list的形式传入，后续使用Numpy进行处理
        r: float = 0.2,  # 每一个判定为与原始窗口相同的阈值
        m: int = 2  # 每一个滑动小窗口的窗口大小，默认为2
      ) -> float:

    n = len(nums)
    threshold = r * np.std(nums)

    def get_distance(nums1: list, nums2: list) -> int:
        res = 0
        for num1, num2 in zip(nums1, nums2):
            res = max(res, np.abs(num1 - num2))
        return res

    def calculate(window_size: int) -> float:
        slices = [nums[i:i + window_size] for i in range(n - window_size + 1)]  # 一共会有n-m+1个小窗口
        B = [0] * (n - window_size + 1)

        for i in range(n - window_size + 1):
            for j in range(n - window_size + 1):
                if i != j:
                    dis_ij = get_distance(slices[i], slices[j])
                    if dis_ij <= threshold:
                        B[i] += 1

        B = [num / (n - window_size) for num in B]
        return sum(B) / (n - window_size + 1)

    return -np.log(calculate(m + 1) / calculate(m))


def FSE(
        nums: list,  # 传入的时间序列对象,以list的形式传入，后续使用Numpy进行处理
        n: int,  # 计算模糊隶属度函数时内部的次方
        r: float = 0.2,  # 每一个判定为与原始窗口相同的阈值
        m: int = 2  # 每一个滑动小窗口的窗口大小，默认为2
      ) -> float:

    N = len(nums)

    def get_distance(nums1: list, nums2: list) -> int:
        mean1, mean2 = np.mean(nums1), np.mean(nums2)
        res = 0
        for num1, num2 in zip(nums1, nums2):
            res = max(res, np.abs((num1 - mean1) - (num2 - mean2)))
        return res

    def calculate(window_size: int) -> float:
        slices = [nums[i: i+window_size] for i in range(0, N - window_size + 1)]
        B = [0] * (N - window_size + 1)

        for i in range(N - window_size + 1):
            for j in range(N - window_size + 1):
                if i != j:
                    dis_ij = np.exp(-(get_distance(slices[i], slices[j]) ** n) / r)
                    B[i] += dis_ij

        B = [num / (N - window_size - 1) for num in B]
        return sum(B) / (N - window_size + 1)

    return -np.log(calculate(m+1) / calculate(m))

