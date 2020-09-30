import numpy as np


class Utils:
    @staticmethod
    def to_one_hot(a, len_):
        b = np.zeros((a.size, len_))
        b[np.arange(a.size), a] = 1
        return b

    @staticmethod
    def to_one_hot_obs(obs, len_):
        return np.array([Utils.to_one_hot(np.array(x), len_) for x in obs]).reshape(-1, len_ * 2)

    @staticmethod
    def to_one_hot_flatten(a, len_):
        b = np.zeros((a.shape[0], len_ * len_))

        def to_idx(el):
            return el[0] * el[1] + el[1]

        idx = np.apply_along_axis(to_idx, axis=1, arr=a)
        b[np.arange(a.shape[0]), idx] = 1
        return b

    @staticmethod
    def to_one_hot_flatten_obs(obs, len_):
        return Utils.to_one_hot_flatten(np.array(obs), len_)


class UtilsRooms:
    @staticmethod
    def to_one_hot(a, len_, rooms):
        b = np.zeros((1, len_ * 2 + rooms))
        b[0, [a[0], a[1] + len_, a[2] + len_ * 2]] = 1
        return b

    @staticmethod
    def to_one_hot_obs(obs, len_, rooms):
        return np.array([UtilsRooms.to_one_hot(obs, len_, rooms)])

    @staticmethod
    def to_one_hot_flatten(a, len_):
        b = np.zeros((a.shape[0], len_ * len_))

        def to_idx(el):
            return el[0] * el[1] + el[1]

        idx = np.apply_along_axis(to_idx, axis=1, arr=a)
        b[np.arange(a.shape[0]), idx] = 1
        return b

    @staticmethod
    def to_one_hot_flatten_obs(obs, len_):
        return Utils.to_one_hot_flatten(np.array(obs), len_)
