import abc

class DissimilarityModel(abc.ABC):
    @abstractmethod
    def get_realization():
        pass

    @abstractmethod
    def mean_variance_along_path():
        pass