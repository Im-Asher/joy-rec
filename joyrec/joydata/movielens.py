import re
import zipfile
import numpy as np

from torch.utils.data import Dataset
from joydata.common.utils import _check_exists_and_download

URL = 'https://dataset.bj.bcebos.com/movielens%2Fml-1m.zip'
MD5 = 'c4d9eecfca2ab87c1945afe126590906'

__all__ = []

age_table = [1, 18, 25, 35, 45, 50, 56]


class MovieInfo():
    """
    Movie id, title and categories information are stored in MovieInfo.
    """

    def __init__(self, index, categories, title):
        self.index = int(index)
        self.categories = categories
        self.title = title

    def value(self, categories_dict, movie_title_dict):
        """
        Get information from a movie.
        """
        return [
            [self.index],
            [categories_dict[c] for c in self.categories],
            [movie_title_dict[w.lower()] for w in self.title.split()],
        ]

    def __str__(self):
        return "<MovieInfo id(%d), title(%s), categories(%s)>" % (
            self.index,
            self.title,
            self.categories,
        )

    def __repr__(self):
        return self.__str__()


class UserInfo:
    """
    User id, gender, age, and job information are stored in UserInfo.
    """

    def __init__(self, index, gender, age, job_id):
        self.index = int(index)
        self.is_male = gender == 'M'
        self.age = age_table.index(int(age))
        self.job_id = int(job_id)

    def value(self):
        """
        Get information from a user.
        """
        return [
            [self.index],
            [0 if self.is_male else 1],
            [self.age],
            [self.job_id],
        ]

    def __str__(self):
        return "<UserInfo id(%d), gender(%s), age(%d), job(%d)>" % (
            self.index,
            "M" if self.is_male else "F",
            age_table[self.age],
            self.job_id,
        )

    def __repr__(self):
        return str(self)


class Movielens(Dataset):
    def __init__(self,
                 data_file: str=None,
                 mode: str = 'train',
                 test_ratio: float = 0.2,
                 rand_seed: int = 0,
                 download=True) -> None:
        assert mode.lower() in [
            "train", "test"], f"mode should be 'test' & 'train',but got {mode}"

        self.data_file = data_file

        self.mode = mode
        if self.data_file is None:
            assert(download),"data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(data_file, URL, MD5, 'sentiment', download)

        self.split_ratio = test_ratio
        self.rand_seed = rand_seed

        self._load_meta_info()
        self._load_data()

    def _load_meta_info(self):
        pattern = re.compile(r'^(.*)\((\d+)\)$')
        self.movie_info = {}
        self.movie_title_dict = {}
        self.categories_dict = {}
        self.user_info = {}
        with zipfile.ZipFile(self.data_file) as package:
            for info in package.infolist():
                assert isinstance(info, zipfile.ZipInfo)
                title_word_set = set()
                categories_set = set()
                with package.open('ml-1m/movies.dat') as movie_file:
                    for i, line in enumerate(movie_file):
                        line = line.decode(encoding='latin')
                        movie_id, title, categories = line.strip().split('::')
                        categories = categories.split('|')
                        for c in categories:
                            categories_set.add(c)
                        title = pattern.match(title).group(1)
                        self.movie_info[int(movie_id)] = MovieInfo(
                            index=movie_id, categories=categories, title=title
                        )
                        for w in title.split():
                            title_word_set.add(w.lower())

                for i, w in enumerate(title_word_set):
                    self.movie_title_dict[w] = i

                for i, c in enumerate(categories_set):
                    self.categories_dict[c] = i

                with package.open('ml-1m/users.dat') as user_file:
                    for line in user_file:
                        line = line.decode(encoding='latin')
                        uid, gender, age, job, _ = line.strip().split("::")
                        self.user_info[int(uid)] = UserInfo(
                            index=uid, gender=gender, age=age, job_id=job
                        )

    def _load_data(self):
        self.data = []

        is_test = self.mode == "test"

        with zipfile.ZipFile(self.data_file) as file:
            with file.open("ml-1m/ratings.dat") as rating:
                for line in rating:
                    line = line.decode(encoding='latin')
                    if (np.random.random() < self.split_ratio) == is_test:
                        uid, mov_id, rating, _ = line.strip().split("::")
                        uid = int(uid)
                        mov_id = int(mov_id)
                        rating = float(rating) * 2 - 5.0

                        mov = self.movie_info[mov_id]
                        user = self.user_info[uid]
                        self.data.append(
                            user.value()+mov.value(self.categories_dict, self.movie_title_dict)+[[rating]])

    def __getitem__(self, index):
        data = self.data[index]
        return tuple([np.array(d) for d in data])

    def __len__(self):
        return len(self.data)
