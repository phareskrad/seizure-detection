import os
import os.path
import shutil
from common.utils.io import load_hkl_file, save_hkl_file


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


class JSDict(dict):
    def __init__(self, data):
        self.__dict__ = data

    def __unicode__(self):
        return unicode(repr(self.__dict__))

    def __repr__(self):
        return repr(self.__dict__)


class CachedDataLoader:

    def __init__(self, dir):
        self.dir = dir
        makedirs(dir)

    # try to load data from filename, if it doesn't exist then run the func()
    # and save the data to filename
    def load(self, filename, func):
        def wrap_data(data):
            if isinstance(data, list):
                return [JSDict(x) for x in data]
            else:
                return JSDict(data)

        if filename is not None:
            filename = os.path.join(self.dir, filename)
            data = load_hkl_file(filename)
            if data is not None:
                return wrap_data(data)

            # data = load_json_file(filename)
            # if data is not None:
            #     return wrap_data(data)

        data = func()

        if filename is not None:
            save_hkl_file(filename, data)
            # save_json_file(filename, data)
        return wrap_data(data)

    @staticmethod
    def clean_cache(path, cache_type):
        if cache_type == 'all':
            shutil.rmtree(path)
            makedirs(path)
        else:
            for f in os.listdir(path):
                file_path = os.path.join(path, f)
                try:
                    if cache_type in f:
                        print file_path
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

