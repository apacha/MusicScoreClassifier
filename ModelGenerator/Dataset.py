import os
import urllib.parse as urlparse
import urllib.request as urllib2
from abc import ABC, abstractmethod

filename = download_file(url)
print(filename)


class Dataset(ABC):
    """ The abstract base class for the datasets used to train the model """

    def __init__(self,
                 directory: str):
        """
        
        :param directory: The root directory that will contain the data. Inside of this directory, the following
         subdirectories will be created: 
         
         directory
         |- training
         |   |- other
         |   |- scores
         |
         |- validation
         |   |- other
         |   |- scores
         
        """
        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            os.makedirs(os.path.join(self.directory, "training"))
            os.makedirs(os.path.join(self.directory, "training", "other"))
            os.makedirs(os.path.join(self.directory, "training", "scores"))
            os.makedirs(os.path.join(self.directory, "validation"))
            os.makedirs(os.path.join(self.directory, "validation", "other"))
            os.makedirs(os.path.join(self.directory, "validation", "scores"))

    @abstractmethod
    def is_dataset_cached_on_disk(self) -> bool:
        pass

    @abstractmethod
    def download_dataset(self):
        """ Starts the download of the dataset and extracts it into the directory specified in the constructor """
        pass

    def download_file(self, url, desc=None):
        u = urllib2.urlopen(url)
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'
        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)
                print(status, end="")
            print()

        return filename
