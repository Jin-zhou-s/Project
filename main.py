import os.path

from Py_Class import Data_process


def main():
    data_process = Data_process()
    data_process.download_and_unzipped_file()
    lin = data_process.read_directory()
    lin1 = data_process.read_directory(lin[0])
    path = os.path.join(lin[0], lin1[0])
    data_process.read_directory(path)


if __name__ == '__main__':
    main()
