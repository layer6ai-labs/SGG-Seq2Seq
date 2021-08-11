import pickle


def load_object_from_file(filename):
    with open(filename, 'rb') as file_in:
        return pickle.load(file_in)


def dump_object_to_file(data, filename):
    with open(filename, 'wb+') as file_out:
        pickle.dump(data, file_out)
