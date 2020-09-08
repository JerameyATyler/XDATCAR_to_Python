"""
    Authors: Emily de Stefanis, Jeramey Tyler
    Description: Convert XDATCAR files to various file formats
    Derived from :
        source url: http://home.ustc.edu.cn/~lipai/scripts/ml_scripts/xdat2xyz_unwraped.html
        author: lipai@mail.ustc.edu.cn
        description: convert XDATCAR to unwraped xyz file

"""

import json
import h5py
import numpy as np
from pathlib import Path
from copy import deepcopy

# Paths to input/output directories
input_path = Path('data')
output_path = Path('output')


def parse_file(file_name):
    # Set the scope for reading the input file
    with open(file_name, 'r') as f:
        # Skip the first line of the file since it contains system environment metadata
        next(f)

        # Fetch the first line of data, strip white spaces from the ends, and cast it to a floating point number
        scale = float(f.readline().strip())

        # Get lattice vectors
        a1 = [float(s) * scale for s in f.readline().strip().split()]
        a2 = [float(s) * scale for s in f.readline().strip().split()]
        a3 = [float(s) * scale for s in f.readline().strip().split()]

        lattice = [a1, a2, a3]

        # Get the elements names and the quantity of each
        element_names = f.readline().strip().split()
        element_numbers = [int(s) for s in f.readline().strip().split()]

        # Total number of atoms
        n_atoms = int(sum(element_numbers))
        # Array for storing and indexing atoms
        n_names = []
        # For each type of atom...
        for i in range(len(element_names)):
            # Add the appropriate number of atoms to n_names as described in element_numbers
            n_names += [element_names[i]] * element_numbers[i]

        # Read in the remaining lines of the XDATCAR file
        lines = f.readlines()

        configurations = []

        # Each configuration is preceded by a line with 'Direct configuration=    <configuration number>', so a
        # configuration is n_atoms + 1 lines longs. The loop below iterates over configurations and parses each
        # individually
        for i in range(0, len(lines), n_atoms + 1):
            configuration = []
            for j in lines[i + 1: i + 1 + n_atoms]:
                configuration.append([float(s) for s in j.strip().split()])
            configurations.append(configuration)

    return {'lattice_vectors': lattice, 'atom_names': n_names, 'atom_configurations': configurations}


def write_json(file_name, data):
    with open(file_name + '.json', 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def write_hdf5(file_name, data):
    lattice_vectors = np.asarray(data['lattice_vectors'])
    atom_names = [s.encode('ascii', 'ignore') for s in data['atom_names']]
    atom_configurations = np.asarray(data['atom_configurations'])

    f = h5py.File(file_name + '.hdf5', 'w')

    f.create_dataset('lattice_vectors', (3, 3), data=lattice_vectors)
    f.create_dataset('atom_names', (len(atom_names),), data=atom_names, dtype=h5py.string_dtype())
    f.create_dataset('atom_configurations', atom_configurations.shape, data=atom_configurations)


def write_xyz(file_name, data):
    n_atoms = len(data['atom_names'])

    atom_names = data['atom_names']
    atom_configurations = data['atom_configurations']
    lattice_vectors = data['lattice_vectors']

    # Numpy array so we can calculate the dot product
    v1 = np.array([s for s in lattice_vectors[0]])
    v2 = np.array([s for s in lattice_vectors[1]])
    v3 = np.array([s for s in lattice_vectors[2]])

    # String representation of lattice matrix
    s1 = ' '.join([str(i) for i in v1])
    s2 = ' '.join([str(i) for i in v2])
    s3 = ' '.join([str(i) for i in v3])

    v_prev = np.zeros([n_atoms, 3])
    v_next = np.zeros([n_atoms, 3])

    with open(file_name + '.xyz', 'w') as f:
        for c in atom_configurations:
            f.write('{}\n'.format(n_atoms))
            f.write('Lattice="{} {} {}"\n'.format(s1, s2, s3))

            for i in range(len(c)):
                v_next[i, :] = np.array([v for v in c[i]])

                for j in range(3):
                    if v_next[i, j] - v_prev[i, j] < -0.5:
                        v_next[i, j] += 1
                    elif v_next[i, j] - v_prev[i, j] > 0.5:
                        v_next[i, j] -= 1

                coordinates = np.dot(v_next[i], np.array([v1, v2, v3]))
                f.write('{} {} {} {}\n'.format(atom_names[i], str(coordinates[0]), str(coordinates[1]),
                                               str(coordinates[2])))

            v_prev = deepcopy(v_next)


if __name__ == '__main__':
    input_filename = str(Path(input_path / 'XDATCAR_19234-20000').absolute().resolve())
    output_filename = str(Path(output_path / 'XDATCAR_19234-20000').absolute().resolve())

    file_data = parse_file(input_filename)

    write_json(output_filename, file_data)
    write_hdf5(output_filename, file_data)
    write_xyz(output_filename, file_data)
