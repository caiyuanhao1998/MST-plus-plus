import argparse
import os
import glob
import zipfile
import numpy as np
from tqdm import tqdm
import h5py
import hdf5storage

NOISE = 750
JPEG_QUALITY = 65
ANALOG_CHANNEL_GAIN = np.array([2.2933984, 1, 1.62308182])
TYPICAL_SCENE_REFLECTIVITY = 0.18
MAX_VAL_8_BIT = (2 ** 8 - 1)
MAX_VAL_12_BIT = (2 ** 12 - 1)
SIZE = 512
QUARTER = SIZE // 4
CROP = np.s_[QUARTER:-QUARTER, QUARTER:-QUARTER]  # keep only the center 50% of the image
SUBMISSION_SIZE_LIMIT = 5*10**8  # (500MB)

def loadCube(path):
    with h5py.File(path, 'r') as mat:
        cube = np.array(mat['cube']).T
    return cube

def saveCube(path, cube, bands=None, norm_factor=None):
    fields = {'cube': cube}
    if bands is not None:
        fields['bands'] = bands
    if norm_factor is not None:
        fields['norm_factor'] = norm_factor
    hdf5storage.savemat(path, fields, format='7.3', store_python_metadata=False)


def main(argv=None):
    # Argument parser
    parser = argparse.ArgumentParser(description="NTIRE2022 Spectral Submission Prep Utility")

    parser.add_argument('-i', '--in_dir',    help='Input directory for the evaluated images', required=True)
    parser.add_argument('-o', '--out_dir',    help='Empty output directory for the evaluated images (will be created)', required=True)
    parser.add_argument('-k', '--keep', help="Keep temporary files", action='store_true', default=False)

    args = parser.parse_args(argv)

    out_dir = args.out_dir
    in_dir = args.in_dir
    keep = args.keep

    print(in_dir)
    print(out_dir)

    input_files = sorted(glob.glob(f'{in_dir}/*.mat'))
    if not input_files:
        raise FileNotFoundError(f'No reconstructed MAT files found in {in_dir}')

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over files
    print("Cropping files from input directory")
    for file in tqdm(input_files):

        # Load file
        cube = loadCube(file)

        # Crop center area
        cube = cube[CROP]

        # Save cropped file
        saveCube(os.path.join(out_dir, f'{os.path.basename(file)[:-4]}_crop.mat'), cube)

    # Compress cropped files
    print("Compressing submission")
    outfile = os.path.join(out_dir, f'submission.zip')
    with zipfile.ZipFile(outfile, mode="w", compression=zipfile.ZIP_DEFLATED) as zip:
        for file in tqdm(glob.glob(f'{out_dir}/*_crop.mat')):
            zip.write(file, os.path.basename(file))

    # Remove cropped files
    if not keep:
        print("Removing temporary files")
        for file in tqdm(glob.glob(f'{out_dir}/*_crop.mat')):
            os.remove(file)

    # Verify that archive is < 500MB

    if os.path.getsize(outfile) > SUBMISSION_SIZE_LIMIT:
        print("Verifying submission size - ERROR:")
        print("Submission over 500MB and unlikely to be accepted by CodaLab platform")
    else:
        print("Verifying submission size -  SUCCESS!")
        print(f'Submission generated @ {outfile}')


if __name__ == "__main__":
    main()
