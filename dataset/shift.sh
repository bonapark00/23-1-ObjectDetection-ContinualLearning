#python3 -m pip install -r ./requirements.txt

# Example commands to downaload from all 1fps images and convert them into hdf5.
#python download_shift.py --view "front" --group "[img, det_2d]" --split "all" --framerate "[images]" --shift "discrete" dataset/SHIFT_dataset/
python -m utils.shift_dev.io.to_hdf5 "dataset/SHIFT_dataset/**/*.zip" --zip -j 4