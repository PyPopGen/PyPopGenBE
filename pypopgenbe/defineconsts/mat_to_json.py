import scipy.io as sio
from pathlib import Path
import sys
import json
from defineconsts.numpyencoder import NumpyEncoder

if __name__ == '__main__':
    mat_file = Path(sys.argv[1]).resolve()
    json_file = mat_file.parent.joinpath(mat_file.stem + '.json')

    mat_contents = sio.loadmat(mat_file, squeeze_me=True)
    mat_contents['__header__'] = str(mat_contents['__header__'])
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(mat_contents, f, ensure_ascii=False,
                  indent=4, cls=NumpyEncoder)
