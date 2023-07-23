import os
import sys
import httpx
import shutil
import hashlib

from tqdm import tqdm


HOME = os.path.expanduser('~')

DATA_HOME = os.path.join(HOME, '.cache', 'joyrec', 'dataset')


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url: str, module_name, md5sum, save_name=None):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(
        dirname, url.split('/')[-1] if save_name is None else save_name
    )

    if os.path.exists(filename) and md5file(filename) == md5sum:
        return filename

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write(f"file {md5file(filename)}  md5 {md5sum}\n")
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError(
                "Cannot download {} within retry limit {}".format(
                    url, retry_limit
                )
            )
        sys.stderr.write(
            f"Cache file {filename} not found, downloading {url} \n"
        )
        sys.stderr.write("Begin to download\n")
        try:
            # (risemeup1):use httpx to replace requests
            with httpx.stream(
                "GET", url, timeout=None, follow_redirects=True
            ) as r:
                total_length = r.headers.get('content-length')
                if total_length is None:
                    with open(filename, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                else:
                    with open(filename, 'wb') as f:
                        chunk_size = 4096
                        total_length = int(total_length)
                        total_iter = total_length / chunk_size + 1
                        log_interval = (
                            total_iter // 20 if total_iter > 20 else 1
                        )
                        log_index = 0
                        bar = tqdm(
                            total_iter, desc='item', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|'
                        )
                        for data in r.iter_bytes(chunk_size=chunk_size):
                            f.write(data)
                            log_index += 1
                            bar.update(log_index, {})
                            if log_index % log_interval == 0:
                                bar.update(log_index)

        except Exception as e:
            # re-try
            continue
    sys.stderr.write("\nDownload finished\n")
    sys.stdout.flush()
    return filename

def _check_exists_and_download(path, url, md5, module_name, to_download=True):
    if path and os.path.exists(path):
        return path

    if to_download:
        return download(url, module_name, md5)
    else:
        raise ValueError(f'{path} not exists and auto download disabled')