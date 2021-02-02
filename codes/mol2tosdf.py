import os
import multiprocessing
import argparse
path_marker = '/'


def mol2tosdf(file):
    try:
        sdf_file = sdfpath+path_marker+file.split('/')[-1].replace('.mol2', '.sdf')
        cmdline = 'babel -imol2 %s -osdf %s -h' % (file, sdf_file)
        os.system(cmdline)
    except:
        print('converting error for %s'%file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mol2path', type=str, default='./examples/mol2_files', help="the path for storing mol2 files")
    argparser.add_argument('--sdfpath', type=str, default='./examples/sdf_files', help="the  path for storing converted sdf files")
    argparser.add_argument('--num_process', type=int, default=12, help="the number of process for converting molecule format")
    args = argparser.parse_args()
    mol2path, sdfpath, num_process = args.mol2path, args.sdfpath, args.num_process
    if not os.path.exists(sdfpath):
        os.makedirs(sdfpath)

    entries = os.listdir(mol2path)
    files = [mol2path + path_marker + entry for entry in entries]
    pool = multiprocessing.Pool(24)
    pool.starmap(mol2tosdf, zip(files))
    pool.close()
    pool.join()
