#!/bin/bash

set -e

has_cuda=0

while : ; do
    case "$1" in
        "")
            break;;
        -use-cuda|--use-cuda)
         has_cuda=1;;
        *)
            echo "unknown option $1, try -help"
            exit 2;;
    esac
    shift
done


wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm m4-1.4.1.tar.gz



wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz



wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
make install
cd ..
rm mpfr-4.1.0.tar.xz

wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar -xvf cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
make install
cd ..
rm cddlib-0.94m.tar.gz

cd ELINA
./configure -use-deeppoly
make
make install
cd ..

wget https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz
tar -xvf gurobi9.5.0_linux64.tar.gz
cd gurobi950/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cp ../../lib/libgurobi95.so /usr/lib
cd ../..
python3 setup.py install
cd ../..

export GUROBI_HOME="$(pwd)/gurobi950/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib

ldconfig

