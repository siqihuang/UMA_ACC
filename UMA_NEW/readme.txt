way to compile file in Linux Ubuntu

1 compile cpp file
2 use swig(already installed) to compile and get a cpp interface file
3 use gcc to mix compile the cpp in 2
4 use cuda compiler(nvcc) to compile cuda file and link all .o file and get .so

Say if I have Agent.cpp and Snapshot.cpp and kernel.cu

g++ -fPIC -std=c++11 -c Agent.cpp -o Agent.o//get the agent obj
g++ -fPIC -std=c++11 -c Snapshot.cpp -o Snapshot.o//get the Snapshot obj
swig -c++ -python -o UMA_NEW_wrap.cpp UMA_NEW.i//swig auto generate the UMA_NEW_wrap.cpp based on UMA_NEW.i, the UMA_NEW.i is like a interface protocal, I will maintain it.
gcc -fPIC -c UMA_NEW_wrap.cpp -o UMA_NEW_wrap.o -I/usr/include/python2.7//use python and gcc c compiler to mix compile
nvcc -shared -Xcompiler -fPIC kernel.cu Agent.o Snapshot.o UMA_NEW_wrap.o -o _UMA_NEW.so//use nvcc to compile cuda file and link all obj together

note: 
1 -fPIC is used to craete module with no specified location in memory(Position-Independent-Code)
2 -shared means use shared library
3 -Xcompiler is a tag in nvcc
4 all the library is installed(python2.7,nvcc,swig)
5 the name of the .so is _UMA_NEW.so and that really matters, if name changes, python may not find the module