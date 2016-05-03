%module UMA_ACC

%include "std_vector.i"
%include "std_string.i"
%{
#include "../../UMA_Plugin/test.h"
#include "../../UMA_Plugin/snapshot_platform.h"
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
   %template(BoolVector) vector<bool>;
   %template(IntVector2D) vector<vector<int>>;
   %template(DoubleVector2D) vector<vector<double>>;
   %template(StringVector2D) vector<vector<string>>;
   %template(ConstCharVector2D) vector<vector<const char*>>;
   %template(BoolVector2D) vector<vector<bool>>;
}

%include "../../UMA_Plugin/test.h"
%include "../../UMA_Plugin/snapshot_platform.h"