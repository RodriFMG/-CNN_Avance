#include "CNN.h"

double CNN::Derivada_Relu(double x){
    return (x>0) ? 1 : 0;
}