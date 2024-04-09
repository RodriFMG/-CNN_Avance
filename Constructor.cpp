#include "CNN.h"

CNN::CNN(const std::string& cifar_10_train, const std::string& cifar_10_test, int n2_, int n3_, int n4_, int n5_,
         int n6_, int n7_, int file) {

    n2 = n2_;
    n3 = n3_;
    n4 = n4_;
    n5 = n5_;
    n6 = n6_;
    n7 = n7_;

    CNN::Establecer_Filtros();

    CNN::Dimensiones_Contenedores();

    CNN::Datos_Img(cifar_10_train);

    CNN::Resultados(cifar_10_test);
}
