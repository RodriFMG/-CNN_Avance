#include "CNN.h"


pair<cv::Mat, Index> CNN::Sacar_img_prediccion(const std::string &cifar_10_test) {
    pair<cv::Mat, Index> falta_completar{};

    return falta_completar;
}

void CNN::Prediccion(const std::string& cifar_10_test) {

    CNN::MatToVector(CNN::Sacar_img_prediccion(cifar_10_test));

}

void CNN::Resultados(const std::string& cifar_10_test) {

    Prediccion(cifar_10_test);
}