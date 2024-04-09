#include "CNN.h"

void CNN::Datos_Img(const string& cifar_10_train) {
    ifstream file(R"(cifar_10_train)", ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error abriendo el archivo." << std::endl;
    }

    const int img_original = 32*32*3; // 32 = dimenciones espaciales de la imagen || 3 = RGB
    const int etiqueta = 1; // 1 espacio para la etiqueta

    for(int imagen_i = 0; imagen_i < 10000; ++imagen_i){
        uint8_t imagen_bin[img_original + etiqueta];

        file.read(reinterpret_cast<char*>(imagen_bin), sizeof(imagen_bin));

        cv::Mat imagen_cifar_10(32,32,CV_8UC3);
        memcpy(imagen_cifar_10.data, imagen_bin, img_original);

        CNN::MatToVector(make_pair(imagen_cifar_10, static_cast<Index>(imagen_bin[0])));
        CNN::Normalizar_Dimenciones_Espaciales(static_cast<Index>(imagen_bin[0]));
    }

}
