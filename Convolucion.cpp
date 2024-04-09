#include "CNN.h"


// En esta función estableceré los filtros para las convolcuión
void CNN::Normalizar_Dimenciones_Espaciales(const Index& index) {

    // Contenedor que guardará las capas de convoluciones hechas en 1 sola imagen.
    vector<vector<vector<Vector3i>>> filtros_img(6);

    // Primera capa convolutional

    filtros_img[0] = CNN::Convolucion_Filtro(imagen, Sobel_Horiz);
    filtros_img[1] = CNN::Convolucion_Filtro(imagen, Sobel_Verti);
    filtros_img[2] = CNN::Convolucion_Filtro(imagen, Gabor_Horiz);
    filtros_img[3] = CNN::Convolucion_Filtro(imagen, Gabor_Verti);
    filtros_img[4] = CNN::Convolucion_Filtro(imagen, Harriz_Esquinas);
    filtros_img[5] = CNN::Convolucion_Filtro(imagen, Suavizado);

    for(auto& filtro : filtros_img){
        filtro = CNN::max_pooling(filtro);
    }


    // 2da capa convolutional
    filtros_img[0] = CNN::Convolucion_Filtro(imagen, Sobel_Verti);
    filtros_img[1] = CNN::Convolucion_Filtro(imagen, Sobel_Horiz);
    filtros_img[2] = CNN::Convolucion_Filtro(imagen, Gabor_Verti);
    filtros_img[3] = CNN::Convolucion_Filtro(imagen, Gabor_Horiz);
    filtros_img[4] = CNN::Convolucion_Filtro(imagen, Suavizado);
    filtros_img[5] = CNN::Convolucion_Filtro(imagen, Harriz_Esquinas);

    for(auto& filtro : filtros_img){
        filtro = CNN::max_pooling(filtro);
    }

    // 3era capa convolutional
    filtros_img[0] = CNN::Convolucion_Filtro(imagen, Suavizado);
    filtros_img[1] = CNN::Convolucion_Filtro(imagen, Sobel_Verti);
    filtros_img[2] = CNN::Convolucion_Filtro(imagen, Harriz_Esquinas);
    filtros_img[3] = CNN::Convolucion_Filtro(imagen, Suavizado);
    filtros_img[4] = CNN::Convolucion_Filtro(imagen, Gabor_Horiz);
    filtros_img[5] = CNN::Convolucion_Filtro(imagen, Gabor_Verti);

    for(auto& filtro : filtros_img){
        filtro = CNN::max_pooling(filtro);
    }

    for(const auto& filtro_x_filtro : filtros_img){
        CNN::Entrenamiento(filtro_x_filtro, index);
    }



}