#include "CNN.h"

// Tendré que pasarle como parámetro la etiqueta de cada imagen.
void CNN::Entrenamiento(const vector<vector<Vector3i>>& img_filtrada, const Index& etiqueta){

    size_t filas = img_filtrada.size();
    size_t columnas = img_filtrada[0].size();
    size_t canales = img_filtrada[0][0].size();

    cap_entrada = VectorXd::Zero(static_cast<int>(filas*columnas));

    for(int canal=0; canal < canales; ++canal){
        int px{};

        for(int fila=0; fila < filas; ++fila){
            for(int columna=0; columna < columnas; ++columna){
                cap_entrada[px] = img_filtrada[fila][columna][canal];
                ++px;
            }
        }

        etiqueta_real[etiqueta] = 1;

        CNN::BackPropagation(etiqueta);

        etiqueta_real = VectorXd::Zero(n7);
    }

}