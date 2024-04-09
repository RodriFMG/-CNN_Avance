//
// Created by RODRIGO on 6/01/2024.
//

#include "CNN.h"

void CNN::Establecer_Filtros() {

    Sobel_Horiz << -1,-2,-1, 0,0,0, 1,2,1;
    Sobel_Verti << -1,0,1, -2,0,2, -1,0,1;

    Gabor_Horiz << 1,2,1, 0,0,0, -1,-2,-1;
    Gabor_Verti << 1,0,-1, 2,0,-2, 1,0,-1;

    Harriz_Esquinas << -1,2,-1, -1,2,-1, -1,2,-1;
    Suavizado << 1,1,1, 1,1,1, 1,1,1;

}