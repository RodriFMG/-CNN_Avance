#include "CNN.h"

//Función para pasar la escala de píxeles (rgb) de la imagen a una estructura de datos equivalente.
void CNN::MatToVector(const pair<cv::Mat, Index>& date){
    imagen.resize(date.first.rows);

    for(int i=0; i<date.first.rows; ++i){
        vector<Vector3i> retorno(date.first.cols);

        for(int j=0; j<date.first.cols; ++j){
            Vector3i rgb;

            for(int k=0; k< static_cast<int>(date.first.channels()); ++k){
                rgb[k] = date.first.at<Vec3b>(i,j)[k];
            }

            retorno[j] = rgb;
        }
        imagen[i] = retorno;
    }
}

// Función en la cual se aplicará los filtros (convoluciones), quería hacerlo genérico, pero tendría que declararlo
// en la clase, cosa que quedaría poco entendible.
vector<vector<Vector3i>>CNN::Convolucion_Filtro(const vector<vector<Eigen::Vector3i>>& mtz_img, Matrix<int, 3, 3> filtro) {

    if(mtz_img.size() < filtro.rows() or mtz_img[0].size() < filtro.cols()) {
        cerr << "La imagen es demasiado pequeña para aplicar el filtro." << endl;
        return mtz_img;
    }

    //Determino el tamaño previo de almacenamiento
    vector<vector<Vector3i>> convolusion(mtz_img.size() - filtro.rows() + 1,
                                         vector<Vector3i>(mtz_img[0].size() - filtro.cols() + 1, Vector3i(0, 0, 0)));

    int controlar_fila{}, controlar_columna{};

    size_t cant_iter = (mtz_img.size() - filtro.rows() + 1) * (mtz_img[0].size() - filtro.cols() + 1);

    for(int iter=0; iter < cant_iter; ++iter){

        for(int k=0; k < mtz_img[0][0].size();++k){
            int sumar{};

            for(int i=0; i < filtro.rows(); ++i){
                for(int j=0; j < filtro.cols(); ++j){

                    sumar += filtro(i,j) * mtz_img[i+controlar_fila][j+controlar_columna][k];

                }
            }

            sumar = static_cast<int>(Relu(static_cast<double>(sumar)));
            convolusion[controlar_fila][controlar_columna][k] = sumar;

        }

        ++controlar_columna;

        if(controlar_columna == mtz_img[0].size() - filtro.cols() + 1){
            ++controlar_fila;
            controlar_columna = 0;
        }

    }

    return convolusion;
}

// Función en la cual se aplicará el pooling (max_pooling)
vector<vector<Vector3i>> CNN::max_pooling(const vector<vector<Eigen::Vector3i>>& mtz_img) {

    int fil_pooling = 2, col_pooling = 2;

    if(mtz_img.size()<=fil_pooling or mtz_img[0].size()<=col_pooling){
        cerr << "La imagen es demasiado pequeña para aplicar el pooling." << endl;
        return mtz_img;
    }


    size_t fil_retorno = mtz_img.size()/fil_pooling;
    size_t col_retorno = mtz_img[0].size()/col_pooling;

    vector<vector<Vector3i>> retorno(fil_retorno, vector<Vector3i>(col_retorno, Vector3i(0,0,0)));

    int cant_iter = static_cast<int>(fil_retorno * col_retorno);

    int controlar_fila{}, controlar_columna{};

    for(int iter=0; iter<cant_iter; ++iter){

        int fil_iter_pool;
        int col_iter_pool;

        //---
        if(controlar_fila >= mtz_img.size() - 2*fil_pooling){
            fil_iter_pool = (mtz_img.size()%2==1) ? fil_pooling + 1 : fil_pooling;
        }
        else{
            fil_iter_pool = fil_pooling;
        }

        if(controlar_columna  >= mtz_img[0].size() - 2*col_pooling){
            col_iter_pool = (mtz_img[0].size()%2==1) ? col_pooling + 1 : col_pooling;
        }
        else{
            col_iter_pool = col_pooling;
        }
        //---

        for(int canal = 0; canal < mtz_img[0][0].size(); ++canal){
            int max_pol = INT_MIN;

            for(int i = 0; i < fil_iter_pool; ++i){
                for(int j = 0; j < col_iter_pool; ++j) {

                    //Para pasar de max_pooling a AVG_pooling, nomas habría que sumar todos los elementos y...
                    max_pol = max(max_pol, mtz_img[i + controlar_fila][j + controlar_columna][canal]);

                }
            }

            // acá dividir la suma entre fil_iter_pool*col_iter_pool (antes de asignarle al vector de retorno).
            retorno[controlar_fila / fil_pooling][controlar_columna / col_pooling][canal] = max_pol;
        }

        controlar_columna += col_pooling;


        if(mtz_img[0].size()%2==1){
            if(controlar_columna + 1 == mtz_img[0].size() ){
                controlar_columna = 0;
                controlar_fila+=fil_pooling;
            }
        }
        else{
            if(controlar_columna == mtz_img[0].size() ){
                controlar_columna = 0;
                controlar_fila+=fil_pooling;
            }
        }

    }

    return retorno;
}