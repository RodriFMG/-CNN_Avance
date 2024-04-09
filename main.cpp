#include "CNN.h"

vector<vector<Vector3i>> MatToVector(const Mat& img){
    vector<vector<Vector3i>> total(img.rows);


    for(int i=0; i<img.rows; ++i){
        vector<Vector3i> retorno(img.cols);

        for(int j=0; j<img.cols; ++j){
            Vector3i rgb;

            for(int k=0; k< static_cast<int>(img.channels()); ++k){
                rgb[k] = img.at<Vec3b>(i,j)[k];
            }

            retorno[j] = rgb;
        }
        total[i] = retorno;
    }

    return total;
}

// maybe tengo que recorrer la imagen y detectar posicion por posicion con el pair y asi con los filtros
// recorrer la imagen denuevo, eso devo hacer.

template <typename T>
T Relu(T x) {

    const T max_val = 1e9;
    const T min_val = -1e9;
    x = max(min_val, min(max_val, x));

    return (x>=0) ? x : 0;
}

template<typename T, int fil, int col>
vector<vector<Vector3i>> Convolucion_Filtro(const vector<vector<Vector3i>>& imagen, Matrix<T, fil, col> filtro){

    if(imagen.size() < filtro.rows() or imagen[0].size() < filtro.cols()) {
        cerr << "La imagen es demasiado pequeña para aplicar el filtro." << endl;
        return imagen;
    }

    //Determino el tamaño previo de almacenamiento
    vector<vector<Vector3i>> convolusion(imagen.size() - filtro.rows() + 1,
                                         vector<Vector3i>(imagen[0].size() - filtro.cols() + 1, Vector3i(0, 0, 0)));

    int controlar_fila{}, controlar_columna{};

    int cant_iter = (imagen.size() - filtro.rows() + 1) * (imagen[0].size() - filtro.cols() + 1);

    for(int iter=0; iter < cant_iter; ++iter){

        for(int k=0; k < imagen[0][0].size();++k){
            T sumar{};

            for(int i=0; i < filtro.rows(); ++i){
                for(int j=0; j < filtro.cols(); ++j){

                    sumar += filtro(i,j) * imagen[i+controlar_fila][j+controlar_columna][k];

                }
            }

            sumar = Relu(sumar);
            convolusion[controlar_fila][controlar_columna][k] = sumar;

        }

        ++controlar_columna;

        if(controlar_columna == imagen[0].size() - filtro.cols() + 1){
            ++controlar_fila;
            controlar_columna = 0;
        }

    }

    return convolusion;
}

vector<vector<Vector3i>> max_pooling(const vector<vector<Vector3i>>& imagen){

    int fil_pooling = 2, col_pooling = 2;

    if(imagen.size()<=fil_pooling or imagen[0].size()<=col_pooling){
        cerr << "La imagen es demasiado pequeña para aplicar el pooling." << endl;
        return imagen;
    }


    size_t fil_retorno = imagen.size()/fil_pooling;
    size_t col_retorno = imagen[0].size()/col_pooling;

    vector<vector<Vector3i>> retorno(fil_retorno, vector<Vector3i>(col_retorno, Vector3i(0,0,0)));

    int cant_iter = static_cast<int>(fil_retorno * col_retorno);

    int controlar_fila{}, controlar_columna{};

    for(int iter=0; iter<cant_iter; ++iter){

        int fil_iter_pool;
        int col_iter_pool;

        //---
        if(controlar_fila >= imagen.size() - 2*fil_pooling){
            fil_iter_pool = (imagen.size()%2==1) ? fil_pooling + 1 : fil_pooling;
        }
        else{
            fil_iter_pool = fil_pooling;
        }

        if(controlar_columna  >= imagen[0].size() - 2*col_pooling){
            col_iter_pool = (imagen[0].size()%2==1) ? col_pooling + 1 : col_pooling;
        }
        else{
            col_iter_pool = col_pooling;
        }
        //---

        for(int canal = 0; canal < imagen[0][0].size(); ++canal){
            int max_pol = INT_MIN;

            for(int i = 0; i < fil_iter_pool; ++i){
                for(int j = 0; j < col_iter_pool; ++j) {
                    max_pol = max(max_pol, imagen[i + controlar_fila][j + controlar_columna][canal]);
                }
            }

            retorno[controlar_fila / fil_pooling][controlar_columna / col_pooling][canal] = max_pol;
        }

        controlar_columna += col_pooling;


        if(imagen[0].size()%2==1){
            if(controlar_columna + 1 == imagen[0].size() ){
                controlar_columna = 0;
                controlar_fila+=fil_pooling;
            }
        }
        else{
            if(controlar_columna == imagen[0].size() ){
                controlar_columna = 0;
                controlar_fila+=fil_pooling;
            }
        }

    }

    return retorno;
}


void ejemplo_1(){
    cv::Mat img = cv::imread(R"(C:\Users\RODRIGO\Pictures\Saved Pictures\gorda.png)", cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "No se pudo cargar la imagen desde el archivo." << std::endl;
    }

    vector<vector<Vector3i>> a = MatToVector(img);

    Matrix<int, 3, 3> Sobel_Horiz;
    Sobel_Horiz << -1, -2, -1, 0, 0, 0, 1, 2, 1;

    // Cambia a Vector3i para la matriz del filtro

    cout<<a.size()<<endl;
    cout<<a[0].size()<<endl<<"---"<<endl;

    auto result = Convolucion_Filtro(a, Sobel_Horiz);


    auto result2 = max_pooling(result);

    cout<<result.size()<<endl;
    cout<<result[0].size()<<endl;

    cout<<"-----------"<<endl;
    cout<<result2.size()<<endl;
    cout<<result2[0].size()<<endl;





    /*
    vector<vector<Vector3i>> matriz_9x9 = {
            {{166, 2, 3}, {4, 5, 65}, {77, 8, 9}, {10, 11, 172}, {173, 14, 15}, {16, 17, 168}, {1999, 20, 21}, {22, 23, 24}, {25, 26, 1000}},
            {{258, 29, 30}, {31, 732, 33}, {345, 35, 36}, {37, 38, 39}, {407, 41, 42}, {43, 44, 45}, {46, 47, 48}, {49, 5077, 51}, {52, 53, 54}},
            {{55, 56, 57}, {58, 59, 60}, {61, 62, 63}, {64, 65, 66}, {67, 68, 69}, {70, 71, 72}, {73, 748, 75}, {76, 767, 789}, {79, 80, 81}},
            {{82, 83, 84}, {85, 86, 587}, {88, 89, 90}, {91, 922, 93}, {94, 95, 926}, {979, 98, 99}, {100, 101, 102}, {1039, 104, 105}, {106, 107, 108}},
            {{109, 110, 111}, {112, 1133, 114}, {1154, 116, 117}, {118, 119, 1240}, {121, 122, 123}, {124, 125, 126}, {127, 128, 129}, {130, 131, 132}, {133, 134, 135}},
            {{1364, 137, 1384}, {139, 140, 1431}, {142, 143, 144}, {145, 146, 147}, {148, 455, 150}, {151, 152, 153}, {154, 155, 156}, {157, 158, 159}, {160, 161, 162}},
            {{455, 164, 165}, {166, 167, 168}, {169, 170, 171}, {172, 173, 174}, {546, 176, 177}, {178, 179, 180}, {1999, 182, 183}, {184, 185, 186}, {187, 188, 189}},
            {{190, 191, 192}, {193, 1934, 195}, {196, 197, 198}, {199, 200, 201}, {202, 203, 204}, {205, 206, 207}, {208, 209, 210}, {211, 212, 213}, {214, 215, 216}},
            {{217, 218, 219}, {220, 221, 2252}, {223, 224, 225}, {226, 227, 228}, {229, 230, 231}, {232, 233, 644}, {235, 236, 237}, {238, 239, 240}, {241, 242, 243}}
    };

    auto result = max_pooling(matriz_9x9);

    // Imprimir el resultado
    for (const auto& row : result) {
        for (const auto& val : row) {
            for (int i : val) {
                cout << i << " ";
            }
            cout << "| ";
        }
        cout << endl;
    }
     */
}

int main() {
    /*
    // Abre el archivo binario de CIFAR-10
    std::ifstream file(R"(C:\Users\RODRIGO\Downloads\cifar10_si\cifar-10-binary\cifar-10-batches-bin\data_batch_1.bin)", std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error abriendo el archivo." << std::endl;
            return 1;
        }

    const int original_image_size = 32 * 32 * 3;  // 32x32 píxeles, 3 canales (RGB)
    const int label_size = 1;  // 1 byte para la etiqueta

    // Bucle para leer múltiples ejemplos
    for (int example = 0; example < 100; ++example) {
        uint8_t buffer[original_image_size + label_size];

        // Lee una imagen junto con su etiqueta
        file.read(reinterpret_cast<char*>(buffer), sizeof(buffer));

        // Extrae la etiqueta
        int label = buffer[0];


        // Convierte el buffer en una matriz OpenCV
        cv::Mat original_image(32, 32, CV_8UC3);

        std::memcpy(original_image.data, buffer, original_image_size);

        // Muestra la etiqueta y la imagen (esto puede variar según tu entorno y configuración)
        std::cout << "Etiqueta: " << label << std::endl;
        cv::imshow("Original CIFAR-10 Image", original_image);
        cv::waitKey(0);
    }

    return 0;
     */

    cout<<"Hola pe";
}

