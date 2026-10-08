// Filtros - Vision Artificial
// uso: ./filtros imagen.jpg   (opcional: --nogui)
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;

bool GUI = true;
string OUT = "salida";

// pone la imagen dentro de una celda con su titulo
Mat aCelda(const Mat& m, Size celda, string titulo) {
    Mat t;
    if (m.depth() != CV_8U) {
        normalize(m, t, 0, 255, NORM_MINMAX);
        t.convertTo(t, CV_8U);
    } else {
        t = m.clone();
    }
    if (t.channels() == 1) cvtColor(t, t, COLOR_GRAY2BGR);

    double s = min((double)celda.width / t.cols, (double)celda.height / t.rows);
    resize(t, t, Size(), s, s, INTER_AREA);

    Mat lienzo = Mat::zeros(celda, CV_8UC3);
    t.copyTo(lienzo(Rect((celda.width - t.cols) / 2, (celda.height - t.rows) / 2, t.cols, t.rows)));
    putText(lienzo, titulo, Point(6, 17), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    return lienzo;
}

// junta varias imagenes en una ventana y las guarda en la carpeta salida
bool mostrar(string nombre, vector<pair<string, Mat>> items, int cols = 3) {
    Size celda(360, 290);
    int filas = ((int)items.size() + cols - 1) / cols;
    Mat grande = Mat::zeros(filas * celda.height, cols * celda.width, CV_8UC3);
    for (int i = 0; i < (int)items.size(); i++) {
        Mat c = aCelda(items[i].second, celda, items[i].first);
        c.copyTo(grande(Rect((i % cols) * celda.width, (i / cols) * celda.height, celda.width, celda.height)));
    }
    imwrite(OUT + "/" + nombre + ".png", grande);
    if (!GUI) return true;
    imshow(nombre, grande);
    int k = waitKey(0);
    destroyWindow(nombre);
    return k != 27;  // 27 = ESC
}

// imagen de prueba por si no pasan ninguna
Mat imagenPrueba() {
    Mat img(420, 560, CV_8UC3);
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            img.at<Vec3b>(y, x) = Vec3b(60 + x * 120 / img.cols, 90 + y * 100 / img.rows, 140);
    circle(img, Point(150, 160), 90, Scalar(30, 200, 240), FILLED);
    rectangle(img, Rect(300, 80, 190, 150), Scalar(240, 240, 240), FILLED);
    rectangle(img, Rect(320, 100, 150, 110), Scalar(40, 40, 40), 4);
    line(img, Point(0, 380), Point(560, 280), Scalar(255, 255, 255), 3);
    putText(img, "OpenCV", Point(60, 360), FONT_HERSHEY_DUPLEX, 2.0, Scalar(20, 20, 20), 3);
    return img;
}

// ---------- convolucion ----------

vector<float> conv1D(vector<float> x, vector<float> k, int pad, int stride) {
    vector<float> xp(pad, 0);
    xp.insert(xp.end(), x.begin(), x.end());
    xp.insert(xp.end(), pad, 0);
    vector<float> y;
    for (int i = 0; i + (int)k.size() <= (int)xp.size(); i += stride) {
        float suma = 0;
        for (int j = 0; j < (int)k.size(); j++) suma += xp[i + j] * k[j];
        y.push_back(suma);
    }
    return y;
}

void imprimir(string texto, vector<float> v) {
    cout << texto << " (tam " << v.size() << "): ";
    for (float f : v) cout << f << " ";
    cout << endl;
}

// convolucion 2D a mano, con ceros en los bordes
Mat conv2D(Mat src, Mat k) {
    int ph = k.rows / 2, pw = k.cols / 2;
    Mat pad;
    copyMakeBorder(src, pad, ph, ph, pw, pw, BORDER_CONSTANT, Scalar(0));
    Mat dst(src.size(), CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float suma = 0;
            for (int i = 0; i < k.rows; i++)
                for (int j = 0; j < k.cols; j++)
                    suma += pad.at<float>(y + i, x + j) * k.at<float>(i, j);
            dst.at<float>(y, x) = suma;
        }
    }
    return dst;
}

void convolucion(Mat gray) {
    cout << "--- convolucion 1D ---" << endl;
    vector<float> x = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
    imprimir("entrada", x);
    imprimir("filtro 1x1", conv1D(x, {2}, 0, 1));
    imprimir("filtro 2x1", conv1D(x, {1, -1}, 0, 1));
    imprimir("filtro 3x1 sin padding", conv1D(x, {1, 0, -1}, 0, 1));
    imprimir("filtro 3x1 con padding", conv1D(x, {1, 0, -1}, 1, 1));
    imprimir("filtro 3x1 stride 3", conv1D(x, {1, 0, -1}, 1, 3));

    cout << "--- convolucion 2D (ejemplo de la diapositiva) ---" << endl;
    Mat img = (Mat_<float>(5, 5) <<
        10,  20,  30,  40,  50,
        60, 170, 245,   0,  90,
       110, 234,  42,  64, 130,
       150,  32,  53, 128, 170,
       190, 200, 210, 220, 230);
    Mat k = (Mat_<float>(3, 3) <<
        -1,  0, 1,
         2,  1, 2,
         1, -2, 0);
    Mat manual = conv2D(img, k);
    Mat opencv;
    filter2D(img, opencv, CV_32F, k, Point(-1, -1), 0, BORDER_CONSTANT);
    cout << "manual:\n" << manual << endl;
    cout << "filter2D:\n" << opencv << endl;
    cout << "pixel del centro: " << manual.at<float>(2, 2) << " (en la diapositiva da 394)" << endl;

    // lo mismo pero con la imagen real
    Mat g32;
    gray.convertTo(g32, CV_32F);
    Mat caja = Mat::ones(5, 5, CV_32F) / 25.0f;
    Mat a = conv2D(g32, caja), b;
    filter2D(g32, b, CV_32F, caja, Point(-1, -1), 0, BORDER_CONSTANT);
    double dif;
    minMaxLoc(abs(a - b), 0, &dif);
    cout << "diferencia maxima manual vs filter2D: " << dif << endl;
}

// ---------- filtros espaciales ----------

Mat salYPimienta(Mat g, double p = 0.05) {
    Mat r = g.clone();
    RNG rng(42);
    for (int y = 0; y < r.rows; y++) {
        for (int x = 0; x < r.cols; x++) {
            double n = rng.uniform(0.0, 1.0);
            if (n < p / 2) r.at<uchar>(y, x) = 0;
            else if (n < p) r.at<uchar>(y, x) = 255;
        }
    }
    return r;
}

bool espaciales(Mat gray) {
    // suavizado
    Mat caja, gauss;
    blur(gray, caja, Size(7, 7));
    GaussianBlur(gray, gauss, Size(7, 7), 1.5);
    if (!mostrar("2a_suavizado", {{"Original", gray}, {"Caja 7x7", caja}, {"Gaussiano 7x7", gauss}})) return false;

    // mediana contra los otros con ruido
    Mat ruido = salYPimienta(gray);
    Mat rCaja, rGauss, rMed;
    blur(ruido, rCaja, Size(5, 5));
    GaussianBlur(ruido, rGauss, Size(5, 5), 1.5);
    medianBlur(ruido, rMed, 5);
    if (!mostrar("2b_mediana", {{"Con ruido", ruido}, {"Caja 5x5", rCaja}, {"Gaussiano 5x5", rGauss}, {"Mediana 5x5", rMed}}, 2)) return false;

    // maximo y minimo
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat maximo, minimo;
    dilate(gray, maximo, kernel);
    erode(gray, minimo, kernel);
    if (!mostrar("2c_max_min", {{"Original", gray}, {"Maximo", maximo}, {"Minimo", minimo}})) return false;

    // bordes
    Mat suave, gx, gy, mag, lap, lapAbs, canny;
    GaussianBlur(gray, suave, Size(3, 3), 0);
    Sobel(suave, gx, CV_32F, 1, 0, 3);
    Sobel(suave, gy, CV_32F, 0, 1, 3);
    magnitude(gx, gy, mag);
    Laplacian(suave, lap, CV_16S, 3);
    convertScaleAbs(lap, lapAbs);
    Canny(suave, canny, 80, 160);
    if (!mostrar("2d_bordes", {{"Sobel X", abs(gx)}, {"Sobel Y", abs(gy)}, {"Sobel magnitud", mag},
                               {"Laplaciano", lapAbs}, {"Canny", canny}, {"Original", gray}})) return false;

    // enfoque
    Mat borroso, unsharp, enfoque;
    GaussianBlur(gray, borroso, Size(0, 0), 3);
    addWeighted(gray, 1.8, borroso, -0.8, 0, unsharp);
    Mat kEnf = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(gray, enfoque, -1, kEnf);
    return mostrar("2e_enfoque", {{"Original", gray}, {"Unsharp mask", unsharp}, {"Kernel de enfoque", enfoque}});
}

// ---------- transformaciones afines ----------

bool afines(Mat src) {
    int w = src.cols, h = src.rows;
    Mat refX, refY, escala, rot, cizalla;

    Mat Mx = (Mat_<double>(2, 3) << -1, 0, w, 0, 1, 0);
    warpAffine(src, refX, Mx, src.size());

    Mat My = (Mat_<double>(2, 3) << 1, 0, 0, 0, -1, h);
    warpAffine(src, refY, My, src.size());

    double sx = 1.5, sy = 0.7;
    Mat Ms = (Mat_<double>(2, 3) << sx, 0, 0, 0, sy, 0);
    warpAffine(src, escala, Ms, Size(cvRound(w * sx), cvRound(h * sy)));

    Mat Mr = getRotationMatrix2D(Point2f(w / 2.0f, h / 2.0f), 30, 1.0);
    warpAffine(src, rot, Mr, src.size());

    double shx = 0.4;
    Mat Mc = (Mat_<double>(2, 3) << 1, shx, 0, 0, 1, 0);
    warpAffine(src, cizalla, Mc, Size(cvRound(w + h * shx), h));

    return mostrar("3_afines", {{"Original", src}, {"Reflexion X", refX}, {"Reflexion Y", refY},
                                {"Escala", escala}, {"Rotacion 30", rot}, {"Cizallamiento", cizalla}});
}

// ---------- frecuencia ----------

// tamano bueno para la DFT y que sea par
int tamanoDFT(int n) {
    n = getOptimalDFTSize(n);
    while (n % 2 != 0) n = getOptimalDFTSize(n + 1);
    return n;
}

// cambia los cuadrantes para que el centro quede en medio
void cambiarCuadrantes(Mat& m) {
    int cx = m.cols / 2, cy = m.rows / 2;
    Mat q0(m, Rect(0, 0, cx, cy));
    Mat q1(m, Rect(cx, 0, cx, cy));
    Mat q2(m, Rect(0, cy, cx, cy));
    Mat q3(m, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

Mat espectro(Mat gray) {
    // 1. gris, 2. padding
    Mat padded;
    int m = tamanoDFT(gray.rows), n = tamanoDFT(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));
    // 3. imagen compleja
    Mat planos[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complejo;
    merge(planos, 2, complejo);
    // 4. dft
    dft(complejo, complejo);
    // 5. magnitud
    split(complejo, planos);
    magnitude(planos[0], planos[1], planos[0]);
    Mat mag = planos[0];
    // 6. logaritmo
    mag += Scalar::all(1);
    log(mag, mag);
    // 7. cuadrantes
    cambiarCuadrantes(mag);
    // 8. normalizar
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    return mag;
}

// tipo: 0 = paso bajo, 1 = paso alto, 2 = paso banda
Mat crearMascara(Size s, int tipo, double r1, double r2) {
    Mat mask(s, CV_32F);
    for (int y = 0; y < s.height; y++) {
        for (int x = 0; x < s.width; x++) {
            double d = hypot(x - s.width / 2.0, y - s.height / 2.0);
            bool pasa;
            if (tipo == 0) pasa = d <= r1;
            else if (tipo == 1) pasa = d > r1;
            else pasa = (d >= r1 && d <= r2);
            mask.at<float>(y, x) = pasa ? 1.0f : 0.0f;
        }
    }
    return mask;
}

Mat filtroFrecuencia(Mat gray, int tipo, double r1, double r2, Mat& mascara) {
    Mat padded;
    int m = tamanoDFT(gray.rows), n = tamanoDFT(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planos[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat c;
    merge(planos, 2, c);
    dft(c, c);
    split(c, planos);

    mascara = crearMascara(padded.size(), tipo, r1, r2);
    Mat mascara2 = mascara.clone();
    cambiarCuadrantes(mascara2);
    multiply(planos[0], mascara2, planos[0]);
    multiply(planos[1], mascara2, planos[1]);

    merge(planos, 2, c);
    Mat res;
    idft(c, res, DFT_SCALE | DFT_REAL_OUTPUT);
    return res(Rect(0, 0, gray.cols, gray.rows)).clone();
}

bool frecuencia(Mat gray) {
    if (!mostrar("4a_espectro", {{"Original", gray}, {"Espectro", espectro(gray)}}, 2)) return false;

    Mat m1, m2, m3;
    Mat bajo = filtroFrecuencia(gray, 0, 30, 0, m1);
    Mat alto = filtroFrecuencia(gray, 1, 30, 0, m2);
    Mat banda = filtroFrecuencia(gray, 2, 20, 60, m3);
    return mostrar("4b_filtros_frecuencia", {{"Mascara paso bajo", m1}, {"Mascara paso alto", m2}, {"Mascara paso banda", m3},
                                             {"Paso bajo", bajo}, {"Paso alto", alto}, {"Paso banda", banda}});
}

int main(int argc, char** argv) {
    string ruta = "";
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--nogui") GUI = false;
        else ruta = a;
    }
    filesystem::create_directories(OUT);

    Mat color;
    if (ruta != "") {
        color = imread(ruta);
        if (color.empty()) {
            cout << "no se pudo abrir la imagen" << endl;
            return 1;
        }
    } else {
        color = imagenPrueba();
    }
    // si es muy grande la achico para que la convolucion manual no se demore
    if (color.cols > 640) {
        double s = 640.0 / color.cols;
        resize(color, color, Size(), s, s, INTER_AREA);
    }
    Mat gray;
    cvtColor(color, gray, COLOR_BGR2GRAY);

    convolucion(gray);
    if (!espaciales(gray)) return 0;
    if (!afines(color)) return 0;
    frecuencia(gray);

    cout << "listo, las imagenes quedaron en la carpeta " << OUT << endl;
    return 0;
}
