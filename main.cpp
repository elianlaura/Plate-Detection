#include "alpr.h"
#include "compared.h"

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace alpr;

struct Plate
{
    Rect box;
    short int porc_acep;
    short int porc_vida;
    Ptr<Tracker> tracker;
    std::string candidatos[3];
    float confidence[3];
    bool showed;
    bool tracking;
};

//%----------------------------------------------------------
//%----------------------------------------------------------
double SPOSMIN_MAT = 20.0 ; // sobreposición mínima de matching entre objetos //Mayor valor->más estricto
                            // Difícilmente dos placas van a ser contiguas, esto no sucede en vehículos.d
int INC_DEC_VIDA = 20;

int w =  720, h = 480, c = 0; //int w =  1920, h = 1080, c = 0;
int izqx = 1; // el 20% del lado izquierdo de la escena debe ser excluida en la detección.
int derx = 99; // Más allá del 78% del ancho de la imagen se excluye de la detección.
int supy_track = 2; // Menos del 1% de la altura de la imagen, se excluye para el tracking.
int infy_track = 99; // Más del 88% de la altura de la imagen, se excluye para el tracking.
int x_lim_izq = (w * izqx) / 100;
int x_lim_der = (w * derx) / 100;
int y_lim_sup_track = (h * supy_track) / 100;
int y_lim_inf_track = (h * infy_track) / 100;

//%----------------------------------------------------------
//%----------------------------------------------------------
int getResultsFrame(alpr::Alpr* openalpr, cv::Mat &frame, std::vector<Plate> &mem_objs, string n_frm_str);
void function_tracking(Mat& frame, Mat& frame_vis,
                       std::vector<Plate>& det_objs,
                       std::vector<Plate>& mem_objs);
void dibujaObjs(cv::Rect, Mat &img, alpr::AlprPlateResult plate, int i);
bool overlapTotal(Rect r1, Rect r2);
bool overlapParcial( Rect r1, Rect r2, double min_overlap );
bool matchingObjs (Plate mylist, Rect curr_rect );
bool enelAmbitoGlobal(Point punto , int width, int height);
void dibujaObj( Rect, Mat&, Scalar color );

//%----------------------------------------------------------
//%----------------------------------------------------------
int main()
{
    alpr::Alpr openalpr("us", "/home/elian/Documents/app/openalpr/config/openalpr.conf");


    openalpr.setTopN(3);

    openalpr.setDefaultRegion("base");
    cout<<"Iteration"<<endl;

    if (openalpr.isLoaded() == false)
    {
         std::cerr << "Error loading OpenALPR" << std::endl;
         return 1;
    }


    //__________ Read video ___________
    VideoCapture *cap;
    cap = new cv::VideoCapture("MVI_0031_downsize.avi");
    //cap = new cv::VideoCapture("reconocimiento_placas.");
    cv::Mat frame;
    if ( !cap->isOpened() ) {
        cout << "Cannot open the video file" << endl;
        return -1;
    }

    for ( int i = 0; i < 2 ; i++ )
        cap->read(frame);
    //_________________________________

    const int KEY_SPACE  = 32;
    const int KEY_ESC    = 27;
    int key = 0;
    //________________________________

    std::vector<Plate> mem_objs;
    //________________________________
    int n_frm = -1;
    string n_frm_str = "";

    do{
        if ( ! cap->read(frame))
            break;
        if (frame.empty()) {                             // if unable to open image
            std::cout << "error: image not read from file\n\n";     // show error message on command line
            //_getch();                                               // may have to modify this line if not using Windows
            return(0);                                              // and exit program
        }

        // Write frames
        n_frm++;
        //cout<<"frm: "<<n_frm<<"......................................"<<endl;
        //{stringstream ss;    ss<<n_frm;      n_frm_str = ss.str();}
        //imwrite("frames/frame_"+n_frm_str+".jpg", frame);


        key = cvWaitKey(10); // hold windows open until user presses a key
        if(key == KEY_SPACE)
          key = cvWaitKey(0);
        if(key == KEY_ESC)
          break;

        getResultsFrame(&openalpr, frame, mem_objs, n_frm_str);


    }while(1);

    /*cv::Mat frame = imread("/home/elian/guille.jpg", 1);
    getResultsFrame(&openalpr, frame);
    cv::imshow("imgOriginalScene", frame);
    imwrite("frame.jpg", frame);
    cv::waitKey(0);*/
    //results = openalpr.recognize("/home/elian/test2.jpg");




    return 0;
}

int getResultsFrame(alpr::Alpr* openalpr, cv::Mat &frame, std::vector<Plate> &mem_objs,  string n_frm_str) {

    std::vector<alpr::AlprRegionOfInterest> regionsOfInterest;

    alpr::AlprResults results = openalpr->recognize(frame.data, frame.elemSize(),
                                                frame.cols, frame.rows, regionsOfInterest);
    std::vector<Plate> det_objs;
    cv::Mat frame_vis = frame.clone();
    int n_plates = results.plates.size();

    for (int i = 0; i < n_plates; i++)
    {
       Plate my_plate;

       alpr::AlprPlateResult plate = results.plates[i];

       int x, y, w, h;
       alpr::AlprCoordinate coord = plate.plate_points[0];

       x = coord.x;     y = coord.y;
       coord = plate.plate_points[1];
       w = coord.x - x;
       coord = plate.plate_points[2];
       h = coord.y - y;

       my_plate.box = Rect(x, y, w, h);


       // Iterate by the candidates. plate.topNPlates = 3 este valor en openalpr.conf
       alpr::AlprPlate candidate = plate.topNPlates[0];
       if ( candidate.characters.size() == 6 ) {
           for (int k = 0; k < plate.topNPlates.size(); k++)
            {
                alpr::AlprPlate candidate = plate.topNPlates[k];
                //std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
                my_plate.candidatos[k] = candidate.characters;
                my_plate.confidence[k] = candidate.overall_confidence;
            }

           det_objs.push_back(my_plate);
           dibujaObj( my_plate.box, frame_vis, Scalar(0,255,0));
       }

     } // Fin del for

    // Seguimiento de las placas
    function_tracking(frame, frame_vis, det_objs, mem_objs);

    // Mostrar placas y sus caracteres
    for ( auto &obj : mem_objs ) {
        if (obj.tracking) {
            cv::rectangle( frame_vis, obj.box, Scalar(0,200,0), 2);
            string text = obj.candidatos[0];
            cv::putText(frame_vis, text, cv::Point(obj.box.x, obj.box.y - 15),
                                CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0.0, 255.0, 255.0), 2);


            if (!obj.showed ) {
                for ( int i = 0; i < 3; i++ ) {
                    string text = obj.candidatos[i];
                    cout << "Candidato "<<i<<": " << text << " \t " << obj.confidence[i] << endl;
                }
                obj.showed = true;
                cout<<endl;

            }

        }

    }


    cv::imshow("imgOriginalScene", frame_vis);
    //cv::imwrite("frame_vis_"+n_frm_str+".jpg", frame_vis);
     return 0;
}


//%------------------------------------------------------------------
//%------------------------------------------------------------------
void function_tracking(Mat& frame, Mat& frame_vis,
                       std::vector<Plate>& det_objs,
                       std::vector<Plate>& mem_objs) {

    char n_case = 'n';
    int det_objs_sz = det_objs.size();
    int mem_objs_sz = mem_objs.size();

    if( det_objs_sz > 0 && mem_objs.empty())
        n_case = 'A';
    if( mem_objs_sz > 0 && det_objs_sz > 0)
        n_case = 'B';
    if ( mem_objs_sz > 0 && det_objs_sz == 0)
        n_case = 'C';

    switch ( n_case ) {

        case 'A':
                // * I *  Almacenando en memoria los primeros objetos encontrados.

                for ( auto &obj : det_objs ) {
                    Plate a_plate;
                    a_plate = obj; // obj ya contiene el box y candidatos
                    a_plate.porc_acep = 0;
                    a_plate.porc_vida = 100;
                    Ptr<Tracker> tracker;
                    a_plate.tracker = tracker;
                    a_plate.showed = false;
                    a_plate.tracking = false;
                    mem_objs.push_back( a_plate );

                }
                break;

        case 'B':
        {   // * II * Cuando ya se tiene objetos en memoria y se tiene objetos detectados en el frame actual        
        // Matching entre objetos de memoria y objetos actualmente detectados

                    for( int i = 0 ; i < mem_objs_sz ; i++) {
                            bool anymatching = true;
                            Plate curr_mem_obj = mem_objs[i];
                            for( int j = 0 ; j < det_objs_sz ; j++) {
                                Plate myplate = det_objs[j];
                                Rect curr_det_obj = myplate.box;
                                if ( matchingObjs( curr_mem_obj , curr_det_obj) )  {
                                    // Debería ser con quien tiene el máximo overlapping, es decir el más cercano
                                     if ( !curr_mem_obj.tracking ) {
                                         curr_mem_obj.porc_acep = curr_mem_obj.porc_acep + INC_DEC_VIDA ;
                                         curr_mem_obj.box = curr_det_obj;
                                     }
                                     anymatching = false;
                                     det_objs.erase( det_objs.begin() + j );
                                     det_objs_sz = det_objs.size();
                                     j--;
                                }
                            }
                            if( anymatching && !curr_mem_obj.tracking ) {
                                   curr_mem_obj.porc_vida = curr_mem_obj.porc_vida - INC_DEC_VIDA;
                            }
                            mem_objs[i] = curr_mem_obj;
                    }

                    // Crea objetos nuevos para memoria
                    for ( auto &obj : det_objs ) {
                        Plate a_plate;
                        a_plate = obj; // obj ya contiene el box y candidatos
                        a_plate.porc_acep = 0;
                        a_plate.porc_vida = 100;
                        Ptr<Tracker> tracker;
                        a_plate.tracker = tracker;                        
                        a_plate.showed = false;
                        a_plate.tracking = false;
                        //#pragma omp critical
                        mem_objs.push_back( a_plate );
                    }
                    mem_objs_sz = mem_objs.size();

                    // Continua trackings respectivos, to update trackings
                    std::vector<bool> eliminar = std::vector<bool>(mem_objs_sz, false);
                    int i;
                    for( i = 0 ; i < mem_objs_sz ; i++) {
                        Plate curr_mem_obj = mem_objs[i];
                        if ( curr_mem_obj.tracking ) {
                            // verifica si el objeto está aún en escena, sino eliminarlo
                            if ( enelAmbitoGlobal(Point(curr_mem_obj.box.x, curr_mem_obj.box.y),
                                                  curr_mem_obj.box.width, curr_mem_obj.box.height) ) {
                                cv::Rect2d new_box = curr_mem_obj.box;

                                curr_mem_obj.tracker->update(frame, new_box);

                                curr_mem_obj.box = new_box;
                                mem_objs[i] = curr_mem_obj;
                            } else {
                                eliminar[i] = true;
                            }
                        }
                    }

                    // PORQUE ELIMINA SIN VERIFICAR ALGO ANTES
                    for( i = 0 ; i < mem_objs_sz ; i++)
                        if ( eliminar[i] == true ) {
                            mem_objs.erase( mem_objs.begin() + i );
                        }
                    mem_objs_sz = mem_objs.size();
                    break;


                } // Fin del case B





    case 'C':
    { // * III * No hay detección en el frame actual

        // Verifica ambito global
        std::vector<bool> eliminar = std::vector<bool>(mem_objs_sz, false);
        for( int i = 0 ; i < mem_objs_sz ; i++) {
            Plate curr_mem_obj = mem_objs[i];
            if ( curr_mem_obj.tracking ) {
                if ( !enelAmbitoGlobal(Point(curr_mem_obj.box.x, curr_mem_obj.box.y),
                                      curr_mem_obj.box.width, curr_mem_obj.box.height) )
                    eliminar[i] = true;
            }
        }


        // Elimina aquellos fuera del ambito global
        for( int i = 0 ; i < mem_objs_sz ; i++)
            if ( eliminar[i] == true ) {
                mem_objs.erase( mem_objs.begin() + i );
            }
        mem_objs_sz = mem_objs.size();


        // Continua tracking con aquellos objetos en estado tracking = true.
        for( int i = 0 ; i < mem_objs_sz ; i++) {            
            Plate curr_mem_obj = mem_objs[i];
            if (curr_mem_obj.tracking ) {
                cv::Rect2d new_box = curr_mem_obj.box;
                curr_mem_obj.tracker->update(frame, new_box);
                curr_mem_obj.box = new_box;
            }
            else {

            // Reduce porcentaje de vida de cada objeto en mem_objs
            curr_mem_obj.porc_vida =  mem_objs[i].porc_vida - INC_DEC_VIDA;
            }
            mem_objs[i] = curr_mem_obj;
        }
            break;
    } // Fin del caso C

    default:
            break;

    }//F: fin del SWITCH




    // Verifica porcentajes de los objetos en memoria
    for( int i = 0 ; i < mem_objs_sz ; i++) {
        Plate curr_mem_obj = mem_objs[i];
        // Inicializa un tracker
        if ( curr_mem_obj.porc_acep >= 100 && !curr_mem_obj.tracking ) {
            Rect2d this_rect = curr_mem_obj.box;
            curr_mem_obj.tracker = Tracker::create( "KCF" );
            curr_mem_obj.tracker->init(frame, this_rect);
            curr_mem_obj.tracking = true;
            mem_objs[i] = curr_mem_obj;
        }

        // Elimina un elemento
        if ( curr_mem_obj.porc_vida <= 0 ) {
            mem_objs.erase( mem_objs.begin() + i );
            mem_objs_sz = mem_objs.size();
            i--;
        }
    }


} // Fin de function_tracking


//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void dibujaObjs(cv::Rect plate_rect, Mat &frame, alpr::AlprPlateResult plate, int i){

    cv::rectangle( frame, plate_rect, Scalar(0,255,0), 2);

    //std::cout << "plate" << i << ": " << plate.topNPlates.size() << " results" << std::endl;

    // Iterate by the candidates
    for (int k = 0; k < plate.topNPlates.size(); k++)
     {
         alpr::AlprPlate candidate = plate.topNPlates[k];
         //std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
         string text = candidate.characters;
         if ( k == 0)
             cv::putText(frame, text, cv::Point(plate_rect.x, plate_rect.y - 15), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0.0, 255.0, 255.0), 2);
         //std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
     }

}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool matchingObjs (Plate mylist, Rect curr_rect ) {
    Rect mem_rect = mylist.box;
    // Verificar una sobreposición fuertemente exigente, con un 70% de área de sobreposición mínima
    if ( overlapTotal( mem_rect, curr_rect ) ||  overlapParcial( mem_rect, curr_rect, SPOSMIN_MAT ) )
      return true;

    return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool overlapTotal(Rect r1, Rect r2) {
    int area_inter = ( r1 & r2 ).area();

    if( area_inter == r1.area() ||  area_inter == r2.area()  )
        return true;
    return false;

}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
///
/// El ratio entre el área de la intersección y el área de la región meno, debe ser
/// mayor a min_overlap
bool overlapParcial( Rect r1, Rect r2, double min_overlap ) {
    double propMenor = 0.0;
    int area_inter = ( r1 & r2 ).area();
    int res = compareD((double)r1.area(), (double)r2.area()); // Si r1 es menor entonces -1
    if(  res < 1  ) //Buscamos el menor
        propMenor = (double)area_inter/(double)r1.area();
    else
        propMenor = (double)area_inter/(double)r2.area();

    if ( (propMenor*100.0) > min_overlap )
        return true;

    return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool enelAmbitoGlobal(Point punto, int width, int height){

    if ( punto.x > x_lim_izq &&
         punto.x < x_lim_der &&
         punto.y > y_lim_sup_track &&
         (punto.y + height) < y_lim_inf_track  )
        return true;
    return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void dibujaObj(Rect obj, Mat& img, Scalar color){
       cv::rectangle( img, Rect(obj.x, obj.y, obj.width, obj.height), color, 3);
}
