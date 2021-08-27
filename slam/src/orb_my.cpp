/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>
#include <nmmintrin.h>
#include <bitset>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Custom Libraries */
#include "include/libUtils_basic.h"
#include "include/libUtils_eigen.h"
#include "include/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* Global Variables */
string image1_filepath = "../carlaData/image/id00044.png";
string image2_filepath = "../carlaData/image/id00045.png";

string semantic1_filepath = "../carlaData/semantic/id00044.png";
string semantic1color_filepath = "../carlaData/semantic/idcolor00044.png";
string semantic2_filepath = "../carlaData/semantic/id00045.png";


const int nfeatures = 500;
const int nrBrief = 256;
const int nSemrBrief = 48;
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 24;

double matches_lower_bound = 30.0;


// ORB pattern
int ORB_pattern[256 * 4] = {
    8, -3, 9, 5         /*mean (0), correlation (0)*/,
    4, 2, 7, -12        /*mean (1.12461e-05), correlation (0.0437584)*/,
    -11, 9, -8, 2       /*mean (3.37382e-05), correlation (0.0617409)*/,
    7, -12, 12, -13     /*mean (5.62303e-05), correlation (0.0636977)*/, // FIXME: https://translate.google.com/translate?sl=auto&tl=en&u=https://github.com/gaoxiang12/slambook2/issues/142
    2, -13, 2, 12       /*mean (0.000134953), correlation (0.085099)*/,
    1, -7, 1, 6         /*mean (0.000528565), correlation (0.0857175)*/,
    -2, -10, -2, -4     /*mean (0.0188821), correlation (0.0985774)*/,
    -13, -13, -11, -8   /*mean (0.0363135), correlation (0.0899616)*/, // FIXME: https://translate.google.com/translate?sl=auto&tl=en&u=https://github.com/gaoxiang12/slambook2/issues/142
    -13, -3, -12, -9    /*mean (0.121806), correlation (0.099849)*/,
    10, 4, 11, 9        /*mean (0.122065), correlation (0.093285)*/,
    -13, -8, -8, -9     /*mean (0.162787), correlation (0.0942748)*/,
    -11, 7, -9, 12      /*mean (0.21561), correlation (0.0974438)*/,
    7, 7, 12, 6         /*mean (0.160583), correlation (0.130064)*/,
    -4, -5, -3, 0       /*mean (0.228171), correlation (0.132998)*/,
    -13, 2, -12, -3     /*mean (0.00997526), correlation (0.145926)*/,
    -9, 0, -7, 5        /*mean (0.198234), correlation (0.143636)*/,
    12, -6, 12, -1      /*mean (0.0676226), correlation (0.16689)*/,
    -3, 6, -2, 12       /*mean (0.166847), correlation (0.171682)*/,
    -6, -13, -4, -8     /*mean (0.101215), correlation (0.179716)*/,
    11, -13, 12, -8     /*mean (0.200641), correlation (0.192279)*/,
    4, 7, 5, 1          /*mean (0.205106), correlation (0.186848)*/,
    5, -3, 10, -3       /*mean (0.234908), correlation (0.192319)*/,
    3, -7, 6, 12        /*mean (0.0709964), correlation (0.210872)*/,
    -8, -7, -6, -2      /*mean (0.0939834), correlation (0.212589)*/,
    -2, 11, -1, -10     /*mean (0.127778), correlation (0.20866)*/,
    -13, 12, -8, 10     /*mean (0.14783), correlation (0.206356)*/,
    -7, 3, -5, -3       /*mean (0.182141), correlation (0.198942)*/,
    -4, 2, -3, 7        /*mean (0.188237), correlation (0.21384)*/,
    -10, -12, -6, 11    /*mean (0.14865), correlation (0.23571)*/,
    5, -12, 6, -7       /*mean (0.222312), correlation (0.23324)*/,
    5, -6, 7, -1        /*mean (0.229082), correlation (0.23389)*/,
    1, 0, 4, -5         /*mean (0.241577), correlation (0.215286)*/,
    9, 11, 11, -13      /*mean (0.00338507), correlation (0.251373)*/,
    4, 7, 4, 12         /*mean (0.131005), correlation (0.257622)*/,
    2, -1, 4, 4         /*mean (0.152755), correlation (0.255205)*/,
    -4, -12, -2, 7      /*mean (0.182771), correlation (0.244867)*/,
    -8, -5, -7, -10     /*mean (0.186898), correlation (0.23901)*/,
    4, 11, 9, 12        /*mean (0.226226), correlation (0.258255)*/,
    0, -8, 1, -13       /*mean (0.0897886), correlation (0.274827)*/,
    -13, -2, -8, 2      /*mean (0.148774), correlation (0.28065)*/,
    -3, -2, -2, 3       /*mean (0.153048), correlation (0.283063)*/,
    -6, 9, -4, -9       /*mean (0.169523), correlation (0.278248)*/,
    8, 12, 10, 7        /*mean (0.225337), correlation (0.282851)*/,
    0, 9, 1, 3          /*mean (0.226687), correlation (0.278734)*/,
    7, -5, 11, -10      /*mean (0.00693882), correlation (0.305161)*/,
    -13, -6, -11, 0     /*mean (0.0227283), correlation (0.300181)*/,
    10, 7, 12, 1        /*mean (0.125517), correlation (0.31089)*/,
    -6, -3, -6, 12      /*mean (0.131748), correlation (0.312779)*/,
    10, -9, 12, -4      /*mean (0.144827), correlation (0.292797)*/,
    -13, 8, -8, -12     /*mean (0.149202), correlation (0.308918)*/,
    -13, 0, -8, -4      /*mean (0.160909), correlation (0.310013)*/,
    3, 3, 7, 8          /*mean (0.177755), correlation (0.309394)*/,
    5, 7, 10, -7        /*mean (0.212337), correlation (0.310315)*/,
    -1, 7, 1, -12       /*mean (0.214429), correlation (0.311933)*/,
    3, -10, 5, 6        /*mean (0.235807), correlation (0.313104)*/,
    2, -4, 3, -10       /*mean (0.00494827), correlation (0.344948)*/,
    -13, 0, -13, 5      /*mean (0.0549145), correlation (0.344675)*/,
    -13, -7, -12, 12    /*mean (0.103385), correlation (0.342715)*/,
    -13, 3, -11, 8      /*mean (0.134222), correlation (0.322922)*/,
    -7, 12, -4, 7       /*mean (0.153284), correlation (0.337061)*/,
    6, -10, 12, 8       /*mean (0.154881), correlation (0.329257)*/,
    -9, -1, -7, -6      /*mean (0.200967), correlation (0.33312)*/,
    -2, -5, 0, 12       /*mean (0.201518), correlation (0.340635)*/,
    -12, 5, -7, 5       /*mean (0.207805), correlation (0.335631)*/,
    3, -10, 8, -13      /*mean (0.224438), correlation (0.34504)*/,
    -7, -7, -4, 5       /*mean (0.239361), correlation (0.338053)*/,
    -3, -2, -1, -7      /*mean (0.240744), correlation (0.344322)*/,
    2, 9, 5, -11        /*mean (0.242949), correlation (0.34145)*/,
    -11, -13, -5, -13   /*mean (0.244028), correlation (0.336861)*/,
    -1, 6, 0, -1        /*mean (0.247571), correlation (0.343684)*/,
    5, -3, 5, 2         /*mean (0.000697256), correlation (0.357265)*/,
    -4, -13, -4, 12     /*mean (0.00213675), correlation (0.373827)*/,
    -9, -6, -9, 6       /*mean (0.0126856), correlation (0.373938)*/,
    -12, -10, -8, -4    /*mean (0.0152497), correlation (0.364237)*/,
    10, 2, 12, -3       /*mean (0.0299933), correlation (0.345292)*/,
    7, 12, 12, 12       /*mean (0.0307242), correlation (0.366299)*/,
    -7, -13, -6, 5      /*mean (0.0534975), correlation (0.368357)*/,
    -4, 9, -3, 4        /*mean (0.099865), correlation (0.372276)*/,
    7, -1, 12, 2        /*mean (0.117083), correlation (0.364529)*/,
    -7, 6, -5, 1        /*mean (0.126125), correlation (0.369606)*/,
    -13, 11, -12, 5     /*mean (0.130364), correlation (0.358502)*/,
    -3, 7, -2, -6       /*mean (0.131691), correlation (0.375531)*/,
    7, -8, 12, -7       /*mean (0.160166), correlation (0.379508)*/,
    -13, -7, -11, -12   /*mean (0.167848), correlation (0.353343)*/,
    1, -3, 12, 12       /*mean (0.183378), correlation (0.371916)*/,
    2, -6, 3, 0         /*mean (0.228711), correlation (0.371761)*/,
    -4, 3, -2, -13      /*mean (0.247211), correlation (0.364063)*/,
    -1, -13, 1, 9       /*mean (0.249325), correlation (0.378139)*/,
    7, 1, 8, -6         /*mean (0.000652272), correlation (0.411682)*/,
    1, -1, 3, 12        /*mean (0.00248538), correlation (0.392988)*/,
    9, 1, 12, 6         /*mean (0.0206815), correlation (0.386106)*/,
    -1, -9, -1, 3       /*mean (0.0364485), correlation (0.410752)*/,
    -13, -13, -10, 5    /*mean (0.0376068), correlation (0.398374)*/,
    7, 7, 10, 12        /*mean (0.0424202), correlation (0.405663)*/,
    12, -5, 12, 9       /*mean (0.0942645), correlation (0.410422)*/,
    6, 3, 7, 11         /*mean (0.1074), correlation (0.413224)*/,
    5, -13, 6, 10       /*mean (0.109256), correlation (0.408646)*/,
    2, -12, 2, 3        /*mean (0.131691), correlation (0.416076)*/,
    3, 8, 4, -6         /*mean (0.165081), correlation (0.417569)*/,
    2, 6, 12, -13       /*mean (0.171874), correlation (0.408471)*/,
    9, -12, 10, 3       /*mean (0.175146), correlation (0.41296)*/,
    -8, 4, -7, 9        /*mean (0.183682), correlation (0.402956)*/,
    -11, 12, -4, -6     /*mean (0.184672), correlation (0.416125)*/,
    1, 12, 2, -8        /*mean (0.191487), correlation (0.386696)*/,
    6, -9, 7, -4        /*mean (0.192668), correlation (0.394771)*/,
    2, 3, 3, -2         /*mean (0.200157), correlation (0.408303)*/,
    6, 3, 11, 0         /*mean (0.204588), correlation (0.411762)*/,
    3, -3, 8, -8        /*mean (0.205904), correlation (0.416294)*/,
    7, 8, 9, 3          /*mean (0.213237), correlation (0.409306)*/,
    -11, -5, -6, -4     /*mean (0.243444), correlation (0.395069)*/,
    -10, 11, -5, 10     /*mean (0.247672), correlation (0.413392)*/,
    -5, -8, -3, 12      /*mean (0.24774), correlation (0.411416)*/,
    -10, 5, -9, 0       /*mean (0.00213675), correlation (0.454003)*/,
    8, -1, 12, -6       /*mean (0.0293635), correlation (0.455368)*/,
    4, -6, 6, -11       /*mean (0.0404971), correlation (0.457393)*/,
    -10, 12, -8, 7      /*mean (0.0481107), correlation (0.448364)*/,
    4, -2, 6, 7         /*mean (0.050641), correlation (0.455019)*/,
    -2, 0, -2, 12       /*mean (0.0525978), correlation (0.44338)*/,
    -5, -8, -5, 2       /*mean (0.0629667), correlation (0.457096)*/,
    7, -6, 10, 12       /*mean (0.0653846), correlation (0.445623)*/,
    -9, -13, -8, -8     /*mean (0.0858749), correlation (0.449789)*/,
    -5, -13, -5, -2     /*mean (0.122402), correlation (0.450201)*/,
    8, -8, 9, -13       /*mean (0.125416), correlation (0.453224)*/,
    -9, -11, -9, 0      /*mean (0.130128), correlation (0.458724)*/,
    1, -8, 1, -2        /*mean (0.132467), correlation (0.440133)*/,
    7, -4, 9, 1         /*mean (0.132692), correlation (0.454)*/,
    -2, 1, -1, -4       /*mean (0.135695), correlation (0.455739)*/,
    11, -6, 12, -11     /*mean (0.142904), correlation (0.446114)*/,
    -12, -9, -6, 4      /*mean (0.146165), correlation (0.451473)*/,
    3, 7, 7, 12         /*mean (0.147627), correlation (0.456643)*/,
    5, 5, 10, 8         /*mean (0.152901), correlation (0.455036)*/,
    0, -4, 2, 8         /*mean (0.167083), correlation (0.459315)*/,
    -9, 12, -5, -13     /*mean (0.173234), correlation (0.454706)*/,
    0, 7, 2, 12         /*mean (0.18312), correlation (0.433855)*/,
    -1, 2, 1, 7         /*mean (0.185504), correlation (0.443838)*/,
    5, 11, 7, -9        /*mean (0.185706), correlation (0.451123)*/,
    3, 5, 6, -8         /*mean (0.188968), correlation (0.455808)*/,
    -13, -4, -8, 9      /*mean (0.191667), correlation (0.459128)*/,
    -5, 9, -3, -3       /*mean (0.193196), correlation (0.458364)*/,
    -4, -7, -3, -12     /*mean (0.196536), correlation (0.455782)*/,
    6, 5, 8, 0          /*mean (0.1972), correlation (0.450481)*/,
    -7, 6, -6, 12       /*mean (0.199438), correlation (0.458156)*/,
    -13, 6, -5, -2      /*mean (0.211224), correlation (0.449548)*/,
    1, -10, 3, 10       /*mean (0.211718), correlation (0.440606)*/,
    4, 1, 8, -4         /*mean (0.213034), correlation (0.443177)*/,
    -2, -2, 2, -13      /*mean (0.234334), correlation (0.455304)*/,
    2, -12, 12, 12      /*mean (0.235684), correlation (0.443436)*/,
    -2, -13, 0, -6      /*mean (0.237674), correlation (0.452525)*/,
    4, 1, 9, 3          /*mean (0.23962), correlation (0.444824)*/,
    -6, -10, -3, -5     /*mean (0.248459), correlation (0.439621)*/,
    -3, -13, -1, 1      /*mean (0.249505), correlation (0.456666)*/,
    7, 5, 12, -11       /*mean (0.00119208), correlation (0.495466)*/,
    4, -2, 5, -7        /*mean (0.00372245), correlation (0.484214)*/,
    -13, 9, -9, -5      /*mean (0.00741116), correlation (0.499854)*/,
    7, 1, 8, 6          /*mean (0.0208952), correlation (0.499773)*/,
    7, -8, 7, 6         /*mean (0.0220085), correlation (0.501609)*/,
    -7, -4, -7, 1       /*mean (0.0233806), correlation (0.496568)*/,
    -8, 11, -7, -8      /*mean (0.0236505), correlation (0.489719)*/,
    -13, 6, -12, -8     /*mean (0.0268781), correlation (0.503487)*/,
    2, 4, 3, 9          /*mean (0.0323324), correlation (0.501938)*/,
    10, -5, 12, 3       /*mean (0.0399235), correlation (0.494029)*/,
    -6, -5, -6, 7       /*mean (0.0420153), correlation (0.486579)*/,
    8, -3, 9, -8        /*mean (0.0548021), correlation (0.484237)*/,
    2, -12, 2, 8        /*mean (0.0616622), correlation (0.496642)*/,
    -11, -2, -10, 3     /*mean (0.0627755), correlation (0.498563)*/,
    -12, -13, -7, -9    /*mean (0.0829622), correlation (0.495491)*/,
    -11, 0, -10, -5     /*mean (0.0843342), correlation (0.487146)*/,
    5, -3, 11, 8        /*mean (0.0929937), correlation (0.502315)*/,
    -2, -13, -1, 12     /*mean (0.113327), correlation (0.48941)*/,
    -1, -8, 0, 9        /*mean (0.132119), correlation (0.467268)*/,
    -13, -11, -12, -5   /*mean (0.136269), correlation (0.498771)*/,
    -10, -2, -10, 11    /*mean (0.142173), correlation (0.498714)*/,
    -3, 9, -2, -13      /*mean (0.144141), correlation (0.491973)*/,
    2, -3, 3, 2         /*mean (0.14892), correlation (0.500782)*/,
    -9, -13, -4, 0      /*mean (0.150371), correlation (0.498211)*/,
    -4, 6, -3, -10      /*mean (0.152159), correlation (0.495547)*/,
    -4, 12, -2, -7      /*mean (0.156152), correlation (0.496925)*/,
    -6, -11, -4, 9      /*mean (0.15749), correlation (0.499222)*/,
    6, -3, 6, 11        /*mean (0.159211), correlation (0.503821)*/,
    -13, 11, -5, 5      /*mean (0.162427), correlation (0.501907)*/,
    11, 11, 12, 6       /*mean (0.16652), correlation (0.497632)*/,
    7, -5, 12, -2       /*mean (0.169141), correlation (0.484474)*/,
    -1, 12, 0, 7        /*mean (0.169456), correlation (0.495339)*/,
    -4, -8, -3, -2      /*mean (0.171457), correlation (0.487251)*/,
    -7, 1, -6, 7        /*mean (0.175), correlation (0.500024)*/,
    -13, -12, -8, -13   /*mean (0.175866), correlation (0.497523)*/,
    -7, -2, -6, -8      /*mean (0.178273), correlation (0.501854)*/,
    -8, 5, -6, -9       /*mean (0.181107), correlation (0.494888)*/,
    -5, -1, -4, 5       /*mean (0.190227), correlation (0.482557)*/,
    -13, 7, -8, 10      /*mean (0.196739), correlation (0.496503)*/,
    1, 5, 5, -13        /*mean (0.19973), correlation (0.499759)*/,
    1, 0, 10, -13       /*mean (0.204465), correlation (0.49873)*/,
    9, 12, 10, -1       /*mean (0.209334), correlation (0.49063)*/,
    5, -8, 10, -9       /*mean (0.211134), correlation (0.503011)*/,
    -1, 11, 1, -13      /*mean (0.212), correlation (0.499414)*/,
    -9, -3, -6, 2       /*mean (0.212168), correlation (0.480739)*/,
    -1, -10, 1, 12      /*mean (0.212731), correlation (0.502523)*/,
    -13, 1, -8, -10     /*mean (0.21327), correlation (0.489786)*/,
    8, -11, 10, -6      /*mean (0.214159), correlation (0.488246)*/,
    2, -13, 3, -6       /*mean (0.216993), correlation (0.50287)*/,
    7, -13, 12, -9      /*mean (0.223639), correlation (0.470502)*/,
    -10, -10, -5, -7    /*mean (0.224089), correlation (0.500852)*/,
    -10, -8, -8, -13    /*mean (0.228666), correlation (0.502629)*/,
    4, -6, 8, 5         /*mean (0.22906), correlation (0.498305)*/,
    3, 12, 8, -13       /*mean (0.233378), correlation (0.503825)*/,
    -4, 2, -3, -3       /*mean (0.234323), correlation (0.476692)*/,
    5, -13, 10, -12     /*mean (0.236392), correlation (0.475462)*/,
    4, -13, 5, -1       /*mean (0.236842), correlation (0.504132)*/,
    -9, 9, -4, 3        /*mean (0.236977), correlation (0.497739)*/,
    0, 3, 3, -9         /*mean (0.24314), correlation (0.499398)*/,
    -12, 1, -6, 1       /*mean (0.243297), correlation (0.489447)*/,
    3, 2, 4, -8         /*mean (0.00155196), correlation (0.553496)*/,
    -10, -10, -10, 9    /*mean (0.00239541), correlation (0.54297)*/,
    8, -13, 12, 12      /*mean (0.0034413), correlation (0.544361)*/,
    -8, -12, -6, -5     /*mean (0.003565), correlation (0.551225)*/,
    2, 2, 3, 7          /*mean (0.00835583), correlation (0.55285)*/,
    10, 6, 11, -8       /*mean (0.00885065), correlation (0.540913)*/,
    6, 8, 8, -12        /*mean (0.0101552), correlation (0.551085)*/,
    -7, 10, -6, 5       /*mean (0.0102227), correlation (0.533635)*/,
    -3, -9, -3, 9       /*mean (0.0110211), correlation (0.543121)*/,
    -1, -13, -1, 5      /*mean (0.0113473), correlation (0.550173)*/,
    -3, -7, -3, 4       /*mean (0.0140913), correlation (0.554774)*/,
    -8, -2, -8, 3       /*mean (0.017049), correlation (0.55461)*/,
    4, 2, 12, 12        /*mean (0.01778), correlation (0.546921)*/,
    2, -5, 3, 11        /*mean (0.0224022), correlation (0.549667)*/,
    6, -9, 11, -13      /*mean (0.029161), correlation (0.546295)*/,
    3, -1, 7, 12        /*mean (0.0303081), correlation (0.548599)*/,
    11, -1, 12, 4       /*mean (0.0355151), correlation (0.523943)*/,
    -3, 0, -3, 6        /*mean (0.0417904), correlation (0.543395)*/,
    4, -11, 4, 12       /*mean (0.0487292), correlation (0.542818)*/,
    2, -4, 2, 1         /*mean (0.0575124), correlation (0.554888)*/,
    -10, -6, -8, 1      /*mean (0.0594242), correlation (0.544026)*/,
    -13, 7, -11, 1      /*mean (0.0597391), correlation (0.550524)*/,
    -13, 12, -11, -13   /*mean (0.0608974), correlation (0.55383)*/,
    6, 0, 11, -13       /*mean (0.065126), correlation (0.552006)*/,
    0, -1, 1, 4         /*mean (0.074224), correlation (0.546372)*/,
    -13, 3, -9, -2      /*mean (0.0808592), correlation (0.554875)*/,
    -9, 8, -6, -3       /*mean (0.0883378), correlation (0.551178)*/,
    -13, -6, -8, -2     /*mean (0.0901035), correlation (0.548446)*/,
    5, -9, 8, 10        /*mean (0.0949843), correlation (0.554694)*/,
    2, 7, 3, -9         /*mean (0.0994152), correlation (0.550979)*/,
    -1, -6, -1, -1      /*mean (0.10045), correlation (0.552714)*/,
    9, 5, 11, -2        /*mean (0.100686), correlation (0.552594)*/,
    11, -3, 12, -8      /*mean (0.101091), correlation (0.532394)*/,
    3, 0, 3, 5          /*mean (0.101147), correlation (0.525576)*/,
    -1, 4, 0, 10        /*mean (0.105263), correlation (0.531498)*/,
    3, -6, 4, 5         /*mean (0.110785), correlation (0.540491)*/,
    -13, 0, -10, 5      /*mean (0.112798), correlation (0.536582)*/,
    5, 8, 12, 11        /*mean (0.114181), correlation (0.555793)*/,
    8, 9, 9, -6         /*mean (0.117431), correlation (0.553763)*/,
    7, -4, 8, -12       /*mean (0.118522), correlation (0.553452)*/,
    -10, 4, -10, 9      /*mean (0.12094), correlation (0.554785)*/,
    7, 3, 12, 4         /*mean (0.122582), correlation (0.555825)*/,
    9, -7, 10, -2       /*mean (0.124978), correlation (0.549846)*/,
    7, 0, 12, -2        /*mean (0.127002), correlation (0.537452)*/,
    -1, -6, 0, -11      /*mean (0.127148), correlation (0.547401)*/
};


void computeDesc(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor){
    GaussianBlur(img, img, Size(7, 7), 2, 2, BORDER_REFLECT_101);
    
    descriptor = Mat::zeros(keypoints.size(), 32, CV_8UC1);
    // descriptor.create(500, 32, CV_8UC1);
    int count_kp = 0;
    
    for(auto &kp: keypoints){
        float m01=0.0, m10=0.0;

        // for(int y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; y++){
        //     for(int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; x++){
        //         m01 += y*img.at<uchar>(kp.pt.y+y, kp.pt.y+x);
        //         m10 += x*img.at<uchar>(kp.pt.y+y, kp.pt.y+x);
        //     }
        // }
        
        // kp.angle = atan2(m01, m10);
        // float angle = cvRound((kp.angle)/(CV_PI/15.0));
        
        // angle *= CV_PI/15.0;

        // float sin_kp = sin(angle);
        // float cos_kp = cos(angle);

        float sin_kp = sin(kp.angle*CV_PI/180.0);
        float cos_kp = cos(kp.angle*CV_PI/180.0);

        float a = cos_kp;
        float b = sin_kp;

        #define GET_VALUE(idx_p) \
            center[cvRound(ORB_pattern[idx_p]*b + ORB_pattern[idx_p+1]*a)*step + \
                   cvRound(ORB_pattern[idx_p]*a - ORB_pattern[idx_p+1]*b)]
        
        for(int i = 0; i < 32; i++){
            u_char desc = 0;
            int idx = i*8*4;
            


            const u_char* center = &img.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x));
            const int step = (int)img.step;
            
            for(int pt = 0; pt < 8*4; pt+=4){
            // for(int pt = 0; pt < 8*4; pt+=4){
                
                
                // Point2f p(ORB_pattern[idx + pt], ORB_pattern[idx + pt + 1]);
                // Point2f q(ORB_pattern[idx + pt + 2],ORB_pattern[idx + pt + 3]);
                
                // Point2f pp = Point2f(cvRound(cos_kp*p.x - sin_kp*p.y + kp.pt.x), cvRound(sin_kp*p.x + cos_kp*p.y + kp.pt.y));
                // Point2f qq = Point2f(cvRound(cos_kp*q.x - sin_kp*q.y + kp.pt.x), cvRound(sin_kp*q.x + cos_kp*q.y + kp.pt.y));

                // Point2f pp = Point2f((cos_kp*p.x - sin_kp*p.y), (sin_kp*p.x + cos_kp*p.y)) + kp.pt;
                // Point2f qq = Point2f((cos_kp*q.x - sin_kp*q.y), (sin_kp*q.x + cos_kp*q.y)) + kp.pt;

                desc |= (GET_VALUE(idx + pt) < GET_VALUE(idx + pt + 2)) << (int)(pt*0.25);
                // if(img.at<uchar>(pp.y, pp.x) < img.at<uchar>(qq.y, qq.x))
                //     desc |= 1 << (int)(pt*0.25);

            }
            descriptor.at<uchar>(count_kp, i) = desc;
        }
        count_kp++;
        #undef GET_VALUE
    }

}


void computeSemanticDesc(const Mat &sem_img, const vector<KeyPoint> &keypoints, Mat &descriptor){
    // descriptor.create(nfeatures, nrBrief/32 + nSemrBrief/6, CV_32SC1);

    int n, cols, idxSemDescCol, count_kp = 0;

    const float factorPI = CV_PI/180.0;

    for(auto &kp: keypoints){
        float sin_kp = sin(kp.angle*factorPI);
        float cos_kp = cos(kp.angle*factorPI);
    
        for(int i = 0; i < nSemrBrief/6; i++){
            int32_t desc = 0;
            int idx = i*6*2;
            for(int pt = 0; pt < 6*2; pt+=2){
                Point2f p(ORB_pattern[idx + pt], ORB_pattern[idx + pt + 1]);

                Point2f pp = Point2f(round(cos_kp*p.x - sin_kp*p.y + kp.pt.x), round(sin_kp*p.x + cos_kp*p.y + kp.pt.y));
                desc |= sem_img.at<uchar>(pp.y, pp.x*3 + 2) << int(pt*3); /*int(pt*0.5*6)*/
            }
            descriptor.at<u_int32_t>(count_kp, nrBrief/32 + i) = desc;
        }
        count_kp++;
    }
}


void convertDesc(Mat &descriptor, Mat &sem_descriptor, Mat semantic_img){

    int n, cols, idxSemDescCol, idxSemBitOffset;
    // sem_descriptor = Mat::zeros(Size(nfeatures, (int)nrBrief/32/* + nSemrBrief/6*/), CV_32SC1);

    sem_descriptor.create(nfeatures, nrBrief/32 + nSemrBrief/6, CV_32SC1);

    for(n = 0; n < descriptor.rows; n++){
        idxSemDescCol = -1;
        // cout << "linha: " << n << endl;
        for(cols = 0; cols < descriptor.cols; cols++, idxSemBitOffset++){
            // cout << "coluna: " << n;
            // cout << (int)semantic_img.at<uchar>(n,cols*3 + 2) << endl;
            if(!(cols % 4)){
                idxSemDescCol++;
                sem_descriptor.at<int32_t>(n,idxSemDescCol) = 0;
    
                if(!(cols % 8))
                    idxSemBitOffset = 0;
            }
            
            sem_descriptor.at<int32_t>(n,idxSemDescCol) |= descriptor.at<uchar>(n, cols) << (7-idxSemBitOffset)*8;
            // sem_descriptor.at<int32_t>(n,idxSemDescCol + nrBrief/32)
        }    
    }
}


void matchDesc(Mat &descriptor1, Mat &descriptor2, vector<DMatch> &matches){

    // const int d_max = 40;

    for (int i1 = 0; i1 < nfeatures; ++i1) {
        cv::DMatch m{i1, 0, 0, 256};

        for (int i2 = 0; i2 < nfeatures; ++i2) {

            int distance = 0;

            for (int k = 0; k < 8; k++) {
                distance += _mm_popcnt_u32((uint32_t)(descriptor1.at<int32_t>(i1,k) ^ descriptor2.at<int32_t>(i2,k)));
            }

            for (int j = nrBrief/32; j < nrBrief/32 + nSemrBrief/6; j++) {
                for (int k = 0; k < 6; k++) {
                    // cout << ((int)((descriptor1.at<int32_t>(i1,j) >> k*6 ^ descriptor1.at<int32_t>(i2,j) >> k*6) & 63) ? 8 : 0) << endl;
                    distance += ((descriptor1.at<int32_t>(i1,j) >> k*6 ^ descriptor2.at<int32_t>(i2,j) >> k*6) & 63) ? 6 : 0;
                }
            }

            // if (distance < d_max && distance < m.distance) {
            if (distance < m.distance) {
                m.distance = distance;
                m.trainIdx = i2;
            }
        }

        // if (m.distance < d_max) {
        matches.push_back(m);
        // }
    }
}

/* ====== */
/*  Main  */
/* ====== */
/* This program demonstrates how to extract ORB features and perform matching using the OpenCV library. */
int main(int argc, char **argv) {
    cout << "[orb_cv] Hello!" << endl << endl;

    /* Load the images */
    Mat image1 = imread(image1_filepath, IMREAD_COLOR);
    Mat image2 = imread(image2_filepath, IMREAD_COLOR);
    Mat image_gray1(image1.size().height, image1.size().width, CV_8UC1);
    Mat image_gray2(image2.size().height, image2.size().width, CV_8UC1);
    cvtColor(image1, image_gray1, COLOR_BGR2GRAY);
    cvtColor(image2, image_gray2, COLOR_BGR2GRAY);

    Mat semantic1 = imread(semantic1_filepath, IMREAD_COLOR);
    // Mat semantic1color = imread(semantic1color_filepath, IMREAD_COLOR);
    
    Mat semantic2 = imread(semantic2_filepath, IMREAD_COLOR);
    // assert(image1.data != nullptr && image2.data != nullptr);  // FIXME: I think this its not working!

    /* Initialization */
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Mat descriptors_gray1, descriptors_gray2;
    // Mat descriptors_my1, descriptors_my2;

    /* --------------------- */
    /*  Features Extraction  */
    /* --------------------- */
    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //--- Step 1: Detect the position of the Oriented FAST keypoints (Corner Points)
    Timer t1 = chrono::steady_clock::now();
    detector->detect(image_gray1, keypoints1);
    detector->detect(image_gray2, keypoints2);
    Timer t2 = chrono::steady_clock::now();

    //--- Step 2: Calculate the BRIEF descriptors based on the position of Oriented FAST keypoints
    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    
    Mat sem_descriptor1, sem_descriptor2;
    // convertDesc(descriptors1, sem_descriptor1, semantic1);
    // convertDesc(descriptors2, sem_descriptor2, semantic2);
    // computeSemanticDesc(semantic1, keypoints1, sem_descriptor1);
    // computeSemanticDesc(semantic2, keypoints2, sem_descriptor2);


    computeDesc(image_gray1, keypoints1, sem_descriptor1);
    computeDesc(image_gray2, keypoints2, sem_descriptor2);
    Timer t3 = chrono::steady_clock::now();

    // ComputeORB(image1, keypoints1, descriptors1_self);
    // ComputeORB(image2, keypoints2, descriptors2_self);

    // cout << keypoints1.size() << endl;
    cout << descriptors1.row(0) << endl;
    cout << sem_descriptor1.row(0) << endl;
    // cout << "--------------------------------" << endl;
    // cout << descriptors_my1 << endl;
    // cout << descriptors_gray1 << endl;
    // cout << descriptors2 << endl;


    /* ------------------- */
    /*  Features Matching  */
    /* ------------------- */
    //--- Step 3: Match the BRIEF descriptors of the two images using the Hamming distance
    vector<DMatch> matches;

    Timer t4 = chrono::steady_clock::now();
    // matchDesc(sem_descriptor1, sem_descriptor2, matches);
    matcher->match(descriptors1, descriptors2, matches);
    Timer t5 = chrono::steady_clock::now();

    /* -------------------- */
    /*  Features Filtering  */
    /* -------------------- */
    //--- Step 4: Correct matching selection
    /* Calculate the min & max distances */
    /** Parameters: 
     * @param[in] __first – Start of range.
    /* @param[in] __last – End of range.
    /* @param[in] __comp – Comparison functor.
    /* @param[out] make_pair(m,M) Return a pair of iterators pointing to the minimum and maximum elements in a range.
     */
    Timer t6 = chrono::steady_clock::now();
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
        //cout << m1.distance << " " << m2.distance << endl;
        return m1.distance < m2.distance;
    });  

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    /* Perform Filtering */
    // Rule of Thumb: When the distance between the descriptors is greater than 2 times the min distance, we treat the matching
    // as wrong. But sometimes the min distance could be very small, set an experience value of 30 as the lower bound.
    vector<DMatch> goodMatches;

    Timer t7 = chrono::steady_clock::now();
    for (int i=0; i<sem_descriptor1.rows; i++){
        // cout << matches[i].distance << endl;
        if (matches[i].distance <= max(2*min_dist, matches_lower_bound)){
            goodMatches.push_back(matches[i]);
        }
    }
    Timer t8 = chrono::steady_clock::now();

    //--- Step 5: Visualize the Matching result
    Mat outImage1, outImage2;
    Mat outImage_gray1, outImage_gray2;
    Mat image_matches;
    Mat image_goodMatches;

    drawKeypoints(image1, keypoints1, outImage1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image2, keypoints2, outImage2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    // drawKeypoints(image_gray1, keypoints_gray1, outImage_gray1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // drawKeypoints(image_gray2, keypoints_gray2, outImage_gray2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    drawMatches(image1, keypoints1, image2, keypoints2, matches, image_matches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, image_goodMatches,
        Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;


    // /* Results */
    printElapsedTime("ORB Features Extraction: ", t1, t3);
    printElapsedTime(" | Oriented FAST Keypoints detection: ", t1, t2);
    printElapsedTime(" | BRIEF descriptors calculation: ", t2, t3);
    cout << "\n-- Number of detected keypoints1: " << keypoints1.size() << endl;
    cout << "-- Number of detected keypoints2: " << keypoints2.size() << endl << endl;

    printElapsedTime("ORB Features Matching: ", t4, t5);
    cout << "-- Number of matches: " << matches.size() << endl << endl;
    
    printElapsedTime("ORB Features Filtering: ", t6, t8);
    printElapsedTime(" | Min & Max Distances Calculation: ", t6, t7);
    printElapsedTime(" | Filtering by Hamming Distance: ", t7, t8);
    cout << "-- Min dist: " << min_dist << endl;
    cout << "-- Max dist: " << max_dist << endl;
    cout << "-- Number of good matches: " << goodMatches.size() << endl << endl;
    cout << "In total, we get " << goodMatches.size() << "/" << matches.size() << " good pairs of feature points." << endl << endl;

    /* Display */
    //input
    // imshow("image1", image1);
    // imshow("image2", image2);

    imshow("semantic1", semantic1);
    // imshow("semantic1color", semantic1color);
    // imshow("semantic2", semantic2);
    //output
    imshow("outImage1", outImage1);
    // imshow("outImage2", outImage2);

    // imshow("outImage_gray1", outImage_gray1);
    // imshow("outImage_gray2", outImage_gray2);
    imshow("image_matches", image_matches);
    imshow("image_goodMatches", image_goodMatches);
    // cout << "\nPress 'ESC' to exit the program..." << endl;
    waitKey(0);

    /* Save */
    // imwrite("../../orb_features/src/results/orb_cv_goodMatches.png", image_goodMatches);

    cout << "Done." << endl;

    return 0;
}
