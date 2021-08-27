import numpy as np

cityscapes_pallete = np.array([[128, 64, 128], [244, 35, 231], [69, 69, 69],
                      # 0 = road, 1 = sidewalk, 2 = building
                      [102, 102, 156], [190, 153, 153], [153, 153, 153],
                      # 3 = wall, 4 = fence, 5 = pole
                      [250, 170, 29], [219, 219, 0], [106, 142, 35],
                      # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                      [152, 250, 152], [69, 129, 180], [219, 19, 60],
                      # 9 = terrain, 10 = sky, 11 = person
                      [255, 0, 0], [0, 0, 142], [0, 0, 69],
                      # 12 = rider, 13 = car, 14 = truck
                      [0, 60, 100], [0, 79, 100], [0, 0, 230],
                      # 15 = bus, 16 = train, 17 = motocycle
                      [119, 10, 32]])
                      # 18 = bicycle

cityscapes_pallete = cityscapes_pallete/float(255.0)


cityscapes_pallete_float = np.array([[0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.07843137, 0.07843137, 0.07843137],
                                     [0.43529412, 0.29019608, 0.        ],
                                     [0.31764706, 0.        , 0.31764706],
                                     [0.50196078, 0.25098039, 0.50196078],
                                     [0.95686275, 0.1372549 , 0.90980392],
                                     [0.98039216, 0.66666667, 0.62745098],
                                     [0.90196078, 0.58823529, 0.54901961],
                                     [0.2745098 , 0.2745098 , 0.2745098 ],
                                     [0.4       , 0.4       , 0.61176471],
                                     [0.74509804, 0.6       , 0.6       ],
                                     [0.70588235, 0.64705882, 0.70588235],
                                     [0.58823529, 0.39215686, 0.39215686],
                                     [0.58823529, 0.47058824, 0.35294118],
                                     [0.6       , 0.6       , 0.6       ],
                                     [0.6       , 0.6       , 0.6       ],
                                     [0.98039216, 0.66666667, 0.11764706],
                                     [0.8627451 , 0.8627451 , 0.        ],
                                     [0.41960784, 0.55686275, 0.1372549 ],
                                     [0.59607843, 0.98431373, 0.59607843],
                                     [0.2745098 , 0.50980392, 0.70588235],
                                     [0.8627451 , 0.07843137, 0.23529412],
                                     [1.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.55686275],
                                     [0.        , 0.        , 0.2745098 ],
                                     [0.        , 0.23529412, 0.39215686],
                                     [0.        , 0.        , 0.35294118],
                                     [0.        , 0.        , 0.43137255],
                                     [0.        , 0.31372549, 0.39215686],
                                     [0.        , 0.        , 0.90196078],
                                     [0.46666667, 0.04313725, 0.1254902 ],
                                     [0.        , 0.        , 0.55686275]])

mapillary_pallete_float = np.asarray([[165,  42,  42],
                                    [  0, 192,   0],
                                    [250, 170,  31],
                                    [250, 170,  32],
                                    [196, 196, 196],
                                    [190, 153, 153],
                                    [180, 165, 180],
                                    [ 90, 120, 150],
                                    [250, 170,  33],
                                    [250, 170,  34],
                                    [128, 128, 128],
                                    [250, 170,  35],
                                    [102, 102, 156],
                                    [128,  64, 255],
                                    [140, 140, 200],
                                    [170, 170, 170],
                                    [250, 170,  36],
                                    [250, 170, 160],
                                    [250, 170,  37],
                                    [ 96,  96,  96],
                                    [230, 150, 140],
                                    [128,  64, 128],
                                    [110, 110, 110],
                                    [110, 110, 110],
                                    [244,  35, 232],
                                    [128, 196, 128],
                                    [150, 100, 100],
                                    [ 70,  70,  70],
                                    [150, 150, 150],
                                    [150, 120,  90],
                                    [220,  20,  60],
                                    [220,  20,  60],
                                    [255,   0,   0],
                                    [255,   0, 100],
                                    [255,   0, 200],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [250, 170,  29],
                                    [250, 170,  28],
                                    [250, 170,  26],
                                    [250, 170,  25],
                                    [250, 170,  24],
                                    [250, 170,  22],
                                    [250, 170,  21],
                                    [250, 170,  20],
                                    [255, 255, 255],
                                    [250, 170,  19],
                                    [250, 170,  18],
                                    [250, 170,  12],
                                    [250, 170,  11],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [250, 170,  16],
                                    [250, 170,  15],
                                    [250, 170,  15],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [255, 255, 255],
                                    [ 64, 170,  64],
                                    [230, 160,  50],
                                    [ 70, 130, 180],
                                    [190, 255, 255],
                                    [152, 251, 152],
                                    [107, 142,  35],
                                    [  0, 170,  30],
                                    [255, 255, 128],
                                    [250,   0,  30],
                                    [100, 140, 180],
                                    [220, 128, 128],
                                    [222,  40,  40],
                                    [100, 170,  30],
                                    [ 40,  40,  40],
                                    [ 33,  33,  33],
                                    [100, 128, 160],
                                    [ 20,  20, 255],
                                    [142,   0,   0],
                                    [ 70, 100, 150],
                                    [250, 171,  30],
                                    [250, 172,  30],
                                    [250, 173,  30],
                                    [250, 174,  30],
                                    [250, 175,  30],
                                    [250, 176,  30],
                                    [210, 170, 100],
                                    [153, 153, 153],
                                    [153, 153, 153],
                                    [128, 128, 128],
                                    [  0,   0,  80],
                                    [210,  60,  60],
                                    [250, 170,  30],
                                    [250, 170,  30],
                                    [250, 170,  30],
                                    [250, 170,  30],
                                    [250, 170,  30],
                                    [250, 170,  30],
                                    [192, 192, 192],
                                    [192, 192, 192],
                                    [192, 192, 192],
                                    [220, 220,   0],
                                    [220, 220,   0],
                                    [  0,   0, 196],
                                    [192, 192, 192],
                                    [220, 220,   0],
                                    [140, 140,  20],
                                    [119,  11,  32],
                                    [150,   0, 255],
                                    [  0,  60, 100],
                                    [  0,   0, 142],
                                    [  0,   0,  90],
                                    [  0,   0, 230],
                                    [  0,  80, 100],
                                    [128,  64,  64],
                                    [  0,   0, 110],
                                    [  0,   0,  70],
                                    [  0,   0, 142],
                                    [  0,   0, 192],
                                    [170, 170, 170],
                                    [ 32,  32,  32],
                                    [111,  74,   0],
                                    [120,  10,  10],
                                    [ 81,   0,  81],
                                    [111, 111,   0],
                                    [  0,   0,   0]])

mapillary_pallete_float = mapillary_pallete_float/255.