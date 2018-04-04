project(Fast_IMAS_IPOL)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

ADD_SUBDIRECTORY(libMatch)
ADD_SUBDIRECTORY(libNumerics)
ADD_SUBDIRECTORY(libOrsa)


FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

IF(MSVC)
ADD_DEFINITIONS(/arch:SSE2)
ENDIF(MSVC)


IF(CMAKE_COMPILER_IS_GNUCXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# Not the MATLAB compiler!!!
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _NOMEX=1")



####### Different Configs For IMAS

set(IMAS_srcs_standalone
    main_standalone.cpp
    imas.cpp IMAS_coverings.cpp

    #TILT SIMULATIONS
    libSimuTilts/digital_tilt.cpp
    libSimuTilts/numerics1.cpp libSimuTilts/frot.cpp libSimuTilts/splines.cpp
    libSimuTilts/fproj.cpp libSimuTilts/library.cpp libSimuTilts/flimage.cpp
    libSimuTilts/filter.cpp


    libMatch/match.cpp

    #SURF
    libLocalDesc/surf/extract_surf.cpp libLocalDesc/surf/descriptor.cpp libLocalDesc/surf/image.cpp
    libLocalDesc/surf/keypoint.cpp libLocalDesc/surf/lib_match_surf.cpp

    #SIFT
    libLocalDesc/sift/demo_lib_sift.cpp

    #ORSA
    libOrsa/orsa_fundamental.cpp libOrsa/conditioning.cpp
    libOrsa/orsa_model.cpp libOrsa/fundamental_model.cpp
    libOrsa/homography_model.hpp libOrsa/homography_model.cpp
    libOrsa/orsa_homography.cpp libOrsa/orsa_homography.hpp

    # Minimal Algebra capabilities
    libNumerics/numerics.cpp

    mex_and_omp.cpp

    io_png/io_png.c
    )

set(IMAS_srcs_standalone_withUSAC
    main_standalone.cpp
    imas.cpp IMAS_coverings.cpp

    #TILT SIMULATIONS
    libSimuTilts/digital_tilt.cpp
    libSimuTilts/numerics1.cpp libSimuTilts/frot.cpp libSimuTilts/splines.cpp
    libSimuTilts/fproj.cpp libSimuTilts/library.cpp libSimuTilts/flimage.cpp
    libSimuTilts/filter.cpp


    libMatch/match.cpp

    #SURF
    libLocalDesc/surf/extract_surf.cpp libLocalDesc/surf/descriptor.cpp libLocalDesc/surf/image.cpp
    libLocalDesc/surf/keypoint.cpp libLocalDesc/surf/lib_match_surf.cpp

    #SIFT
    libLocalDesc/sift/demo_lib_sift.cpp

    #ORSA
    libOrsa/orsa_fundamental.cpp libOrsa/conditioning.cpp
    libOrsa/orsa_model.cpp libOrsa/fundamental_model.cpp
    libOrsa/homography_model.hpp libOrsa/homography_model.cpp
    libOrsa/orsa_homography.cpp libOrsa/orsa_homography.hpp

    # Minimal Algebra capabilities
    libNumerics/numerics.cpp

    mex_and_omp.cpp

    #USAC
    ./libUSAC/config/ConfigFileReader.cpp
    ./libUSAC/config/ConfigParams.cpp
    ./libUSAC/config/ConfigParamsFundmatrix.cpp
    ./libUSAC/config/ConfigParamsHomog.cpp
    ./libUSAC/utils/FundmatrixFunctions.cpp
    ./libUSAC/utils/HomographyFunctions.cpp
    ./libUSAC/utils/MathFunctions.cpp

    io_png/io_png.c
    )

####### END of Different Configs For IMAS


# ACTIVATE USAC - needs libconfig and lapack
set(USAC ON)

message("************* Fast IMAS *************")
ADD_SUBDIRECTORY(io_png)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _NO_OPENCV=1")

include_directories(. ./libOrsa ./libSimuTilts)

if (USAC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _USAC=1")
    message("************* with USAC *************")
    add_executable(main ${IMAS_srcs_standalone_withUSAC})
    include_directories(./libUSAC/config ./libUSAC/estimators ./libUSAC/utils)
    TARGET_LINK_LIBRARIES(main config++ lapack)
else()
    add_executable(main ${IMAS_srcs_standalone})
endif()
include_directories(./io_png ./io_png/libs/png)
TARGET_LINK_LIBRARIES(main png zlib Match Numerics)
