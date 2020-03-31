project(Fast_IMAS_IPOL)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
SET(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")

include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()
message("Optimized Flags for this machine: ${CMAKE_CXX_FLAGS}")

ADD_SUBDIRECTORY(libMatch)
ADD_SUBDIRECTORY(libNumerics)
ADD_SUBDIRECTORY(libOrsa)

set(OPENMP ON)

if (OPENMP)
message("************* OpenMP *************")
FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)
endif()

IF(MSVC)
ADD_DEFINITIONS(/arch:SSE2)
ENDIF(MSVC)


IF(CMAKE_COMPILER_IS_GNUCXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)



####### Base Source files
set(IMAS_srcs
    main.cpp
    imas.cpp IMAS_coverings.cpp

    #TILT SIMULATIONS
    libSimuTilts/digital_tilt.cpp
    libSimuTilts/numerics1.cpp libSimuTilts/frot.cpp libSimuTilts/splines.cpp
    libSimuTilts/fproj.cpp libSimuTilts/library.cpp libSimuTilts/flimage.cpp
    libSimuTilts/filter.cpp


    libMatch/match.cpp

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


# Not the MATLAB compiler!!!
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _NOMEX=1")

# ACTIVATE USAC - needs libconfig and lapack
set(USAC ON)

# OPENCV Control: ON -> Dont uses opencv ; OFF -> Uses opencv
set(opencv OFF)

# ACTIVATE AC Descriptors
set(ACD OFF)
# ACTIVATE old AC-Q Matcher
set(OLD_ACQ OFF)

# ACTIVATE LDAHASH
set(LDAHASH ON)

# ACTIVATE GDAL
set(GDAL OFF)


if (opencv)
    message("************* OPENCV Descriptors *************")
    set (OpenCV_DIR "/home/rdguez-mariano/Sources/opencv_3.2.0/share/OpenCV")
    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/rdguez-mariano/Sources/opencv_3.2.0/share/OpenCV")
    find_package( OpenCV 3.2 REQUIRED )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _NO_OPENCV=1")
    message("************* Standalone Descriptors *************")

    if (ACD)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _ACD=1")
        message("************* Including AC Descriptors *************")
    endif()

    if (OLD_ACQ)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _OLDACQ=1")
        message("************* Including AC Descriptors *************")
    endif()

    set(IMAS_srcs
        ${IMAS_srcs}

        #SURF
        libLocalDesc/surf/extract_surf.cpp libLocalDesc/surf/descriptor.cpp libLocalDesc/surf/image.cpp
        libLocalDesc/surf/keypoint.cpp libLocalDesc/surf/lib_match_surf.cpp

        #SIFT
        libLocalDesc/sift/demo_lib_sift.cpp
        )

    if (LDAHASH)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _LDAHASH=1")
        message("************* Including LDAHASH Descriptors *************")
        set(IMAS_srcs
            ${IMAS_srcs}
            #SIFT
            libLocalDesc/ldahash/lib_ldahash.cpp
            )

    endif()

endif()

if (USAC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _USAC=1")
    message("************* with USAC *************")
    set(IMAS_srcs
        ${IMAS_srcs}

        #USAC
        ./libUSAC/config/ConfigFileReader.cpp
        ./libUSAC/config/ConfigParams.cpp
        ./libUSAC/config/ConfigParamsFundmatrix.cpp
        ./libUSAC/config/ConfigParamsHomog.cpp
        ./libUSAC/utils/FundmatrixFunctions.cpp
        ./libUSAC/utils/HomographyFunctions.cpp
        ./libUSAC/utils/MathFunctions.cpp
        )
    include_directories(./libUSAC/config ./libUSAC/estimators ./libUSAC/utils)
endif()



ADD_SUBDIRECTORY(io_png)
include_directories(./io_png ./io_png/libs/png)
include_directories(. ./libOrsa ./libSimuTilts)

add_executable(main ${IMAS_srcs})

if (USAC)
    TARGET_LINK_LIBRARIES(main config++ lapack)
endif()

if (opencv)
    TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})
endif()

TARGET_LINK_LIBRARIES(main png zlib Match Numerics)

if (GDAL)
    find_package( GDAL )
    include_directories( ${GDAL_INCLUDE_DIRS} )
    message("************* Including GDAL *************")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _GDAL=1")
    TARGET_LINK_LIBRARIES(main gdal)
endif()
